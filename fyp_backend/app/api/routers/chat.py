"""
chat.py — Chat endpoint router
═════════════════════════════════════════════════════════

Handles free-form conversation with the travel-planning agent.
Reads prior session state from LangGraph's Postgres checkpointer
(via ``thread_id``) and streams back the AI reply + draft itinerary.
"""

from __future__ import annotations

import asyncio
import copy
from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
import math
import threading
from collections.abc import AsyncIterator
from typing import Annotated, Any, Final, Literal, Union

from fastapi import APIRouter, Query, Request, Response
from pydantic import BaseModel, Field, field_validator

from app.api.dependencies import (
    RequestId,
    CurrentUser,
    invoke_chat_authenticated,
)
from app.api.errors import (
    PLANNING_UNAVAILABLE_MESSAGE,
    map_exception_to_http,
    planning_unavailable_reason,
)
from app.core.supabase_db import fetch_chat_history_checkpoints
from app.schemas.requests import ChatRequest
from app.schemas.responses import (
    BudgetConfirmationResponse,
    BudgetEvidenceResponse,
    CountryCode,
    DailyItinerary,
    GeoJSONFeatureCollection,
    PlanningUnavailableReason,
    canonicalize_public_daily_maps,
    validated_accepted_plan,
)
from app.services.budget_assessment import (
    BudgetAssessment,
    assessment_matches_trip,
    is_budget_sufficient,
    load_confirmed_budget_assessment,
    primary_destination_city,
)
from app.services.planning_transaction import safe_observability_log
from app.services.output_review import (
    OutputReviewContext,
    deterministic_output_review,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


# ── Response schema ─────────────────────────────────────────────────
class ChatSuccessResponse(BaseModel):
    """A freshly revalidated accepted plan and safe assistant reply."""

    model_config = {"extra": "forbid"}

    status: Literal["success"] = "success"
    chat_reply: str = ""
    draft_itinerary: list[DailyItinerary]
    daily_map_info: dict[int, GeoJSONFeatureCollection] = Field(default_factory=dict)
    destination_country_code: CountryCode
    itinerary_modified: bool = False
    total_budget: float = 0.0
    currency: str = ""
    budget_allocation: dict[str, float] = Field(default_factory=dict)
    budget_confirmation: None = None

    @field_validator("daily_map_info", mode="before")
    @classmethod
    def _canonical_map_keys(cls, value: Any) -> dict[int, Any]:
        return canonicalize_public_daily_maps(value)


class ChatBudgetGateResponse(BaseModel):
    """Existing budget-gate wire shape, explicitly free of plan data."""

    model_config = {"extra": "forbid"}

    status: Literal["budget_confirmation_required", "budget_check_unavailable"]
    chat_reply: str
    draft_itinerary: list[DailyItinerary] = Field(default_factory=list, max_length=0)
    daily_map_info: dict[int, Any] = Field(default_factory=dict, max_length=0)
    itinerary_modified: Literal[False] = False
    total_budget: Literal[0.0] = 0.0
    currency: Literal[""] = ""
    budget_allocation: dict[str, float] = Field(default_factory=dict, max_length=0)
    budget_confirmation: BudgetConfirmationResponse | None = None


class ChatPlanningUnavailableResponse(BaseModel):
    """Chat-compatible unavailable shape with no rejected planning payload."""

    model_config = {"extra": "forbid"}

    status: Literal["planning_unavailable"]
    reason: PlanningUnavailableReason
    chat_reply: str
    retryable: Literal[True] = True
    draft_itinerary: list[DailyItinerary] = Field(default_factory=list, max_length=0)
    daily_map_info: dict[int, Any] = Field(default_factory=dict, max_length=0)
    itinerary_modified: Literal[False] = False
    total_budget: Literal[0.0] = 0.0
    currency: Literal[""] = ""
    budget_allocation: dict[str, float] = Field(default_factory=dict, max_length=0)
    budget_confirmation: None = None


ChatResponse = Annotated[
    Union[
        ChatSuccessResponse,
        ChatBudgetGateResponse,
        ChatPlanningUnavailableResponse,
    ],
    Field(discriminator="status"),
]


class ChatHistorySession(BaseModel):
    """A persisted conversation reconstructed from its latest checkpoint."""

    id: str
    title: str
    destination: str = ""
    updated_at: str
    messages: list[dict[str, Any]] = Field(default_factory=list)


class ChatHistoryResponse(BaseModel):
    status: str = "success"
    sessions: list[ChatHistorySession] = Field(default_factory=list)


@dataclass(slots=True)
class _SessionLockEntry:
    lock: asyncio.Lock
    references: int = 0


class _SessionMutationLockRegistry:
    """Reference-counted async locks keyed by authenticated trip session.

    This registry is intentionally process-local for the single-worker
    ``run.py`` deployment. Multiple workers require a distributed/advisory
    lock using the same owner/session key.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], _SessionLockEntry] = {}
        self._registry_guard = threading.Lock()

    @asynccontextmanager
    async def hold(self, owner: str, session_id: str) -> AsyncIterator[None]:
        key = (owner, session_id)
        with self._registry_guard:
            entry = self._entries.get(key)
            if entry is None:
                entry = _SessionLockEntry(lock=asyncio.Lock())
                self._entries[key] = entry
            entry.references += 1

        acquired = False
        try:
            await entry.lock.acquire()
            acquired = True
            yield
        finally:
            if acquired:
                entry.lock.release()
            with self._registry_guard:
                entry.references -= 1
                if entry.references == 0 and self._entries.get(key) is entry:
                    self._entries.pop(key, None)


_SESSION_MUTATION_LOCKS = _SessionMutationLockRegistry()


# ── Helpers ─────────────────────────────────────────────────────────
_EMPTY_TUPLE: Final[tuple] = ()
_UNSAFE_HISTORY_MESSAGE: Final[str] = (
    "A saved assistant response was hidden because it could not be safely displayed."
)

# Tools whose execution means the trip plan (itinerary, budget or core
# requirements) changed this turn.
_PLAN_MODIFYING_TOOLS: Final[frozenset[str]] = frozenset(
    {
        "edit_itinerary",
        "modify_existing_booking",
        "update_total_budget",
        "update_budget_category",
        "update_trip_details",
    }
)


def _plan_modified_this_turn(messages: Any) -> bool:
    """True if a plan-modifying tool ran after the latest human message."""
    for msg in reversed(messages or _EMPTY_TUPLE):
        msg_type = getattr(msg, "type", "")
        if msg_type == "human":
            break
        if msg_type == "tool" and getattr(msg, "name", "") in _PLAN_MODIFYING_TOOLS:
            return True
    return False


def _extract_text(content: Any) -> str:
    """
    Robustly extract a plain-text representation from a LangChain message's
    ``content`` field, which may be:

      * a plain ``str``
      * a ``list`` of parts: ``str | {"text": ...} | {"type": "text", "text": ...}``
      * no arbitrary objects or non-text content blocks

    Optimised for the hot path: avoids re-entering Python loops when content
    is already a string (the common case for text-only LLM responses).
    """
    # Fast path — covers ~95% of responses
    content_type = type(content)
    if content_type is str:
        return content

    if content_type is list:
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
                continue
            if (
                isinstance(part, dict)
                and set(part).issubset({"type", "text"})
                and part.get("type", "text") == "text"
                and isinstance(part.get("text"), str)
            ):
                parts.append(part["text"])
                continue
            return ""
        return " ".join(parts) if parts else ""

    return ""


def _current_ai_reply(messages: Any) -> str | None:
    """Select a completed final AI message after the current human turn."""
    if not isinstance(messages, (list, tuple)) or not messages:
        return None
    last_message = messages[-1]
    if getattr(last_message, "type", "") != "ai":
        return None

    latest_human_index = next(
        (
            index
            for index in range(len(messages) - 2, -1, -1)
            if getattr(messages[index], "type", "") == "human"
        ),
        None,
    )
    if latest_human_index is None:
        return None

    reply = _extract_text(getattr(last_message, "content", None)).strip()
    return reply or None


def _destination_label(state: dict[str, Any]) -> str:
    """Build the same human-readable destination label used by the UI."""
    country = str(state.get("country") or "").strip()
    raw_cities = state.get("city") or []
    if isinstance(raw_cities, str):
        cities = [raw_cities.strip()] if raw_cities.strip() else []
    else:
        cities = [str(city).strip() for city in raw_cities if str(city).strip()]
    return ", ".join([*cities, country] if country else cities)


_BUDGET_CHECK_UNAVAILABLE_MESSAGE: Final[str] = (
    "Unable to verify the budget right now. Your existing trip plan has not "
    "been changed."
)
_EXPIRED_BUDGET_CONFIRMATION_MESSAGE: Final[str] = (
    "This budget check expired or no longer matches the trip. Request a refreshed "
    "recommendation before planning."
)


def _public_accepted_plan_snapshot(
    raw: Any,
    *,
    state: dict[str, Any],
) -> dict[str, Any] | None:
    """Revalidate and project only the explicit accepted snapshot."""
    accepted = validated_accepted_plan(
        {**state, "accepted_plan_snapshot": raw},
    )
    return accepted.model_dump(mode="python") if accepted is not None else None


def _deterministic_confirmation_message(
    confirmation: BudgetConfirmationResponse,
) -> str:
    """Construct public copy only from the validated confirmation projection."""
    minimum = (
        f"{confirmation.base_currency} "
        f"{confirmation.recommended_minimum_budget:,.2f}"
    )
    if confirmation.reason == "recommendation_requested":
        return (
            f"The provider-grounded minimum is {minimum}. Use the recommended "
            "budget to continue planning."
        )
    return (
        f"{confirmation.base_currency} {confirmation.stated_budget:,.2f} is below "
        f"the provider-grounded minimum of {minimum}. Use the recommended budget "
        "to continue planning."
    )


def _public_budget_confirmation(
    raw: Any,
    *,
    state: dict[str, Any],
) -> BudgetConfirmationResponse | None:
    """Validate checkpoint data and expose only the confirmation-safe fields."""
    if not isinstance(raw, dict):
        return None
    try:
        assessment = BudgetAssessment.model_validate(raw["assessment"])
        if not assessment_matches_trip(
            assessment,
            origin=str(state.get("origin_country") or ""),
            destination=str(state.get("country") or ""),
            destination_city=primary_destination_city(state.get("city")),
            start_date=str(state.get("start_date") or ""),
            end_date=str(state.get("end_date") or ""),
            num_people=max(1, int(state.get("num_people") or 1)),
        ):
            return None
        evidence = BudgetEvidenceResponse(
            outbound_flight_price=assessment.evidence.outbound_flight_price,
            return_flight_price=assessment.evidence.return_flight_price,
            hotel_price_per_night=assessment.evidence.hotel_price_per_night,
            hotel_nights=assessment.evidence.hotel_nights,
        )
        if (
            raw["budget_assessment_id"] != assessment.assessment_id
            or raw["recommended_minimum_budget"]
            != assessment.recommended_minimum_budget
            or raw["base_currency"] != assessment.base_currency
            or raw["destination_currency"] != assessment.destination_currency
            or raw["expires_at"] != assessment.expires_at
            or raw["evidence"] != evidence.model_dump()
        ):
            return None
        reason = raw["reason"]
        if reason not in {"insufficient_budget", "recommendation_requested"}:
            return None
        stated_budget = raw.get("stated_budget")
        if stated_budget is not None:
            if isinstance(stated_budget, bool) or not isinstance(
                stated_budget, (int, float)
            ):
                return None
            normalized_stated_budget = float(stated_budget)
            if (
                not math.isfinite(normalized_stated_budget)
                or normalized_stated_budget <= 0
            ):
                return None
            stated_budget = normalized_stated_budget
        if reason == "recommendation_requested" and stated_budget is not None:
            return None
        if reason == "insufficient_budget" and (
            stated_budget is None
            or is_budget_sufficient(stated_budget, assessment)
        ):
            return None
        persisted = load_confirmed_budget_assessment(
            assessment_id=assessment.assessment_id,
            origin=str(state.get("origin_country") or ""),
            destination=str(state.get("country") or ""),
            destination_city=primary_destination_city(state.get("city")),
            start_date=str(state.get("start_date") or ""),
            end_date=str(state.get("end_date") or ""),
            num_people=max(1, int(state.get("num_people") or 1)),
        )

        if persisted is None or persisted.model_dump() != assessment.model_dump():
            return None

        # Preserve the verified planning cities associated with this chat session.
        resolved_cities = [
            str(city).strip()
            for city in (state.get("city") or [])
            if str(city).strip()
        ]

        if not resolved_cities:
            return None

        return BudgetConfirmationResponse(
            status="budget_confirmation_required",
            reason=reason,
            chat_reply="",
            budget_assessment_id=assessment.assessment_id,
            resolved_cities=resolved_cities,
            stated_budget=stated_budget,
            recommended_minimum_budget=assessment.recommended_minimum_budget,
            base_currency=assessment.base_currency,
            destination_currency=assessment.destination_currency,
            expires_at=assessment.expires_at,
            evidence=evidence,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _budget_gate_state(
    state: dict[str, Any],
) -> tuple[
    Literal["ordinary", "accepted", "confirmation", "unavailable", "malformed"],
    BudgetConfirmationResponse | None,
    str | None,
]:
    """Classify checkpoint gate fields before exposing a chat response."""
    outcome = state.get("budget_gate_outcome")
    raw_pending = state.get("pending_budget_confirmation")

    if outcome is None:
        return (
            ("ordinary", None, None)
            if raw_pending is None
            else ("malformed", None, None)
        )
    if outcome == "accepted":
        return (
            ("accepted", None, None)
            if raw_pending is None
            else ("malformed", None, None)
        )
    if outcome == "budget_confirmation_required":
        confirmation = _public_budget_confirmation(raw_pending, state=state)
        if confirmation is not None:
            return (
                "confirmation",
                confirmation,
                _deterministic_confirmation_message(confirmation),
            )
        return "malformed", None, None
    if outcome == "budget_check_unavailable":
        if raw_pending is None:
            return "unavailable", None, _BUDGET_CHECK_UNAVAILABLE_MESSAGE
        return "malformed", None, None
    return "malformed", None, None


def _serialize_history_session(record: dict[str, Any]) -> ChatHistorySession:
    """Convert LangChain messages/state into the frontend's display shape."""
    state = record.get("state") or {}
    snapshot = _public_accepted_plan_snapshot(
        state.get("accepted_plan_snapshot"),
        state=state,
    )
    messages: list[dict[str, Any]] = []
    first_ai_index: int | None = None
    plan_ai_indices: dict[int, int] = {}
    latest_user_request = "Restore this saved conversation safely."

    for message in state.get("messages") or _EMPTY_TUPLE:
        message_type = getattr(message, "type", "")
        if message_type == "tool":
            continue
        if message_type not in {"human", "ai"}:
            continue

        content = _extract_text(getattr(message, "content", "")).strip()
        # Tool-calling AI messages commonly have empty content; the completed
        # response later in the same turn is the one users should see.
        if not content:
            continue

        role = "user" if message_type == "human" else "ai"
        if role == "user":
            latest_user_request = content
        else:
            decision = deterministic_output_review(
                OutputReviewContext(
                    latest_user_request=latest_user_request,
                    trusted_requirements={
                        key: state.get(key)
                        for key in (
                            "origin_country",
                            "country",
                            "city",
                            "num_people",
                            "start_date",
                            "end_date",
                            "base_currency_code",
                            "dest_currency_code",
                            "total_base_budget",
                            "total_convert_budget",
                            "budget_allocation",
                        )
                    },
                    normalized_tool_evidence={},
                    proposed_reply=content,
                    candidate_plan=copy.deepcopy(snapshot),
                    deterministic_report={"issues": []},
                    review_stage="history_restore",
                )
            )
            if not decision.approved:
                content = _UNSAFE_HISTORY_MESSAGE
        messages.append({"role": role, "content": content})
        if role == "ai":
            current_index = len(messages) - 1
            if first_ai_index is None:
                first_ai_index = current_index
            message_revision = getattr(message, "additional_kwargs", {}).get(
                "plan_revision"
            )
            if (
                isinstance(message_revision, int)
                and not isinstance(message_revision, bool)
                and message_revision > 0
            ):
                plan_ai_indices[message_revision] = current_index

    gate_state, public_confirmation, gate_message = _budget_gate_state(state)
    latest_ai_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if messages[index]["role"] == "ai"
        ),
        None,
    )
    if latest_ai_index is not None:
        if gate_state == "confirmation":
            messages[latest_ai_index]["content"] = gate_message or ""
            messages[latest_ai_index]["budget_confirmation"] = (
                public_confirmation.model_dump(mode="json")
            )
        elif (
            state.get("budget_gate_outcome") == "budget_confirmation_required"
            and state.get("pending_budget_confirmation") is not None
        ):
            messages[latest_ai_index]["content"] = _EXPIRED_BUDGET_CONFIRMATION_MESSAGE
        elif gate_state in {"unavailable", "malformed"}:
            messages[latest_ai_index]["content"] = (
                gate_message or _BUDGET_CHECK_UNAVAILABLE_MESSAGE
            )

    # Checkpoints retain the latest itinerary, not an itinerary snapshot for
    # every turn. Attach it to the most recent plan-changing response (or the
    # initial planning response) so restored chats use the existing renderer.
    if snapshot is not None:
        itinerary = snapshot["draft_itinerary"]
        maps = snapshot["daily_map_info"]
        total_budget = snapshot["total_convert_budget"]
        currency = snapshot["dest_currency_code"]
        allocation = snapshot["budget_allocation"]
    else:
        itinerary = []
        maps = {}
        total_budget = 0.0
        currency = ""
        allocation = {}
    attach_index = None
    if snapshot is not None:
        attach_index = plan_ai_indices.get(snapshot["plan_revision"])
        if attach_index is None and not plan_ai_indices:
            attach_index = first_ai_index
    if (
        snapshot is not None
        and attach_index is not None
        and (itinerary or maps)
    ):
        messages[attach_index]["itinerary"] = itinerary
        messages[attach_index]["maps"] = maps
        messages[attach_index]["budget"] = {
            "total": total_budget,
            "currency": currency,
            "allocation": allocation,
        }

    destination = _destination_label(state)
    first_user = next(
        (message["content"] for message in messages if message["role"] == "user"),
        "",
    )
    title = destination or " ".join(first_user.split())[:48] or "New trip"
    return ChatHistorySession(
        id=str(record.get("session_id") or ""),
        title=title,
        destination=destination,
        updated_at=str(record.get("updated_at") or ""),
        messages=messages,
    )


@router.get(
    "/history",
    response_model=ChatHistoryResponse,
    status_code=200,
    summary="List persisted chat sessions for the current user",
)
async def get_chat_history(
    user_id: CurrentUser,
    response: Response,
    limit: int = Query(default=10, ge=1, le=20),
) -> ChatHistoryResponse:
    """Load each conversation's latest state from Supabase checkpoints."""
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Vary"] = "X-User-ID"
    try:
        records = await asyncio.to_thread(
            fetch_chat_history_checkpoints,
            user_id,
            limit,
        )
    except Exception as exc:
        safe_observability_log(
            logger,
            "chat.history.error",
            stage="history",
            issue_codes=("storage.failed",),
            outcome="error",
        )
        raise map_exception_to_http(exc) from exc

    sessions = [
        _serialize_history_session(record)
        for record in records
        if record.get("session_id")
    ]
    safe_observability_log(
        logger,
        "chat.history.done",
        stage="history",
        outcome="completed",
    )
    return ChatHistoryResponse(sessions=sessions)


async def _invoke_and_project_chat(
    user_id: CurrentUser,
    chat: ChatRequest,
    request_id: RequestId,
) -> ChatResponse:
    """Run and serialize one mutation while its session lock is held."""
    try:
        invoke_kwargs: dict[str, Any] = {
            "user_message": chat.user_message,
            "thread_id": chat.session_id,
            "user_id": user_id,
            "request_id": request_id,
        }
        if chat.budget_action is not None:
            invoke_kwargs["budget_action"] = chat.budget_action
            invoke_kwargs["budget_assessment_id"] = chat.budget_assessment_id
        final_state: dict[str, Any] = await invoke_chat_authenticated(
            **invoke_kwargs,
        )
    except Exception as exc:
        safe_observability_log(
            logger,
            "chat.message.error",
            request_id=request_id,
            session_id=chat.session_id,
            stage="chat",
            issue_codes=("workflow.failed",),
            outcome="error",
        )
        raise map_exception_to_http(exc) from exc

    messages = final_state.get("messages") or _EMPTY_TUPLE
    ai_reply = _current_ai_reply(messages)

    safe_observability_log(
        logger,
        "chat.message.done",
        request_id=request_id,
        session_id=chat.session_id,
        stage="chat",
        outcome="completed",
    )

    gate_state, confirmation, gate_message = _budget_gate_state(final_state)
    if gate_state == "confirmation":
        assert confirmation is not None
        confirmation = confirmation.model_copy(update={"chat_reply": gate_message})
        return ChatBudgetGateResponse(
            status="budget_confirmation_required",
            chat_reply=gate_message or _BUDGET_CHECK_UNAVAILABLE_MESSAGE,
            budget_confirmation=confirmation,
        )
    if gate_state in {"unavailable", "malformed"}:
        return ChatBudgetGateResponse(
            status="budget_check_unavailable",
            chat_reply=gate_message or _BUDGET_CHECK_UNAVAILABLE_MESSAGE,
        )

    if final_state.get("planning_outcome") == "unavailable":
        return ChatPlanningUnavailableResponse(
            status="planning_unavailable",
            reason=planning_unavailable_reason(final_state),
            chat_reply=PLANNING_UNAVAILABLE_MESSAGE,
        )

    accepted = validated_accepted_plan(final_state)
    if accepted is None or ai_reply is None:
        return ChatPlanningUnavailableResponse(
            status="planning_unavailable",
            reason="validation_failed",
            chat_reply=PLANNING_UNAVAILABLE_MESSAGE,
        )

    return ChatSuccessResponse(
        status="success",
        chat_reply=ai_reply,
        draft_itinerary=accepted.draft_itinerary,
        daily_map_info=accepted.daily_map_info,
        destination_country_code=accepted.destination_country_code,
        itinerary_modified=bool(
            final_state.get("_itinerary_modified")
        ),
        total_budget=accepted.total_convert_budget,
        currency=accepted.dest_currency_code,
        budget_allocation=accepted.budget_allocation,
    )


# ── Route ───────────────────────────────────────────────────────────
@router.post(
    "/message",
    response_model=ChatResponse,
    status_code=200,
    summary="Send a message to the travel-planning agent",
)
async def chat_with_ai(
    user_id: CurrentUser,
    chat: ChatRequest,
    request: Request,
    request_id: RequestId,
) -> ChatResponse:
    """Serialize one authenticated session mutation through response projection."""
    del request
    safe_observability_log(
        logger,
        "chat.message.start",
        request_id=request_id,
        session_id=chat.session_id,
        stage="chat",
        outcome="started",
    )

    async with _SESSION_MUTATION_LOCKS.hold(str(user_id), chat.session_id):
        return await _invoke_and_project_chat(user_id, chat, request_id)
