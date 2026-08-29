"""
graph.py — LangGraph Workflow for the AI Travel Agent
═══════════════════════════════════════════════════════

Defines the complete state machine that powers:
  1. Initial trip setup   (form.py  → POST /form)
  2. Ongoing conversation  (chat.py  → POST /chat)

Graph Topology
──────────────

  New trip: START → process_initial_form → prepare_planning_transaction
                  → execute_validated_transaction → generate_reviewed_response
                  → promote_candidate / END

  Ongoing chat: START → memory_extraction → agent → reviewed final / tools
                                            ▲                    │
                                            └── post-processing ─┘

  Full replans enter ``prepare_planning_transaction`` before provider work.
  Incremental edits become private validated candidates and share the same
  review plus atomic-promotion boundary.
"""

from __future__ import annotations

from langgraph.checkpoint.postgres import PostgresSaver

import asyncio
import atexit
import copy
import json
import logging
import math
import pycountry
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack, ExitStack
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

# ── LangGraph ──────────────────────────────────────────
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# ── psycopg (checkpoint-migration self-healing + pooling) ──
from psycopg import OperationalError
from psycopg.errors import DuplicateColumn, DuplicateObject, DuplicateTable
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

# ── LangChain ──────────────────────────────────────────
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# ── Internal: State & Config ───────────────────────────
from app.agents.state import AgentState
from app.core.config import settings
from app.services.budget_assessment import (
    DEFAULT_BUDGET_RATIOS,
    BudgetAssessment,
    assessment_matches_trip,
    is_budget_sufficient,
    load_confirmed_budget_assessment,
    primary_destination_city,
)
from app.services.chat_budget_gate import ChatBudgetDecision, evaluate_chat_budget
from app.services.itinerary_quality import (
    TripRequirements,
    validate_itinerary_candidate,
)
from app.services.output_review import (
    OutputReviewContext,
    OutputReviewDecision,
    deterministic_output_review,
    is_read_only_itinerary_history_request,
    review_public_output,
)
from app.services.planning_transaction import (
    build_validated_plan,
    planning_observability_context,
    safe_observability_log,
)
from app.core.session_access import (
    SessionAccessDenied,
    backfill_existing_session_owner,
    complete_new_session_reservation,
    finalize_expired_new_session_reservation,
    reclaim_expired_new_session_reservation,
    release_new_session_reservation,
    reserve_new_session,
    verify_session_owner,
)

# ── Internal: Prompts & Memory ─────────────────────────
from app.agents.prompts import (
    build_budget_decision_messages,
    build_prompt_messages,
)
from app.agents.itinerary_history import (
    append_previous_itinerary,
    recover_itinerary_history,
)
from app.memory.extractor import run_memory_extraction_task
from app.core.supabase_db import fetch_user_memory_context

# ── Internal: Currency Pipeline & Tools ────────────────
from app.tools.currency import (
    currency_pipeline,
    chat_currency_conversion,
    update_total_budget,  # noqa: F401 - preserved for legacy checkpoint imports
    update_budget_category,
)

# ── Internal: Flights, Hotels & Tools ──────────────────
from app.tools.flights_hotels import (
    plan_flight_hotel,
    search_alternative_opt,
    edit_itinerary,
)

# ── Internal: Trip-requirement updates ─────────────────
from app.tools.trip_details import update_trip_details
from app.tools.budget_chat import propose_budget_change, confirm_recommended_budget

# ── Internal: Places / Attractions & Tools ─────────────
from app.tools.attractions import (
    plan_activities,
    search_places,
    search_places_provider,
)

# ── Internal: Mapbox & Tools ───────────────────────────
from app.tools.mapbox import (
    generate_daily_map,
    search_nearby_amenities,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# 1. CONSTANTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════

# LLM configuration
_LLM_MODEL: str = settings.GEMINI_CHAT_MODEL
_LLM_TEMPERATURE: float = 0.2
_LLM_MAX_RETRIES: int = 3

# Graph recursion limit — max node transitions per invoke
_DEFAULT_RECURSION_LIMIT: int = 50

# Hard ceilings on a single graph invocation. Without these a hung
# LLM/API/DB call leaves the HTTP request (and the frontend) frozen forever;
# asyncio.TimeoutError is mapped to HTTP 504 by app/api/errors.py.
_CHAT_INVOKE_TIMEOUT_S: float = 240.0
_NEW_TRIP_INVOKE_TIMEOUT_S: float = 480.0
_SESSION_CLEANUP_ATTEMPTS: int = 3

# Background memory-extraction thread pool — bounded to prevent
# thread explosion under high concurrency.
_MEMORY_POOL_MAX_WORKERS: int = 4

_PLAN_MODIFYING_TOOLS = {
    "edit_itinerary",
    "modify_existing_booking",
    "update_budget_category",
    "update_trip_details",
}
_BUDGET_PROPOSAL_TOOLS = {
    "propose_budget_change",
    "confirm_recommended_budget",
}
_NON_ACCEPTED_BUDGET_OUTCOMES = {
    "budget_confirmation_required",
    "budget_check_unavailable",
}
_BUDGET_GATE_SAFE_FALLBACK = (
    "Unable to verify the budget right now. Your existing trip plan has not "
    "been changed."
)
_BUDGET_CONFIRMATION_SAFE_COPY = (
    "A provider-grounded budget confirmation is required before planning can "
    "continue."
)
_BUDGET_REPLAN_STAGES = {
    "assessment",
    "currency",
    "itinerary",
    "activities",
    "maps",
    "commit",
}
_BUDGET_REPLAN_ROUTE = {
    "currency": "prepare_budget_replan",
    "itinerary": "prepare_budget_replan",
    "activities": "prepare_budget_replan",
    "maps": "prepare_budget_replan",
    "commit": "prepare_budget_replan",
}
_CANDIDATE_FINANCIAL_FIELDS = {
    "base_currency_code",
    "dest_currency_code",
    "exchange_rate",
    "total_convert_budget",
    "budget_allocation",
    "currency_fetched_at",
}
_OUTPUT_REVIEW_MAX_ATTEMPTS = 3

_EXPLICIT_MUTATION_REQUEST_PATTERN = re.compile(
    # Direct command, optionally prefixed by an explicit itinerary day:
    # "replace activity 1 ...", "Day 1, replace ...", "On Day 2: swap ...".
    r"^\s*(?:"
    r"(?:(?:on|for)\s+)?"
    r"(?:day\s*(?:#\s*)?0*[1-9]\d*|[1-9]\d*(?:st|nd|rd|th)\s+day)"
    r"\s*[,;:\-]?\s*"
    r")?"
    r"(?:please\s+)?"
    r"(?:add|change|edit|modify|move|remove|replace|reschedule|swap|delete)\b"
    r"|"
    r"\b(?:can|could|would|will)\s+you\s+(?:please\s+)?"
    r"(?:add|change|edit|modify|move|remove|replace|reschedule|swap|delete)\b"
    r"|"
    r"\b(?:i\s+want|i(?:'d|\s+would)\s+like)\s+(?:you\s+to\s+)?"
    r"(?:add|change|edit|modify|move|remove|replace|reschedule|swap|delete)\b",
    re.IGNORECASE,
)

_ITINERARY_DISLIKE_MUTATION_PATTERN = re.compile(
    r"\bi\s+(?:also\s+)?"
    r"(?:(?:do\s+not|don't|dont)\s+(?:like|want)|dislike|hate)\b"
    r".{0,96}\b(?:activit(?:y|ies)|attraction|restaurant|hotel|flight|booking|stop)\b",
    re.IGNORECASE | re.DOTALL,
)

_PLACE_ITEM_REFERENCE_PATTERN = re.compile(
    r"\b(?:activity|restaurant)\s*(?:#\s*)?\d+\b"
    r"|\b(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"\d+(?:st|nd|rd|th))\s+(?:activity|restaurant)\b",
    re.IGNORECASE,
)

_PLACE_EDIT_INTENT_PATTERN = re.compile(
    r"\b(?:remove|delete|replace|change|swap)\b"
    r"|\bi\s+(?:also\s+)?"
    r"(?:(?:do\s+not|don't|dont)\s+(?:like|want)|dislike|hate)\b",
    re.IGNORECASE,
)

_EXPLICIT_DAY_REFERENCE_PATTERN = re.compile(
    r"\bday\s*(?:#\s*)?0*[1-9]\d*\b"
    r"|\b[1-9]\d*(?:st|nd|rd|th)\s+day\b",
    re.IGNORECASE,
)

_AFFIRMATIVE_CONFIRMATION_PATTERN = re.compile(
    r"^\s*(?:yes|yes\s+please|sure|okay|ok|go\s+ahead|"
    r"please\s+do|do\s+it)\b",
    re.IGNORECASE,
)

_DAY_CLARIFICATION_REPLY_PATTERN = re.compile(
    r"^\s*(?:(?:i\s+mean|it(?:'s|\s+is))\s+)?"
    r"(?:day\s*(?:#\s*)?)?0*(?P<day>[1-9]\d*)\s*[.!]?\s*$",
    re.IGNORECASE,
)

_PLANNING_TRANSACTION_MAX_ATTEMPTS = 3

_OUTPUT_REVIEW_SAFE_FALLBACK = (
    "I couldn't verify a complete response for that request, so no "
    "unverified changes were saved to your trip. Please try again or "
    "provide the specific trip detail you would like me to update."
)

_OUTPUT_REVIEW_READ_ONLY_FALLBACK = (
    "I couldn't verify that historical trip detail from the trusted itinerary "
    "information available in this conversation. Your current itinerary has "
    "not been changed."
)

def _qualified_plan_fallback_text(
    state: AgentState,
) -> str:
    """Return safe public text appropriate to the current planning action."""

    if _current_turn_used_tool(
        state,
        "edit_itinerary",
    ):
        return (
            "Your requested itinerary change has been applied successfully. "
            "The affected trip details, route, map, and totals have been "
            "updated where necessary, while unrelated itinerary items "
            "have been preserved."
        )

    return (
        "Your itinerary has been generated successfully. "
        "The latest trip details, activities, accommodation, routes, "
        "and estimated costs are available in the itinerary."
    )

# ═══════════════════════════════════════════════════════════
# 2. LLM & TOOLS SETUP
# ═══════════════════════════════════════════════════════════

# ── Tool Registry ──────────────────────────────────────
# `edit_itinerary` supersedes `modify_existing_booking` (it also handles
# activities/restaurants and multiple edits per turn); the older tool is kept
# importable for backward-compatibility but is no longer bound to the LLM.
tools = [
    search_alternative_opt,
    search_places,
    edit_itinerary,
    search_nearby_amenities,
    chat_currency_conversion,
    propose_budget_change,
    update_budget_category,
    update_trip_details,
]

# ── Primary Reasoning Model ────────────────────────────
llm = ChatGoogleGenerativeAI(
    model=_LLM_MODEL,
    temperature=_LLM_TEMPERATURE,
    google_api_key=settings.GEMINI_API_KEY,
    max_retries=_LLM_MAX_RETRIES,
    # Keep Gemini "thinking" minimal for the tool-calling agent.
    #
    # NOTE: this does NOT fix the thought_signature 400. Gemini 3.x models
    # ignore `thinking_budget` (they use `thinking_level`) and cannot disable
    # thinking, so function calls always carry a `thought_signature` that must
    # be echoed back on the next turn — and langchain-google-genai 2.1.x drops
    # it. The actual fix is `_flatten_tool_history()` in prompts.py, which
    # replays tool turns as plain text so no functionCall parts (and thus no
    # signatures) are ever sent back. Both this flag and the flattening can be
    # removed after upgrading to langchain-core>=1.0 + langchain-google-genai
    # >=3.0, which round-trip signatures natively.
    thinking_budget=0,
)

llm_with_tools = llm.bind_tools(tools)

# Once a current-turn place search has completed, the only remaining
# plan-mutating action for an activity/restaurant add/replace is
# edit_itinerary. Restrict the recovery model to that tool so it cannot
# wander into unrelated tools or produce a prose-only success claim.
edit_continuation_llm = llm.bind_tools([edit_itinerary])

budget_decision_tools = [
    propose_budget_change,
    confirm_recommended_budget,
]

budget_decision_llm = llm.bind_tools(
    budget_decision_tools
)


# ═══════════════════════════════════════════════════════════
# 3. RESOURCE POOLS & LIFECYCLE MANAGEMENT
# ═══════════════════════════════════════════════════════════

# ExitStack manages sync resources; AsyncExitStack manages the
# AsyncPostgresSaver context (opened/closed on the app's event loop).
_exit_stack = ExitStack()
_async_exit_stack = AsyncExitStack()

# Bounded thread pool for fire-and-forget memory extraction tasks.
# Prevents thread explosion under concurrent request load.
_memory_pool = ThreadPoolExecutor(
    max_workers=_MEMORY_POOL_MAX_WORKERS,
    thread_name_prefix="mem0-extractor",
)

# Thread-safe lazy-init locks (double-checked locking pattern).
_checkpointer_lock = threading.Lock()
_workflow_lock = threading.Lock()

# Async lazy-init guard for the workflow/checkpointer (async setup can't run
# under a threading.Lock without risking event-loop blocking).
_async_init_lock = asyncio.Lock()


# ═══════════════════════════════════════════════════════════
# 4. HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float safely, returning *default* on failure.

    Handles ``None``, strings with currency symbols, and other non-numeric
    types that may appear in unstructured API responses.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        if isinstance(value, str):
            # Provider/display values sometimes contain an ISO code, currency
            # symbol, or thousands separators ("RM 88.20", "¥12,345").
            cleaned = value.replace(",", "")
            match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", cleaned)
            if match:
                try:
                    return float(match.group(0))
                except ValueError:
                    pass
        return default


def _initialize_budget_allocation(total_budget: float) -> Dict[str, float]:
    """Distribute *total_budget* across canonical categories.

    NOTE: no ``"flight"`` alias is injected anymore. The alias duplicated
    ``"transportation"``, so ``update_budget_category`` treated it as an extra
    independent category — inflating the allocation total and corrupting the
    proportional trim. ``flights_hotels.py`` now reads ``"transportation"``
    (with a legacy ``"flight"`` fallback for old checkpoints).
    """
    return {
        cat: round(total_budget * ratio, 2)
        for cat, ratio in DEFAULT_BUDGET_RATIOS.items()
    }


def _parse_tool_content(content: Any) -> Optional[dict]:
    """Coerce a ``ToolMessage.content`` value into a ``dict``.

    Handles:
      - ``dict``   → returned as-is
      - ``str``    → JSON-parsed (returns ``None`` if invalid)
      - ``list``   → iterates elements, returns first valid dict
      - other      → ``None``
    """
    if isinstance(content, dict):
        return content

    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None

    if isinstance(content, list):
        # Try each element until we find a parseable dict.
        # Previous implementation returned None on the first non-dict
        # string element, skipping potentially valid later elements.
        for part in content:
            parsed = _parse_tool_content(part)
            if parsed is not None:
                return parsed

    return None


def _recalculate_day_cost(day: dict, *, is_checkout_day: bool = False) -> float:
    """Recompute ``day_total_cost`` from flight and hotel prices.

    Uses ``_safe_float`` to handle string / None prices from APIs. The planner
    retains the selected hotel on the final day for display and routing, but
    that checkout day is not an overnight stay and must not be charged.
    """
    flight_cost = 0.0
    hotel_cost = 0.0

    # Boundary flights may both occur on a one-day itinerary. Charge every
    # retained leg; editing one leg must not make the opposite leg free.
    flights = day.get("flight")
    if isinstance(flights, list):
        flight_cost = sum(
            _safe_float(flight.get("price", 0))
            for flight in flights
            if isinstance(flight, dict)
        )
    elif isinstance(flights, dict):
        flight_cost = _safe_float(flights.get("price", 0))

    # Hotel (present on every day with an overnight stay)
    hotel = day.get("hotel")
    if isinstance(hotel, dict) and not is_checkout_day:
        hotel_cost = _safe_float(hotel.get("price_per_night", 0))

    return round(flight_cost + hotel_cost, 2)


def _recalculate_day_cost_full(
    day: dict, *, is_checkout_day: bool = False
) -> float:
    """Day cost including estimated activity/restaurant costs (flight+hotel+places)."""
    base = _recalculate_day_cost(day, is_checkout_day=is_checkout_day)
    acts = day.get("activities") or []
    act_cost = sum(
        _safe_float(a.get("estimated_cost", 0)) for a in acts if isinstance(a, dict)
    )
    return round(base + act_cost, 2)


def _is_checkout_day(itinerary: List[dict], target_day: dict) -> bool:
    """Return whether *target_day* is the final valid day in itinerary order."""
    days = [day for day in itinerary if isinstance(day, dict)]
    # A one-day legacy snapshot has no distinct arrival/checkout boundary, so
    # preserve its historical hotel charging behavior.
    return len(days) > 1 and days[-1] is target_day


def _transportation_budget(allocation: Any) -> float:
    """Read the canonical transportation cap with legacy-checkpoint fallback."""
    if not isinstance(allocation, dict):
        return 0.0
    key = "transportation" if "transportation" in allocation else "flight"
    return max(0.0, _safe_float(allocation.get(key, 0.0)))


def _reassess_boundary_flight_budget(
    itinerary: List[dict], allocation: Any
) -> None:
    """Clear and recompute the shared trip cap on available boundary legs."""
    days = [day for day in itinerary if isinstance(day, dict)]
    if not days:
        return

    boundary_days = [days[0]]
    if days[-1] is not days[0]:
        boundary_days.append(days[-1])

    flights: List[dict] = []
    for day in boundary_days:
        raw_flights = day.get("flight")
        raw_flight = (
            raw_flights[0]
            if isinstance(raw_flights, list) and raw_flights
            else raw_flights
        )
        if isinstance(raw_flight, dict):
            raw_flight.pop("over_budget", None)
            flights.append(raw_flight)

    total = sum(_safe_float(flight.get("price", 0.0)) for flight in flights)
    budget_cap = _transportation_budget(allocation)
    if budget_cap > 0 and total > budget_cap:
        for flight in flights:
            flight["over_budget"] = True


def _validated_trip_updates(state: AgentState, raw_updates: Any) -> Dict[str, Any]:
    """Validate an update transaction against the current trip boundaries."""
    if not isinstance(raw_updates, dict):
        return {}

    allowed = {
        "num_people",
        "start_date",
        "end_date",
        "country",
        "city",
        "origin_country",
    }
    updates = {
        key: value
        for key, value in raw_updates.items()
        if key in allowed and value is not None
    }

    if {"start_date", "end_date"} & updates.keys():
        merged_start = updates.get("start_date", state.start_date)
        merged_end = updates.get("end_date", state.end_date)
        if merged_start and merged_end:
            try:
                start = datetime.strptime(str(merged_start), "%Y-%m-%d")
                end = datetime.strptime(str(merged_end), "%Y-%m-%d")
            except (TypeError, ValueError):
                return {}
            if end < start:
                return {}

    return updates

def _incremental_edit_target_error(
    state: AgentState,
    edit: Mapping[str, Any],
) -> str | None:
    """Validate an edit target before mutating a candidate itinerary.

    Incremental edits are atomic. Invalid day/index requests must never
    trigger a full itinerary regeneration.
    """
    day_num = edit.get("day")

    if (
        isinstance(day_num, bool)
        or not isinstance(day_num, int)
        or day_num < 1
    ):
        return (
            "I couldn't apply that change because the requested "
            "day number is invalid. Your itinerary has not been changed."
        )

    day = next(
        (
            item
            for item in state.draft_itinerary
            if (
                isinstance(item, dict)
                and item.get("day") == day_num
            )
        ),
        None,
    )

    if day is None:
        return (
            f"Day {day_num} does not exist in your current itinerary. "
            "Your itinerary has not been changed."
        )

    category = edit.get("category")
    action = edit.get("action")

    if (
        category in {"activity", "restaurant"}
        and action in {"remove", "replace"}
    ):
        # Destructive/targeted place edits must be grounded in an explicit day
        # from the CURRENT user message. Never trust a day guessed by the LLM
        # from an ordinal such as "activity 4" or from an earlier assistant reply.
        if not _mutation_target_day_is_authorized(
            state,
            day_num,
        ):
            return (
                "Before I change that activity, please tell me which day "
                "you mean. Your itinerary has not been changed."
            )

        activities = day.get("activities") or []

        if not isinstance(activities, list):
            activities = []

        index = edit.get("index")

        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 1
        ):
            return (
                f"Please specify which activity on Day {day_num} "
                "you want to change."
            )

        if index > len(activities):
            requested_name = None

            new_details = edit.get("new_details")

            if isinstance(new_details, Mapping):
                raw_name = new_details.get("name")

                if isinstance(raw_name, str) and raw_name.strip():
                    requested_name = raw_name.strip()

            if requested_name:
                return (
                    f"Day {day_num} currently has only "
                    f"{len(activities)} activities, so activity "
                    f"{index} does not exist. Please choose an "
                    f"existing activity number, or ask me to add "
                    f"{requested_name} as a new activity."
                )

            return (
                f"Day {day_num} currently has only "
                f"{len(activities)} activities, so activity "
                f"{index} does not exist. Please choose an "
                "existing activity number."
            )

    return None

def _normalise_place_edit(new_data: Any, category: str) -> Optional[dict]:
    """Validate a raw agent place edit and return a safe canonical copy.

    This is a second line of defence behind ``ItineraryEdit`` validation. Tool
    messages are persisted data and may predate the current schema, so state
    updates must never trust their free-form dictionaries blindly.
    """
    if not isinstance(new_data, dict):
        return None
    name = new_data.get("name")
    location = new_data.get("location")
    if not isinstance(name, str) or not name.strip() or not isinstance(location, dict):
        return None
    requested_city = location.get("requested_city")
    if not isinstance(requested_city, str) or not requested_city.strip():
        return None

    try:
        lat = float(location.get("latitude"))
        lng = float(location.get("longitude"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(lat) or not math.isfinite(lng):
        return None

    normalised = copy.deepcopy(new_data)
    normalised["name"] = name.strip()
    normalised["type"] = (
        "restaurant"
        if category == "restaurant"
        else "attraction"
    )

    # Provider metadata is not guaranteed to be scalar.
    # SerpAPI may return a list of place categories instead of
    # a single description string. ActivityResult.description
    # accepts only str | None, so normalise the value here as
    # a second line of defence for current and persisted results.
    raw_description = normalised.get("description")

    if isinstance(raw_description, str):
        description = raw_description.strip()

    elif isinstance(raw_description, (list, tuple, set)):
        description = ", ".join(
            str(item).strip()
            for item in raw_description
            if str(item).strip()
        )

    elif raw_description is None:
        description = ""

    else:
        description = str(raw_description).strip()

    normalised["description"] = description or None


    # search_places may not have a numeric provider price.
    # ActivityResult nevertheless requires a finite non-negative
    # estimated_cost, so normalise legacy/current tool results here.
    raw_cost = normalised.get("estimated_cost", 0.0)

    try:
        estimated_cost = float(raw_cost)
    except (TypeError, ValueError):
        estimated_cost = 0.0

    if (
        not math.isfinite(estimated_cost)
        or estimated_cost < 0
    ):
        estimated_cost = 0.0

    normalised["estimated_cost"] = round(
        estimated_cost,
        2,
    )

    normalised["is_estimated"] = bool(
        normalised.get("is_estimated", True)
    )

    normalised["location"] = {
        **location,
        "place_name": (
            location.get("place_name")
            or name.strip()
        ),
        "latitude": lat,
        "longitude": lng,
        "requested_city": requested_city.strip(),
    }

    return normalised


def _trusted_place_options(state: AgentState) -> list[dict[str, Any]]:
    """Return only place results bound to a current-turn search tool call."""
    current_turn: list[BaseMessage] = []
    for message in reversed(state.messages):
        current_turn.insert(0, message)
        if message.type == "human":
            break

    request_ids: set[str] = set()
    options: list[dict[str, Any]] = []
    for message in current_turn:
        if message.type == "ai":
            for tool_call in getattr(message, "tool_calls", []):
                call_id = tool_call.get("id")
                if (
                    tool_call.get("name") == "search_places"
                    and isinstance(call_id, str)
                    and call_id
                    and isinstance(tool_call.get("args"), dict)
                ):
                    request_ids.add(call_id)
            continue
        if (
            message.type != "tool"
            or getattr(message, "name", None) != "search_places"
            or getattr(message, "tool_call_id", None) not in request_ids
        ):
            continue
        content = _parse_tool_content(message.content) or {}
        raw_results = content.get("results")
        if isinstance(raw_results, list):
            options.extend(
                copy.deepcopy(option)
                for option in raw_results
                if isinstance(option, dict)
            )
    return options


def _matching_trusted_place_option(
    edit: Mapping[str, Any],
    trusted_options: Sequence[Mapping[str, Any]],
) -> Optional[dict[str, Any]]:
    """Return the current-turn provider result selected by the model.

    The model is allowed to re-serialise provider data, so identity matching
    must not depend on byte-perfect floating-point coordinates. The returned
    object is always the exact server-trusted provider payload.
    """
    if edit.get("category") not in {"activity", "restaurant"} or edit.get(
        "action"
    ) not in {"add", "replace"}:
        return None

    details = edit.get("new_details")

    if not isinstance(details, Mapping):
        return None

    name = details.get("name")
    location = details.get("location")

    if not isinstance(name, str) or not isinstance(location, Mapping):
        return None

    try:
        latitude = float(location.get("latitude"))
        longitude = float(location.get("longitude"))
    except (TypeError, ValueError):
        return None

    if not math.isfinite(latitude) or not math.isfinite(longitude):
        return None

    name_key = " ".join(name.split()).casefold()

    # Collect only trusted CURRENT-TURN provider results
    # that have the same normalised place name.
    same_name_options: list[
        tuple[dict[str, Any], float, float]
    ] = []

    for option in trusted_options:
        if not isinstance(option, Mapping):
            continue

        option_name = option.get("name")
        option_location = option.get("location")

        if not isinstance(option_name, str) or not isinstance(
            option_location,
            Mapping,
        ):
            continue

        # The provider name must still match.
        if " ".join(option_name.split()).casefold() != name_key:
            continue

        try:
            option_latitude = float(
                option_location.get("latitude")
            )
            option_longitude = float(
                option_location.get("longitude")
            )
        except (TypeError, ValueError):
            continue

        if (
            not math.isfinite(option_latitude)
            or not math.isfinite(option_longitude)
        ):
            continue

        same_name_options.append(
            (
                copy.deepcopy(dict(option)),
                option_latitude,
                option_longitude,
            )
        )

    # ---------------------------------------------------------
    # Most common case:
    #
    # The fresh provider search returned exactly one result
    # with this name.
    #
    # That current-turn provider result is sufficient identity
    # evidence. We return the EXACT provider object rather than
    # trusting coordinates reconstructed by the LLM.
    # ---------------------------------------------------------
    if len(same_name_options) == 1:
        return same_name_options[0][0]

    # ---------------------------------------------------------
    # If the provider returned multiple places with the same
    # name, coordinates are used only to disambiguate them.
    #
    # 1e-4 degrees is approximately <= 11 metres in latitude.
    # This safely tolerates normal decimal rounding by the LLM
    # while remaining strict enough for same-name results.
    # ---------------------------------------------------------
    for (
        option,
        option_latitude,
        option_longitude,
    ) in same_name_options:

        if (
            math.isclose(
                latitude,
                option_latitude,
                abs_tol=1e-4,
            )
            and math.isclose(
                longitude,
                option_longitude,
                abs_tol=1e-4,
            )
        ):
            return option

    # The requested place cannot be connected to any fresh
    # provider result, so fail closed and preserve the itinerary.
    return None

def _recover_trusted_place_options_for_edit(
    state: AgentState,
    edit: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Perform a fresh server-side place lookup when model tool order skipped it.

    Prompt instructions ask the model to call ``search_places`` before every
    activity/restaurant add or replacement, but tool ordering is not a trust
    boundary. If the model calls ``edit_itinerary`` directly from working
    memory, the server performs the missing provider lookup itself in the same
    user turn and still fails closed when the provider cannot verify the place.
    """
    category = edit.get("category")
    action = edit.get("action")

    if category not in {"activity", "restaurant"} or action not in {
        "add",
        "replace",
    }:
        return []

    details = edit.get("new_details")
    if not isinstance(details, Mapping):
        return []

    name = details.get("name")
    if not isinstance(name, str) or not name.strip():
        return []

    configured_cities = [
        city.strip()
        for city in (state.city or [])
        if isinstance(city, str) and city.strip()
    ]
    city_hint = configured_cities[0] if configured_cities else ""

    # A model-supplied requested_city is only accepted when it exactly matches
    # a server-configured city. It can never expand the trusted destination.
    location = details.get("location")
    if isinstance(location, Mapping):
        requested_city = location.get("requested_city")

        if isinstance(requested_city, str) and requested_city.strip():
            requested_key = " ".join(
                requested_city.split()
            ).casefold()

            matched_city = next(
                (
                    city
                    for city in configured_cities
                    if " ".join(
                        city.split()
                    ).casefold() == requested_key
                ),
                None,
            )

            if matched_city is not None:
                city_hint = matched_city

    provider_category = (
        "restaurant"
        if category == "restaurant"
        else "attraction"
    )

    try:
        response = search_places_provider(
            category=provider_category,
            query=name.strip(),
            state=state,
            city=city_hint,
        )

    except Exception:
        logger.warning(
            "Server-side place grounding recovery failed for %r",
            name.strip(),
            exc_info=True,
        )
        return []

    if not isinstance(response, Mapping):
        return []

    raw_results = response.get("results")

    if not isinstance(raw_results, list):
        logger.info(
            "Server-side place grounding recovery returned no results: "
            "name=%r error=%r",
            name.strip(),
            response.get("error"),
        )
        return []

    options = [
        copy.deepcopy(option)
        for option in raw_results
        if isinstance(option, dict)
    ]

    logger.info(
        "Server-side place grounding recovery: name=%r options=%d",
        name.strip(),
        len(options),
    )

    return options


def _trusted_booking_options(
    state: AgentState,
) -> list[tuple[str, int, dict[str, Any]]]:
    """Bind current-turn provider options to their requested category and day."""
    current_turn: list[BaseMessage] = []
    for message in reversed(state.messages):
        current_turn.insert(0, message)
        if message.type == "human":
            break

    requests: dict[str, tuple[str, int]] = {}
    options: list[tuple[str, int, dict[str, Any]]] = []
    for message in current_turn:
        if message.type == "ai":
            for tool_call in getattr(message, "tool_calls", []):
                if tool_call.get("name") != "search_alternative_opt":
                    continue
                call_id = tool_call.get("id")
                args = tool_call.get("args")
                if not isinstance(call_id, str) or not isinstance(args, dict):
                    continue
                category = args.get("category")
                category = "hotel" if category == "accommodation" else category
                day_number = args.get("day_num")
                if category not in {"flight", "hotel"} or (
                    not isinstance(day_number, int) or isinstance(day_number, bool)
                ):
                    continue
                requests[call_id] = (category, day_number)
            continue
        if message.type != "tool" or getattr(message, "name", None) != (
            "search_alternative_opt"
        ):
            continue
        request = requests.get(getattr(message, "tool_call_id", None))
        if request is None:
            continue
        content = _parse_tool_content(message.content) or {}
        raw_options = content.get("options")
        if not isinstance(raw_options, list):
            continue
        category, day_number = request
        options.extend(
            (category, day_number, copy.deepcopy(option))
            for option in raw_options
            if isinstance(option, dict)
        )
    return options


def _country_alpha_2(country: Any) -> str | None:
    if not isinstance(country, str) or not country.strip():
        return None
    try:
        return pycountry.countries.lookup(country).alpha_2
    except LookupError:
        return None


def _booking_edit_is_grounded(
    state: AgentState,
    edit: Mapping[str, Any],
    trusted_options: Sequence[tuple[str, int, Mapping[str, Any]]],
) -> bool:
    """Require a flight/hotel replacement to exactly match current evidence."""
    category = edit.get("category")
    if category not in {"flight", "hotel"} or edit.get("action") != "replace":
        return True
    details = edit.get("new_details")
    day_number = edit.get("day")
    if not isinstance(details, dict) or not any(
        category == option_category
        and day_number == option_day
        and details == option
        for option_category, option_day, option in trusted_options
    ):
        return False

    price_field = "price" if category == "flight" else "price_per_night"
    price = details.get(price_field)
    if (
        isinstance(price, bool)
        or not isinstance(price, (int, float))
        or not math.isfinite(price)
        or price <= 0
    ):
        return False

    destination_code = _country_alpha_2(state.country)
    origin_code = _country_alpha_2(state.origin_country)
    if category == "hotel":
        name = details.get("hotel_name")
        location = details.get("location")
        if not isinstance(name, str) or not name.strip() or not isinstance(
            location, Mapping
        ):
            return False
        latitude = location.get("latitude", location.get("lat"))
        longitude = location.get("longitude", location.get("lng"))
        return bool(
            destination_code
            and location.get("country_code") == destination_code
            and isinstance(latitude, (int, float))
            and not isinstance(latitude, bool)
            and math.isfinite(latitude)
            and isinstance(longitude, (int, float))
            and not isinstance(longitude, bool)
            and math.isfinite(longitude)
        )

    if not all(
        isinstance(details.get(field), str) and details[field].strip()
        for field in ("airline", "flight_number")
    ):
        return False
    departure = details.get("departure_airport")
    arrival = details.get("arrival_airport")
    if not isinstance(departure, Mapping) or not isinstance(arrival, Mapping):
        return False
    if not any(
        isinstance(departure.get(field), str) and departure[field].strip()
        for field in ("id", "name")
    ) or not any(
        isinstance(arrival.get(field), str) and arrival[field].strip()
        for field in ("id", "name")
    ):
        return False

    configured_days = [
        day.get("day")
        for day in state.draft_itinerary
        if isinstance(day, dict) and isinstance(day.get("day"), int)
    ]
    first_day = min(configured_days) if configured_days else 1
    last_day = max(configured_days) if configured_days else first_day
    day_number = edit.get("day")
    if day_number == first_day == last_day:
        expected_pairs = {
            (origin_code, destination_code),
            (destination_code, origin_code),
        }
    elif day_number == first_day:
        expected_pairs = {(origin_code, destination_code)}
    elif day_number == last_day:
        expected_pairs = {(destination_code, origin_code)}
    else:
        return False
    departure_code = departure.get("country_code")
    arrival_code = arrival.get("country_code")
    return any(
        departure_code in {None, expected_departure}
        and arrival_code in {None, expected_arrival}
        for expected_departure, expected_arrival in expected_pairs
    )


def _apply_single_edit(
    itinerary: List[dict],
    edit: dict,
    *,
    origin_country_code: str | None = None,
    destination_country_code: str | None = None,
) -> None:
    """Apply one ``ItineraryEdit`` dict (from the edit_itinerary tool) in place.

    Supports flight/hotel replace and activity/restaurant add/remove/replace.
    Silently ignores edits that reference a non-existent day or index (the LLM
    can occasionally hallucinate positions).
    """
    day_num = edit.get("day")
    action = edit.get("action")
    category = edit.get("category")
    index = edit.get("index")
    # The model and user see activity labels starting at 1. Convert that public
    # number to Python's zero-based list position only at the mutation boundary.
    list_index = (
        index - 1 if isinstance(index, int) and not isinstance(index, bool) else None
    )
    new_data = edit.get("new_details")

    day = next(
        (d for d in itinerary if isinstance(d, dict) and d.get("day") == day_num),
        None,
    )
    if day is None:
        return

    if category == "flight":
        if action == "replace" and new_data:
            existing = day.get("flight")
            flights = list(existing) if isinstance(existing, list) else []
            if len(flights) <= 1:
                day["flight"] = [new_data]
            else:
                departure = new_data.get("departure_airport", {})
                arrival = new_data.get("arrival_airport", {})
                departure_code = (
                    departure.get("country_code")
                    if isinstance(departure, Mapping)
                    else None
                )
                arrival_code = (
                    arrival.get("country_code")
                    if isinstance(arrival, Mapping)
                    else None
                )
                replacement_index = 0
                if (
                    departure_code == destination_country_code
                    and arrival_code == origin_country_code
                ):
                    replacement_index = len(flights) - 1
                flights[replacement_index] = new_data
                day["flight"] = flights
        elif action == "remove":
            day["flight"] = None
    elif category == "hotel":
        if action == "replace" and new_data:
            day["hotel"] = new_data
        elif action == "remove":
            day["hotel"] = None
    else:  # activity | restaurant — both live in the activities list
        acts = list(day.get("activities") or [])
        if action in {"add", "replace"}:
            new_data = _normalise_place_edit(new_data, category)
            if new_data is None:
                return
        if action == "add" and new_data:
            acts.append(new_data)
        elif (
            action == "remove"
            and list_index is not None
            and 0 <= list_index < len(acts)
        ):
            acts.pop(list_index)
        elif (
            action == "replace"
            and list_index is not None
            and new_data
            and 0 <= list_index < len(acts)
        ):
            acts[list_index] = new_data
        for position, activity in enumerate(acts, start=1):
            if isinstance(activity, dict):
                activity["order"] = position
        day["activities"] = acts

    day["day_total_cost"] = _recalculate_day_cost_full(
        day, is_checkout_day=_is_checkout_day(itinerary, day)
    )


def _extract_text(content: Any) -> str:
    """Extract plain text from message content (str or list of parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                parts.append(str(part["text"]))
        return " ".join(parts)
    return str(content)


# ═══════════════════════════════════════════════════════════
# 5. NODE DEFINITIONS
# ═══════════════════════════════════════════════════════════

# ── 5.1  Initial Form Processing ───────────────────────


def process_initial_form_node(
    state: AgentState,
    config: RunnableConfig | None = None,
) -> dict:
    """New-trip setup: currency conversion + budget allocation.

    Runs the currency pipeline, initialises budget allocation across
    canonical categories, and records the fetch timestamp for
    TTL-based rate refresh.
    """
    log_context = _planning_log_context(config, state)
    safe_observability_log(
        logger,
        "planning.initial_form.start",
        stage="initial_form",
        outcome="started",
        **log_context,
    )

    # Step 1: Reuse the exact preflight rate when it belongs to this trip.
    # Otherwise preserve the provider-backed currency pipeline for legacy
    # checkpoints and chat replans whose trip-defining fields changed.
    assessment_data = state.budget_assessment
    if assessment_data and assessment_matches_trip(
        assessment_data,
        origin=state.origin_country or "",
        destination=state.country or "",
        destination_city=primary_destination_city(state.city),
        start_date=state.start_date or "",
        end_date=state.end_date or "",
        num_people=max(1, int(state.num_people or 1)),
    ):
        assessment = BudgetAssessment.model_validate(assessment_data)
        converted_budget = round(
            float(state.total_base_budget or 0.0) * assessment.exchange_rate,
            2,
        )
        result = {
            "base_currency_code": assessment.base_currency,
            "dest_currency_code": assessment.destination_currency,
            "exchange_rate": {
                assessment.base_currency: 1.0,
                assessment.destination_currency: assessment.exchange_rate,
            },
            "total_convert_budget": converted_budget,
        }
    else:
        # Provider failure must stop planning instead of silently applying a
        # mathematically false 1:1 exchange rate.
        result = currency_pipeline(state)

    # ── Step 2: Initialise budget allocation ──
    total_converted = result.get("total_convert_budget") or 0.0
    result["budget_allocation"] = _initialize_budget_allocation(total_converted)

    # ── Step 3: Record fetch timestamp ──
    result["currency_fetched_at"] = datetime.now(timezone.utc).isoformat()

    # The accepted amount is now entering the established planning pipeline;
    # do not leave a stale gate outcome or proposal in the checkpoint.
    result.update(
        {
            "pending_budget_proposal": None,
            "accepted_budget_decision": None,
            "budget_gate_outcome": None,
            "budget_gate_reason": None,
            "budget_gate_message": None,
        }
    )

    safe_observability_log(
        logger,
        "planning.initial_form.done",
        stage="initial_form",
        outcome="completed",
        **log_context,
    )

    return result


def _effective_plan_revision(state: AgentState) -> int:
    """Give legacy committed plans a stable first revision during migration."""
    if state.plan_revision > 0:
        return state.plan_revision
    if state.draft_itinerary or state.daily_map_info:
        return 1
    return 0


def _build_accepted_plan_snapshot(
    state: AgentState,
    *,
    revision: int,
    values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Project only committed plan fields into the durable public snapshot."""
    source = values or {}

    def value(name: str, fallback: Any) -> Any:
        return source[name] if name in source else fallback

    return {
        "plan_revision": revision,
        "total_base_budget": float(
            value("total_base_budget", state.total_base_budget) or 0.0
        ),
        "base_currency_code": value(
            "base_currency_code", state.base_currency_code
        ),
        "dest_currency_code": value(
            "dest_currency_code", state.dest_currency_code
        ),
        "total_convert_budget": float(
            value("total_convert_budget", state.total_convert_budget) or 0.0
        ),
        "budget_allocation": copy.deepcopy(
            value("budget_allocation", state.budget_allocation) or {}
        ),
        "draft_itinerary": copy.deepcopy(
            value("draft_itinerary", state.draft_itinerary) or []
        ),
        "daily_map_info": copy.deepcopy(
            value("daily_map_info", state.daily_map_info) or {}
        ),
    }


def capture_accepted_plan_node(state: AgentState) -> dict[str, Any]:
    """Legacy checkpoint shim that can no longer publish accepted state."""
    del state
    return {}


# ── 5.2  Itinerary Planning ────────────────────────────


def plan_initial_itinerary_node(state: AgentState) -> dict:
    """Fetch baseline flights and hotels via SerpAPI, assemble itinerary."""
    try:
        return plan_flight_hotel(state)
    except Exception:
        return {"draft_itinerary": []}


# ── 5.2b  Places / Attractions / Restaurants Planning ──


async def plan_activities_node(state: AgentState, config: RunnableConfig) -> dict:
    """Brainstorm places (LLM) + fetch real data (SerpAPI), fill each day's
    activities. Runs AFTER flights/hotels and BEFORE the map node so routes can
    be drawn through the day's stops.

    Personalisation: pulls the user's long-term profile (interests, dietary,
    pacing) so the brainstorm reflects learned preferences.
    """
    user_id = config.get("configurable", {}).get("__user_id")

    user_profile: Optional[dict] = None

    if user_id:
        try:
            user_profile = await asyncio.to_thread(
                fetch_user_memory_context,
                user_id,
            )
        except Exception:
            user_profile = None

    try:
        # plan_activities is sync + network-bound (LLM + SerpAPI) — offload it
        # off the event loop.
        return await asyncio.to_thread(plan_activities, state, user_profile)
    except Exception:
        return {}


async def plan_validated_transaction_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Build a qualified private candidate without changing the public plan."""
    log_context = _planning_log_context(config, state)
    safe_observability_log(
        logger,
        "planning.transaction.dispatch",
        attempt=0,
        stage="transaction",
        outcome="started",
        **log_context,
    )
    user_id = config.get("configurable", {}).get("__user_id")
    user_profile: Optional[dict] = None

    if user_id:
        try:
            user_profile = await asyncio.to_thread(
                fetch_user_memory_context,
                user_id,
            )
        except Exception:
            safe_observability_log(
                logger,
                "planning.profile",
                attempt=0,
                stage="planning_profile",
                issue_codes=("profile.unavailable",),
                outcome="unavailable",
                **log_context,
            )

    if state.planning_attempts >= _PLANNING_TRANSACTION_MAX_ATTEMPTS:
        return {
            "planning_outcome": "unavailable",
            "candidate_plan": None,
            "planning_attempts": state.planning_attempts,
            "planning_issue_codes": list(state.planning_issue_codes),
        }

    transaction_state = _planning_transaction_state(state)
    with planning_observability_context(log_context):
        result = await build_validated_plan(
            transaction_state,
            user_profile,
            max_attempts=1,
            attempt_offset=state.planning_attempts,
        )
    candidate_plan: Optional[dict[str, Any]] = None
    if result.candidate is not None:
        candidate_plan = {
            "itinerary": copy.deepcopy(result.candidate.itinerary),
            "maps": copy.deepcopy(result.candidate.maps),
            "attempt": result.candidate.attempt,
            "validation": result.candidate.validation.model_dump(mode="json"),
        }
        if state.accepted_budget_decision is not None:
            candidate_plan["financials"] = {
                key: copy.deepcopy(getattr(transaction_state, key))
                for key in _CANDIDATE_FINANCIAL_FIELDS
            }
            candidate_plan["financials"].update(
                {
                    "total_base_budget": transaction_state.total_base_budget,
                    "budget_assessment": copy.deepcopy(
                        transaction_state.budget_assessment
                    ),
                }
            )
    return {
        "planning_outcome": result.status,
        "candidate_plan": candidate_plan,
        "planning_attempts": result.attempts,
        "planning_issue_codes": [
            *state.planning_issue_codes,
            *(issue.code for issue in result.issues),
        ],
    }


def prepare_planning_transaction_node(_state: AgentState) -> dict[str, Any]:
    """Checkpoint a cleared private transaction before any provider await."""
    return {
        "planning_outcome": None,
        "candidate_plan": None,
        "planning_attempts": 0,
        "planning_issue_codes": [],
        "output_review_attempts": 0,
        "output_review_issue_codes": [],
    }


def _planning_log_context(
    config: Mapping[str, Any] | None,
    state: AgentState,
) -> dict[str, Any]:
    config_values = config if isinstance(config, Mapping) else {}
    configurable = config_values.get("configurable")
    configurable = configurable if isinstance(configurable, Mapping) else {}
    metadata = config_values.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    try:
        destination_country_code: str | None = (
            TripRequirements.from_state(state).destination_country_code
        )
    except (TypeError, ValueError):
        destination_country_code = None
    return {
        "request_id": configurable.get("request_id")
        or metadata.get("request_id"),
        "session_id": configurable.get("thread_id")
        or configurable.get("session_id")
        or metadata.get("thread_id"),
        "destination_country_code": destination_country_code,
    }


def _latest_user_request(state: AgentState) -> str:
    for message in reversed(state.messages):
        if message.type == "human":
            text = _extract_text(message.content).strip()
            if text:
                return text
    return "Create a response for the current server-owned trip request."


def _explicit_itinerary_mutation_request(
    state: AgentState,
) -> bool:
    """Return True when the current user asks to mutate the itinerary."""

    if not state.draft_itinerary:
        return False

    request = _latest_user_request(state)

    # "day 2" may be the immediate answer to a
    # server-owned mutation clarification.
    if _pending_day_reply(state) is not None:
        return True

    # "yes" may confirm the immediately preceding proposal,
    # provided the day is corroborated by the user's earlier message.
    if _confirmation_continues_itinerary_mutation(state):
        return True

    if is_read_only_itinerary_history_request(request):
        return False

    return bool(
        _EXPLICIT_MUTATION_REQUEST_PATTERN.search(request)
        or _ITINERARY_DISLIKE_MUTATION_PATTERN.search(request)
    )

def _place_edit_day_clarification(
    state: AgentState,
) -> str | None:
    """Return a fixed clarification when an indexed place edit omits the day."""

    if not state.draft_itinerary:
        return None

    request = _latest_user_request(state)

    if is_read_only_itinerary_history_request(request):
        return None

    item_match = _PLACE_ITEM_REFERENCE_PATTERN.search(request)

    if (
        item_match is None
        or _PLACE_EDIT_INTENT_PATTERN.search(request) is None
        or _EXPLICIT_DAY_REFERENCE_PATTERN.search(request) is not None
    ):
        return None

    item_reference = item_match.group(0).strip()

    return (
        f"Sure — which day is {item_reference} on? "
        "Your itinerary has not been changed."
    )


def _trusted_review_requirements(
    state: AgentState,
    user_profile: Optional[dict] = None,
) -> dict[str, Any]:
    # The semantic reviewer must see BOTH the authoritative current itinerary
    # and older accepted revisions. Previously it received only
    # ``itinerary_history``. Because that list intentionally excludes the
    # live revision, the reviewer could mistake the newest historical revision
    # for the current trip and reject a correct answer about "current Day 2".
    current_revision = _effective_plan_revision(state)

    accepted = (
        state.accepted_plan_snapshot
        if isinstance(state.accepted_plan_snapshot, dict)
        else {}
    )

    accepted_revision = accepted.get("plan_revision")
    accepted_itinerary = accepted.get("draft_itinerary")

    # Prefer the durable accepted snapshot only when it represents the same
    # revision as the live state. The fallback also supports legacy checkpoints
    # or temporarily inconsistent migrated state.
    if (
        isinstance(accepted_revision, int)
        and not isinstance(accepted_revision, bool)
        and accepted_revision == current_revision
        and isinstance(accepted_itinerary, list)
    ):
        current_itinerary = accepted_itinerary
    else:
        current_itinerary = state.draft_itinerary

    return {
        "origin_country": state.origin_country,
        "country": state.country,
        "city": copy.deepcopy(state.city),
        "num_people": state.num_people,
        "start_date": state.start_date,
        "end_date": state.end_date,
        "base_currency_code": state.base_currency_code,
        "dest_currency_code": state.dest_currency_code,
        "total_base_budget": state.total_base_budget,
        "total_convert_budget": state.total_convert_budget,
        "budget_allocation": copy.deepcopy(
            state.budget_allocation
        ),

        # Persisted long-term memory is trusted server-side context too.
        # Without this, the semantic output reviewer rejects correct
        # preference statements as unsupported even when the main agent
        # successfully retrieved them.
        "long_term_user_preferences": copy.deepcopy(
            user_profile or {}
        ),

        # Current itinerary is authoritative for requests using words such as
        # "current", "latest", and "now".
        "current_plan_revision": current_revision,
        "current_itinerary": copy.deepcopy(
            current_itinerary
        ),

        # Historical revisions are evidence only for questions such as
        # "original", "before", "previous", or revision comparisons.
        "itinerary_history": copy.deepcopy(
            state.itinerary_history
        ),
    }


def _normalized_tool_evidence(state: AgentState) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for message in reversed(state.messages):
        if message.type == "human":
            break
        if message.type != "tool":
            continue
        evidence.insert(
            0,
            {
                "name": getattr(message, "name", None),
                "result": _parse_tool_content(message.content)
                or _extract_text(message.content),
            },
        )
    return evidence


def _output_review_context(
    state: AgentState,
    reply: str,
    candidate: dict[str, Any] | None,
    *,
    config: Mapping[str, Any] | None = None,
    review_attempt: int = 0,
    review_stage: str = "response_review",
    user_profile: Optional[dict] = None,
) -> OutputReviewContext:
    validation = candidate.get("validation", {}) if candidate else {}
    review_state = state
    if candidate is not None and isinstance(candidate.get("financials"), dict):
        allowed_financials = _CANDIDATE_FINANCIAL_FIELDS | {
            "total_base_budget",
            "budget_assessment",
        }
        review_state = state.model_copy(
            update={
                key: copy.deepcopy(value)
                for key, value in candidate["financials"].items()
                if key in allowed_financials
            },
            deep=True,
        )
    log_context = _planning_log_context(config, review_state)
    return OutputReviewContext(
        latest_user_request=_latest_user_request(state),
        trusted_requirements=_trusted_review_requirements(
            review_state,
            user_profile,
        ),
        normalized_tool_evidence=_normalized_tool_evidence(state),
        proposed_reply=reply,
        candidate_plan=copy.deepcopy(candidate),
        deterministic_report=copy.deepcopy(validation),
        request_id=log_context["request_id"],
        session_id=log_context["session_id"],
        destination_country_code=log_context["destination_country_code"],
        review_attempt=review_attempt,
        review_stage=review_stage,
    )


async def _review_reply_attempts(
    state: AgentState,
    config: RunnableConfig,
    base_messages: list[BaseMessage],
    *,
    candidate: dict[str, Any] | None,
    initial_response: AIMessage | None = None,
    max_attempts: int = _OUTPUT_REVIEW_MAX_ATTEMPTS,
    user_profile: Optional[dict] = None,
) -> tuple[str | None, OutputReviewDecision, int, list[str]]:
    """Generate and privately review a bounded number of proposed replies."""
    if max_attempts < 1 or max_attempts > _OUTPUT_REVIEW_MAX_ATTEMPTS:
        raise ValueError("max_attempts is outside the output-review budget")
    response = initial_response
    previous_decision: OutputReviewDecision | None = None
    observed_issue_codes: list[str] = []

    for attempt in range(1, max_attempts + 1):
        if response is None:
            attempt_messages = list(base_messages)
            if previous_decision is not None:
                attempt_messages.append(
                    HumanMessage(
                        content=(
                            "Private output review rejected the prior draft with "
                            f"codes {', '.join(previous_decision.issue_codes)}. "
                            f"Feedback: {previous_decision.feedback} Write a new "
                            "answer as concise public prose. Do not repeat or mention "
                            "the rejected draft or this review instruction."
                        )
                    )
                )
            try:
                response = await llm.ainvoke(attempt_messages, config=config)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                raise
            except Exception:
                safe_observability_log(
                    logger,
                    "planning.review_generation",
                    attempt=attempt,
                    stage="response_generation",
                    issue_codes=("reply.generation.unavailable",),
                    outcome="failed",
                    **_planning_log_context(config, state),
                )
                response = AIMessage(content="")

        reply = _extract_text(response.content).strip()
        try:
            decision = await review_public_output(
                _output_review_context(
                    state,
                    reply,
                    candidate,
                    config=config,
                    review_attempt=attempt,
                    review_stage="response_review",
                    user_profile=user_profile,
                )
            )
        except (asyncio.CancelledError, asyncio.TimeoutError):
            raise
        except Exception:
            safe_observability_log(
                logger,
                "planning.review",
                attempt=attempt,
                stage="response_review",
                issue_codes=("review.unavailable",),
                outcome="failed",
                **_planning_log_context(config, state),
            )
            decision = OutputReviewDecision(
                approved=False,
                issue_codes=("review.unavailable",),
                feedback="Output review is temporarily unavailable.",
            )
        observed_issue_codes.extend(decision.issue_codes)
        if decision.approved:
            return reply, decision, attempt, observed_issue_codes

        # review.unavailable means the reviewer infrastructure failed.
        # It does not mean the already-qualified itinerary is invalid.
        # Stop spending additional Gemini quota and allow the caller to
        # use the deterministic server-owned fallback response.
        if "review.unavailable" in decision.issue_codes:
            return None, decision, attempt, observed_issue_codes

        previous_decision = decision
        response = None

    assert previous_decision is not None
    return None, previous_decision, max_attempts, observed_issue_codes


def _completed_planning_cleanup() -> dict[str, Any]:
    return {
        "accepted_budget_decision": None,
        "pending_budget_proposal": None,
        "pending_budget_confirmation": None,
        "pending_itinerary_mutation": None,
        "budget_gate_outcome": None,
        "budget_gate_reason": None,
        "budget_gate_message": None,
        "chat_budget_action": None,
        "chat_budget_assessment_id": None,
    }

def _current_turn_used_tool(
    state: AgentState,
    tool_name: str,
) -> bool:
    """Return True when a named tool was used in the current chat turn."""

    for message in reversed(state.messages):
        message_type = getattr(
            message,
            "type",
            "",
        )

        if message_type == "human":
            break

        if (
            message_type == "tool"
            and getattr(
                message,
                "name",
                "",
            ) == tool_name
        ):
            return True

    return False

def _review_exhaustion_update(
    state: AgentState,
    attempts: int,
    issue_codes: list[str],
) -> dict[str, Any]:
    update: dict[str, Any] = {
        "output_review_attempts": attempts,
        "output_review_issue_codes": list(issue_codes),
    }

    # The planning transaction has already produced a provider-grounded,
    # deterministically validated candidate. Do not discard that candidate
    # merely because the separate semantic output reviewer is unavailable.
    if isinstance(state.candidate_plan, dict):
        candidate = copy.deepcopy(state.candidate_plan)

        fallback_text = _qualified_plan_fallback_text(
            state,
        )

        fallback_decision = deterministic_output_review(
            _output_review_context(
                state,
                fallback_text,
                candidate,
            )
        )

        if not fallback_decision.approved:
            raise RuntimeError(
                "qualified-plan fallback failed deterministic output review"
            )

        candidate["reply"] = fallback_text
        candidate["review"] = fallback_decision.model_dump(mode="json")

        # This marker is trusted only together with the exact server-owned
        # fallback text during promotion.
        candidate["reply_source"] = "server_owned_qualified_fallback"

        update["candidate_plan"] = candidate
        return update

    # No qualified candidate exists. In this situation planning really is
    # unavailable, so retain the original fail-closed behaviour.
    update["messages"] = [_planning_unavailable_message(state)]

    if state.planning_outcome is not None:
        update.update(
            {
                "planning_outcome": "unavailable",
                "candidate_plan": None,
                **_completed_planning_cleanup(),
            }
        )

    return update


def _planning_unavailable_message(
    state: AgentState,
) -> AIMessage:
    """Build a fixed fail-closed reply appropriate to the current request."""

    fallback = (
        _OUTPUT_REVIEW_SAFE_FALLBACK
        if _explicit_itinerary_mutation_request(state)
        else _OUTPUT_REVIEW_READ_ONLY_FALLBACK
    )

    decision = deterministic_output_review(
        _output_review_context(
            state,
            fallback,
            None,
        )
    )

    if not decision.approved:
        raise RuntimeError(
            "server-owned planning fallback failed deterministic review"
        )

    return AIMessage(
        content=fallback,
        additional_kwargs={
            "server_owned": "planning_unavailable"
        },
    )


async def generate_reviewed_response_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Keep candidate reply attempts private until one passes review."""
    user_id = config.get("configurable", {}).get("__user_id")
    user_profile: Optional[dict] = None

    if user_id:
        try:
            user_profile = await asyncio.to_thread(
                fetch_user_memory_context,
                user_id,
            )
        except Exception:
            safe_observability_log(
                logger,
                "planning.profile",
                attempt=0,
                stage="response_profile",
                issue_codes=("profile.unavailable",),
                outcome="unavailable",
                **_planning_log_context(config, state),
            )

    candidate = copy.deepcopy(state.candidate_plan)

    if candidate is None:
        if (
            state.planning_attempts
            < _PLANNING_TRANSACTION_MAX_ATTEMPTS
        ):
            # A deterministic/provider rejection consumed this shared
            # attempt. No public text is generated from the rejected
            # transaction.
            return {
                "planning_outcome": "unavailable",
                "candidate_plan": None,
            }

        return _review_exhaustion_update(
            state,
            state.output_review_attempts,
            list(
                state.output_review_issue_codes
                or state.planning_issue_codes
            ),
        )

    base_messages = build_prompt_messages(
        state,
        user_profile,
    )

    # Distinguish a successful budget replan, an itinerary edit,
    # and first-time/general itinerary generation.
    is_itinerary_edit = _current_turn_used_tool(
        state,
        "edit_itinerary",
    )

    is_budget_replan = (
        state.budget_gate_outcome == "accepted"
        and isinstance(state.accepted_budget_decision, dict)
    )

    if is_budget_replan:
        instruction = (
            "The user's budget change has already been accepted and the "
            "qualified candidate is the completed itinerary after that budget "
            "change. Write a concise confirmation that the accepted budget "
            "has been applied and that the itinerary was recalculated. Mention "
            "the destination and useful current trip information from the "
            "qualified candidate. Do not ask the user to confirm the budget again. "
            "Do not claim that a booking was purchased or finalized. Do not "
            "mention internal validation, candidate plans, output review, or "
            "internal state."
        )

    elif is_itinerary_edit:
        instruction = (
            "Write a clear user-facing confirmation of the itinerary "
            "change. Explicitly state: "
            "(1) the day and activity/booking position changed, "
            "(2) the previous item and the new item when available, "
            "(3) that the affected day's route, map, and totals were "
            "recalculated when applicable, and "
            "(4) that unrelated itinerary items were preserved. "
            "Do not merely say that the itinerary is ready. "
            "Do not mention internal validation, candidate plans, "
            "output review, or internal state. "
            "Keep the response around 2 to 4 useful sentences."
        )

    else:
        instruction = (
            "Write a clear user-facing summary of the qualified "
            "itinerary. Mention the destination and the most important "
            "trip information that is useful to the traveller. "
            "Do not claim that an existing itinerary was changed unless "
            "the user actually requested a modification. "
            "Do not merely say that the itinerary is ready. "
            "Do not mention internal validation, candidate plans, "
            "output review, or internal state. "
            "Keep the response concise but informative."
        )

    # Append the instruction ONCE only.
    base_messages.append(
        HumanMessage(
            content=instruction,
        )
    )

    reply, decision, attempts, issue_codes = (
        await _review_reply_attempts(
            state,
            config,
            base_messages,
            candidate=candidate,
            max_attempts=_OUTPUT_REVIEW_MAX_ATTEMPTS,
            user_profile=user_profile,
        )
    )

    if reply is None:
        cumulative_attempts = (
            state.output_review_attempts
            + attempts
        )

        cumulative_issues = [
            *state.output_review_issue_codes,
            *issue_codes,
        ]

        # The itinerary itself has already passed deterministic
        # validation. If semantic response review is unavailable,
        # retain the qualified candidate and use the safe
        # server-owned response rather than rebuilding the trip.
        return _review_exhaustion_update(
            state,
            cumulative_attempts,
            cumulative_issues,
        )

    candidate["reply"] = reply
    candidate["review"] = decision.model_dump(
        mode="json",
    )

    return {
        "candidate_plan": candidate,
        "output_review_attempts": (
            state.output_review_attempts
            + attempts
        ),
        "output_review_issue_codes": [
            *state.output_review_issue_codes,
            *issue_codes,
        ],
    }


def _promotion_failure_update(
    state: AgentState,
    issue_codes: list[str],
) -> dict[str, Any]:
    """Handle candidate-promotion failure without corrupting accepted state."""

    # ---------------------------------------------------------
    # Incremental itinerary edits are atomic.
    #
    # If the edited candidate fails promotion/review, preserve
    # the currently accepted itinerary and STOP.
    #
    # Never convert an edit failure into a full planning retry,
    # because that could regenerate unrelated days, flights,
    # hotels, activities, or restaurants.
    # ---------------------------------------------------------
    if _current_turn_used_tool(
        state,
        "edit_itinerary",
    ):
        return {
            "messages": [
                AIMessage(
                    content=(
                        "I found the requested itinerary change, but I "
                        "couldn't verify the final updated plan, so I did "
                        "not save the change. Your existing itinerary has "
                        "been preserved exactly as it was."
                    ),
                    additional_kwargs={
                        "server_owned":
                            "incremental_edit_rejected",
                    },
                )
            ],

            # This is NOT a full-planning failure.
            "planning_outcome": None,
            "candidate_plan": None,
            "planning_attempts": 0,

            "planning_issue_codes": [
                f"planning.incremental.promotion.{code}"
                for code in issue_codes
            ],

            "output_review_attempts": 0,
            "output_review_issue_codes": [],

            **_completed_planning_cleanup(),
        }

    # ---------------------------------------------------------
    # Full itinerary generation may retry normally.
    # ---------------------------------------------------------
    if (
        state.planning_attempts
        < _PLANNING_TRANSACTION_MAX_ATTEMPTS
    ):
        return {
            "planning_outcome": "unavailable",
            "candidate_plan": None,
            "planning_issue_codes": [
                *state.planning_issue_codes,
                *issue_codes,
            ],
        }

    return {
        "planning_outcome": "unavailable",
        "candidate_plan": None,
        "planning_issue_codes": [
            *state.planning_issue_codes,
            *issue_codes,
        ],
        "messages": [
            _planning_unavailable_message(state)
        ],
        **_completed_planning_cleanup(),
    }

async def promote_candidate_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Defensively revalidate and atomically publish one candidate."""
    promotion_started = time.perf_counter()
    log_context = _planning_log_context(config, state)
    candidate = copy.deepcopy(state.candidate_plan)
    candidate_attempt = candidate.get("attempt", 0) if isinstance(candidate, dict) else 0

    def log_promotion(
        event: str,
        outcome: str,
        issue_codes: Sequence[str] = (),
    ) -> None:
        safe_observability_log(
            logger,
            event,
            attempt=candidate_attempt,
            stage="promotion",
            issue_codes=issue_codes,
            elapsed_ms=(time.perf_counter() - promotion_started) * 1000,
            outcome=outcome,
            **log_context,
        )

    log_promotion("planning.promotion", "started")
    if not isinstance(candidate, dict):
        log_promotion(
            "planning.promotion",
            "rejected",
            ("promotion.candidate.missing",),
        )
        return _promotion_failure_update(state, ["promotion.candidate.missing"])

    try:
        prior_review = OutputReviewDecision.model_validate(candidate.get("review"))
        reply = candidate.get("reply")
        if not prior_review.approved or not isinstance(reply, str):
            raise ValueError("candidate reply is not approved")
        local_review = deterministic_output_review(
            _output_review_context(
                state,
                reply,
                candidate,
                config=config,
                review_attempt=state.output_review_attempts + 1,
                review_stage="promotion_review",
            )
        )
        if not local_review.approved:
            log_promotion(
                "planning.promotion",
                "rejected",
                local_review.issue_codes,
            )
            return _promotion_failure_update(
                state,
                list(local_review.issue_codes),
            )

        financials = candidate.get("financials") or {}
        if not isinstance(financials, dict):
            raise ValueError("candidate financials must be a mapping")
        allowed_financials = _CANDIDATE_FINANCIAL_FIELDS | {
            "total_base_budget",
            "budget_assessment",
        }
        candidate_financials = {
            key: copy.deepcopy(value)
            for key, value in financials.items()
            if key in allowed_financials
        }
        candidate_state = state.model_copy(update=candidate_financials, deep=True)
        requirements = TripRequirements.from_state(candidate_state)
        validation = validate_itinerary_candidate(
            requirements,
            candidate.get("itinerary"),
            candidate.get("maps"),
        )
        if not validation.qualified:
            log_promotion(
                "planning.promotion",
                "rejected",
                tuple(issue.code for issue in validation.issues),
            )
            return _promotion_failure_update(
                state,
                [issue.code for issue in validation.issues],
            )

        revision = _effective_plan_revision(state) + 1

        # Preserve the currently accepted itinerary BEFORE overwriting it.
        # LangGraph checkpoints retain the latest state for normal execution,
        # but the agent/reviewer do not automatically read older checkpoints.
        # This compact history makes previous itinerary facts trusted state.
        previous_snapshot = state.accepted_plan_snapshot

        if (
            not isinstance(previous_snapshot, dict)
            and state.draft_itinerary
        ):
            previous_snapshot = {
                "plan_revision": _effective_plan_revision(state),
                "draft_itinerary": copy.deepcopy(
                    state.draft_itinerary
                ),
            }

        itinerary_history = append_previous_itinerary(
            state.itinerary_history,
            previous_snapshot,
        )

        committed: dict[str, Any] = {
            **candidate_financials,
            "draft_itinerary": copy.deepcopy(
                candidate["itinerary"]
            ),
            "daily_map_info": copy.deepcopy(
                candidate["maps"]
            ),
            "plan_revision": revision,
            "itinerary_history": itinerary_history,
        }

        snapshot = _build_accepted_plan_snapshot(
            state,
            revision=revision,
            values=committed,
        )
        
        # ---------------------------------------------------------
        # Do NOT run a second semantic Gemini review here.
        #
        # The exact public reply stored in this candidate has already
        # passed semantic review in generate_reviewed_response_node().
        # The approved review is stored inside candidate["review"] and
        # prior_review.approved was verified earlier in this function.
        #
        # Promotion still remains fail-closed because immediately before
        # committing we have already:
        #
        #   1. validated candidate["review"];
        #   2. checked prior_review.approved;
        #   3. re-run deterministic_output_review();
        #   4. rebuilt TripRequirements from the candidate state; and
        #   5. re-run validate_itinerary_candidate().
        #
        # If the original semantic reviewer was unavailable,
        # _review_exhaustion_update() replaces the response with fixed
        # server-owned fallback text that has independently passed the
        # deterministic output review.
        #
        # Calling review_public_output() for a second time here duplicates
        # the Gemini request and can incorrectly reject a fully validated
        # itinerary when the provider's per-minute quota is exhausted
        # between response_review and promotion_review.
        # ---------------------------------------------------------
    except asyncio.CancelledError:
        raise
    except Exception:
        log_promotion(
            "planning.promotion",
            "failed",
            ("promotion.candidate.invalid",),
        )
        return _promotion_failure_update(state, ["promotion.candidate.invalid"])

    log_promotion("planning.commit", "committed")
    return {
        **committed,
        "accepted_plan_snapshot": snapshot,
        "messages": [
            AIMessage(
                content=reply,
                additional_kwargs={"plan_revision": revision},
            )
        ],
        "planning_outcome": None,
        "candidate_plan": None,
        "planning_attempts": 0,
        "planning_issue_codes": [],
        "output_review_attempts": 0,
        "output_review_issue_codes": [],
        **_completed_planning_cleanup(),
    }


# ── 5.3  Map / GeoJSON Generation ──────────────────────


def run_map_update_node(state: AgentState) -> dict:
    """Geocode all locations, fetch Mapbox routing, build per-day GeoJSON."""
    try:
        return generate_daily_map(state)
    except Exception:
        return {"daily_map_info": {}}


# ── 5.4  Memory Extraction (Background, Non-Blocking) ──


async def memory_extraction_node(state: AgentState, config: RunnableConfig) -> dict:
    """Fire-and-forget memory extraction in a bounded thread pool.

    Analyzes the latest human message for long-term preferences
    (dietary restrictions, interests, accommodation style, etc.)
    and persists them to Supabase.  Never blocks graph execution.
    """
    new_turn_reset = _new_chat_turn_reset(state)
    configurable = config.get("configurable", {})
    user_id = configurable.get("__user_id")
    session_id = configurable.get("thread_id")

    if not state.messages or not user_id:
        return new_turn_reset

    last_msg = state.messages[-1]
    if last_msg.type != "human":
        return new_turn_reset

    user_message = _extract_text(last_msg.content)
    if not user_message.strip():
        return new_turn_reset

    safe_observability_log(
        logger,
        "planning.memory_extraction",
        stage="memory",
        outcome="queued",
        **_planning_log_context(config, state),
    )

    future = _memory_pool.submit(
        run_memory_extraction_task,
        session_id,
        user_id,
        user_message,
    )

    try:
        await asyncio.wrap_future(future)

    except Exception:
        safe_observability_log(
            logger,
            "planning.memory_extraction",
            stage="memory",
            issue_codes=(
                "memory.persistence_failed",
            ),
            outcome="unavailable",
            **_planning_log_context(
                config,
                state,
            ),
        )

    return new_turn_reset


def _new_chat_turn_reset(state: AgentState) -> dict[str, Any]:
    """Clear a finished unavailable transaction when the next chat starts."""
    if state.planning_outcome != "unavailable":
        return {}
    return {
        "planning_outcome": None,
        "candidate_plan": None,
        "planning_attempts": 0,
        "planning_issue_codes": [],
        "output_review_attempts": 0,
        "output_review_issue_codes": [],
    }


# ── 5.5  Core Agent (LLM Reasoning) ────────────────────


async def agent_node(state: AgentState, config: RunnableConfig) -> dict:
    """The core LLM decision engine.

    1. Fetches long-term user preferences from Supabase (async, non-blocking).
    2. Builds the system prompt with full trip context via the cached
       ``ChatPromptTemplate`` (compiled once via ``lru_cache``).
    3. Invokes the LLM asynchronously.
    4. Returns the AI response (which may contain ``tool_calls``).
    """
    user_id = config.get("configurable", {}).get("__user_id")

    # ── Fetch merged long-term memory ──
    user_profile: Optional[dict] = None

    if user_id:
        try:
            # Load both relational profile + Mem0 preferences.
            user_profile = await asyncio.to_thread(
                fetch_user_memory_context,
                user_id,
            )

        except Exception:
            safe_observability_log(
                logger,
                "planning.profile",
                stage="agent_profile",
                issue_codes=("profile.unavailable",),
                outcome="unavailable",
                **_planning_log_context(config, state),
            )

            user_profile = None

    # ── Build formatted prompt messages ──
    messages = build_prompt_messages(state, user_profile)

    # If the current user message is the immediate answer to a
    # server-owned day clarification, reconstruct the pending mutation
    # intent for the model. Provider evidence is NOT reused here.
    pending_instruction = (
        _pending_itinerary_mutation_instruction(state)
    )

    if pending_instruction is not None:
        messages = [
            *messages,
            HumanMessage(
                content=pending_instruction,
            ),
        ]

    # ---------------------------------------------------------
    # Deterministic protection for indexed activity/restaurant edits.
    #
    # Never let the LLM infer a destructive target day from an old
    # assistant message or previous edit. If the user names an indexed
    # activity/restaurant but does not identify the day, ask for the day
    # BEFORE calling the model or any mutation tool.
    # ---------------------------------------------------------
    clarification = _place_edit_day_clarification(state)

    if clarification is not None:
        return {
            "messages": [
                AIMessage(
                    content=clarification,
                    additional_kwargs={
                        "server_owned":
                            "mutation_target_day_required",
                    },
                )
            ],

            "pending_itinerary_mutation": {
                "status": "awaiting_day",
                "source_request":
                    _latest_user_request(state),
                "edits": [],
            },

            "output_review_attempts": 0,
            "output_review_issue_codes": [],
        }

    safe_observability_log(
        logger,
        "planning.agent",
        stage="agent",
        outcome="started",
        **_planning_log_context(config, state),
    )

    # ── Async LLM invocation ──
    try:
        response = await llm_with_tools.ainvoke(messages, config=config)
    except Exception:
        safe_observability_log(
            logger,
            "planning.agent",
            stage="agent",
            issue_codes=("model.unavailable",),
            outcome="unavailable",
            **_planning_log_context(config, state),
        )
        response = AIMessage(
            content=(
                "I apologise, but I encountered a temporary issue while "
                "processing your request. Could you please try again?"
            )
        )

    if response.tool_calls:
        return {"messages": [response]}

    # A direct mutation request must not end as prose-only "success".
    # Give the tool-enabled model one bounded correction attempt so it can
    # execute the required mutation path instead of merely describing it.
    if _explicit_itinerary_mutation_request(state):
        place_search_completed = _current_turn_used_tool(
            state,
            "search_places",
        )

        if place_search_completed:
            # Provider evidence already exists in this same user turn.
            # At this point the remaining mutation tool is edit_itinerary.
            correction_model = edit_continuation_llm

            correction_instruction = (
                "Internal mutation continuation: search_places has already "
                "completed successfully in this same user turn. Inspect the "
                "most recent search_places result and the current itinerary. "
                "For activity/restaurant remove or replace, use the server's "
                "authorized target-day rules: an explicit day in the current "
                "message, a server-owned day-clarification reply, or an immediate "
                "yes-confirmation where the same day was already explicitly "
                "supplied by the user before the proposal. Never use an "
                "assistant-only guessed day. If no authorized day exists, ask "
                "exactly one concise clarification question and do not call "
                "edit_itinerary. Otherwise, call edit_itinerary "
                "using the exact provider-grounded result or results. If the "
                "current request contains multiple edits, include EVERY requested "
                "edit in ONE edit_itinerary call; do not apply only the first "
                "change. Do not output JSON, raw tool data, or a prose success "
                "message."
            )

        else:
            # No provider place evidence exists yet, so the normal tool-enabled
            # agent is still required to perform the search first.
            correction_model = llm_with_tools

            correction_instruction = (
                "Internal execution requirement: the current user message "
                "asks to continue or perform a change to the existing trip. "
                "Do not claim that anything was changed in prose unless you "
                "execute the appropriate mutation tool. Use edit_itinerary for "
                "itinerary items; for an activity/restaurant add or replacement, "
                "call search_places first in this turn, then edit_itinerary. "
                "Honor the server-authorized continuation rules for an immediate "
                "yes-confirmation or a server-owned day clarification reply; "
                "do not re-ask for a day that the server has already authorized. "
                "If essential target information is genuinely missing, ask one "
                "concise clarification question instead."
            )

        correction_messages = [
            *messages,
            HumanMessage(
                content=correction_instruction,
            ),
        ]

        try:
            corrected = await correction_model.ainvoke(
                correction_messages,
                config=config,
            )

        except Exception:
            safe_observability_log(
                logger,
                "planning.agent_mutation_retry",
                stage="agent",
                issue_codes=("model.unavailable",),
                outcome="unavailable",
                **_planning_log_context(config, state),
            )

            if place_search_completed:
                return {
                    "messages": [
                        AIMessage(
                            content=(
                                "I found matching place options, but I could not "
                                "complete the itinerary update right now. "
                                "Please try the modification again."
                            ),
                            additional_kwargs={
                                "server_owned":
                                    "mutation_model_unavailable",
                            },
                        )
                    ],
                    "output_review_attempts": 0,
                    "output_review_issue_codes": [
                        "model.unavailable",
                    ],
                }

        else:
            if corrected.tool_calls:
                safe_observability_log(
                    logger,
                    "planning.agent_mutation_retry",
                    stage="agent",
                    outcome="tool_call_recovered",
                    **_planning_log_context(config, state),
                )

                return {
                    "messages": [
                        corrected
                    ]
                }

            if place_search_completed:
                # search_places already succeeded, but the tool-enabled model
                # still failed to produce edit_itinerary.
                #
                # Do NOT enter _review_reply_attempts(candidate=None), because
                # that review loop has no tools and therefore cannot create the
                # required private itinerary candidate.
                safe_observability_log(
                    logger,
                    "planning.agent_mutation_retry",
                    stage="agent",
                    issue_codes=(
                        "mutation.tool_call_missing",
                    ),
                    outcome="clarification_required",
                    **_planning_log_context(
                        config,
                        state,
                    ),
                )

                return {
                    "messages": [
                        AIMessage(
                            content=(
                                "I found matching place options, but I still "
                                "need the exact itinerary target before I can "
                                "save the change safely. Please specify the day "
                                "and whether you want to add or replace an "
                                "activity/restaurant; include the item number "
                                "if you want to replace an existing item."
                            ),
                            additional_kwargs={
                                "server_owned":
                                    "mutation_target_required",
                            },
                        )
                    ],
                    "output_review_attempts": 0,
                    "output_review_issue_codes": [
                        "mutation.tool_call_missing",
                    ],
                }

            # Without a completed provider search, a prose response may
            # legitimately be a clarification question. Let normal output
            # review validate it.
            response = corrected

    reply, _decision, attempts, issue_codes = await _review_reply_attempts(
        state,
        config,
        messages,
        candidate=None,
        initial_response=response,
        user_profile=user_profile,
    )
    if reply is None:
        return _review_exhaustion_update(state, attempts, issue_codes)
    return {
        "messages": [AIMessage(content=reply)],
        "output_review_attempts": attempts,
        "output_review_issue_codes": list(issue_codes),
    }


async def budget_decision_agent_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict:
    """Interpret a pending reply with only the two non-mutating budget tools."""
    messages = build_budget_decision_messages(state)
    safe_observability_log(
        logger,
        "planning.budget_decision",
        stage="budget_decision",
        outcome="started",
        **_planning_log_context(config, state),
    )
    try:
        response = await budget_decision_llm.ainvoke(messages, config=config)
    except Exception:
        safe_observability_log(
            logger,
            "planning.budget_decision",
            stage="budget_decision",
            issue_codes=("model.unavailable",),
            outcome="unavailable",
            **_planning_log_context(config, state),
        )
        response = AIMessage(
            content=(
                "I could not interpret that budget reply. Please confirm the "
                "recommended amount, provide another positive amount, or request "
                "a refreshed recommendation."
            )
        )
    if response.tool_calls:
        return {"messages": [response]}
    reply, _decision, attempts, issue_codes = await _review_reply_attempts(
        state,
        config,
        messages,
        candidate=None,
        initial_response=response,
    )
    if reply is None:
        return _review_exhaustion_update(state, attempts, issue_codes)
    return {
        "messages": [AIMessage(content=reply)],
        "output_review_attempts": attempts,
        "output_review_issue_codes": list(issue_codes),
    }


class _InvalidAcceptedDecision(ValueError):
    """The evaluator returned acceptance without authoritative evidence."""


def _budget_assessment_failure_update(
    state: AgentState,
    reason: str,
) -> dict[str, Any]:
    return {
        "pending_budget_proposal": None,
        "pending_budget_confirmation": state.pending_budget_confirmation,
        "accepted_budget_decision": None,
        "budget_gate_outcome": "budget_check_unavailable",
        "budget_gate_reason": reason,
        "budget_gate_message": _BUDGET_GATE_SAFE_FALLBACK,
        "chat_budget_action": None,
        "chat_budget_assessment_id": None,
    }


def _validated_accepted_budget_update(
    state: AgentState,
    proposal: Any,
    decision: ChatBudgetDecision,
) -> dict[str, Any]:
    amount = decision.accepted_total_base_budget
    assessment = decision.assessment
    if (
        isinstance(amount, bool)
        or amount is None
        or not math.isfinite(float(amount))
        or float(amount) <= 0
        or assessment is None
        or not isinstance(proposal, dict)
    ):
        raise _InvalidAcceptedDecision

    accepted_amount = float(amount)
    if not assessment_matches_trip(
        assessment,
        origin=state.origin_country or "",
        destination=state.country or "",
        destination_city=primary_destination_city(state.city),
        start_date=state.start_date or "",
        end_date=state.end_date or "",
        num_people=max(1, int(state.num_people or 1)),
    ):
        raise _InvalidAcceptedDecision
    if not is_budget_sufficient(accepted_amount, assessment):
        raise _InvalidAcceptedDecision

    mode = proposal.get("mode")
    if mode == "amount":
        proposed_amount = proposal.get("total_base_budget")
        if isinstance(proposed_amount, bool):
            raise _InvalidAcceptedDecision
        if accepted_amount != float(proposed_amount):
            raise _InvalidAcceptedDecision
    elif mode == "confirm":
        pending = state.pending_budget_confirmation
        if not isinstance(pending, dict):
            raise _InvalidAcceptedDecision
        recommended = float(pending.get("recommended_minimum_budget"))
        pending_id = pending.get("budget_assessment_id")
        supplied_id = proposal.get("assessment_id")
        if (
            pending_id != assessment.assessment_id
            or supplied_id not in {None, assessment.assessment_id}
            or recommended != assessment.recommended_minimum_budget
            or accepted_amount != recommended
        ):
            raise _InvalidAcceptedDecision
    else:
        raise _InvalidAcceptedDecision

    return {
        "accepted_budget_decision": {
            "stage": "assessment",
            "accepted_total_base_budget": accepted_amount,
            "assessment_id": assessment.assessment_id,
            "assessment": assessment.model_dump(),
            "target_plan_revision": _effective_plan_revision(state) + 1,
            "candidate": {},
        },
        "pending_budget_confirmation": None,
    }


def _local_accepted_budget_decision(
    state: AgentState,
    *,
    stages: set[str],
) -> tuple[dict[str, Any], float, BudgetAssessment] | None:
    """Parse the accepted record without treating checkpoint data as authority."""
    record = state.accepted_budget_decision
    if (
        state.budget_gate_outcome != "accepted"
        or state.pending_budget_confirmation is not None
        or not isinstance(record, dict)
        or record.get("stage") not in stages
    ):
        return None
    amount = record.get("accepted_total_base_budget")
    if isinstance(amount, bool):
        return None
    try:
        accepted_amount = float(amount)
        assessment = BudgetAssessment.model_validate(record.get("assessment"))
    except (TypeError, ValueError):
        return None
    target_revision = record.get("target_plan_revision")
    candidate = record.get("candidate")
    if (
        not math.isfinite(accepted_amount)
        or accepted_amount <= 0
        or isinstance(target_revision, bool)
        or not isinstance(target_revision, int)
        or target_revision != _effective_plan_revision(state) + 1
        or not isinstance(candidate, dict)
        or record.get("assessment_id") != assessment.assessment_id
        or not assessment_matches_trip(
            assessment,
            origin=state.origin_country or "",
            destination=state.country or "",
            destination_city=primary_destination_city(state.city),
            start_date=state.start_date or "",
            end_date=state.end_date or "",
            num_people=max(1, int(state.num_people or 1)),
        )
        or not is_budget_sufficient(accepted_amount, assessment)
    ):
        return None
    return record, accepted_amount, assessment


async def validate_accepted_budget_node(state: AgentState) -> dict[str, Any]:
    """Reload the exact accepted assessment before entering planning."""
    parsed = _local_accepted_budget_decision(
        state,
        stages=_BUDGET_REPLAN_STAGES,
    )
    if parsed is None:
        return _budget_assessment_failure_update(
            state,
            "invalid_accepted_decision",
        )
    record, accepted_amount, assessment = parsed
    persisted = await asyncio.to_thread(
        load_confirmed_budget_assessment,
        assessment_id=assessment.assessment_id,
        origin=state.origin_country or "",
        destination=state.country or "",
        destination_city=primary_destination_city(state.city),
        start_date=state.start_date or "",
        end_date=state.end_date or "",
        num_people=max(1, int(state.num_people or 1)),
    )
    if persisted is None or persisted.model_dump() != assessment.model_dump():
        return _budget_assessment_failure_update(
            state,
            "assessment_expired_or_invalid",
        )
    return {
        "accepted_budget_decision": {
            **record,
            "stage": "currency",
            "assessment": persisted.model_dump(),
            "candidate": {},
        },
    }


def _budget_replan_parts(
    state: AgentState,
    expected_stage: str,
) -> tuple[dict[str, Any], float, BudgetAssessment, dict[str, Any]]:
    parsed = _local_accepted_budget_decision(state, stages={expected_stage})
    if parsed is None:
        raise _InvalidAcceptedDecision
    record, amount, assessment = parsed
    candidate = copy.deepcopy(record["candidate"])
    return record, amount, assessment, candidate


def _advanced_budget_replan_record(
    record: dict[str, Any],
    *,
    stage: str,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    return {
        **record,
        "stage": stage,
        "candidate": candidate,
    }


def _budget_replan_failure_update(state: AgentState, reason: str) -> dict[str, Any]:
    return {
        "accepted_budget_decision": None,
        "pending_budget_proposal": None,
        "pending_budget_confirmation": None,
        "budget_gate_outcome": "budget_check_unavailable",
        "budget_gate_reason": reason,
        "budget_gate_message": _BUDGET_GATE_SAFE_FALLBACK,
        "chat_budget_action": None,
        "chat_budget_assessment_id": None,
    }


def _candidate_agent_state(
    state: AgentState,
    *,
    amount: float,
    assessment: BudgetAssessment,
    candidate: dict[str, Any],
) -> AgentState:
    return state.model_copy(
        update={
            **candidate,
            "total_base_budget": amount,
            "budget_assessment": assessment.model_dump(),
        },
        deep=True,
    )


def _planning_transaction_state(state: AgentState) -> AgentState:
    """Apply private accepted-budget financials to transaction input only."""
    parsed = _local_accepted_budget_decision(
        state,
        stages={"itinerary", "activities", "maps", "commit"},
    )
    if parsed is None:
        return state
    record, amount, assessment = parsed
    return _candidate_agent_state(
        state,
        amount=amount,
        assessment=assessment,
        candidate=copy.deepcopy(record["candidate"]),
    )


def restart_budget_replan_node(state: AgentState) -> dict[str, Any]:
    """Safely restart a persisted legacy budget-planning checkpoint."""
    parsed = _local_accepted_budget_decision(
        state,
        stages={"itinerary", "activities", "maps", "commit"},
    )
    if parsed is None:
        return _budget_replan_failure_update(state, "invalid_accepted_decision")
    record, _, _ = parsed
    return {
        "accepted_budget_decision": {
            **record,
            "stage": "currency",
            "candidate": {},
        }
    }


def prepare_budget_replan_node(state: AgentState) -> dict[str, Any]:
    """Build candidate currency/allocation data without touching the old plan."""
    try:
        record, amount, assessment, candidate = _budget_replan_parts(
            state,
            "currency",
        )
        prepared = process_initial_form_node(
            _candidate_agent_state(
                state,
                amount=amount,
                assessment=assessment,
                candidate=candidate,
            )
        )
        candidate.update(
            {
                key: copy.deepcopy(prepared[key])
                for key in _CANDIDATE_FINANCIAL_FIELDS
                if key in prepared
            }
        )
        candidate["total_base_budget"] = amount
        candidate["budget_assessment"] = assessment.model_dump()
        return {
            "accepted_budget_decision": _advanced_budget_replan_record(
                record,
                stage="itinerary",
                candidate=candidate,
            )
        }
    except Exception:
        return _budget_replan_failure_update(state, "budget_replan_currency_failed")


def plan_budget_replan_itinerary_node(state: AgentState) -> dict[str, Any]:
    """Build candidate flights/hotels while committed itinerary stays visible."""
    try:
        record, amount, assessment, candidate = _budget_replan_parts(
            state,
            "itinerary",
        )
        candidate_state = _candidate_agent_state(
            state,
            amount=amount,
            assessment=assessment,
            candidate={**candidate, "draft_itinerary": [], "daily_map_info": {}},
        )
        planned = plan_initial_itinerary_node(candidate_state)
        itinerary = planned.get("draft_itinerary")
        if not isinstance(itinerary, list) or not itinerary:
            return _budget_replan_failure_update(
                state,
                "budget_replan_itinerary_failed",
            )
        candidate["draft_itinerary"] = copy.deepcopy(itinerary)
        return {
            "accepted_budget_decision": _advanced_budget_replan_record(
                record,
                stage="activities",
                candidate=candidate,
            )
        }
    except Exception:
        return _budget_replan_failure_update(state, "budget_replan_itinerary_failed")


async def plan_budget_replan_activities_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Build candidate activities without exposing a partial itinerary."""
    try:
        record, amount, assessment, candidate = _budget_replan_parts(
            state,
            "activities",
        )
        candidate_state = _candidate_agent_state(
            state,
            amount=amount,
            assessment=assessment,
            candidate=candidate,
        )
        planned = await plan_activities_node(candidate_state, config)
        itinerary = planned.get("draft_itinerary", candidate.get("draft_itinerary"))
        if not isinstance(itinerary, list) or not itinerary:
            return _budget_replan_failure_update(
                state,
                "budget_replan_activities_failed",
            )
        candidate["draft_itinerary"] = copy.deepcopy(itinerary)
        return {
            "accepted_budget_decision": _advanced_budget_replan_record(
                record,
                stage="maps",
                candidate=candidate,
            )
        }
    except Exception:
        return _budget_replan_failure_update(state, "budget_replan_activities_failed")


def plan_budget_replan_maps_node(state: AgentState) -> dict[str, Any]:
    """Build candidate maps and advance only the private replan record."""
    try:
        record, amount, assessment, candidate = _budget_replan_parts(state, "maps")
        candidate_state = _candidate_agent_state(
            state,
            amount=amount,
            assessment=assessment,
            candidate=candidate,
        )
        mapped = run_map_update_node(candidate_state)
        maps = mapped.get("daily_map_info")
        if not isinstance(maps, dict):
            return _budget_replan_failure_update(state, "budget_replan_maps_failed")
        candidate["daily_map_info"] = copy.deepcopy(maps)
        return {
            "accepted_budget_decision": _advanced_budget_replan_record(
                record,
                stage="commit",
                candidate=candidate,
            )
        }
    except Exception:
        return _budget_replan_failure_update(state, "budget_replan_maps_failed")


def commit_budget_replan_node(state: AgentState) -> dict[str, Any]:
    """Legacy checkpoint shim; reviewed promotion is the only commit boundary."""
    return restart_budget_replan_node(state)


async def assess_chat_budget_node(state: AgentState) -> dict[str, Any]:
    """Run the deterministic gate and expose only its authorized state update."""
    proposal = state.pending_budget_proposal
    if state.chat_budget_action == "accept_recommended":
        proposal = {
            "mode": "confirm",
            "total_base_budget": None,
            "assessment_id": state.chat_budget_assessment_id,
        }

    try:
        raw_decision = await asyncio.to_thread(
            evaluate_chat_budget,
            proposal=proposal or {},
            pending_confirmation=state.pending_budget_confirmation,
            current_assessment=state.budget_assessment,
            origin=state.origin_country or "",
            destination=state.country or "",
            destination_city=primary_destination_city(state.city),
            start_date=state.start_date or "",
            end_date=state.end_date or "",
            num_people=max(1, int(state.num_people or 1)),
        )
        decision = ChatBudgetDecision.model_validate(raw_decision)

        logger.info(
            "Chat budget gate decision: status=%s reason=%s",
            decision.status,
            decision.reason,
        )

        updates: Dict[str, Any] = {
            "pending_budget_proposal": None,
            "budget_gate_outcome": decision.status,
            "budget_gate_reason": decision.reason,
            "budget_gate_message": (
                _BUDGET_CONFIRMATION_SAFE_COPY
                if decision.status == "budget_confirmation_required"
                else _BUDGET_GATE_SAFE_FALLBACK
            ),
            "chat_budget_action": None,
            "chat_budget_assessment_id": None,
        }
        if decision.status == "accepted":
            updates.update(
                _validated_accepted_budget_update(state, proposal, decision)
            )
        else:
            updates["pending_budget_confirmation"] = (
                decision.pending_confirmation
                if decision.pending_confirmation is not None
                else state.pending_budget_confirmation
            )
        return updates
    except _InvalidAcceptedDecision:
        return _budget_assessment_failure_update(
            state,
            "invalid_accepted_decision",
        )
    except Exception:
        return _budget_assessment_failure_update(state, "budget_gate_exception")


async def budget_gate_response_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Review a non-accepted budget reply before ending the turn."""
    content = (
        _BUDGET_CONFIRMATION_SAFE_COPY
        if state.budget_gate_outcome == "budget_confirmation_required"
        else _BUDGET_GATE_SAFE_FALLBACK
    )
    base_messages = build_prompt_messages(state)
    base_messages.append(
        HumanMessage(
            content=(
                "Write a concise public response for the current budget gate. "
                "Do not claim planning succeeded or expose internal state."
            )
        )
    )
    reply, _decision, attempts, issue_codes = await _review_reply_attempts(
        state,
        config,
        base_messages,
        candidate=None,
        initial_response=AIMessage(content=content),
    )
    if reply is None:
        update = _review_exhaustion_update(state, attempts, issue_codes)
        update["budget_gate_message"] = _OUTPUT_REVIEW_SAFE_FALLBACK
        return update
    return {
        "budget_gate_message": reply,
        "messages": [AIMessage(content=reply)],
        "output_review_attempts": attempts,
        "output_review_issue_codes": list(issue_codes),
    }


# ── 5.6  Post-Tool Processing (State Application) ──────


def _budget_proposal_from_tool_message(message: BaseMessage) -> Optional[dict]:
    """Validate and normalize a safe budget-tool result."""
    tool_name = getattr(message, "name", None)
    if tool_name not in _BUDGET_PROPOSAL_TOOLS:
        return None
    content = _parse_tool_content(message.content)
    if not content or content.get("status") != "budget_proposal":
        return None
    proposal = content.get("proposal")
    if not isinstance(proposal, dict):
        return None
    mode = proposal.get("mode")
    if tool_name == "confirm_recommended_budget":
        if mode != "confirm":
            return None
        return {"mode": "confirm", "total_base_budget": None}
    if mode == "recommendation":
        return {"mode": "recommendation", "total_base_budget": None}
    if mode != "amount":
        return None
    try:
        amount = float(proposal.get("total_base_budget"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(amount) or amount <= 0:
        return None
    return {"mode": "amount", "total_base_budget": amount}


def _resolve_budget_tool_batch(
    messages: List[BaseMessage],
) -> tuple[str, Optional[dict]]:
    """Classify budget-tool presence without conflating invalid and absent."""
    budget_messages = [
        message
        for message in messages
        if getattr(message, "name", None) in _BUDGET_PROPOSAL_TOOLS
    ]
    if not budget_messages:
        return "absent", None

    proposals: List[dict] = []
    for message in budget_messages:
        proposal = _budget_proposal_from_tool_message(message)
        if proposal is None:
            return "invalid_budget_tool_result", None
        proposals.append(proposal)

    first = proposals[0]
    if any(proposal != first for proposal in proposals[1:]):
        return "conflicting_budget_tool_results", None
    return "coherent", first


def _budget_tool_failure_update(state: AgentState, reason: str) -> dict[str, Any]:
    return {
        "pending_budget_proposal": None,
        "pending_budget_confirmation": state.pending_budget_confirmation,
        "budget_gate_outcome": "budget_check_unavailable",
        "budget_gate_reason": reason,
        "budget_gate_message": _BUDGET_GATE_SAFE_FALLBACK,
        "chat_budget_action": None,
        "chat_budget_assessment_id": None,
    }


def _budget_gate_is_unresolved(state: AgentState) -> bool:
    return (
        state.pending_budget_confirmation is not None
        or state.budget_gate_outcome in _NON_ACCEPTED_BUDGET_OUTCOMES
    )


def post_tool_processing_node(state: AgentState) -> dict:
    """Normalize tool results into requirements or a private plan candidate.

    Several tools return state-update dicts, but ``ToolNode`` only wraps
    their outputs as ``ToolMessage`` objects — it does **not** merge
    returned dicts into state fields.  This node intercepts tool outputs
    from the **current turn** (all ``ToolMessage`` objects after the most
    recent ``AIMessage``) and applies the returned state updates.

    Trip-requirement tools update only their server-owned requirement fields.
    Itinerary and category-budget edits are deep-copied, mapped, validated,
    and returned as a private candidate for the shared review/promotion gate.
    """
    updates: Dict[str, Any] = {}
    candidate_financials: Dict[str, Any] = {}

    # ── Collect tool messages from the current turn only ──
    recent_tool_msgs: List[BaseMessage] = []
    for msg in reversed(state.messages):
        if msg.type == "ai":
            break
        if msg.type == "tool":
            recent_tool_msgs.insert(0, msg)

    # Any budget-tool presence dominates the entire tool batch, independent
    # of message order. Only one coherent intent (or identical duplicates) may
    # proceed to assessment; malformed and conflicting results fail closed.
    batch_status, proposal = _resolve_budget_tool_batch(recent_tool_msgs)
    if batch_status == "coherent":
        return {
            "pending_budget_proposal": proposal,
            "budget_gate_outcome": None,
            "budget_gate_reason": None,
            "budget_gate_message": None,
        }
    if batch_status != "absent":
        return _budget_tool_failure_update(state, batch_status)

    # While a recommendation is pending, the restricted interpreter is the
    # only authorized decision-maker. Ignore any injected/stale tool outputs.
    if _budget_gate_is_unresolved(state):
        return {}

    # ── Lazy-copy itinerary (only if a booking modification occurs) ──
    itinerary_copy: Optional[List[dict]] = None
    flight_edit_requested = False
    trusted_booking_options = _trusted_booking_options(state)
    trusted_place_options = _trusted_place_options(state)

    logger.info(
        "Incremental edit grounding: "
        "search_places_used=%s trusted_place_options=%d",
        _current_turn_used_tool(
            state,
            "search_places",
        ),
        len(trusted_place_options),
    )

    origin_country_code = _country_alpha_2(state.origin_country)
    destination_country_code = _country_alpha_2(state.country)

    for msg in recent_tool_msgs:
        content = _parse_tool_content(msg.content)
        if not content:
            continue

        tool_name = getattr(msg, "name", None)

        # ── Trip-requirement updates (party size, dates, destination…) ──
        if tool_name == "update_trip_details":
            trip_updates = _validated_trip_updates(
                state, content.get("updates") or {}
            )
            updates.update(trip_updates)
            continue

        # ── Budget-category updates ──
        if tool_name == "update_budget_category":
            if "budget_allocation" in content:
                candidate_financials["budget_allocation"] = copy.deepcopy(
                    content["budget_allocation"]
                )

        # ── Unified itinerary edits (flight/hotel/activity/restaurant) ──
        elif tool_name == "edit_itinerary":
            edits = content.get("edits") or []

            if not edits:
                continue

            edit_error: str | None = None

            # ---------------------------------------------------------
            # Validate ALL requested edits before changing the candidate.
            #
            # This makes a multi-edit request atomic:
            #
            #   all valid   -> apply all
            #   one invalid -> apply none
            #
            # The currently accepted itinerary therefore cannot be
            # partially or unexpectedly modified.
            # ---------------------------------------------------------
            for edit in edits:
                if not isinstance(edit, dict):
                    edit_error = (
                        "I couldn't apply that itinerary change because "
                        "the edit request was invalid. Your existing "
                        "itinerary has not been changed."
                    )
                    break

                target_error = _incremental_edit_target_error(
                    state,
                    edit,
                )

                if target_error is not None:
                    edit_error = target_error
                    break

                if (
                    edit.get("category") in {"activity", "restaurant"}
                    and edit.get("action") in {"add", "replace"}
                ):
                    details = edit.get("new_details") or {}

                    logger.info(
                        "Place grounding check: name=%r trusted_options=%d",
                        details.get("name")
                        if isinstance(details, dict)
                        else None,
                        len(trusted_place_options),
                    )

                if not _booking_edit_is_grounded(
                    state,
                    edit,
                    trusted_booking_options,
                ):
                    edit_error = (
                        "I found the requested change, but I couldn't "
                        "verify the selected replacement against the "
                        "current provider results. Your existing itinerary "
                        "has not been changed."
                    )
                    break

                if (
                    edit.get("category") in {"activity", "restaurant"}
                    and edit.get("action") in {"add", "replace"}
                ):
                    trusted_place = _matching_trusted_place_option(
                        edit,
                        trusted_place_options,
                    )

                    # The LLM is instructed to perform search_places first,
                    # but prompt compliance is not a reliable control flow
                    # guarantee. If it skipped that step (or the earlier search
                    # yielded no usable evidence), perform a fresh server-owned
                    # provider lookup in this same user turn before rejecting.
                    if trusted_place is None:
                        recovered_place_options = (
                            _recover_trusted_place_options_for_edit(
                                state,
                                edit,
                            )
                        )

                        trusted_place = _matching_trusted_place_option(
                            edit,
                            recovered_place_options,
                        )

                    if trusted_place is None:
                        edit_error = (
                            "I found the requested change, but I couldn't "
                            "verify the selected replacement against the "
                            "current provider results. Your existing itinerary "
                            "has not been changed."
                        )
                        break

                    # Never save a model-reconstructed place payload.
                    # Once its identity matches fresh provider evidence,
                    # replace it with the exact server-trusted provider result.
                    edit["new_details"] = trusted_place

            if edit_error is not None:
                awaiting_day = edit_error.startswith(
                    "Before I change that activity, please tell me which day "
                )

                return {
                    "messages": [
                        AIMessage(
                            content=edit_error,
                            additional_kwargs={
                                "server_owned": (
                                    "mutation_target_day_required"
                                    if awaiting_day
                                    else "incremental_edit_rejected"
                                )
                            },
                        )
                    ],

                    "pending_itinerary_mutation": (
                        {
                            **_sanitised_pending_itinerary_mutation(
                                edits
                            ),
                            "source_request":
                                _latest_user_request(state),
                        }
                        if awaiting_day
                        else None
                    ),

                    # IMPORTANT:
                    # An incremental edit failure is NOT a planning failure.
                    "planning_outcome": None,
                    "candidate_plan": None,
                    "planning_attempts": 0,
                    "planning_issue_codes": [],
                }

            # All edits are valid. Work only on an isolated copy.
            if itinerary_copy is None:
                itinerary_copy = [
                    copy.deepcopy(day)
                    for day in state.draft_itinerary
                ]

            for edit in edits:
                _apply_single_edit(
                    itinerary_copy,
                    edit,
                    origin_country_code=origin_country_code,
                    destination_country_code=destination_country_code,
                )

                if (
                    edit.get("category") == "flight"
                    and edit.get("action")
                    in {"replace", "remove"}
                ):
                    flight_edit_requested = True

        # ── Booking modifications (legacy single flight/hotel swap) ──
        elif tool_name == "modify_existing_booking":
            day_num = content.get("target_day")
            category = content.get("target_category")
            new_data = content.get("new_data")

            if day_num is None or not new_data:
                continue
            legacy_edit = {
                "day": day_num,
                "action": "replace",
                "category": category,
                "new_details": new_data,
            }
            if not _booking_edit_is_grounded(
                state,
                legacy_edit,
                trusted_booking_options,
            ):
                continue

            # Check if target day exists BEFORE expensive deepcopy.
            # This avoids unnecessary allocation when the LLM hallucinates
            # a day number that doesn't exist in the itinerary.
            day_exists = any(
                isinstance(d, dict) and d.get("day") == day_num
                for d in state.draft_itinerary
            )
            if not day_exists:
                continue

            # Deep-copy itinerary ONCE — reuse for multiple modifications.
            # Previous implementation deep-copied per tool message, causing
            # later modifications to overwrite earlier ones.
            if itinerary_copy is None:
                itinerary_copy = [copy.deepcopy(day) for day in state.draft_itinerary]

            _apply_single_edit(
                itinerary_copy,
                legacy_edit,
                origin_country_code=origin_country_code,
                destination_country_code=destination_country_code,
            )
            if category == "flight":
                flight_edit_requested = True


    if itinerary_copy is not None and flight_edit_requested:
        allocation = candidate_financials.get(
            "budget_allocation", state.budget_allocation
        )
        _reassess_boundary_flight_budget(itinerary_copy, allocation)

    # Requirement changes trigger a clean full transaction on the next node.
    # Do not combine them with an incremental candidate from the same batch.
    if updates:
        return updates

    itinerary_changed = (
        itinerary_copy is not None and itinerary_copy != state.draft_itinerary
    )
    if not itinerary_changed and not candidate_financials:
        return {}

    try:
        candidate_itinerary = copy.deepcopy(
            itinerary_copy if itinerary_changed else state.draft_itinerary
        )
        candidate_state = state.model_copy(
            update={
                **candidate_financials,
                "draft_itinerary": candidate_itinerary,
            },
            deep=True,
        )
        candidate_maps = copy.deepcopy(state.daily_map_info)

        if itinerary_changed:
            mapped = generate_daily_map(candidate_state)

            if not isinstance(mapped, dict):
                raise ValueError("map generation returned an invalid result")

            mapped_itinerary = mapped.get("draft_itinerary")
            mapped_maps = mapped.get("daily_map_info")

            if not isinstance(mapped_itinerary, list):
                raise ValueError(
                    "map generation did not return draft_itinerary"
                )

            if not isinstance(mapped_maps, dict):
                raise ValueError(
                    "map generation did not return daily_map_info"
                )

            # IMPORTANT:
            # Map generation recalculates route metadata inside each itinerary day.
            # Keep the regenerated itinerary and its maps together so deterministic
            # validation compares the same candidate snapshot.
            candidate_itinerary = copy.deepcopy(mapped_itinerary)
            candidate_maps = copy.deepcopy(mapped_maps)

            candidate_state = candidate_state.model_copy(
                update={
                    "draft_itinerary": candidate_itinerary,
                },
                deep=True,
            )

        report = validate_itinerary_candidate(
            TripRequirements.from_state(candidate_state),
            candidate_itinerary,
            candidate_maps,
        )
    except Exception as exc:
        logger.warning(
            "Incremental itinerary edit validation failed.",
            exc_info=True,
        )

        return {
            "messages": [
                AIMessage(
                    content=(
                        "I found the requested change, but I couldn't "
                        "apply it safely to the current itinerary. "
                        "Your existing itinerary has been kept exactly "
                        "as it was."
                    ),
                    additional_kwargs={
                        "server_owned":
                            "incremental_edit_rejected"
                    },
                )
            ],
            "planning_outcome": None,
            "candidate_plan": None,
            "planning_attempts": 0,
            "planning_issue_codes": [
                (
                    "planning.incremental."
                    f"{type(exc).__name__.casefold()}"
                )
            ],
        }

    if not report.qualified:
        issue_codes = [
            issue.code
            for issue in report.issues
        ]

        logger.warning(
            "Incremental itinerary candidate rejected: %s",
            issue_codes,
        )

        return {
            "messages": [
                AIMessage(
                    content=(
                        "I found the requested change, but the updated "
                        "day did not pass itinerary validation, so I "
                        "did not save it. Your existing itinerary is "
                        "unchanged."
                    ),
                    additional_kwargs={
                        "server_owned":
                            "incremental_edit_rejected"
                    },
                )
            ],
            "planning_outcome": None,
            "candidate_plan": None,
            "planning_attempts": 0,
            "planning_issue_codes": [
                f"planning.incremental.{code}"
                for code in issue_codes
            ],
        }
    candidate_plan: dict[str, Any] = {
        "itinerary": candidate_itinerary,
        "maps": candidate_maps,
        "attempt": 1,
        "validation": report.model_dump(mode="json"),
    }
    if candidate_financials:
        candidate_plan["financials"] = copy.deepcopy(candidate_financials)
    return {
        "planning_outcome": "validated",
        "candidate_plan": candidate_plan,
        "planning_attempts": 1,
        "planning_issue_codes": [],
        "output_review_attempts": 0,
        "output_review_issue_codes": [],
    }


# ═══════════════════════════════════════════════════════════
# 6. ROUTING LOGIC
# ═══════════════════════════════════════════════════════════


def route_start(state: AgentState) -> str:
    """Entry router — new trip vs. ongoing chat.

    A new trip has neither a draft itinerary nor a base currency code.
    If either is set, the trip has been at least partially initialised
    → route to ongoing-chat path.
    """
    if state.budget_gate_outcome == "accepted" or state.accepted_budget_decision:
        # A checkpoint marker is never authorization on its own. Route every
        # accepted/replan record through the normalizer so malformed legacy
        # state becomes a retryable unavailable gate instead of a permanent
        # pseudo-success response.
        return "prepare_budget_validation"

    if (
        state.chat_budget_action == "accept_recommended"
        or state.pending_budget_proposal is not None
    ):
        return "assess_chat_budget"

    if _budget_gate_is_unresolved(state):
        return "budget_decision_agent"

    if not state.draft_itinerary and not state.base_currency_code:
        return "process_initial_form"

    return "memory_extraction"


def route_tools_or_end(state: AgentState) -> Literal["tools", "__end__"]:
    """Post-agent router — check for ``tool_calls`` in the last message."""
    if not state.messages:
        return "__end__"

    last_message = state.messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "__end__"


def route_after_budget_assessment(state: AgentState) -> str:
    """Only a deterministic accepted result may enter planning."""
    if _local_accepted_budget_decision(state, stages={"assessment"}):
        return "prepare_budget_validation"
    return "budget_gate_response"


def route_after_accepted_budget_validation(state: AgentState) -> str:
    """Only the cache-validated record may enter currency/planning nodes."""
    parsed = _local_accepted_budget_decision(
        state,
        stages=set(_BUDGET_REPLAN_ROUTE),
    )
    if parsed is not None:
        return _BUDGET_REPLAN_ROUTE[parsed[0]["stage"]]
    return "budget_gate_response"


def route_after_candidate_review(state: AgentState) -> str:
    """Send a qualified private candidate to promotion or retry planning only
    when the planning transaction itself failed.

    Exhausted output-review attempts must not cause an already validated
    itinerary to be regenerated.
    """
    if state.candidate_plan is not None:
        return "promote_candidate"

    if (
        state.planning_outcome == "unavailable"
        and state.planning_attempts < _PLANNING_TRANSACTION_MAX_ATTEMPTS
        and state.output_review_attempts < _OUTPUT_REVIEW_MAX_ATTEMPTS
    ):
        return "execute_validated_transaction"

    return "__end__"


def route_after_promotion(
    state: AgentState,
) -> str:
    """Retry full planning only for full-plan failures, never edit failures."""

    # ---------------------------------------------------------
    # A current-turn itinerary edit is atomic.
    #
    # Even if candidate promotion unexpectedly fails, never
    # regenerate the full itinerary. The existing accepted
    # itinerary must remain unchanged.
    # ---------------------------------------------------------
    if _current_turn_used_tool(
        state,
        "edit_itinerary",
    ):
        return "__end__"

    # Full itinerary generation may still use the normal retry
    # mechanism when its candidate fails.
    if (
        state.candidate_plan is None
        and state.planning_outcome == "unavailable"
        and state.planning_attempts
        < _PLANNING_TRANSACTION_MAX_ATTEMPTS
    ):
        return "execute_validated_transaction"

    return "__end__"


def route_after_post_tool(state: AgentState) -> str:
    """Post-tool router — re-plan or review a private incremental candidate.

    Priority:
      1. ``update_trip_details`` changed the destination
         → full re-init (currency conversion + allocation + flights/hotels +
         activities + maps) via ``process_initial_form``.
      2. ``update_trip_details`` changed dates / party size / cities
         → build an isolated validated plan transaction while currency and
         allocation stay valid.
      3. A validated incremental candidate enters output review.
      4. Otherwise → back to the agent.
    """

    # A deterministic incremental-edit rejection is already a
    # complete user-facing response. Do not send it back to the
    # agent and NEVER enter the full planning transaction.
    if state.messages:
        last_message = state.messages[-1]

        if (
            getattr(last_message, "type", "") == "ai"
            and getattr(
                last_message,
                "additional_kwargs",
                {},
            ).get("server_owned")
            == "incremental_edit_rejected"
        ):
            return "__end__"

    _FULL_REINIT_FIELDS = {"country", "origin_country"}

    if state.pending_budget_proposal is not None:
        return "assess_chat_budget"

    recent_tool_msgs: List[BaseMessage] = []
    for msg in reversed(state.messages):
        if msg.type == "ai":
            break
        if msg.type == "tool":
            recent_tool_msgs.insert(0, msg)

    batch_status, _ = _resolve_budget_tool_batch(recent_tool_msgs)
    if batch_status == "coherent":
        return "assess_chat_budget"
    if batch_status != "absent":
        return "budget_gate_response"

    if _budget_gate_is_unresolved(state):
        return "budget_gate_response"

    if state.planning_outcome is not None:
        return "generate_reviewed_response"

    for msg in reversed(state.messages):
        if msg.type == "ai":
            break
        if msg.type != "tool":
            continue
        name = getattr(msg, "name", "")
        if name == "update_trip_details":
            content = _parse_tool_content(msg.content) or {}
            changed = set(
                _validated_trip_updates(
                    state, content.get("updates") or {}
                ).keys()
            )
            if changed & _FULL_REINIT_FIELDS:
                return "process_initial_form"
            if changed:
                return "prepare_planning_transaction"

    return "agent"


# ═══════════════════════════════════════════════════════════
# 7. GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════

builder = StateGraph(AgentState)

# ── Register all nodes ─────────────────────────────────
builder.add_node("process_initial_form", process_initial_form_node)
builder.add_node("prepare_planning_transaction", prepare_planning_transaction_node)
builder.add_node("execute_validated_transaction", plan_validated_transaction_node)
# Safe aliases preserve pending legacy checkpoints without running public
# candidate writers.
builder.add_node("plan_initial_itinerary", prepare_planning_transaction_node)
builder.add_node("plan_activities", prepare_planning_transaction_node)
builder.add_node("plan_validated_transaction", prepare_planning_transaction_node)
builder.add_node("run_map_update", prepare_planning_transaction_node)
builder.add_node("capture_accepted_plan", prepare_planning_transaction_node)
builder.add_node("generate_reviewed_response", generate_reviewed_response_node)
builder.add_node("promote_candidate", promote_candidate_node)
builder.add_node("memory_extraction", memory_extraction_node)
builder.add_node("agent", agent_node)
builder.add_node("budget_decision_agent", budget_decision_agent_node)
builder.add_node("assess_chat_budget", assess_chat_budget_node)
builder.add_node("prepare_budget_validation", prepare_planning_transaction_node)
builder.add_node("validate_accepted_budget", prepare_planning_transaction_node)
builder.add_node("validate_accepted_budget_async", validate_accepted_budget_node)
builder.add_node("prepare_budget_replan", prepare_budget_replan_node)
builder.add_node(
    "plan_budget_replan_itinerary",
    restart_budget_replan_node,
)
builder.add_node(
    "plan_budget_replan_activities",
    restart_budget_replan_node,
)
builder.add_node("plan_budget_replan_maps", restart_budget_replan_node)
builder.add_node("commit_budget_replan", restart_budget_replan_node)
builder.add_node("budget_gate_response", budget_gate_response_node)
builder.add_node("tools", ToolNode(tools))
builder.add_node("budget_decision_tools", ToolNode(budget_decision_tools))
builder.add_node("post_tool_processing", post_tool_processing_node)

# ── New-trip setup flow (linear) ───────────────────────
builder.add_conditional_edges(START, route_start)

builder.add_edge("process_initial_form", "prepare_planning_transaction")
builder.add_edge("prepare_planning_transaction", "execute_validated_transaction")
builder.add_edge("plan_initial_itinerary", "execute_validated_transaction")
builder.add_edge("plan_activities", "execute_validated_transaction")
builder.add_edge("plan_validated_transaction", "execute_validated_transaction")
builder.add_edge("execute_validated_transaction", "generate_reviewed_response")
builder.add_edge("run_map_update", "execute_validated_transaction")
builder.add_edge("capture_accepted_plan", "execute_validated_transaction")
builder.add_conditional_edges(
    "generate_reviewed_response",
    route_after_candidate_review,
    {
        "execute_validated_transaction": "execute_validated_transaction",
        "promote_candidate": "promote_candidate",
        "__end__": END,
    },
)
builder.add_conditional_edges(
    "promote_candidate",
    route_after_promotion,
    {"execute_validated_transaction": "execute_validated_transaction", "__end__": END},
)
builder.add_conditional_edges(
    "assess_chat_budget",
    route_after_budget_assessment,
    {
        "prepare_budget_validation": "prepare_budget_validation",
        "budget_gate_response": "budget_gate_response",
    },
)
builder.add_edge("prepare_budget_validation", "validate_accepted_budget_async")
builder.add_edge("validate_accepted_budget", "validate_accepted_budget_async")
builder.add_conditional_edges(
    "validate_accepted_budget_async",
    route_after_accepted_budget_validation,
    {
        "prepare_budget_replan": "prepare_budget_replan",
        "budget_gate_response": "budget_gate_response",
    },
)
builder.add_edge("prepare_budget_replan", "prepare_planning_transaction")
builder.add_edge("plan_budget_replan_itinerary", "prepare_budget_replan")
builder.add_edge("plan_budget_replan_activities", "prepare_budget_replan")
builder.add_edge("plan_budget_replan_maps", "prepare_budget_replan")
builder.add_edge("commit_budget_replan", "prepare_budget_replan")
builder.add_edge("budget_gate_response", END)

# ── Ongoing chat flow ──────────────────────────────────
builder.add_edge("memory_extraction", "agent")

# ── ReAct loop (agent ↔ tools) ─────────────────────────
builder.add_conditional_edges(
    "agent",
    route_tools_or_end,
    {"tools": "tools", "__end__": END},
)
builder.add_conditional_edges(
    "budget_decision_agent",
    route_tools_or_end,
    {"tools": "budget_decision_tools", "__end__": END},
)

# Tools → post-processing → conditional routing
builder.add_edge("tools", "post_tool_processing")
builder.add_edge("budget_decision_tools", "post_tool_processing")

builder.add_conditional_edges(
    "post_tool_processing",
    route_after_post_tool,
    {
        "run_map_update": "run_map_update",
        "generate_reviewed_response": "generate_reviewed_response",
        "agent": "agent",
        # update_trip_details re-plan paths (full pipeline re-entry)
        "process_initial_form": "process_initial_form",
        "prepare_planning_transaction": "prepare_planning_transaction",
        "assess_chat_budget": "assess_chat_budget",
        "budget_gate_response": "budget_gate_response",

        # Incremental edit rejected -> preserve existing plan.
        "__end__": END,
    },
)


# ═══════════════════════════════════════════════════════════
# 8. LAZY INITIALIZATION (Thread-Safe, Double-Checked Locking)
# ═══════════════════════════════════════════════════════════

_checkpointer: Optional[PostgresSaver] = None
_workflow: Optional[Any] = None  # CompiledStateGraph


# Errors Postgres raises when a migration tries to (re)create a schema object
# that already exists — i.e. the live schema is *ahead* of the version counter
# recorded in ``checkpoint_migrations``.  See ``_setup_checkpointer_resilient``.
_MIGRATION_DESYNC_ERRORS = (DuplicateColumn, DuplicateTable, DuplicateObject)


async def _record_applied_migration(saver: AsyncPostgresSaver) -> bool:
    """Advance ``checkpoint_migrations`` by one version.

    Reads the current max recorded version and records ``max + 1`` (capped at
    the highest migration the installed langgraph knows about).  Returns
    ``True`` if a new version row was written, ``False`` if already fully
    caught up (nothing left to reconcile).

    ``saver.conn`` is an ``AsyncConnectionPool`` — check a connection out for
    the duration of the two statements.
    """
    n_migrations = len(AsyncPostgresSaver.MIGRATIONS)
    async with saver.conn.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT COALESCE(MAX(v), -1) AS v FROM checkpoint_migrations"
            )
            row = await cur.fetchone()
            current = row["v"] if isinstance(row, dict) else row[0]
            nxt = current + 1
            if nxt >= n_migrations:
                return False
            # ON CONFLICT keeps this idempotent under concurrent startups.
            await cur.execute(
                "INSERT INTO checkpoint_migrations (v) VALUES (%s) "
                "ON CONFLICT (v) DO NOTHING",
                (nxt,),
            )
    return True


async def _setup_checkpointer_resilient(saver: AsyncPostgresSaver) -> None:
    """Run ``saver.setup()``, self-healing a desynced migration counter.

    LangGraph replays every migration after the highest version recorded in
    ``checkpoint_migrations``.  If a previous run applied a migration's DDL but
    crashed before recording the version — e.g. the process was killed between
    the ``ALTER TABLE ... ADD COLUMN task_path`` (migration #9) and its
    bookkeeping ``INSERT`` (both run under autocommit) — then setup replays that
    DDL and Postgres raises "column/table/object already exists", aborting
    startup even though the schema is actually correct.

    A duplicate-object error is *proof* that the next pending migration is
    already applied, so we record it and retry.  ``setup()`` runs migrations in
    order starting from ``max_recorded + 1``, so the first one to fail is always
    the next pending one — recording it can never mask an unapplied migration.
    Each duplicate advances the counter by exactly one, so the loop is bounded
    by the migration count and non-duplicate errors propagate untouched.
    """
    for _ in range(len(AsyncPostgresSaver.MIGRATIONS) + 1):
        try:
            await saver.setup()
            return
        except _MIGRATION_DESYNC_ERRORS:
            if not await _record_applied_migration(saver):
                raise
            safe_observability_log(
                logger,
                "planning.checkpoint_migration",
                stage="storage",
                issue_codes=("checkpoint.migration_desync",),
                outcome="retrying",
            )
    raise RuntimeError(
        "Checkpoint setup did not converge after reconciling migration versions."
    )


_CHECKPOINT_RETRY_ATTEMPTS = 3
_CHECKPOINT_RETRY_BASE_DELAY_S = 0.25


class ResilientAsyncPostgresSaver(AsyncPostgresSaver):
    """``AsyncPostgresSaver`` that retries transient connection failures.

    The Supabase session pooler (and residential NAT) silently drops TCP
    connections while the graph is stalled on slow LLM calls; the pool's
    checkout health-check can't catch a socket that dies *after* checkout,
    so a checkpoint write can still hit ``OperationalError`` ("SSL error:
    bad length" / "EOF detected").  Checkpoint statements are idempotent
    upserts keyed by checkpoint id, so retrying is safe — the pool discards
    the dead connection and hands the retry a fresh one.
    """

    async def _retry(self, op, *args):
        delay = _CHECKPOINT_RETRY_BASE_DELAY_S
        for attempt in range(1, _CHECKPOINT_RETRY_ATTEMPTS + 1):
            try:
                return await op(*args)
            except OperationalError:
                if attempt == _CHECKPOINT_RETRY_ATTEMPTS:
                    raise
                safe_observability_log(
                    logger,
                    "planning.checkpoint_retry",
                    attempt=attempt,
                    stage="storage",
                    issue_codes=("checkpoint.connection_dropped",),
                    outcome="retrying",
                )
                await asyncio.sleep(delay)
                delay *= 2

    async def aget_tuple(self, config):
        return await self._retry(super().aget_tuple, config)

    async def aput(self, config, checkpoint, metadata, new_versions):
        return await self._retry(
            super().aput, config, checkpoint, metadata, new_versions
        )

    async def aput_writes(self, config, writes, task_id, task_path=""):
        return await self._retry(
            super().aput_writes, config, writes, task_id, task_path
        )


async def _init_checkpointer() -> AsyncPostgresSaver:
    """Initialise a persistent ``AsyncPostgresSaver`` checkpointer.

    The graph contains async nodes (``agent_node`` awaits the LLM), so it is
    always driven via ``workflow.ainvoke()`` — which requires an *async*
    checkpointer (the sync ``PostgresSaver`` only implements the blocking
    ``get_tuple``/``put`` and raises ``NotImplementedError`` on ``aget_tuple``).

    Backed by an ``AsyncConnectionPool`` rather than ``from_conn_string``:
    the latter holds ONE async connection for the process lifetime, and when
    Supabase/router kills it after an idle period every subsequent request
    blocks on a dead socket — the "server randomly freezes" symptom.  The pool
    health-checks connections on checkout (``check_connection``) and retires
    idle ones before the upstream can kill them (``max_idle``).
    """
    pool: AsyncConnectionPool = AsyncConnectionPool(
        settings.SUPABASE_POSTGRES_URI,
        min_size=1,
        max_size=4,
        # dict_row + autocommit mirror from_conn_string's connection setup;
        # prepare_threshold=None disables server-side prepared statements so
        # the pool also works behind PgBouncer (Supabase pooled ports).
        kwargs={
            "autocommit": True,
            "row_factory": dict_row,
            "prepare_threshold": None,
            # TCP keepalives: the Supabase session pooler / home-router NAT
            # silently drops sockets that idle while the graph awaits slow
            # LLM calls; probes every 10s keep the mapping alive and surface
            # dead peers quickly instead of via "SSL error: bad length".
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 3,
        },
        check=AsyncConnectionPool.check_connection,
        max_idle=240.0,  # retire idle conns before upstream kills them
        timeout=30.0,  # fail fast if the pool can't produce a connection
        open=False,
    )
    # Entering the pool's async context opens it; the exit stack closes it
    # cleanly during shutdown_graph().
    await _async_exit_stack.enter_async_context(pool)
    saver = ResilientAsyncPostgresSaver(pool)
    await _setup_checkpointer_resilient(saver)
    safe_observability_log(
        logger,
        "planning.checkpointer",
        stage="storage",
        outcome="ready",
    )
    return saver


async def get_checkpointer() -> AsyncPostgresSaver:
    """Async lazy accessor for the checkpointer (single initialisation)."""
    global _checkpointer
    if _checkpointer is None:
        async with _async_init_lock:
            if _checkpointer is None:  # Double-check inside lock
                _checkpointer = await _init_checkpointer()
    return _checkpointer


async def get_workflow():
    """Async lazy accessor for the compiled LangGraph workflow.

    The compiled graph is checkpointer-backed and invoked via
    ``workflow.ainvoke()``.
    """
    global _workflow, _checkpointer
    if _workflow is None:
        async with _async_init_lock:
            if _workflow is None:  # Double-check inside lock
                # Call _init_checkpointer directly (not get_checkpointer) to
                # avoid re-acquiring the non-reentrant _async_init_lock.
                if _checkpointer is None:
                    _checkpointer = await _init_checkpointer()
                _workflow = builder.compile(checkpointer=_checkpointer)
                safe_observability_log(
                    logger,
                    "planning.workflow",
                    stage="workflow",
                    outcome="ready",
                )
    return _workflow


# ═══════════════════════════════════════════════════════════
# 9. LIFECYCLE MANAGEMENT
# ═══════════════════════════════════════════════════════════


async def init_graph() -> None:
    """Eagerly initialise checkpointer and compile the workflow.

    Call (awaited) during FastAPI ``lifespan`` startup to avoid first-request
    latency.  Safe to call multiple times (idempotent via lazy init).
    """
    await get_workflow()
    safe_observability_log(
        logger,
        "planning.workflow_startup",
        stage="workflow",
        outcome="ready",
    )


_shutdown_done = False


async def shutdown_graph() -> None:
    """Release all graph resources.

    Call (awaited) during FastAPI ``lifespan`` shutdown.  Idempotent —
    safe to call multiple times.
    """
    global _shutdown_done
    if _shutdown_done:
        return
    _shutdown_done = True
    _memory_pool.shutdown(wait=False)
    await _async_exit_stack.aclose()  # close AsyncPostgresSaver connection
    _exit_stack.close()
    safe_observability_log(
        logger,
        "planning.workflow_shutdown",
        stage="workflow",
        outcome="completed",
    )


def _shutdown_sync_safety_net() -> None:
    """atexit safety net — can't await, so only release sync resources.

    The async checkpointer connection is normally closed via
    ``shutdown_graph()`` during the FastAPI lifespan; this just ensures the
    memory thread-pool is released if the process exits abnormally.
    """
    global _shutdown_done
    if _shutdown_done:
        return
    _memory_pool.shutdown(wait=False)
    _exit_stack.close()


# Safety net: ensure sync cleanup even if lifespan shutdown is not called.
atexit.register(_shutdown_sync_safety_net)


# ═══════════════════════════════════════════════════════════
# 10. CONVENIENCE INVOKERS
# ═══════════════════════════════════════════════════════════


async def _release_new_session_with_retries(
    checkpointer: Any,
    thread_id: str,
    user_id: str,
    request_id: Optional[str],
) -> None:
    """Run one bounded cleanup sequence without exposing cleanup failures."""
    for _attempt in range(1, _SESSION_CLEANUP_ATTEMPTS + 1):
        try:
            await release_new_session_reservation(checkpointer, thread_id, user_id)
            return
        except (Exception, asyncio.CancelledError):
            continue
    safe_observability_log(
        logger,
        "planning.session_cleanup",
        request_id=request_id,
        session_id=thread_id,
        attempt=_SESSION_CLEANUP_ATTEMPTS,
        stage="session_cleanup",
        issue_codes=("session.cleanup_exhausted",),
        outcome="error",
    )


def _start_new_session_cleanup(
    checkpointer: Any,
    thread_id: str,
    user_id: str,
    request_id: Optional[str],
) -> asyncio.Task[None]:
    """Start exactly one shieldable cleanup sequence for a failed initializer."""
    return asyncio.create_task(
        _release_new_session_with_retries(
            checkpointer,
            thread_id,
            user_id,
            request_id,
        )
    )


async def invoke_new_trip(
    initial_state: dict,
    thread_id: str,
    user_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> dict:
    """Invoke the graph for a **new trip** (called from ``form.py``).

    Parameters
    ----------
    initial_state : dict
        AgentState fields from the form submission. Must include
        ``origin_country``, ``country``, ``city``, ``num_people``,
        ``total_base_budget``, ``start_date``, ``end_date``,
        and an initial ``messages`` list.
    thread_id : str
        Unique UUID for this trip's conversation thread.
    user_id : str, optional
        Authenticated user's ID (enables memory personalisation).

    Returns
    -------
    dict
        Final ``AgentState`` after graph execution.
    """
    if not thread_id:
        raise ValueError("thread_id is required")

    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": _DEFAULT_RECURSION_LIMIT,
        "metadata": {
            "thread_id": thread_id,
        },
    }

    if user_id:
        # Internal graph access for memory/profile nodes.
        config["configurable"]["__user_id"] = user_id

        # Persist the authenticated owner in checkpoint metadata so
        # /api/chat/history can safely retrieve this user's sessions.
        config["metadata"]["user_id"] = user_id

    if request_id:
        config["configurable"]["request_id"] = request_id
        config["metadata"]["request_id"] = request_id

    workflow = await get_workflow()
    if user_id:
        checkpointer = await get_checkpointer()
        checkpoint_config = {"configurable": {"thread_id": thread_id}}
        existing = await checkpointer.aget_tuple(checkpoint_config)
        if existing is not None:
            if await finalize_expired_new_session_reservation(
                checkpointer,
                thread_id,
                user_id,
            ):
                checkpoint = getattr(existing, "checkpoint", None)
                channel_values = (
                    checkpoint.get("channel_values")
                    if isinstance(checkpoint, dict)
                    else None
                )
                if isinstance(channel_values, dict):
                    return copy.deepcopy(channel_values)
                raise SessionAccessDenied
            metadata = getattr(existing, "metadata", None)
            owner = metadata.get("user_id") if isinstance(metadata, dict) else None
            if not isinstance(owner, str) or owner != user_id:
                raise SessionAccessDenied
            if not await backfill_existing_session_owner(
                checkpointer, thread_id, user_id
            ):
                raise SessionAccessDenied
            raise SessionAccessDenied
        if not await reserve_new_session(checkpointer, thread_id, user_id):
            if await checkpointer.aget_tuple(checkpoint_config) is not None:
                raise SessionAccessDenied
            if not await reclaim_expired_new_session_reservation(
                checkpointer,
                thread_id,
                user_id,
            ):
                raise SessionAccessDenied

    # Hard ceiling — a hung LLM/API/DB call must surface as a 504 instead of
    # leaving the request (and the browser) frozen indefinitely.
    try:
        result = await asyncio.wait_for(
            workflow.ainvoke(initial_state, config=config),
            timeout=_NEW_TRIP_INVOKE_TIMEOUT_S,
        )
        if user_id and not await complete_new_session_reservation(
            checkpointer, thread_id, user_id
        ):
            raise SessionAccessDenied
    except asyncio.CancelledError as cancellation:
        if user_id:
            cleanup_task = _start_new_session_cleanup(
                checkpointer,
                thread_id,
                user_id,
                request_id,
            )
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                # A repeated cancellation must not duplicate or wait for the
                # already-running shielded cleanup sequence.
                pass
        raise cancellation
    except Exception:
        if user_id:
            cleanup_task = _start_new_session_cleanup(
                checkpointer,
                thread_id,
                user_id,
                request_id,
            )
            await asyncio.shield(cleanup_task)
        raise
    return result


async def invoke_chat(
    user_message: str,
    thread_id: str,
    user_id: Optional[str] = None,
    budget_action: Optional[str] = None,
    budget_assessment_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> dict:
    """Invoke the graph for an **ongoing conversation** (called from
    ``chat.py``).

    The graph resumes from the checkpoint associated with
    ``thread_id``, so all prior state is automatically available.

    Parameters
    ----------
    user_message : str
        The user's latest message text.
    thread_id : str
        Existing trip's thread ID.
    user_id : str, optional
        Authenticated user's ID.

    Returns
    -------
    dict
        Final ``AgentState`` after graph execution.
    """
    if not thread_id:
        raise ValueError("thread_id is required")

    config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": _DEFAULT_RECURSION_LIMIT,
        "metadata": {
            "thread_id": thread_id,
        },
    }

    if user_id:
        # Internal graph access for memory/profile nodes.
        config["configurable"]["__user_id"] = user_id

        # Persist ownership with every new checkpoint written for this session.
        config["metadata"]["user_id"] = user_id

    if request_id:
        config["configurable"]["request_id"] = request_id
        config["metadata"]["request_id"] = request_id

    workflow = await get_workflow()
    existing = None
    if user_id:
        checkpointer = await get_checkpointer()
        existing = await checkpointer.aget_tuple(
            {"configurable": {"thread_id": thread_id}}
        )
        if existing is None:
            raise SessionAccessDenied
        if not await verify_session_owner(checkpointer, thread_id, user_id):
            metadata = getattr(existing, "metadata", None)
            legacy_owner = (
                metadata.get("user_id") if isinstance(metadata, dict) else None
            )
            if not isinstance(legacy_owner, str) or legacy_owner != user_id:
                raise SessionAccessDenied
            if not await backfill_existing_session_owner(
                checkpointer,
                thread_id,
                user_id,
            ):
                raise SessionAccessDenied

    input_state = {
        "messages": [HumanMessage(content=user_message)],
        "chat_budget_action": budget_action,
        "chat_budget_assessment_id": budget_assessment_id,
    }

    # Backward-compatible one-time hydration for sessions created before
    # ``itinerary_history`` existed. LangGraph's Postgres saver still retains
    # older checkpoints, but normal graph execution resumes only from the latest
    # checkpoint. Recover previous accepted revisions once, then let the new
    # state field persist them on subsequent checkpoints.
    checkpoint = getattr(existing, "checkpoint", None)

    channel_values = (
        checkpoint.get("channel_values")
        if isinstance(checkpoint, dict)
        else None
    )

    if isinstance(channel_values, dict):
        current_revision = channel_values.get(
            "plan_revision",
            0,
        )

        existing_history = channel_values.get(
            "itinerary_history"
        )

        if (
            isinstance(current_revision, int)
            and not isinstance(current_revision, bool)
            and current_revision > 1
            and not existing_history
            and user_id
        ):
            try:
                legacy_states: list[dict[str, Any]] = []

                async for saved in checkpointer.alist(
                    {
                        "configurable": {
                            "thread_id": thread_id
                        }
                    },
                    limit=200,
                ):
                    saved_checkpoint = getattr(
                        saved,
                        "checkpoint",
                        None,
                    )

                    saved_values = (
                        saved_checkpoint.get(
                            "channel_values"
                        )
                        if isinstance(
                            saved_checkpoint,
                            dict,
                        )
                        else None
                    )

                    if isinstance(saved_values, dict):
                        legacy_states.append(
                            saved_values
                        )

                recovered_history = (
                    recover_itinerary_history(
                        legacy_states,
                        current_revision=current_revision,
                    )
                )

                if recovered_history:
                    input_state[
                        "itinerary_history"
                    ] = recovered_history

                    safe_observability_log(
                        logger,
                        "planning.itinerary_history_recovered",
                        stage="storage",
                        outcome="completed",
                        session_id=thread_id,
                    )

            except Exception:
                # Historical recovery is helpful context, not authorization for
                # changing the trip. Never make ordinary chat unavailable just
                # because a legacy-history scan failed.
                logger.warning(
                    "Legacy itinerary-history recovery failed.",
                    exc_info=True,
                )

    # Hard ceiling — see invoke_new_trip. asyncio.TimeoutError → HTTP 504.
    result = await asyncio.wait_for(
        workflow.ainvoke(
            input_state,
            config=config,
        ),
        timeout=_CHAT_INVOKE_TIMEOUT_S,
    )
    if not isinstance(result, dict):
        return result

    checkpoint = getattr(existing, "checkpoint", None)
    channel_values = (
        checkpoint.get("channel_values")
        if isinstance(checkpoint, dict)
        else None
    )
    before_revision = (
        channel_values.get("plan_revision", 0)
        if isinstance(channel_values, dict)
        else 0
    )
    after_revision = result.get("plan_revision", 0)
    if (
        isinstance(before_revision, bool)
        or not isinstance(before_revision, int)
        or before_revision < 0
    ):
        before_revision = 0
    if (
        isinstance(after_revision, bool)
        or not isinstance(after_revision, int)
        or after_revision < 0
    ):
        after_revision = 0
    return {
        **result,
        "_itinerary_modified": existing is not None
        and after_revision > before_revision,
    }


# ═══════════════════════════════════════════════════════════
# 11. MODULE EXPORTS
# ═══════════════════════════════════════════════════════════

__all__: List[str] = [
    # Lazy accessors (use these in API routes)
    "get_workflow",
    "get_checkpointer",
    # Lifecycle management
    "init_graph",
    "shutdown_graph",
    # Convenience invokers
    "invoke_new_trip",
    "invoke_chat",
    # Underlying components (for testing / introspection)
    "tools",
    "llm",
    "llm_with_tools",
    "builder",
    # Constants
    "DEFAULT_BUDGET_RATIOS",
    "_DEFAULT_RECURSION_LIMIT",
]

def _latest_user_message_index(state: AgentState) -> int | None:
    for index in range(len(state.messages) - 1, -1, -1):
        if state.messages[index].type == "human":
            return index
    return None


def _explicit_day_number(text: str) -> int | None:
    match = _EXPLICIT_DAY_REFERENCE_PATTERN.search(text or "")

    if match is None:
        return None

    number = re.search(
        r"[1-9]\d*",
        match.group(0),
    )

    return (
        int(number.group(0))
        if number is not None
        else None
    )


def _confirmed_prior_day(
    state: AgentState,
    expected_day: int,
) -> bool:
    """
    Permit an affirmative continuation only when the user had already
    supplied the same day before the immediately preceding proposal.
    """

    if _AFFIRMATIVE_CONFIRMATION_PATTERN.search(
        _latest_user_request(state)
    ) is None:
        return False

    current_index = _latest_user_message_index(state)

    if current_index is None or current_index < 2:
        return False

    previous_ai_index: int | None = None

    for index in range(
        current_index - 1,
        -1,
        -1,
    ):
        if state.messages[index].type == "ai":
            previous_ai_index = index
            break

    if previous_ai_index is None:
        return False

    proposal = _extract_text(
        state.messages[previous_ai_index].content
    ).strip()

    if (
        _explicit_day_number(proposal) != expected_day
        or re.search(
            r"\b(?:add|change|edit|modify|move|remove|replace|"
            r"reschedule|swap|update)\b",
            proposal,
            re.IGNORECASE,
        )
        is None
    ):
        return False

    previous_user = ""

    for index in range(
        previous_ai_index - 1,
        -1,
        -1,
    ):
        if state.messages[index].type == "human":
            previous_user = _extract_text(
                state.messages[index].content
            ).strip()
            break

    return (
        _explicit_day_number(previous_user)
        == expected_day
    )


def _confirmation_continues_itinerary_mutation(
    state: AgentState,
) -> bool:
    """
    Return True when the current user affirmatively confirms the
    immediately preceding itinerary-edit proposal and the proposal's
    day is corroborated by the user's earlier message.
    """

    request = _latest_user_request(state)

    # The current message must actually be an affirmative confirmation,
    # e.g. "yes", "yes please", "go ahead", or
    # "yes, and make that day more relaxed".
    if _AFFIRMATIVE_CONFIRMATION_PATTERN.search(request) is None:
        return False

    current_index = _latest_user_message_index(state)

    if current_index is None:
        return False

    previous_ai = ""
    previous_user = ""
    previous_ai_index: int | None = None

    # Find the assistant message immediately preceding
    # the current user's confirmation.
    for index in range(
        current_index - 1,
        -1,
        -1,
    ):
        if state.messages[index].type == "ai":
            previous_ai_index = index
            previous_ai = _extract_text(
                state.messages[index].content
            ).strip()
            break

    if previous_ai_index is None:
        return False

    # Find the user's message immediately before that
    # assistant proposal.
    for index in range(
        previous_ai_index - 1,
        -1,
        -1,
    ):
        if state.messages[index].type == "human":
            previous_user = _extract_text(
                state.messages[index].content
            ).strip()
            break

    proposal_day = _explicit_day_number(previous_ai)

    return bool(
        proposal_day is not None
        and _explicit_day_number(previous_user) == proposal_day
        and re.search(
            r"\b(?:add|change|edit|modify|move|remove|replace|"
            r"reschedule|swap|update)\b",
            previous_ai,
            re.IGNORECASE,
        )
    )

def _mutation_target_day_is_authorized(
    state: AgentState,
    day_num: int,
) -> bool:
    request = _latest_user_request(state)

    if re.search(
        rf"\b(?:day\s*(?:#\s*)?0*{day_num}|"
        rf"{day_num}(?:st|nd|rd|th)\s+day)\b",
        request,
        flags=re.IGNORECASE,
    ) is not None:
        return True

    if _pending_day_reply(state) == day_num:
        return True

    return _confirmed_prior_day(
        state,
        day_num,
    )


def _pending_day_reply(
    state: AgentState,
) -> int | None:
    pending = state.pending_itinerary_mutation

    if (
        not isinstance(pending, dict)
        or pending.get("status") != "awaiting_day"
    ):
        return None

    current_index = _latest_user_message_index(state)

    if current_index is None:
        return None

    previous_ai = None

    for index in range(
        current_index - 1,
        -1,
        -1,
    ):
        if state.messages[index].type == "ai":
            previous_ai = state.messages[index]
            break

    # Important: only allow the IMMEDIATE reply to
    # our server-owned clarification.
    if (
        previous_ai is None
        or getattr(
            previous_ai,
            "additional_kwargs",
            {},
        ).get("server_owned")
        != "mutation_target_day_required"
    ):
        return None

    match = _DAY_CLARIFICATION_REPLY_PATTERN.fullmatch(
        _latest_user_request(state)
    )

    return (
        int(match.group("day"))
        if match is not None
        else None
    )


def _sanitised_pending_itinerary_mutation(
    edits: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Store only pending edit intent.

    Provider coordinates, prices, ratings, and other provider payloads
    must not be persisted as trusted evidence across turns.
    """

    pending_edits: list[dict[str, Any]] = []

    for edit in edits:
        if not isinstance(edit, dict):
            continue

        item: dict[str, Any] = {
            "action": edit.get("action"),
            "category": edit.get("category"),
        }

        index = edit.get("index")

        if (
            isinstance(index, int)
            and not isinstance(index, bool)
            and index > 0
        ):
            item["index"] = index

        details = edit.get("new_details")

        if isinstance(details, dict):
            name = details.get("name")

            if isinstance(name, str) and name.strip():
                item["place_name"] = name.strip()

        pending_edits.append(item)

    return {
        "status": "awaiting_day",
        "source_request": None,
        "edits": pending_edits,
    }


def _pending_itinerary_mutation_instruction(
    state: AgentState,
) -> str | None:
    """
    Reconstruct the server-owned pending edit after the user supplies
    the missing day.

    The actual provider evidence must still be fetched again in the
    current turn.
    """

    day_num = _pending_day_reply(state)
    pending = state.pending_itinerary_mutation

    if (
        day_num is None
        or not isinstance(pending, dict)
    ):
        return None

    edits = pending.get("edits")
    edit_lines: list[str] = []

    if isinstance(edits, list):
        for edit in edits:
            if not isinstance(edit, dict):
                continue

            parts = [
                str(edit.get("action") or "change"),
                str(edit.get("category") or "activity"),
            ]

            if edit.get("index") is not None:
                parts.append(
                    f"index {edit['index']}"
                )

            if edit.get("place_name"):
                parts.append(
                    f"place {edit['place_name']}"
                )

            edit_lines.append("; ".join(parts))

    source_request = pending.get("source_request")

    source_text = (
        source_request.strip()
        if isinstance(source_request, str)
        and source_request.strip()
        else ""
    )

    intent_parts: list[str] = []

    if source_text:
        intent_parts.append(
            f"original user request: {source_text}"
        )

    if edit_lines:
        intent_parts.append(
            "pending edits: " + " | ".join(edit_lines)
        )

    if not intent_parts:
        return None

    intent_text = " ; ".join(intent_parts)

    return (
        "Internal server-owned clarification continuation: "
        "the user has now supplied "
        f"Day {day_num} for the pending itinerary edit. "
        f"Pending intent: {intent_text}. "
        "Do not reuse provider payloads from an earlier turn. "
        "For every activity/restaurant add or replacement, "
        "perform a fresh search_places lookup in THIS turn, "
        "then call edit_itinerary with Day "
        f"{day_num}. Complete all pending edits atomically "
        "when possible."
    )