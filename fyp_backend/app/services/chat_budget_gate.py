"""Deterministic budget decisions for ongoing travel-planning chats."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ValidationError

from app.services.budget_assessment import (
    BudgetAssessment,
    BudgetAssessmentUnavailable,
    assessment_matches_trip,
    get_or_create_budget_assessment,
    is_budget_sufficient,
    load_confirmed_budget_assessment,
)


class ChatBudgetDecision(BaseModel):
    """A non-mutating decision for one chat budget proposal."""

    status: Literal[
        "accepted", "budget_confirmation_required", "budget_check_unavailable"
    ]
    chat_reply: str
    reason: Optional[str] = None
    accepted_total_base_budget: Optional[float] = None
    assessment: Optional[BudgetAssessment] = None
    pending_confirmation: Optional[dict[str, Any]] = None


def _unavailable(reason: str) -> ChatBudgetDecision:
    return ChatBudgetDecision(
        status="budget_check_unavailable",
        chat_reply="Unable to verify the budget using provider-backed data.",
        reason=reason,
    )


def _pending_confirmation(
    assessment: BudgetAssessment,
    *,
    reason: Literal["insufficient_budget", "recommendation_requested"],
    stated_budget: float | None,
) -> dict[str, Any]:
    evidence = assessment.evidence
    return {
        "budget_assessment_id": assessment.assessment_id,
        "reason": reason,
        "stated_budget": stated_budget,
        "recommended_minimum_budget": assessment.recommended_minimum_budget,
        "base_currency": assessment.base_currency,
        "destination_currency": assessment.destination_currency,
        "expires_at": assessment.expires_at,
        "evidence": {
            "outbound_flight_price": evidence.outbound_flight_price,
            "return_flight_price": evidence.return_flight_price,
            "hotel_price_per_night": evidence.hotel_price_per_night,
            "hotel_nights": evidence.hotel_nights,
        },
        "assessment": assessment.model_dump(),
    }


def _confirmation_assessment(
    proposal: dict[str, Any],
    pending_confirmation: dict[str, Any] | None,
    *,
    origin: str,
    destination: str,
    destination_city: str,
    start_date: str,
    end_date: str,
    num_people: int,
) -> BudgetAssessment | None:
    if not isinstance(pending_confirmation, dict):
        return None
    try:
        pending_assessment = BudgetAssessment.model_validate(
            pending_confirmation["assessment"]
        )
    except (KeyError, TypeError, ValidationError):
        return None

    if not assessment_matches_trip(
        pending_assessment,
        origin=origin,
        destination=destination,
        destination_city=destination_city,
        start_date=start_date,
        end_date=end_date,
        num_people=num_people,
    ):
        return None
    if (
        pending_confirmation.get("budget_assessment_id")
        != pending_assessment.assessment_id
    ):
        return None
    supplied_assessment_id = proposal.get("assessment_id")
    if (
        supplied_assessment_id is not None
        and supplied_assessment_id != pending_assessment.assessment_id
    ):
        return None
    persisted_assessment = load_confirmed_budget_assessment(
        assessment_id=pending_assessment.assessment_id,
        origin=origin,
        destination=destination,
        destination_city=destination_city,
        start_date=start_date,
        end_date=end_date,
        num_people=num_people,
    )
    if persisted_assessment is None:
        return None
    if persisted_assessment.model_dump() != pending_assessment.model_dump():
        return None
    return persisted_assessment


def _trip_inputs_are_valid(
    *,
    origin: Any,
    destination: Any,
    destination_city: Any,
    start_date: Any,
    end_date: Any,
    num_people: Any,
) -> bool:
    basic_inputs_are_valid = (
        isinstance(origin, str)
        and bool(origin.strip())
        and isinstance(destination, str)
        and bool(destination.strip())
        and isinstance(destination_city, str)
        and bool(destination_city.strip())
        and isinstance(start_date, str)
        and bool(start_date.strip())
        and isinstance(end_date, str)
        and bool(end_date.strip())
        and isinstance(num_people, int)
        and not isinstance(num_people, bool)
        and num_people >= 1
    )
    if not basic_inputs_are_valid:
        return False
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError:
        return False
    return (
        start_date == start.isoformat()
        and end_date == end.isoformat()
        and end >= start
    )


def _valid_amount(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(amount) or amount <= 0:
        return None
    return amount


def _matching_current_assessment(
    current_assessment: dict[str, Any] | None,
    *,
    origin: str,
    destination: str,
    destination_city: str,
    start_date: str,
    end_date: str,
    num_people: int,
) -> BudgetAssessment | None:
    if current_assessment is None:
        return None
    try:
        assessment = BudgetAssessment.model_validate(current_assessment)
    except (TypeError, ValidationError):
        return None
    if not assessment_matches_trip(
        assessment,
        origin=origin,
        destination=destination,
        destination_city=destination_city,
        start_date=start_date,
        end_date=end_date,
        num_people=num_people,
    ):
        return None
    persisted = load_confirmed_budget_assessment(
        assessment_id=assessment.assessment_id,
        origin=origin,
        destination=destination,
        destination_city=destination_city,
        start_date=start_date,
        end_date=end_date,
        num_people=num_people,
    )
    if persisted is None or persisted.model_dump() != assessment.model_dump():
        return None
    return persisted


def _confirmation_required(
    assessment: BudgetAssessment,
    *,
    reason: Literal["insufficient_budget", "recommendation_requested"],
    stated_budget: float | None,
) -> ChatBudgetDecision:
    return ChatBudgetDecision(
        status="budget_confirmation_required",
        chat_reply="Please confirm the provider-backed recommended minimum budget.",
        reason=reason,
        assessment=assessment,
        pending_confirmation=_pending_confirmation(
            assessment,
            reason=reason,
            stated_budget=stated_budget,
        ),
    )


def evaluate_chat_budget(
    *,
    proposal: dict[str, Any],
    pending_confirmation: dict[str, Any] | None,
    current_assessment: dict[str, Any] | None,
    origin: str,
    destination: str,
    destination_city: str,
    start_date: str,
    end_date: str,
    num_people: int,
) -> ChatBudgetDecision:
    """Evaluate a chat proposal without changing itinerary or session state."""
    if not isinstance(proposal, dict):
        return _unavailable("invalid_budget_proposal")
    if not _trip_inputs_are_valid(
        origin=origin,
        destination=destination,
        destination_city=destination_city,
        start_date=start_date,
        end_date=end_date,
        num_people=num_people,
    ):
        return _unavailable("invalid_trip_details")
    mode = proposal.get("mode")
    if mode == "confirm":
        # A confirmation is valid only while a server-owned recommendation
        # is actually pending. This also distinguishes a stale/duplicate
        # confirmation from a genuinely expired assessment.
        if not isinstance(pending_confirmation, dict):
            return _unavailable("no_pending_budget_confirmation")

        assessment = _confirmation_assessment(
            proposal,
            pending_confirmation,
            origin=origin,
            destination=destination,
            destination_city=destination_city,
            start_date=start_date,
            end_date=end_date,
            num_people=num_people,
        )

        if assessment is None:
            return _unavailable("assessment_expired_or_invalid")

        return ChatBudgetDecision(
            status="accepted",
            chat_reply="Budget accepted.",
            accepted_total_base_budget=assessment.recommended_minimum_budget,
            assessment=assessment,
        )

    if mode not in {"amount", "recommendation"}:
        return _unavailable("invalid_budget_proposal")
    entered_budget = None
    if mode == "amount":
        entered_budget = _valid_amount(proposal.get("total_base_budget"))
        if entered_budget is None:
            return _unavailable("invalid_budget_proposal")

    assessment = _matching_current_assessment(
        current_assessment,
        origin=origin,
        destination=destination,
        destination_city=destination_city,
        start_date=start_date,
        end_date=end_date,
        num_people=num_people,
    )
    assessment_persisted = assessment is not None
    if assessment is None:
        try:
            result = get_or_create_budget_assessment(
                origin=origin,
                destination=destination,
                destination_city=destination_city,
                start_date=start_date,
                end_date=end_date,
                num_people=num_people,
            )
        except (BudgetAssessmentUnavailable, ValidationError):
            return _unavailable("provider_data_unavailable")
        assessment = result.assessment
        assessment_persisted = result.persisted

    if mode == "recommendation":
        if not assessment_persisted:
            return _unavailable("assessment_cache_unavailable")
        return _confirmation_required(
            assessment,
            reason="recommendation_requested",
            stated_budget=None,
        )

    if is_budget_sufficient(entered_budget, assessment):
        return ChatBudgetDecision(
            status="accepted",
            chat_reply="Budget accepted.",
            accepted_total_base_budget=entered_budget,
            assessment=assessment,
        )
    if not assessment_persisted:
        return _unavailable("assessment_cache_unavailable")
    return _confirmation_required(
        assessment,
        reason="insufficient_budget",
        stated_budget=entered_budget,
    )
