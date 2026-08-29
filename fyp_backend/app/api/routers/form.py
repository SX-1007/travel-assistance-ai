"""
form.py — Initial trip-form submission router
═════════════════════════════════════════════════════════

Accepts the structured initial form (origin, destination, dates, budget, …)
and bootstraps a brand-new LangGraph thread.  Returns the first-pass
draft itinerary produced by the workflow.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, status

from app.api.dependencies import (
    CurrentUser,
    RequestId,
    invoke_new_trip_authenticated,
)
from app.api.errors import (
    PLANNING_UNAVAILABLE_MESSAGE,
    map_exception_to_http,
    planning_unavailable_reason,
)
from app.core.supabase_db import fetch_user_profile
from app.schemas.requests import InitialFormRequest
from app.schemas.responses import (
    BudgetCheckUnavailableResponse,
    BudgetConfirmationResponse,
    FinalResponse,
    PlanningUnavailableResponse,
    TripSubmissionResponse,
    validated_accepted_plan,
)
from app.services.budget_assessment import (
    BudgetAssessment,
    BudgetAssessmentUnavailable,
    get_or_create_budget_assessment,
    is_budget_sufficient,
    load_confirmed_budget_assessment,
)
from app.services.planning_transaction import safe_observability_log
from app.services.output_review import (
    OutputReviewContext,
    deterministic_output_review,
)
from app.services.destination_resolution import (
    DestinationResolutionInvalid,
    DestinationResolutionUnavailable,
    resolve_destination_cities,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["form"])


def _promoted_plan_reply(
    final_state: dict[str, Any],
    *,
    plan_revision: int,
) -> str | None:
    """Return only the reviewed AI message atomically tagged at promotion."""
    messages = final_state.get("messages")
    if not isinstance(messages, (list, tuple)) or not messages:
        return None
    latest = messages[-1]
    if getattr(latest, "type", "") != "ai":
        return None
    metadata = getattr(latest, "additional_kwargs", {})
    if not isinstance(metadata, dict) or metadata.get("plan_revision") != plan_revision:
        return None
    content = getattr(latest, "content", None)
    if not isinstance(content, str) or not content.strip():
        return None
    reply = content.strip()
    review = deterministic_output_review(
        OutputReviewContext(
            latest_user_request="",
            trusted_requirements={
                "country": final_state.get("country"),
                "city": final_state.get("city"),
                "start_date": final_state.get("start_date"),
                "end_date": final_state.get("end_date"),
            },
            proposed_reply=reply,
            candidate_plan={"plan_revision": plan_revision},
            deterministic_report={"issues": []},
        )
    )
    return reply if review.approved else None


@router.post(
    "/submit",
    response_model=TripSubmissionResponse,
    status_code=200,
    summary="Submit the initial trip form and bootstrap a new planning session",
)
async def process_form(
    request: InitialFormRequest,
    user_id: CurrentUser,
    request_id: RequestId,
) -> TripSubmissionResponse:
    """
    Check the provider-grounded minimum before invoking any planning node.

    Notes
    -----
    * ``thread_id`` is created only after the submitted budget is sufficient.
    * Field names here **must** match ``AgentState`` in
    ``app/agents/state.py``. The request schema validates the destination,
    dates and budget; origin_country and origin_state are loaded from the
    onboarding profile below.
    """
    # ── Load origin from the onboarding profile (Req 2) ────────────
    # The trip form no longer collects origin_country/origin_state; they are
    # captured once during onboarding (POST /api/profile).
    profile = await asyncio.to_thread(fetch_user_profile, user_id)

    profile_name = str(profile.get("name") or "").strip()
    origin_country = str(profile.get("home_country") or "").strip()
    origin_state = str(profile.get("home_state") or "").strip()

    if not profile_name or not origin_country or not origin_state:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Onboarding incomplete: submit your profile "
                "(name + origin country + state/province) via /api/profile "
                "before planning a trip."
            ),
        )

    # A budget confirmation must echo the exact server-resolved city list from
    # the previous budget response. Re-resolving a blank city here could select
    # a different locality and would invalidate the assessment identity.
    if request.budget_assessment_id and not request.city:
        return BudgetCheckUnavailableResponse(
            status="budget_check_unavailable",
            reason="assessment_expired_or_invalid",
            chat_reply=(
                "This budget confirmation is missing its verified planning city. "
                "Refresh the budget check before planning."
            ),
        )

    # Public city input is optional, but every downstream provider and planning
    # operation receives a non-empty, country-verified server-owned city list.
    try:
        planning_cities = await asyncio.to_thread(
            resolve_destination_cities,
            country=request.country,
            requested_cities=list(request.city),
            start_date=request.start_date,
            end_date=request.end_date,
            num_people=request.num_people,
        )

    except DestinationResolutionInvalid as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    except DestinationResolutionUnavailable as exc:
        # Preserve the safe public response, but log the bounded internal reason
        # so validation failures can be traced to AI selection vs Mapbox
        # verification without exposing provider details to the client.
        logger.warning(
            "Destination resolution unavailable: %s",
            str(exc)[:200],
        )

        safe_observability_log(
            logger,
            "form.submit.destination_resolution_unavailable",
            request_id=request_id,
            stage="destination_resolution",
            issue_codes=("destination.resolution.unavailable",),
            outcome="unavailable",
        )

        return BudgetCheckUnavailableResponse(
            status="budget_check_unavailable",
            reason="destination_resolution_unavailable",
            chat_reply=(
                "I could not safely verify a planning city for this country "
                "right now. Please retry before itinerary planning."
            ),
        )

    primary_city = planning_cities[0]

    logger.info(
        "Budget check destination resolved: country=%s city=%s",
        request.country,
        primary_city,
    )

    # ── Hard budget gate: no thread or graph exists before this passes ──
    assessment: BudgetAssessment
    persisted = True
    if request.budget_assessment_id:
        assessment = await asyncio.to_thread(
            load_confirmed_budget_assessment,
            assessment_id=request.budget_assessment_id,
            origin=origin_country,
            destination=request.country,
            destination_city=primary_city,
            start_date=request.start_date,
            end_date=request.end_date,
            num_people=request.num_people,
        )
        if assessment is None:
            return BudgetCheckUnavailableResponse(
                status="budget_check_unavailable",
                reason="assessment_expired_or_invalid",
                chat_reply=(
                    "This budget check expired or no longer matches the trip. "
                    "Refresh it before planning."
                ),
            )
    else:
        try:
            assessment_result = await asyncio.to_thread(
                get_or_create_budget_assessment,
                origin=origin_country,
                destination=request.country,
                destination_city=primary_city,
                start_date=request.start_date,
                end_date=request.end_date,
                num_people=request.num_people,
            )
        except BudgetAssessmentUnavailable as exc:
            logger.warning("Budget assessment unavailable: %s", exc)

            safe_observability_log(
                logger,
                "form.submit.budget_data_unavailable",
                request_id=request_id,
                stage="budget_gate",
                issue_codes=("provider.data.unavailable",),
                outcome="unavailable",
            )
            return BudgetCheckUnavailableResponse(
                status="budget_check_unavailable",
                reason="provider_data_unavailable",
                chat_reply=(
                    "Current flight, hotel, or exchange-rate data is unavailable. "
                    "Retry the budget check before planning."
                ),
            )
        assessment = assessment_result.assessment
        persisted = assessment_result.persisted

    entered_budget = request.total_budget
    sufficient = entered_budget is not None and is_budget_sufficient(
        entered_budget,
        assessment,
    )
    confirmed_displayed_minimum = (
        entered_budget is not None
        and entered_budget >= assessment.recommended_minimum_budget
    )
    if request.budget_assessment_id and (
        not sufficient or not confirmed_displayed_minimum
    ):
        return BudgetCheckUnavailableResponse(
            status="budget_check_unavailable",
            reason="assessment_expired_or_invalid",
            chat_reply=(
                "The confirmed amount no longer satisfies this budget check. "
                "Refresh it before planning."
            ),
        )

    if not sufficient:
        if not persisted:
            return BudgetCheckUnavailableResponse(
                status="budget_check_unavailable",
                reason="assessment_cache_unavailable",
                chat_reply=(
                    "The grounded prices could not be saved for confirmation. "
                    "Retry the budget check before planning."
                ),
            )
        recommendation_requested = entered_budget is None
        return BudgetConfirmationResponse(
            status="budget_confirmation_required",
            reason=(
                "recommendation_requested"
                if recommendation_requested
                else "insufficient_budget"
            ),
            chat_reply=(
                "Here is the grounded minimum you requested. Confirm it or "
                "enter your own budget before itinerary planning."
                if recommendation_requested
                else "Your entered budget is below the grounded minimum. "
                "Confirm the recommendation or enter another amount before "
                "itinerary planning."
            ),
            budget_assessment_id=assessment.assessment_id,

            # Important: preserve the city selected/verified for this assessment.
            resolved_cities=list(planning_cities),

            stated_budget=entered_budget,
            recommended_minimum_budget=assessment.recommended_minimum_budget,
            base_currency=assessment.base_currency,
            destination_currency=assessment.destination_currency,
            expires_at=assessment.expires_at,
            evidence={
                "outbound_flight_price": (
                    assessment.evidence.outbound_flight_price
                ),
                "return_flight_price": (
                    assessment.evidence.return_flight_price
                ),
                "hotel_price_per_night": (
                    assessment.evidence.hotel_price_per_night
                ),
                "hotel_nights": assessment.evidence.hotel_nights,
            },
        )

    # A sufficient amount is the only path that creates a session and invokes
    # currency, itinerary, activity, map, and agent nodes.
    thread_id = str(uuid.uuid4())
    safe_observability_log(
        logger,
        "form.submit.start",
        request_id=request_id,
        session_id=thread_id,
        stage="form",
        outcome="started",
    )

    # ── Build the initial graph state ──────────────────────────────
    # Field names mirror AgentState exactly. Keys with empty defaults
    # are populated by downstream nodes (currency, map, etc.).
    initial_state: dict[str, Any] = {
        # ── user message (drives the planner LLM) ──
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Please analyse all the information and plan a perfect "
                    f"trip to {request.country}"
                ),
            }
        ],
        # ── user origin (from onboarding profile) ──
        "origin_country": origin_country,
        "origin_state": origin_state,
        # ── destination ──
        "country": request.country,
        "city": list(planning_cities),  # verified server-owned planning scope
        "num_people": request.num_people,
        # ── budget (raw, in user's currency) ──
        "total_base_budget": float(entered_budget),
        # ── dates (already validated by InitialFormRequest) ──
        "start_date": request.start_date,
        "end_date": request.end_date,
        # ── placeholders filled by downstream nodes ──
        "budget_allocation": {},
        "exchange_rate": {},
        "currency_fetched_at": None,
        "budget_assessment": assessment.model_dump(),
        "draft_itinerary": [],
    }

    try:
        # ── Invoke the authenticated graph workflow ────────────────────
        final_state: dict[str, Any] = await invoke_new_trip_authenticated(
            initial_state=initial_state,
            thread_id=thread_id,
            user_id=user_id,
            request_id=request_id,
        )
    except Exception as exc:
        safe_observability_log(
            logger,
            "form.submit.error",
            request_id=request_id,
            session_id=thread_id,
            stage="form",
            issue_codes=("workflow.failed",),
            outcome="error",
        )
        raise map_exception_to_http(exc) from exc

    accepted = validated_accepted_plan(final_state)
    promoted_reply = (
        _promoted_plan_reply(final_state, plan_revision=accepted.plan_revision)
        if accepted is not None
        else None
    )
    if (
        final_state.get("planning_outcome") == "unavailable"
        or accepted is None
        or promoted_reply is None
    ):
        safe_observability_log(
            logger,
            "form.submit.planning_unavailable",
            request_id=request_id,
            session_id=thread_id,
            stage="form",
            issue_codes=("planning.unavailable",),
            outcome="unavailable",
        )
        return PlanningUnavailableResponse(
            status="planning_unavailable",
            reason=planning_unavailable_reason(final_state),
            chat_reply=PLANNING_UNAVAILABLE_MESSAGE,
        )

    safe_observability_log(
        logger,
        "form.submit.done",
        request_id=request_id,
        session_id=thread_id,
        stage="form",
        destination_country_code=accepted.destination_country_code,
        outcome="completed",
    )

    return FinalResponse(
        status="success",
        chat_reply=promoted_reply,
        itinerary=accepted.draft_itinerary,
        daily_geojson_maps=accepted.daily_map_info,
        destination_country_code=accepted.destination_country_code,
        # Preserve the verified city actually used for planning.
        resolved_cities=list(planning_cities),
        total_budget=accepted.total_convert_budget,
        currency=accepted.dest_currency_code,
        budget_allocation=accepted.budget_allocation,
        session_id=thread_id,
    )
