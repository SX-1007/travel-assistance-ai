from __future__ import annotations

import copy
import importlib
import importlib.util
import json
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.services.budget_assessment import (
    BudgetAssessment,
    BudgetAssessmentResult,
    BudgetAssessmentUnavailable,
)


@pytest.fixture
def assessment() -> BudgetAssessment:
    return BudgetAssessment.model_validate(
        {
            "assessment_id": "assessment-1",
            "calculation_version": "allocation-v1",
            "origin": "Malaysia",
            "destination": "China",
            "destination_city": "Shanghai",
            "start_date": "2026-08-17",
            "end_date": "2026-08-20",
            "num_people": 1,
            "base_currency": "MYR",
            "destination_currency": "CNY",
            "exchange_rate": 2.0,
            "minimum_destination_budget": 7000.0,
            "recommended_minimum_budget": 3500.0,
            "evidence": {
                "outbound_flight": {"price": 900},
                "return_flight": {"price": 850},
                "hotel": {"price_per_night": 400},
                "outbound_flight_price": 900,
                "return_flight_price": 850,
                "hotel_price_per_night": 400,
                "hotel_nights": 3,
            },
            "created_at": "2026-08-16T12:00:00+00:00",
            "expires_at": "2999-08-16T13:00:00+00:00",
        }
    )


@pytest.fixture(autouse=True)
def authoritative_assessment_cache(assessment, monkeypatch):
    """Make the ordinary fixture represent an exact loadable cache entry."""
    gate = load_gate_module()

    def load_authoritative(*, assessment_id: str, **trip) -> BudgetAssessment | None:
        if assessment_id != assessment.assessment_id:
            return None
        if trip != trip_args(assessment):
            return None
        return assessment

    monkeypatch.setattr(
        gate,
        "load_confirmed_budget_assessment",
        load_authoritative,
    )


def load_gate_module():
    spec = importlib.util.find_spec("app.services.chat_budget_gate")
    assert spec is not None, "deterministic chat budget service is not implemented"
    return importlib.import_module("app.services.chat_budget_gate")


def trip_args(assessment: BudgetAssessment) -> dict[str, str | int]:
    return {
        "origin": assessment.origin,
        "destination": assessment.destination,
        "destination_city": assessment.destination_city,
        "start_date": assessment.start_date,
        "end_date": assessment.end_date,
        "num_people": assessment.num_people,
    }


@pytest.mark.unit
def test_insufficient_amount_returns_pending_without_accepted_amount(assessment):
    gate = load_gate_module()
    with patch.object(gate, "get_or_create_budget_assessment") as lookup:
        decision = gate.evaluate_chat_budget(
            proposal={"mode": "amount", "total_base_budget": 500},
            pending_confirmation=None,
            current_assessment=assessment.model_dump(),
            **trip_args(assessment),
        )
    lookup.assert_not_called()
    assert decision.status == "budget_confirmation_required"
    assert decision.accepted_total_base_budget is None
    assert decision.pending_confirmation["stated_budget"] == 500
    assert decision.pending_confirmation["recommended_minimum_budget"] == 3500


@pytest.mark.unit
def test_sufficient_amount_is_accepted_without_mutating_itinerary_data(assessment):
    gate = load_gate_module()
    proposal = {"mode": "amount", "total_base_budget": 4000}
    current_assessment = assessment.model_dump()
    proposal_before = copy.deepcopy(proposal)
    assessment_before = copy.deepcopy(current_assessment)
    decision = gate.evaluate_chat_budget(
        proposal=proposal,
        pending_confirmation=None,
        current_assessment=current_assessment,
        **trip_args(assessment),
    )
    assert decision.status == "accepted"
    assert decision.accepted_total_base_budget == 4000
    assert decision.assessment.assessment_id == assessment.assessment_id
    assert decision.pending_confirmation is None
    assert proposal == proposal_before
    assert current_assessment == assessment_before


@pytest.mark.unit
def test_unknown_budget_returns_recommendation_reason_without_stated_amount(assessment):
    gate = load_gate_module()
    decision = gate.evaluate_chat_budget(
        proposal={"mode": "recommendation", "total_base_budget": None},
        pending_confirmation=None,
        current_assessment=assessment.model_dump(),
        **trip_args(assessment),
    )
    assert decision.status == "budget_confirmation_required"
    assert decision.reason == "recommendation_requested"
    assert decision.pending_confirmation["stated_budget"] is None


@pytest.mark.unit
def test_valid_confirmation_uses_exact_pending_amount_without_provider_calls(assessment):
    gate = load_gate_module()
    first = gate.evaluate_chat_budget(
        proposal={"mode": "amount", "total_base_budget": 500},
        pending_confirmation=None,
        current_assessment=assessment.model_dump(),
        **trip_args(assessment),
    )
    with (
        patch.object(
            gate,
            "get_or_create_budget_assessment",
            side_effect=AssertionError("confirmation must not research again"),
        ),
        patch.object(
            gate,
            "load_confirmed_budget_assessment",
            return_value=assessment,
        ) as load_assessment,
    ):
        confirmed = gate.evaluate_chat_budget(
            proposal={
                "mode": "confirm",
                "total_base_budget": None,
                "assessment_id": assessment.assessment_id,
            },
            pending_confirmation=first.pending_confirmation,
            current_assessment=None,
            **trip_args(assessment),
        )
    assert confirmed.status == "accepted"
    assert confirmed.accepted_total_base_budget == assessment.recommended_minimum_budget
    load_assessment.assert_called_once_with(
        assessment_id=assessment.assessment_id,
        **trip_args(assessment),
    )


@pytest.mark.unit
def test_confirmation_rejects_a_fully_forged_pending_assessment(assessment):
    gate = load_gate_module()
    first = gate.evaluate_chat_budget(
        proposal={"mode": "amount", "total_base_budget": 500},
        pending_confirmation=None,
        current_assessment=assessment.model_dump(),
        **trip_args(assessment),
    )
    forged = copy.deepcopy(first.pending_confirmation)
    forged_assessment = copy.deepcopy(forged["assessment"])
    forged_assessment.update(
        {
            "assessment_id": "forged-assessment",
            "minimum_destination_budget": 60 / 7,
            "recommended_minimum_budget": 4.29,
            "evidence": {
                "outbound_flight": {"price": 1},
                "return_flight": {"price": 1},
                "hotel": {"price_per_night": 1},
                "outbound_flight_price": 1,
                "return_flight_price": 1,
                "hotel_price_per_night": 1,
                "hotel_nights": 3,
            },
        }
    )
    forged.update(
        {
            "budget_assessment_id": "forged-assessment",
            "recommended_minimum_budget": 4.29,
            "evidence": {
                "outbound_flight_price": 1,
                "return_flight_price": 1,
                "hotel_price_per_night": 1,
                "hotel_nights": 3,
            },
            "assessment": forged_assessment,
        }
    )
    with patch.object(
        gate,
        "load_confirmed_budget_assessment",
        return_value=assessment,
    ):
        decision = gate.evaluate_chat_budget(
            proposal={"mode": "confirm", "assessment_id": "forged-assessment"},
            pending_confirmation=forged,
            current_assessment=None,
            **trip_args(assessment),
        )
    assert decision.status == "budget_check_unavailable"
    assert decision.accepted_total_base_budget is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "entered_budget",
    [True, False, float("nan"), float("inf"), float("-inf"), 0, -1, "invalid"],
)
def test_invalid_amounts_fail_closed_with_json_portable_decisions(
    assessment,
    entered_budget,
):
    gate = load_gate_module()
    with patch.object(gate, "get_or_create_budget_assessment") as lookup:
        decision = gate.evaluate_chat_budget(
            proposal={"mode": "amount", "total_base_budget": entered_budget},
            pending_confirmation=None,
            current_assessment=assessment.model_dump(),
            **trip_args(assessment),
        )
    lookup.assert_not_called()
    assert decision.status == "budget_check_unavailable"
    assert decision.accepted_total_base_budget is None
    assert decision.pending_confirmation is None
    json.dumps(decision.model_dump(mode="json"), allow_nan=False)


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_trip",
    [
        {"origin": None},
        {"destination": 12},
        {"destination_city": ""},
        {"destination_city": 12},
        {"num_people": "not-a-number"},
    ],
)
def test_confirmation_with_malformed_trip_fails_closed_without_provider_calls(
    assessment,
    invalid_trip,
):
    gate = load_gate_module()
    first = gate.evaluate_chat_budget(
        proposal={"mode": "amount", "total_base_budget": 500},
        pending_confirmation=None,
        current_assessment=assessment.model_dump(),
        **trip_args(assessment),
    )
    with (
        patch.object(
            gate,
            "get_or_create_budget_assessment",
            side_effect=AssertionError("confirmation must not research again"),
        ),
        patch.object(
            gate,
            "load_confirmed_budget_assessment",
            side_effect=AssertionError("malformed trips must not load evidence"),
        ),
    ):
        decision = gate.evaluate_chat_budget(
            proposal={"mode": "confirm", "assessment_id": assessment.assessment_id},
            pending_confirmation=first.pending_confirmation,
            current_assessment=None,
            **{**trip_args(assessment), **invalid_trip},
        )
    assert decision.status == "budget_check_unavailable"
    assert decision.accepted_total_base_budget is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_trip",
    [
        {"origin": None},
        {"destination": 12},
        {"destination_city": ""},
        {"destination_city": 12},
        {"num_people": "not-a-number"},
    ],
)
def test_amount_with_current_assessment_and_malformed_trip_fails_closed(
    assessment,
    invalid_trip,
):
    gate = load_gate_module()
    with patch.object(
        gate,
        "get_or_create_budget_assessment",
        side_effect=AssertionError("malformed trips must not research"),
    ):
        decision = gate.evaluate_chat_budget(
            proposal={"mode": "amount", "total_base_budget": 500},
            pending_confirmation=None,
            current_assessment=assessment.model_dump(),
            **{**trip_args(assessment), **invalid_trip},
        )
    assert decision.status == "budget_check_unavailable"
    assert decision.accepted_total_base_budget is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_dates",
    [
        {"start_date": "17-08-2026"},
        {"start_date": "2026-02-30"},
        {"start_date": "2026-08-17", "end_date": "2026-08-17"},
        {"start_date": "2026-08-20", "end_date": "2026-08-17"},
    ],
)
@pytest.mark.parametrize("mode", ["amount", "confirm"])
def test_invalid_dates_fail_closed_before_any_assessment_boundary(
    assessment,
    invalid_dates,
    mode,
):
    gate = load_gate_module()
    first = gate.evaluate_chat_budget(
        proposal={"mode": "amount", "total_base_budget": 500},
        pending_confirmation=None,
        current_assessment=assessment.model_dump(),
        **trip_args(assessment),
    )
    proposal = (
        {"mode": "amount", "total_base_budget": 500}
        if mode == "amount"
        else {"mode": "confirm", "assessment_id": assessment.assessment_id}
    )
    with (
        patch.object(gate, "assessment_matches_trip") as matcher,
        patch.object(gate, "load_confirmed_budget_assessment") as load_assessment,
        patch.object(gate, "get_or_create_budget_assessment") as lookup,
    ):
        decision = gate.evaluate_chat_budget(
            proposal=proposal,
            pending_confirmation=first.pending_confirmation,
            current_assessment=assessment.model_dump(),
            **{**trip_args(assessment), **invalid_dates},
        )
    assert decision.status == "budget_check_unavailable"
    assert decision.reason == "invalid_trip_details"
    assert decision.accepted_total_base_budget is None
    assert decision.pending_confirmation is None
    assert decision.assessment is None
    matcher.assert_not_called()
    load_assessment.assert_not_called()
    lookup.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize(
    "change", ["id", "expiry", "destination", "destination_city"]
)
def test_invalid_confirmation_fails_closed(assessment, change):
    gate = load_gate_module()
    first = gate.evaluate_chat_budget(
        proposal={"mode": "amount", "total_base_budget": 500},
        pending_confirmation=None,
        current_assessment=assessment.model_dump(),
        **trip_args(assessment),
    )
    pending = copy.deepcopy(first.pending_confirmation)
    proposal_id = assessment.assessment_id
    args = trip_args(assessment)
    if change == "id":
        proposal_id = "forged"
    elif change == "expiry":
        pending["assessment"]["expires_at"] = "2000-01-01T00:00:00+00:00"
    elif change == "destination":
        args["destination"] = "Japan"
    else:
        args["destination_city"] = "Beijing"
    decision = gate.evaluate_chat_budget(
        proposal={"mode": "confirm", "assessment_id": proposal_id},
        pending_confirmation=pending,
        current_assessment=None,
        **args,
    )
    assert decision.status == "budget_check_unavailable"
    assert decision.accepted_total_base_budget is None


@pytest.mark.unit
def test_provider_failure_is_unavailable(assessment):
    gate = load_gate_module()
    with patch.object(
        gate,
        "get_or_create_budget_assessment",
        side_effect=BudgetAssessmentUnavailable("offline"),
    ):
        decision = gate.evaluate_chat_budget(
            proposal={"mode": "amount", "total_base_budget": 500},
            pending_confirmation=None,
            current_assessment=None,
            **trip_args(assessment),
        )
    assert decision.status == "budget_check_unavailable"
    assert decision.reason == "provider_data_unavailable"


@pytest.mark.unit
def test_assessment_validation_failure_is_unavailable(assessment):
    gate = load_gate_module()
    with pytest.raises(ValidationError) as validation_error:
        BudgetAssessment.model_validate({})
    with patch.object(
        gate,
        "get_or_create_budget_assessment",
        side_effect=validation_error.value,
    ):
        decision = gate.evaluate_chat_budget(
            proposal={"mode": "amount", "total_base_budget": 500},
            pending_confirmation=None,
            current_assessment=None,
            **trip_args(assessment),
        )
    assert decision.status == "budget_check_unavailable"
    assert decision.reason == "provider_data_unavailable"
    assert decision.accepted_total_base_budget is None


@pytest.mark.unit
def test_unpersisted_insufficient_research_cannot_be_confirmed_later(assessment):
    gate = load_gate_module()
    result = BudgetAssessmentResult(assessment=assessment, persisted=False)
    with patch.object(gate, "get_or_create_budget_assessment", return_value=result):
        decision = gate.evaluate_chat_budget(
            proposal={"mode": "amount", "total_base_budget": 500},
            pending_confirmation=None,
            current_assessment=None,
            **trip_args(assessment),
        )
    assert decision.status == "budget_check_unavailable"
    assert decision.reason == "assessment_cache_unavailable"
    assert decision.pending_confirmation is None


@pytest.mark.unit
def test_new_amount_replaces_older_pending_decision(assessment):
    gate = load_gate_module()
    old_pending = {
        "budget_assessment_id": "old",
        "recommended_minimum_budget": 9999,
    }
    decision = gate.evaluate_chat_budget(
        proposal={"mode": "amount", "total_base_budget": 4000},
        pending_confirmation=old_pending,
        current_assessment=assessment.model_dump(),
        **trip_args(assessment),
    )
    assert decision.status == "accepted"
    assert decision.accepted_total_base_budget == 4000
    assert decision.pending_confirmation is None


@pytest.mark.unit
def test_forged_current_assessment_cannot_authorize_a_sufficient_budget(assessment):
    """Catch self-consistent checkpoint evidence being mistaken for cache data."""
    gate = load_gate_module()
    forged = assessment.model_copy(deep=True)
    forged.assessment_id = "forged-low-assessment"
    forged.minimum_destination_budget = 60 / 7
    forged.recommended_minimum_budget = 4.29
    forged.evidence.outbound_flight = {"price": 1}
    forged.evidence.return_flight = {"price": 1}
    forged.evidence.hotel = {"price_per_night": 1}
    forged.evidence.outbound_flight_price = 1
    forged.evidence.return_flight_price = 1
    forged.evidence.hotel_price_per_night = 1
    authoritative = BudgetAssessmentResult(assessment=assessment, persisted=True)

    with (
        patch.object(
            gate,
            "load_confirmed_budget_assessment",
            return_value=None,
        ) as cache_load,
        patch.object(
            gate,
            "get_or_create_budget_assessment",
            return_value=authoritative,
        ) as refresh,
    ):
        decision = gate.evaluate_chat_budget(
            proposal={"mode": "amount", "total_base_budget": 100},
            pending_confirmation=None,
            current_assessment=forged.model_dump(),
            **trip_args(assessment),
        )

    assert decision.status == "budget_confirmation_required"
    assert decision.accepted_total_base_budget is None
    assert decision.assessment.assessment_id == assessment.assessment_id
    cache_load.assert_called_once_with(
        assessment_id="forged-low-assessment",
        **trip_args(assessment),
    )
    refresh.assert_called_once_with(**trip_args(assessment))


@pytest.mark.unit
def test_unpersisted_form_assessment_never_emits_an_unusable_chat_card(assessment):
    """Catch persisted=False form evidence being promoted to accept-later state."""
    gate = load_gate_module()
    with (
        patch.object(gate, "load_confirmed_budget_assessment", return_value=None),
        patch.object(
            gate,
            "get_or_create_budget_assessment",
            return_value=BudgetAssessmentResult(
                assessment=assessment,
                persisted=False,
            ),
        ),
    ):
        decision = gate.evaluate_chat_budget(
            proposal={"mode": "amount", "total_base_budget": 500},
            pending_confirmation=None,
            current_assessment=assessment.model_dump(),
            **trip_args(assessment),
        )

    assert decision.status == "budget_check_unavailable"
    assert decision.reason == "assessment_cache_unavailable"
    assert decision.pending_confirmation is None


@pytest.mark.unit
def test_current_assessment_is_reused_only_after_exact_authoritative_reload(
    assessment,
):
    """Catch cache ID matches that silently tolerate different contents."""
    gate = load_gate_module()
    with (
        patch.object(
            gate,
            "load_confirmed_budget_assessment",
            return_value=assessment,
        ) as cache_load,
        patch.object(gate, "get_or_create_budget_assessment") as refresh,
    ):
        decision = gate.evaluate_chat_budget(
            proposal={"mode": "amount", "total_base_budget": 4000},
            pending_confirmation=None,
            current_assessment=assessment.model_dump(),
            **trip_args(assessment),
        )

    assert decision.status == "accepted"
    assert decision.assessment == assessment
    cache_load.assert_called_once_with(
        assessment_id=assessment.assessment_id,
        **trip_args(assessment),
    )
    refresh.assert_not_called()
