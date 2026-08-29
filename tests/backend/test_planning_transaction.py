from __future__ import annotations

import asyncio
import copy
from typing import Any

import pytest

from app.agents.state import AgentState
from app.services.planning_transaction import (
    PlanningDependencies,
    build_validated_plan,
)


def _activity(name: str, country_code: str = "SG") -> dict[str, Any]:
    return {
        "name": name,
        "type": "attraction",
        "address": f"{name}, Singapore",
        "estimated_cost": 20.0,
        "order": 1,
        "location": {
            "place_name": name,
            "latitude": 1.2966,
            "longitude": 103.7764,
            "country_code": country_code,
            "requested_city": "Singapore",
            "verified_locality": "Singapore",
        },
    }


def _flight(
    departure_id: str,
    departure_name: str,
    arrival_id: str,
    arrival_name: str,
    date: str,
) -> dict[str, Any]:
    return {
        "airline": "Grounded Air",
        "flight_number": f"GA-{departure_id}-{arrival_id}",
        "departure_airport": {
            "id": departure_id,
            "name": departure_name,
            "time": f"{date} 08:00",
        },
        "arrival_airport": {
            "id": arrival_id,
            "name": arrival_name,
            "time": f"{date} 09:30",
        },
        "departure_time": f"{date} 08:00",
        "arrival_time": f"{date} 09:30",
        "price": 100.0,
    }


def _hotel() -> dict[str, Any]:
    return {
        "hotel_name": "Verified Singapore Hotel",
        "price_per_night": 50.0,
        "location": {"lat": 1.31, "lng": 103.82, "country_code": "SG"},
    }


def _itinerary(*, empty_day: bool = False, country_code: str = "SG") -> list[dict]:
    hotel = _hotel()
    return [
        {
            "day": 1,
            "date": "2026-08-01",
            "flight": [
                _flight(
                    "KUL",
                    "Kuala Lumpur International Airport",
                    "SIN",
                    "Singapore Changi Airport",
                    "2026-08-01",
                )
            ],
            "hotel": hotel,
            "activities": [_activity("Gardens by the Bay", country_code)],
            "day_total_cost": 170.0,
        },
        {
            "day": 2,
            "date": "2026-08-02",
            "flight": [
                _flight(
                    "SIN",
                    "Singapore Changi Airport",
                    "KUL",
                    "Kuala Lumpur International Airport",
                    "2026-08-02",
                )
            ],
            "hotel": None,
            "activities": (
                []
                if empty_day
                else [_activity("National Gallery Singapore", country_code)]
            ),
            "day_total_cost": 100.0 if empty_day else 120.0,
        },
    ]


def _maps_for(itinerary: list[dict]) -> dict[int, dict]:
    return {
        day["day"]: {
            "type": "FeatureCollection",
            "features": [
                *(
                    [
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Point",
                                "coordinates": [
                                    day["hotel"]["location"]["lng"],
                                    day["hotel"]["location"]["lat"],
                                ],
                            },
                            "properties": {
                                "name": day["hotel"]["hotel_name"],
                                "type": "hotel",
                                "order": 0,
                            },
                        }
                    ]
                    if day["hotel"] is not None
                    else []
                ),
                *[
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [103.7764, 1.2966],
                        },
                        "properties": {
                            "name": activity["name"],
                            "type": activity["type"],
                            "order": activity["order"],
                        },
                    }
                    for activity in day["activities"]
                ],
            ],
        }
        for day in itinerary
    }


def _skeleton() -> list[dict]:
    return [
        {"day": 1, "date": "2026-08-01", "activities": []},
        {"day": 2, "date": "2026-08-02", "activities": []},
    ]


class FakePlanningProviders:
    def __init__(self, activity_outcomes: list[object]) -> None:
        self.activity_outcomes = activity_outcomes
        self.flight_states: list[AgentState] = []
        self.activity_states: list[AgentState] = []
        self.map_states: list[AgentState] = []
        self.profiles: list[dict | None] = []

    def flight(self, state: AgentState) -> dict[str, Any]:
        self.flight_states.append(state.model_copy(deep=True))
        return {"draft_itinerary": _skeleton()}

    def activities(
        self, state: AgentState, user_profile: dict | None
    ) -> dict[str, Any]:
        self.activity_states.append(state.model_copy(deep=True))
        self.profiles.append(copy.deepcopy(user_profile))
        outcome = self.activity_outcomes[len(self.activity_states) - 1]
        if isinstance(outcome, BaseException):
            raise outcome
        return {"draft_itinerary": copy.deepcopy(outcome)}

    def maps(self, state: AgentState) -> dict[str, Any]:
        self.map_states.append(state.model_copy(deep=True))
        return {
            "draft_itinerary": copy.deepcopy(state.draft_itinerary),
            "daily_map_info": _maps_for(state.draft_itinerary),
        }

    @property
    def dependencies(self) -> PlanningDependencies:
        return PlanningDependencies(
            flight_planner=self.flight,
            activity_planner=self.activities,
            map_planner=self.maps,
        )


@pytest.fixture
def singapore_state() -> AgentState:
    return AgentState(
        origin_country="Malaysia",
        country="Singapore",
        city=["Singapore"],
        num_people=2,
        start_date="2026-08-01",
        end_date="2026-08-02",
        base_currency_code="MYR",
        dest_currency_code="SGD",
        total_convert_budget=3000.0,
        budget_allocation={"activity": 1000.0, "food": 2000.0},
        draft_itinerary=[{"day": 99, "activities": [{"name": "accepted"}]}],
        daily_map_info={99: {"name": "accepted-map"}},
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_transaction_discards_rejected_attempts_and_returns_third_valid_plan(
    singapore_state: AgentState,
):
    providers = FakePlanningProviders(
        [
            _itinerary(empty_day=True),
            _itinerary(country_code="MY"),
            _itinerary(country_code="SG"),
        ]
    )

    result = await build_validated_plan(
        singapore_state,
        user_profile={"interests": ["food"]},
        dependencies=providers.dependencies,
        max_attempts=3,
    )

    assert result.status == "validated"
    assert result.attempts == 3
    assert result.candidate is not None
    assert result.candidate.attempt == 3
    assert result.candidate.validation.qualified
    assert result.candidate.itinerary == _itinerary(country_code="SG")
    assert all(state.draft_itinerary == [] for state in providers.flight_states)
    assert all(state.daily_map_info == {} for state in providers.flight_states)
    assert providers.profiles == [
        {"interests": ["food"]},
        {"interests": ["food"]},
        {"interests": ["food"]},
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_exhaustion_has_no_candidate_and_preserves_original_state(
    singapore_state: AgentState,
):
    original_itinerary = copy.deepcopy(singapore_state.draft_itinerary)
    original_maps = copy.deepcopy(singapore_state.daily_map_info)
    providers = FakePlanningProviders(
        [_itinerary(empty_day=True), _itinerary(country_code="MY")]
    )

    result = await build_validated_plan(
        singapore_state,
        user_profile=None,
        dependencies=providers.dependencies,
        max_attempts=2,
    )

    assert result.status == "unavailable"
    assert result.candidate is None
    assert result.attempts == 2
    assert {issue.code for issue in result.issues} >= {
        "day.activities.empty",
        "activity.country.mismatch",
    }
    assert singapore_state.draft_itinerary == original_itinerary
    assert singapore_state.daily_map_info == original_maps
    assert len(providers.flight_states) == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_exception_is_diagnostic_and_next_attempt_is_clean(
    singapore_state: AgentState,
):
    providers = FakePlanningProviders(
        [RuntimeError("places offline"), _itinerary(country_code="SG")]
    )

    result = await build_validated_plan(
        singapore_state,
        user_profile=None,
        dependencies=providers.dependencies,
    )

    assert result.status == "validated"
    assert result.attempts == 2
    assert result.candidate is not None
    assert "planning.activity_planner.exception" in {
        issue.code for issue in result.issues
    }
    assert len(providers.map_states) == 1
    assert all(state.draft_itinerary == [] for state in providers.flight_states)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_type"),
    [
        (asyncio.TimeoutError("provider timeout"), asyncio.TimeoutError),
        (asyncio.CancelledError(), asyncio.CancelledError),
    ],
)
async def test_timeout_and_cancellation_propagate(
    singapore_state: AgentState,
    error: BaseException,
    expected_type: type[BaseException],
):
    def interrupt(_state: AgentState) -> dict[str, Any]:
        raise error

    dependencies = PlanningDependencies(
        flight_planner=interrupt,
        activity_planner=lambda state, profile: {},
        map_planner=lambda state: {},
    )

    with pytest.raises(expected_type):
        await build_validated_plan(
            singapore_state,
            user_profile=None,
            dependencies=dependencies,
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_default_stops_after_exactly_three_failed_attempts(
    singapore_state: AgentState,
):
    providers = FakePlanningProviders([_itinerary(empty_day=True)] * 4)

    result = await build_validated_plan(
        singapore_state,
        user_profile=None,
        dependencies=providers.dependencies,
    )

    assert result.status == "unavailable"
    assert result.attempts == 3
    assert len(providers.flight_states) == 3
    assert len(providers.activity_states) == 3
    assert len(providers.map_states) == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_max_attempts_must_be_positive(singapore_state: AgentState):
    providers = FakePlanningProviders([_itinerary(country_code="SG")])

    with pytest.raises(ValueError, match="max_attempts must be at least 1"):
        await build_validated_plan(
            singapore_state,
            user_profile=None,
            dependencies=providers.dependencies,
            max_attempts=0,
        )

    assert providers.flight_states == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_updates_cannot_override_authoritative_trip_fields(
    singapore_state: AgentState,
):
    observed: list[AgentState] = []

    def flight(state: AgentState) -> dict[str, Any]:
        return {
            "draft_itinerary": _skeleton(),
            "country": "Malaysia",
            "city": ["Johor Bahru"],
            "start_date": "2030-01-01",
            "budget_allocation": {"activity": 1.0},
        }

    def activities(state: AgentState, _profile: dict | None) -> dict[str, Any]:
        observed.append(state.model_copy(deep=True))
        return {"draft_itinerary": _itinerary(country_code="SG")}

    result = await build_validated_plan(
        singapore_state,
        user_profile=None,
        dependencies=PlanningDependencies(
            flight_planner=flight,
            activity_planner=activities,
            map_planner=lambda state: {
                "daily_map_info": _maps_for(state.draft_itinerary)
            },
        ),
    )

    assert result.status == "validated"
    assert observed[0].country == "Singapore"
    assert observed[0].city == ["Singapore"]
    assert observed[0].start_date == "2026-08-01"
    assert observed[0].budget_allocation == {"activity": 1000.0, "food": 2000.0}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_nested_mutation_isolated_from_original_and_next_attempt(
    singapore_state: AgentState,
):
    starting_cities = copy.deepcopy(singapore_state.city)
    seen_cities: list[list[str]] = []
    activity_outcomes = [
        _itinerary(empty_day=True),
        _itinerary(country_code="SG"),
    ]

    def flight(state: AgentState) -> dict[str, Any]:
        seen_cities.append(copy.deepcopy(state.city))
        state.city.append("Mutated Provider City")
        state.budget_allocation["activity"] = 1.0
        return {"draft_itinerary": _skeleton()}

    def activities(state: AgentState, _profile: dict | None) -> dict[str, Any]:
        return {"draft_itinerary": activity_outcomes.pop(0)}

    result = await build_validated_plan(
        singapore_state,
        user_profile=None,
        dependencies=PlanningDependencies(
            flight_planner=flight,
            activity_planner=activities,
            map_planner=lambda state: {
                "daily_map_info": _maps_for(state.draft_itinerary)
            },
        ),
    )

    assert result.status == "validated"
    assert result.attempts == 2
    assert seen_cities == [["Singapore"], ["Singapore"]]
    assert singapore_state.city == starting_cities
    assert singapore_state.budget_allocation == {
        "activity": 1000.0,
        "food": 2000.0,
    }
