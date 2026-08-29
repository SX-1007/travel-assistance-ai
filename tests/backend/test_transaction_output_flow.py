from __future__ import annotations

import copy
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agents import graph
from app.agents.state import AgentState
from app.services import planning_transaction
from app.services.output_review import OutputReviewDecision
from app.tools import flights_hotels


def _allocation(total: float) -> dict[str, float]:
    return {
        "transportation": total * 0.25,
        "accommodation": total * 0.35,
        "food": total * 0.15,
        "activity": total * 0.15,
        "shopping": total * 0.05,
        "emergency_fund": total * 0.05,
    }


def _itinerary(marker: str) -> list[dict]:
    outbound = {
        "airline": "Grounded Air",
        "flight_number": "GA-KUL-SIN",
        "departure_airport": {
            "id": "KUL",
            "name": "Kuala Lumpur International Airport",
            "time": "2026-09-01 08:00",
        },
        "arrival_airport": {
            "id": "SIN",
            "name": "Singapore Changi Airport",
            "time": "2026-09-01 09:30",
        },
        "departure_time": "2026-09-01 08:00",
        "arrival_time": "2026-09-01 09:30",
        "price": 100.0,
    }
    returning = {
        **outbound,
        "flight_number": "GA-SIN-KUL",
        "departure_airport": outbound["arrival_airport"],
        "arrival_airport": outbound["departure_airport"],
        "departure_time": "2026-09-01 18:00",
        "arrival_time": "2026-09-01 19:30",
    }
    return [
        {
            "day": 1,
            "date": "2026-09-01",
            "flight": [outbound, returning],
            "hotel": None,
            "activities": [
                {
                    "name": marker,
                    "type": "attraction",
                    "address": "18 Marina Gardens Drive, Singapore",
                    "estimated_cost": 20,
                    "order": 1,
                    "location": {
                        "place_name": marker,
                        "latitude": 1.2816,
                        "longitude": 103.8636,
                        "country_code": "SG",
                        "requested_city": "Singapore",
                        "verified_locality": "Singapore",
                    },
                }
            ],
            "day_total_cost": 220,
        }
    ]


def _maps(marker: str) -> dict[int, dict]:
    return {
        1: {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [103.8636, 1.2816],
                    },
                    "properties": {"name": marker, "type": "attraction", "order": 1},
                }
            ],
        }
    }


def _state() -> AgentState:
    accepted_itinerary = _itinerary("Accepted museum")
    accepted_maps = _maps("Accepted museum")
    return AgentState(
        origin_country="Malaysia",
        country="Singapore",
        city=["Singapore"],
        start_date="2026-09-01",
        end_date="2026-09-01",
        num_people=1,
        total_base_budget=3000,
        base_currency_code="MYR",
        dest_currency_code="SGD",
        total_convert_budget=900,
        budget_allocation=_allocation(900),
        draft_itinerary=accepted_itinerary,
        daily_map_info=accepted_maps,
        plan_revision=4,
        accepted_plan_snapshot={
            "plan_revision": 4,
            "total_base_budget": 3000,
            "base_currency_code": "MYR",
            "dest_currency_code": "SGD",
            "total_convert_budget": 900,
            "budget_allocation": _allocation(900),
            "draft_itinerary": accepted_itinerary,
            "daily_map_info": accepted_maps,
        },
        messages=[HumanMessage(content="Plan my Singapore itinerary.")],
    )


class _Providers:
    def __init__(self, markers: list[str]) -> None:
        self.markers = markers
        self.flight_calls: list[AgentState] = []
        self.activity_calls: list[AgentState] = []
        self.map_calls: list[AgentState] = []

    def flight(self, state: AgentState) -> dict:
        self.flight_calls.append(state.model_copy(deep=True))
        return {"draft_itinerary": [{"day": 1, "date": "2026-09-01"}]}

    def activities(self, state: AgentState, _profile: dict | None) -> dict:
        self.activity_calls.append(state.model_copy(deep=True))
        marker = self.markers[len(self.activity_calls) - 1]
        return {"draft_itinerary": _itinerary(marker)}

    def maps(self, state: AgentState) -> dict:
        self.map_calls.append(state.model_copy(deep=True))
        marker = state.draft_itinerary[0]["activities"][0]["name"]
        return {"daily_map_info": _maps(marker)}

    @property
    def dependencies(self) -> planning_transaction.PlanningDependencies:
        return planning_transaction.PlanningDependencies(
            flight_planner=self.flight,
            activity_planner=self.activities,
            map_planner=self.maps,
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_same_day_transaction_reuses_form_evidence_without_hotel_provider():
    itinerary = _itinerary("Merlion Park")
    outbound, returning = itinerary[0]["flight"]
    state = _state().model_copy(
        update={
            "draft_itinerary": [], "daily_map_info": {},
            "accepted_plan_snapshot": None, "plan_revision": 0,
            "budget_assessment": {
                "calculation_version": "allocation-v1", "origin": "Malaysia",
                "destination": "Singapore", "start_date": "2026-09-01",
                "end_date": "2026-09-01", "num_people": 1,
                "expires_at": "2999-09-01T00:00:00+00:00",
                "evidence": {"outbound_flight": outbound,
                    "return_flight": returning, "hotel": {},
                    "hotel_nights": 0, "hotel_price_per_night": 0},
            },
        }, deep=True,
    )

    def activities(candidate: AgentState, _profile: dict | None) -> dict:
        day = copy.deepcopy(candidate.draft_itinerary[0])
        day["activities"] = copy.deepcopy(itinerary[0]["activities"])
        day["day_total_cost"] += 20
        return {"draft_itinerary": [day]}

    def maps(_candidate: AgentState) -> dict:
        return {"daily_map_info": _maps("Merlion Park")}

    dependencies = planning_transaction.PlanningDependencies(
        flight_planner=flights_hotels.plan_flight_hotel,
        activity_planner=activities,
        map_planner=maps,
    )
    with (
        patch.object(flights_hotels, "fetch_flights_api",
                     side_effect=AssertionError("form evidence must be reused")),
        patch.object(flights_hotels, "fetch_hotels_api",
                     side_effect=AssertionError("same-day transaction needs no hotel")) as hotel,
        patch.object(flights_hotels, "resolve_country_code") as resolve,
    ):
        result = await planning_transaction.build_validated_plan(
            state, None, dependencies=dependencies, max_attempts=1
        )

    hotel.assert_not_called()
    resolve.assert_not_called()
    assert result.status == "validated", result.issues
    assert result.candidate is not None
    day = result.candidate.itinerary[0]
    assert len(day["flight"]) == 2
    assert day["hotel"] is None
    assert day["day_total_cost"] == 220


@pytest.mark.unit
@pytest.mark.asyncio
async def test_candidate_rejection_reexecutes_every_provider_with_clean_state():
    """Candidate A must be destroyed; candidate B starts from trusted fields."""
    state = _state()
    providers = _Providers(["Candidate A", "Candidate B"])
    real_build = planning_transaction.build_validated_plan

    async def controlled_build(current, profile, **kwargs):
        return await real_build(
            current,
            profile,
            dependencies=providers.dependencies,
            **kwargs,
        )

    model = Mock()
    model.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content="Candidate A public reply."),
            AIMessage(content="Candidate B public reply."),
        ]
    )
    rejected = OutputReviewDecision(
        approved=False,
        issue_codes=("reply.unrelated",),
        feedback="Redo the complete plan.",
    )
    reviewer = AsyncMock(
        side_effect=[
            rejected,
            OutputReviewDecision(approved=True),
            OutputReviewDecision(approved=True),
        ]
    )

    with (
        patch.object(graph, "build_validated_plan", new=controlled_build),
        patch.object(graph, "fetch_user_profile", return_value=None),
        patch.object(graph, "llm", model),
        patch.object(graph, "review_public_output", reviewer),
    ):
        first_plan = await graph.plan_validated_transaction_node(
            state, {"configurable": {}}
        )
        first_state = state.model_copy(update=first_plan, deep=True)
        candidate_a = copy.deepcopy(first_state.candidate_plan)

        first_review = await graph.generate_reviewed_response_node(
            first_state, {"configurable": {}}
        )
        rejected_state = first_state.model_copy(update=first_review, deep=True)

        assert rejected_state.candidate_plan is None
        assert graph.route_after_candidate_review(rejected_state) == (
            "execute_validated_transaction"
        )
        assert rejected_state.draft_itinerary == state.draft_itinerary
        assert rejected_state.accepted_plan_snapshot == state.accepted_plan_snapshot
        assert rejected_state.plan_revision == 4

        second_plan = await graph.plan_validated_transaction_node(
            rejected_state, {"configurable": {}}
        )
        second_state = rejected_state.model_copy(update=second_plan, deep=True)
        assert second_state.candidate_plan is not candidate_a
        assert second_state.candidate_plan["itinerary"][0]["activities"][0]["name"] == (
            "Candidate B"
        )
        assert "Candidate A" not in repr(second_state.candidate_plan)

        second_review = await graph.generate_reviewed_response_node(
            second_state, {"configurable": {}}
        )
        reviewed_state = second_state.model_copy(update=second_review, deep=True)
        promoted = await graph.promote_candidate_node(reviewed_state, {})

    assert len(providers.flight_calls) == 2
    assert len(providers.activity_calls) == 2
    assert len(providers.map_calls) == 2
    assert all(call.draft_itinerary == [] for call in providers.flight_calls)
    assert all(call.daily_map_info == {} for call in providers.flight_calls)
    assert promoted["plan_revision"] == 5
    assert promoted["draft_itinerary"][0]["activities"][0]["name"] == "Candidate B"
    assert promoted["messages"][0].content == "Candidate B public reply."


@pytest.mark.unit
@pytest.mark.asyncio
async def test_three_candidate_rejections_share_one_budget_and_preserve_acceptance():
    state = _state()
    providers = _Providers(["Candidate A", "Candidate B", "Candidate C", "Forbidden D"])
    real_build = planning_transaction.build_validated_plan

    async def controlled_build(current, profile, **kwargs):
        return await real_build(
            current,
            profile,
            dependencies=providers.dependencies,
            **kwargs,
        )

    model = Mock()
    model.ainvoke = AsyncMock(
        side_effect=[AIMessage(content=f"Reply {letter}") for letter in "ABC"]
    )
    rejected = OutputReviewDecision(
        approved=False,
        issue_codes=("reply.unrelated",),
        feedback="Reject this candidate.",
    )

    current = state
    with (
        patch.object(graph, "build_validated_plan", new=controlled_build),
        patch.object(graph, "fetch_user_profile", return_value=None),
        patch.object(graph, "llm", model),
        patch.object(
            graph,
            "review_public_output",
            AsyncMock(side_effect=[rejected, rejected, rejected]),
        ),
    ):
        for attempt in range(1, 4):
            planned = await graph.plan_validated_transaction_node(
                current, {"configurable": {}}
            )
            current = current.model_copy(update=planned, deep=True)
            reviewed = await graph.generate_reviewed_response_node(
                current, {"configurable": {}}
            )
            current = current.model_copy(update=reviewed, deep=True)
            assert current.planning_attempts == attempt

    assert len(providers.flight_calls) == 3
    assert len(providers.activity_calls) == 3
    assert len(providers.map_calls) == 3
    assert current.planning_outcome == "unavailable"
    assert current.candidate_plan is None
    assert current.plan_revision == 4
    assert current.draft_itinerary == state.draft_itinerary
    assert current.accepted_plan_snapshot == state.accepted_plan_snapshot
    assert [message.content for message in current.messages if message.type == "ai"] == [
        graph._OUTPUT_REVIEW_SAFE_FALLBACK
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_transaction_attempt_offset_is_global_and_validated():
    state = _state()
    providers = _Providers(["Candidate B"])

    result = await planning_transaction.build_validated_plan(
        state,
        None,
        dependencies=providers.dependencies,
        max_attempts=1,
        attempt_offset=1,
    )

    assert result.attempts == 2
    assert result.candidate is not None
    assert result.candidate.attempt == 2

    with pytest.raises(ValueError, match="attempt_offset must be non-negative"):
        await planning_transaction.build_validated_plan(
            state,
            None,
            dependencies=providers.dependencies,
            max_attempts=1,
            attempt_offset=-1,
        )
