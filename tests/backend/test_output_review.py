from __future__ import annotations

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents import graph
from app.agents.state import AgentState
from app.services import output_review
from app.services.output_review import (
    OutputReviewContext,
    OutputReviewDecision,
    deterministic_output_review,
    review_public_output,
)


def valid_context(
    *,
    reply: str = "Your Singapore itinerary is ready for review.",
    candidate_plan: dict | None = None,
) -> OutputReviewContext:
    if candidate_plan is None:
        candidate_plan = {
            "itinerary": [{"day": 1, "date": "2026-09-01"}],
            "maps": {1: {"type": "FeatureCollection", "features": []}},
        }
    return OutputReviewContext(
        latest_user_request="Plan my Singapore trip.",
        trusted_requirements={
            "country": "Singapore",
            "city": ["Singapore"],
            "start_date": "2026-09-01",
            "end_date": "2026-09-01",
        },
        normalized_tool_evidence={"destination": "Singapore"},
        proposed_reply=reply,
        candidate_plan=candidate_plan,
        deterministic_report={"issues": []},
    )


def _qualified_itinerary(name: str = "Gardens by the Bay") -> list[dict]:
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
                    "name": name,
                    "type": "attraction",
                    "address": "18 Marina Gardens Drive, Singapore",
                    "estimated_cost": 20,
                    "order": 1,
                    "location": {
                        "place_name": name,
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


def _qualified_maps(
    name: str = "Gardens by the Bay",
    *,
    latitude: float = 1.2816,
    longitude: float = 103.8636,
) -> dict[int, dict]:
    return {
        1: {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [longitude, latitude],
                    },
                    "properties": {
                        "name": name,
                        "type": "attraction",
                        "order": 1,
                    },
                }
            ],
        }
    }


def _canonical_allocation(total: float) -> dict[str, float]:
    return {
        "transportation": total * 0.25,
        "accommodation": total * 0.35,
        "food": total * 0.15,
        "activity": total * 0.15,
        "shopping": total * 0.05,
        "emergency_fund": total * 0.05,
    }


@pytest.fixture
def valid_candidate_state() -> AgentState:
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
        budget_allocation=_canonical_allocation(900),
        draft_itinerary=_qualified_itinerary("Accepted Museum"),
        daily_map_info=_qualified_maps("Accepted Museum"),
        accepted_plan_snapshot={"plan_revision": 4, "marker": "accepted"},
        plan_revision=4,
        messages=[HumanMessage(content="Plan my Singapore itinerary.")],
        planning_outcome="validated",
        candidate_plan={
            "itinerary": _qualified_itinerary(),
            "maps": _qualified_maps(),
            "attempt": 1,
            "validation": {"issues": []},
        },
        planning_attempts=1,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "reply",
    [
        '{"draft_itinerary": [{"day": 1}]}',
        '```json\n{"daily_map_info": {}}\n```',
        "The activities were not injected into the Working Itinerary Summary.",
    ],
)
def test_deterministic_review_rejects_internal_or_json_output(reply):
    """Catch a raw object or internal interface detail reaching public chat."""
    decision = deterministic_output_review(valid_context(reply=reply))

    assert decision.approved is False
    assert decision.issue_codes


@pytest.mark.unit
@pytest.mark.parametrize(
    "reply",
    [
        'Here are your activities: [{"name":"Merlion Park","day":1}]',
        'Singapore plan details: {"days":[{"day":1,"activities":["Merlion Park"]}]}',
        'Your stops are: ["Merlion Park", "Singapore Zoo"]',
        "No activity records were returned: []",
    ],
)
def test_deterministic_review_rejects_embedded_json_fragments(reply):
    """Catch raw JSON embedded inside otherwise conversational assistant text."""
    decision = deterministic_output_review(valid_context(reply=reply))

    assert decision.approved is False
    assert decision.issue_codes == ("reply.raw_internal_data",)


@pytest.mark.unit
@pytest.mark.parametrize(
    "reply",
    [
        "Meet at Terminal [1], then visit {Gardens by the Bay}.",
        "Bring braces { if useful, and use square brackets [ for notes.",
    ],
)
def test_deterministic_review_allows_ordinary_braces_and_punctuation(reply):
    """Catch the JSON detector blocking human prose that merely uses delimiters."""
    assert deterministic_output_review(valid_context(reply=reply)).approved is True


@pytest.mark.unit
def test_deterministic_review_rejects_empty_output():
    """Catch an empty model response being treated as a reviewed reply."""
    decision = deterministic_output_review(valid_context(reply="   "))

    assert decision == OutputReviewDecision(
        approved=False,
        issue_codes=("reply.requirements_missing",),
        feedback="Write a non-empty user-facing reply.",
    )


@pytest.mark.unit
def test_deterministic_review_rejects_planning_success_without_candidate():
    """Catch a success claim when no qualified private plan exists."""
    decision = deterministic_output_review(
        valid_context(
            reply="I've completed and updated your Singapore itinerary.",
            candidate_plan={},
        )
    )

    assert decision.approved is False
    assert decision.issue_codes == ("reply.unsupported_claim",)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_structured_review_rejects_irrelevant_destination_text():
    """Catch destination drift reported by the isolated semantic reviewer."""
    rejected = OutputReviewDecision(
        approved=False,
        issue_codes=("reply.destination_mismatch",),
        feedback="Discuss Singapore, not Paris.",
    )
    reviewer = Mock()
    reviewer.ainvoke = AsyncMock(return_value=rejected)

    with patch.object(output_review, "output_reviewer_llm", reviewer):
        decision = await review_public_output(
            valid_context(reply="Your Paris itinerary is ready.")
        )

    assert decision == rejected


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reviewer_result",
    [
        {"approved": "yes", "issue_codes": [], "feedback": ""},
        {
            "approved": False,
            "issue_codes": ["reply.not_a_real_code"],
            "feedback": "bad schema",
        },
        {
            "approved": True,
            "issue_codes": [],
            "feedback": "",
            "unexpected": "must not be ignored",
        },
        object(),
    ],
)
async def test_malformed_structured_reviewer_output_fails_closed(reviewer_result):
    """Catch malformed structured output becoming implicit approval."""
    reviewer = Mock()
    reviewer.ainvoke = AsyncMock(return_value=reviewer_result)

    with patch.object(output_review, "output_reviewer_llm", reviewer):
        decision = await review_public_output(valid_context())

    assert decision.approved is False
    assert decision.issue_codes == ("review.unavailable",)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reviewer_exception_fails_closed():
    """Catch reviewer provider failures escaping or approving a reply."""
    reviewer = Mock()
    reviewer.ainvoke = AsyncMock(side_effect=RuntimeError("review failed"))

    with patch.object(output_review, "output_reviewer_llm", reviewer):
        decision = await review_public_output(valid_context())

    assert decision == OutputReviewDecision(
        approved=False,
        issue_codes=("review.unavailable",),
        feedback="Output review is temporarily unavailable.",
    )


@pytest.mark.unit
def test_negated_planning_failure_is_not_misread_as_a_success_claim():
    decision = deterministic_output_review(
        valid_context(
            reply="I could not complete your Singapore itinerary.",
            candidate_plan={},
        )
    )

    assert decision.approved is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_deterministic_rejection_skips_semantic_reviewer():
    """Catch unsafe raw output being sent to a reviewer before local rejection."""
    reviewer = Mock()
    reviewer.ainvoke = AsyncMock(
        return_value=OutputReviewDecision(approved=True)
    )

    with patch.object(output_review, "output_reviewer_llm", reviewer):
        decision = await review_public_output(
            valid_context(reply='{"tool_calls": []}')
        )

    assert decision.approved is False
    reviewer.ainvoke.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_candidate_reply_rejection_discards_candidate_for_full_transaction_retry(
    valid_candidate_state,
):
    """A candidate-bearing rejection cannot be repaired by reusing that plan."""
    replies = [AIMessage(content="First attempt.")]
    main_model = Mock()
    main_model.ainvoke = AsyncMock(side_effect=replies)
    rejected = OutputReviewDecision(
        approved=False,
        issue_codes=("reply.unrelated",),
        feedback="Address only the Singapore itinerary.",
    )
    with (
        patch.object(graph, "llm", main_model),
        patch.object(
            graph,
            "review_public_output",
            AsyncMock(return_value=rejected),
        ),
        patch.object(graph, "fetch_user_profile", return_value=None),
    ):
        result = await graph.generate_reviewed_response_node(
            valid_candidate_state,
            {"configurable": {}},
        )

    assert result["planning_outcome"] == "unavailable"
    assert result["candidate_plan"] is None
    assert "messages" not in result
    assert {
        "draft_itinerary",
        "daily_map_info",
        "accepted_plan_snapshot",
        "plan_revision",
    }.isdisjoint(result)
    assert main_model.ainvoke.await_count == 1
    routed = valid_candidate_state.model_copy(update=result, deep=True)
    assert graph.route_after_candidate_review(routed) == "execute_validated_transaction"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_budget_candidate_review_uses_candidate_financial_requirements(
    valid_candidate_state,
):
    """Catch semantic review comparing a budget replan to stale accepted values."""
    state = valid_candidate_state.model_copy(
        update={
            "candidate_plan": {
                **valid_candidate_state.candidate_plan,
                "financials": {
                    "total_base_budget": 2500,
                    "total_convert_budget": 750,
                    "budget_allocation": _canonical_allocation(750),
                },
            }
        },
        deep=True,
    )
    main_model = Mock()
    main_model.ainvoke = AsyncMock(
        return_value=AIMessage(content="Your revised Singapore plan is ready.")
    )
    observed = []

    async def approve(context):
        observed.append(context)
        return OutputReviewDecision(approved=True)

    with (
        patch.object(graph, "llm", main_model),
        patch.object(graph, "review_public_output", side_effect=approve),
        patch.object(graph, "fetch_user_profile", return_value=None),
    ):
        await graph.generate_reviewed_response_node(state, {"configurable": {}})

    assert observed[0].trusted_requirements["total_base_budget"] == 2500
    assert observed[0].trusted_requirements["total_convert_budget"] == 750


@pytest.mark.unit
@pytest.mark.asyncio
async def test_three_rejected_candidate_replies_fail_closed_and_preserve_acceptance(
    valid_candidate_state,
):
    """Catch review exhaustion publishing a candidate or changing revision."""
    main_model = Mock()
    main_model.ainvoke = AsyncMock(return_value=AIMessage(content="Attempt three."))
    rejected = OutputReviewDecision(
        approved=False,
        issue_codes=("reply.unsupported_claim",),
        feedback="Do not claim unsupported success.",
    )

    with (
        patch.object(graph, "llm", main_model),
        patch.object(
            graph,
            "review_public_output",
            AsyncMock(return_value=rejected),
        ),
        patch.object(graph, "fetch_user_profile", return_value=None),
    ):
        result = await graph.generate_reviewed_response_node(
            valid_candidate_state.model_copy(
                update={
                    "planning_attempts": 3,
                    "candidate_plan": {
                        **valid_candidate_state.candidate_plan,
                        "attempt": 3,
                    },
                },
                deep=True,
            ),
            {"configurable": {}},
        )

    assert result["planning_outcome"] == "unavailable"
    assert result["candidate_plan"] is None
    assert len(result["messages"]) == 1
    assert "existing trip plan has not been changed" in result["messages"][0].content
    assert {
        "draft_itinerary",
        "daily_map_info",
        "accepted_plan_snapshot",
        "plan_revision",
    }.isdisjoint(result)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_promote_candidate_atomically_deep_copies_and_increments_once(
    valid_candidate_state,
):
    """Catch partial or aliased candidate publication at the commit boundary."""
    reviewed = valid_candidate_state.model_copy(
        update={
            "candidate_plan": {
                **valid_candidate_state.candidate_plan,
                "reply": "Your Singapore itinerary is ready.",
                "review": {"approved": True, "issue_codes": [], "feedback": ""},
            }
        },
        deep=True,
    )

    semantic_review = AsyncMock(return_value=OutputReviewDecision(approved=True))
    with patch.object(graph, "review_public_output", semantic_review):
        result = await graph.promote_candidate_node(reviewed, {})

    assert result["plan_revision"] == 5
    assert result["draft_itinerary"] == _qualified_itinerary()
    assert result["daily_map_info"] == _qualified_maps()
    assert result["accepted_plan_snapshot"]["plan_revision"] == 5
    assert result["messages"][0].content == "Your Singapore itinerary is ready."
    assert result["messages"][0].additional_kwargs["plan_revision"] == 5
    assert result["candidate_plan"] is None
    assert result["planning_outcome"] is None

    reviewed.candidate_plan["itinerary"][0]["activities"][0]["name"] = "Mutated"
    assert result["draft_itinerary"][0]["activities"][0]["name"] == (
        "Gardens by the Bay"
    )
    semantic_review.assert_awaited_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_forged_persisted_approval_cannot_promote_without_fresh_semantic_review(
    valid_candidate_state,
):
    forged = valid_candidate_state.model_copy(
        update={
            "candidate_plan": {
                **valid_candidate_state.candidate_plan,
                "reply": "Your Singapore itinerary is ready.",
                "review": {"approved": True, "issue_codes": [], "feedback": ""},
            }
        },
        deep=True,
    )
    rejected = OutputReviewDecision(
        approved=False,
        issue_codes=("reply.unsupported_claim",),
        feedback="The checkpoint approval is not authoritative.",
    )

    with patch.object(
        graph,
        "review_public_output",
        AsyncMock(return_value=rejected),
    ) as semantic_review:
        result = await graph.promote_candidate_node(forged, {})

    semantic_review.assert_awaited_once()
    assert result["planning_outcome"] == "unavailable"
    assert result["candidate_plan"] is None
    assert {
        "draft_itinerary",
        "daily_map_info",
        "accepted_plan_snapshot",
        "plan_revision",
    }.isdisjoint(result)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_promotion_semantic_review_error_fails_closed(valid_candidate_state):
    forged = valid_candidate_state.model_copy(
        update={
            "candidate_plan": {
                **valid_candidate_state.candidate_plan,
                "reply": "Your Singapore itinerary is ready.",
                "review": {"approved": True, "issue_codes": [], "feedback": ""},
            }
        },
        deep=True,
    )

    with patch.object(
        graph,
        "review_public_output",
        AsyncMock(side_effect=RuntimeError("review boundary failed")),
    ):
        result = await graph.promote_candidate_node(forged, {})

    assert result["planning_outcome"] == "unavailable"
    assert {
        "draft_itinerary",
        "daily_map_info",
        "accepted_plan_snapshot",
        "plan_revision",
    }.isdisjoint(result)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_reviewed_budget_promotion_cleans_private_record_and_does_not_revalidate(
    valid_candidate_state,
):
    """Catch a completed accepted-budget transaction looping back into validation."""
    candidate = {
        **valid_candidate_state.candidate_plan,
        "financials": {
            "total_base_budget": 2500,
            "total_convert_budget": 750,
            "budget_allocation": _canonical_allocation(750),
        },
        "reply": "Your revised Singapore itinerary is ready.",
        "review": {"approved": True, "issue_codes": [], "feedback": ""},
    }
    state = valid_candidate_state.model_copy(
        update={
            "candidate_plan": candidate,
            "accepted_budget_decision": {
                "stage": "itinerary",
                "target_plan_revision": 5,
                "candidate": {},
            },
            "budget_gate_outcome": "accepted",
        },
        deep=True,
    )

    with patch.object(
        graph,
        "review_public_output",
        AsyncMock(return_value=OutputReviewDecision(approved=True)),
    ):
        update = await graph.promote_candidate_node(state, {})
    committed = state.model_copy(update=update)

    assert update["total_base_budget"] == 2500
    assert update["total_convert_budget"] == 750
    assert update["accepted_budget_decision"] is None
    assert update["budget_gate_outcome"] is None
    assert graph.route_start(committed) == "memory_extraction"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_promotion_defensively_revalidates_and_preserves_prior_plan(
    valid_candidate_state,
):
    """Catch an approved review bypassing deterministic candidate validation."""
    invalid = valid_candidate_state.model_copy(
        update={
            "candidate_plan": {
                "itinerary": [{"day": 1, "date": "2026-09-01", "activities": []}],
                "maps": {},
                "reply": "Your Singapore itinerary is ready.",
                "review": {"approved": True, "issue_codes": [], "feedback": ""},
            }
        },
        deep=True,
    )

    with patch.object(
        graph,
        "review_public_output",
        AsyncMock(return_value=OutputReviewDecision(approved=True)),
    ):
        result = await graph.promote_candidate_node(invalid, {})

    assert result["planning_outcome"] == "unavailable"
    assert result["candidate_plan"] is None
    assert {
        "draft_itinerary",
        "daily_map_info",
        "accepted_plan_snapshot",
        "plan_revision",
    }.isdisjoint(result)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_promotion_rejects_structurally_and_financially_invalid_candidate(
    valid_candidate_state,
):
    """Catch schema-invalid costs/order/logistics mutating the accepted revision."""
    prior_itinerary = valid_candidate_state.draft_itinerary
    prior_snapshot = valid_candidate_state.accepted_plan_snapshot
    invalid_itinerary = _qualified_itinerary()
    invalid_activity = invalid_itinerary[0]["activities"][0]
    invalid_activity.pop("order", None)
    invalid_activity["estimated_cost"] = -50.0
    invalid_itinerary[0]["flight"] = None
    invalid_itinerary[0]["day_total_cost"] = -999.0
    invalid = valid_candidate_state.model_copy(
        update={
            "candidate_plan": {
                "itinerary": invalid_itinerary,
                "maps": _qualified_maps(),
                "reply": "Your Singapore itinerary is ready.",
                "review": {"approved": True, "issue_codes": [], "feedback": ""},
            }
        },
        deep=True,
    )

    with patch.object(
        graph,
        "review_public_output",
        AsyncMock(return_value=OutputReviewDecision(approved=True)),
    ):
        result = await graph.promote_candidate_node(invalid, {})

    assert result["planning_outcome"] == "unavailable"
    assert result["candidate_plan"] is None
    assert {
        "activity.cost.invalid",
        "activity.order.invalid",
        "day.total_cost.invalid",
        "flight.outbound.missing",
        "flight.return.missing",
    } <= set(result["planning_issue_codes"])
    assert {
        "draft_itinerary",
        "daily_map_info",
        "accepted_plan_snapshot",
        "plan_revision",
    }.isdisjoint(result)
    assert valid_candidate_state.draft_itinerary == prior_itinerary
    assert valid_candidate_state.accepted_plan_snapshot == prior_snapshot


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "allocation",
    [
        {**_canonical_allocation(900), "food": -1, "shopping": 136},
        {**_canonical_allocation(900), "activity": 1},
    ],
)
async def test_promotion_rejects_invalid_candidate_financials(
    valid_candidate_state,
    allocation,
):
    invalid = valid_candidate_state.model_copy(
        update={
            "candidate_plan": {
                **valid_candidate_state.candidate_plan,
                "financials": {"budget_allocation": allocation},
                "reply": "Your Singapore itinerary is ready.",
                "review": {"approved": True, "issue_codes": [], "feedback": ""},
            }
        },
        deep=True,
    )

    with patch.object(
        graph,
        "review_public_output",
        AsyncMock(return_value=OutputReviewDecision(approved=True)),
    ):
        result = await graph.promote_candidate_node(invalid, {})

    assert result["planning_outcome"] == "unavailable"
    assert {
        "draft_itinerary",
        "daily_map_info",
        "accepted_plan_snapshot",
        "plan_revision",
    }.isdisjoint(result)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_non_planning_final_text_is_rewritten_before_entering_messages():
    """Catch ordinary final chat bypassing the same review boundary."""
    initial_model = Mock()
    initial_model.ainvoke = AsyncMock(return_value=AIMessage(content="Paris advice."))
    rewrite_model = Mock()
    rewrite_model.ainvoke = AsyncMock(
        return_value=AIMessage(content="Here is the Singapore advice you requested.")
    )
    rejected = OutputReviewDecision(
        approved=False,
        issue_codes=("reply.destination_mismatch",),
        feedback="Answer for Singapore only.",
    )

    with (
        patch.object(graph, "llm_with_tools", initial_model),
        patch.object(graph, "llm", rewrite_model),
        patch.object(
            graph,
            "review_public_output",
            AsyncMock(side_effect=[rejected, OutputReviewDecision(approved=True)]),
        ),
        patch.object(graph, "fetch_user_profile", return_value=None),
    ):
        result = await graph.agent_node(
            AgentState(
                country="Singapore",
                city=["Singapore"],
                base_currency_code="MYR",
                messages=[HumanMessage(content="What should I pack?")],
            ),
            {"configurable": {}},
        )

    assert [message.content for message in result["messages"]] == [
        "Here is the Singapore advice you requested."
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embedded_json_candidate_reply_fails_promotion_and_preserves_acceptance(
    valid_candidate_state,
):
    """A late raw-data rejection must discard the whole candidate transaction."""
    unsafe_reply = 'Plan: {"days":[{"day":1,"activities":["Merlion Park"]}]}'
    state = valid_candidate_state.model_copy(
        update={
            "candidate_plan": {
                **valid_candidate_state.candidate_plan,
                "reply": unsafe_reply,
                "review": {"approved": True, "issue_codes": [], "feedback": ""},
            }
        },
        deep=True,
    )
    semantic_review = AsyncMock(return_value=OutputReviewDecision(approved=True))

    with patch.object(graph, "review_public_output", semantic_review):
        update = await graph.promote_candidate_node(state, {})

    rejected = state.model_copy(update=update, deep=True)
    assert semantic_review.await_count == 0
    assert rejected.candidate_plan is None
    assert rejected.planning_outcome == "unavailable"
    assert graph.route_after_promotion(rejected) == "execute_validated_transaction"
    assert rejected.plan_revision == state.plan_revision
    assert rejected.draft_itinerary == state.draft_itinerary
    assert rejected.daily_map_info == state.daily_map_info
    assert rejected.accepted_plan_snapshot == state.accepted_plan_snapshot
    assert unsafe_reply not in [
        message.content for message in rejected.messages if message.type == "ai"
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_non_planning_embedded_json_is_rewritten_before_publication():
    unsafe_reply = 'Packing data: {"destination":"Singapore","items":["hat"]}'
    initial_model = Mock()
    initial_model.ainvoke = AsyncMock(return_value=AIMessage(content=unsafe_reply))
    rewrite_model = Mock()
    rewrite_model.ainvoke = AsyncMock(
        return_value=AIMessage(content="Pack light clothing and a rain jacket.")
    )
    semantic_model = Mock()
    semantic_model.ainvoke = AsyncMock(
        return_value=OutputReviewDecision(approved=True)
    )

    with (
        patch.object(graph, "llm_with_tools", initial_model),
        patch.object(graph, "llm", rewrite_model),
        patch.object(output_review, "output_reviewer_llm", semantic_model),
        patch.object(graph, "fetch_user_profile", return_value=None),
    ):
        result = await graph.agent_node(
            AgentState(
                country="Singapore",
                city=["Singapore"],
                base_currency_code="MYR",
                messages=[HumanMessage(content="What should I pack?")],
            ),
            {"configurable": {}},
        )

    assert [message.content for message in result["messages"]] == [
        "Pack light clothing and a rain jacket."
    ]
    assert unsafe_reply not in repr(result)
    assert rewrite_model.ainvoke.await_count == 1
    assert semantic_model.ainvoke.await_count == 1


@pytest.mark.unit
def test_incremental_edit_becomes_private_validated_candidate():
    """Catch edit_itinerary publishing before deterministic review and promotion."""
    new_name = "National Gallery Singapore"
    grounded_place = {
        "name": new_name,
        "type": "attraction",
        "address": "1 St Andrew's Road, Singapore",
        "estimated_cost": 15,
        "location": {
            "place_name": new_name,
            "latitude": 1.2903,
            "longitude": 103.8519,
            "country_code": "SG",
            "requested_city": "Singapore",
            "verified_locality": "Singapore",
        },
    }
    state = AgentState(
        origin_country="Malaysia",
        country="Singapore",
        city=["Singapore"],
        start_date="2026-09-01",
        end_date="2026-09-01",
        num_people=1,
        base_currency_code="MYR",
        dest_currency_code="SGD",
        total_convert_budget=900,
        budget_allocation=_canonical_allocation(900),
        draft_itinerary=_qualified_itinerary(),
        daily_map_info=_qualified_maps(),
        plan_revision=2,
        messages=[
            HumanMessage(content="Replace the first activity with the gallery."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_places",
                        "args": {
                            "query": new_name,
                            "category": "attraction",
                            "city": "Singapore",
                        },
                        "id": "search-place-1",
                    }
                ],
            ),
            ToolMessage(
                name="search_places",
                tool_call_id="search-place-1",
                content=json.dumps(
                    {"category": "attraction", "results": [grounded_place], "count": 1}
                ),
            ),
            AIMessage(content="", tool_calls=[]),
            ToolMessage(
                name="edit_itinerary",
                tool_call_id="edit-1",
                content=json.dumps(
                    {
                        "edits": [
                            {
                                "day": 1,
                                "action": "replace",
                                "category": "activity",
                                "index": 1,
                                "new_details": grounded_place,
                            }
                        ]
                    }
                ),
            ),
        ],
    )

    with patch.object(
        graph,
        "generate_daily_map",
        return_value={
            "daily_map_info": _qualified_maps(
                new_name,
                latitude=1.2903,
                longitude=103.8519,
            )
        },
    ):
        result = graph.post_tool_processing_node(state)

    assert result["planning_outcome"] == "validated", result
    assert result["candidate_plan"]["itinerary"][0]["activities"][0]["name"] == (
        new_name
    )
    assert result["candidate_plan"]["itinerary"][0]["activities"][0]["order"] == 1
    assert result["candidate_plan"]["itinerary"][0]["activities"][0]["location"][
        "requested_city"
    ] == "Singapore"
    assert result["candidate_plan"]["validation"] == {"issues": []}
    assert {
        "draft_itinerary",
        "daily_map_info",
        "accepted_plan_snapshot",
        "plan_revision",
    }.isdisjoint(result)


@pytest.mark.unit
def test_incremental_flight_replacement_must_match_current_trusted_search_evidence(
    valid_candidate_state,
):
    grounded_flight = {
        "airline": "Grounded Air",
        "flight_number": "GA123",
        "price": 220.0,
        "departure_airport": {"name": "Kuala Lumpur", "country_code": "MY"},
        "arrival_airport": {"name": "Singapore", "country_code": "SG"},
    }
    fabricated_flight = {**grounded_flight, "flight_number": "FAKE999"}
    state = valid_candidate_state.model_copy(
        update={
            "messages": [
                HumanMessage(content="Use the grounded flight."),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_alternative_opt",
                            "args": {"category": "flight", "day_num": 1},
                            "id": "search-1",
                        }
                    ],
                ),
                ToolMessage(
                    name="search_alternative_opt",
                    tool_call_id="search-1",
                    content=json.dumps({"options": [grounded_flight], "count": 1}),
                ),
                AIMessage(content=""),
                ToolMessage(
                    name="edit_itinerary",
                    tool_call_id="edit-1",
                    content=json.dumps(
                        {
                            "edits": [
                                {
                                    "day": 1,
                                    "action": "replace",
                                    "category": "flight",
                                    "new_details": fabricated_flight,
                                }
                            ]
                        }
                    ),
                ),
            ]
        },
        deep=True,
    )

    assert graph.post_tool_processing_node(state) == {}


@pytest.mark.unit
def test_incremental_flight_replacement_accepts_exact_current_trusted_evidence(
    valid_candidate_state,
):
    grounded_flight = {
        "airline": "Grounded Air",
        "flight_number": "GA123",
        "price": 220.0,
        "departure_airport": {
            "id": "KUL",
            "name": "Kuala Lumpur",
            "time": "2026-09-01 08:00",
        },
        "arrival_airport": {
            "id": "SIN",
            "name": "Singapore",
            "time": "2026-09-01 09:30",
        },
        "departure_time": "2026-09-01 08:00",
        "arrival_time": "2026-09-01 09:30",
    }
    state = valid_candidate_state.model_copy(
        update={
            "messages": [
                HumanMessage(content="Use the grounded flight."),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_alternative_opt",
                            "args": {"category": "flight", "day_num": 1},
                            "id": "search-1",
                        }
                    ],
                ),
                ToolMessage(
                    name="search_alternative_opt",
                    tool_call_id="search-1",
                    content=json.dumps({"options": [grounded_flight], "count": 1}),
                ),
                AIMessage(content=""),
                ToolMessage(
                    name="edit_itinerary",
                    tool_call_id="edit-1",
                    content=json.dumps(
                        {
                            "edits": [
                                {
                                    "day": 1,
                                    "action": "replace",
                                    "category": "flight",
                                    "new_details": grounded_flight,
                                }
                            ]
                        }
                    ),
                ),
            ]
        },
        deep=True,
    )

    with patch.object(
        graph,
        "generate_daily_map",
        return_value={"daily_map_info": state.daily_map_info},
    ):
        result = graph.post_tool_processing_node(state)

    assert result["planning_outcome"] == "validated", result
    flights = result["candidate_plan"]["itinerary"][0]["flight"]
    assert flights[0] == grounded_flight
    assert len(flights) == 2
    assert flights[1]["flight_number"] == "GA-SIN-KUL"


@pytest.mark.unit
def test_incremental_hotel_replacement_rejects_wrong_destination_evidence(
    valid_candidate_state,
):
    wrong_destination_hotel = {
        "hotel_name": "Paris Hotel",
        "price_per_night": 120.0,
        "location": {"lat": 48.8566, "lng": 2.3522, "country_code": "FR"},
    }
    state = valid_candidate_state.model_copy(
        update={
            "messages": [
                HumanMessage(content="Change my hotel."),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_alternative_opt",
                            "args": {"category": "hotel", "day_num": 1},
                            "id": "search-1",
                        }
                    ],
                ),
                ToolMessage(
                    name="search_alternative_opt",
                    tool_call_id="search-1",
                    content=json.dumps(
                        {"options": [wrong_destination_hotel], "count": 1}
                    ),
                ),
                AIMessage(content=""),
                ToolMessage(
                    name="edit_itinerary",
                    tool_call_id="edit-1",
                    content=json.dumps(
                        {
                            "edits": [
                                {
                                    "day": 1,
                                    "action": "replace",
                                    "category": "hotel",
                                    "new_details": wrong_destination_hotel,
                                }
                            ]
                        }
                    ),
                ),
            ]
        },
        deep=True,
    )

    assert graph.post_tool_processing_node(state) == {}


@pytest.mark.unit
def test_booking_evidence_for_another_day_cannot_ground_replacement(
    valid_candidate_state,
):
    grounded_hotel = {
        "hotel_name": "Singapore Hotel",
        "price_per_night": 120.0,
        "location": {"lat": 1.31, "lng": 103.82, "country_code": "SG"},
    }
    state = valid_candidate_state.model_copy(
        update={
            "messages": [
                HumanMessage(content="Show a hotel for day one."),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_alternative_opt",
                            "args": {"category": "hotel", "day_num": 1},
                            "id": "search-1",
                        }
                    ],
                ),
                ToolMessage(
                    name="search_alternative_opt",
                    tool_call_id="search-1",
                    content=json.dumps({"options": [grounded_hotel], "count": 1}),
                ),
            ]
        },
        deep=True,
    )
    edit = {
        "day": 2,
        "action": "replace",
        "category": "hotel",
        "new_details": grounded_hotel,
    }

    assert graph._booking_edit_is_grounded(
        state,
        edit,
        graph._trusted_booking_options(state),
    ) is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_budget_gate_terminal_text_is_semantically_reviewed_before_publication():
    reviewer = AsyncMock(return_value=OutputReviewDecision(approved=True))
    state = AgentState(
        country="Singapore",
        budget_gate_outcome="budget_check_unavailable",
        messages=[HumanMessage(content="Can we continue with this budget?")],
    )

    with patch.object(graph, "review_public_output", reviewer):
        result = await graph.budget_gate_response_node(state, {})

    reviewer.assert_awaited_once()
    assert result["messages"][0].content == graph._BUDGET_GATE_SAFE_FALLBACK


@pytest.mark.unit
@pytest.mark.asyncio
async def test_budget_gate_review_exhaustion_uses_only_fixed_server_fallback():
    rejected = OutputReviewDecision(
        approved=False,
        issue_codes=("review.unavailable",),
        feedback="Output review is unavailable.",
    )
    reviewer = AsyncMock(side_effect=[rejected, rejected, rejected])
    rewrite_model = Mock()
    rewrite_model.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content="Rejected rewrite one."),
            AIMessage(content="Rejected rewrite two."),
        ]
    )
    state = AgentState(
        country="Singapore",
        budget_gate_outcome="budget_check_unavailable",
        messages=[HumanMessage(content="Can we continue with this budget?")],
    )

    with (
        patch.object(graph, "review_public_output", reviewer),
        patch.object(graph, "llm", rewrite_model),
    ):
        result = await graph.budget_gate_response_node(state, {})

    assert reviewer.await_count == 3
    assert [message.content for message in result["messages"]] == [
        graph._OUTPUT_REVIEW_SAFE_FALLBACK
    ]
    assert result["messages"][0].additional_kwargs == {
        "server_owned": "planning_unavailable"
    }


@pytest.mark.unit
def test_only_fixed_server_fallback_may_skip_semantic_review_after_exhaustion():
    state = AgentState(planning_outcome="validated", candidate_plan={"private": True})

    update = graph._review_exhaustion_update(
        state,
        attempts=3,
        issue_codes=["review.unavailable"],
    )
    message = update["messages"][0]
    context = graph._output_review_context(state, message.content, None)

    assert message.content == graph._OUTPUT_REVIEW_SAFE_FALLBACK
    assert message.additional_kwargs == {"server_owned": "planning_unavailable"}
    assert deterministic_output_review(context).approved is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_new_turn_resets_stale_unavailable_before_current_update_replans(
    valid_candidate_state,
):
    failed = valid_candidate_state.model_copy(
        update={
            "planning_outcome": "unavailable",
            "candidate_plan": None,
            "planning_attempts": 3,
            "planning_issue_codes": ["review.unavailable"],
            "output_review_attempts": 3,
            "output_review_issue_codes": ["review.unavailable"],
            "messages": [HumanMessage(content="Try changing the dates again.")],
        },
        deep=True,
    )
    assert failed.planning_outcome == "unavailable"

    reset = await graph.memory_extraction_node(failed, {"configurable": {}})
    current_turn = failed.model_copy(update=reset, deep=True)
    assert current_turn.planning_outcome is None

    current_turn = current_turn.model_copy(
        update={
            "messages": [
                HumanMessage(content="Start one day later."),
                AIMessage(content=""),
                ToolMessage(
                    name="update_trip_details",
                    tool_call_id="update-1",
                    content=json.dumps({"updates": {"start_date": "2026-09-02"}}),
                ),
            ],
            "end_date": "2026-09-04",
        },
        deep=True,
    )
    update = graph.post_tool_processing_node(current_turn)
    routed = current_turn.model_copy(update=update, deep=True)

    assert graph.route_after_post_tool(routed) == "prepare_planning_transaction"


@pytest.mark.unit
def test_graph_routes_candidates_through_review_then_single_promotion():
    """Catch any validated transaction edge that bypasses output review."""
    assert (
        "execute_validated_transaction",
        "generate_reviewed_response",
    ) in graph.builder.edges
    assert (
        graph.builder.nodes["generate_reviewed_response"].runnable.afunc
        is graph.generate_reviewed_response_node
    )
    assert (
        graph.builder.nodes["promote_candidate"].runnable.afunc
        is graph.promote_candidate_node
    )
    assert ("execute_validated_transaction", "__end__") not in graph.builder.edges
    assert ("run_map_update", "capture_accepted_plan") not in graph.builder.edges
    for legacy_name in ("run_map_update", "capture_accepted_plan"):
        assert (
            graph.builder.nodes[legacy_name].runnable.func
            is graph.prepare_planning_transaction_node
        )
