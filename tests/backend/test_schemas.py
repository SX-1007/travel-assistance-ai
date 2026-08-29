from __future__ import annotations

from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from app.agents.state import AgentState
from app.schemas.requests import ChatRequest, InitialFormRequest, OnboardingRequest
from app.schemas.responses import (
    ActivityLocation,
    ActivityResult,
    BudgetCheckUnavailableResponse,
    BudgetConfirmationResponse,
    DailyItinerary,
    FinalResponse,
    FlightResult,
    HotelResult,
    MapLocation,
    PlanningUnavailableResponse,
)


VALID_FORM = {
    "country": "Japan",
    "city": ["Tokyo", "Kyoto"],
    "num_people": 2,
    "total_budget": 5000.0,
    "start_date": "2026-08-01",
    "end_date": "2026-08-05",
}


def _producer_shaped_success_payload():
    """Public data emitted by the current flight/hotel/map producers."""
    return {
        "status": "success",
        "chat_reply": "Your reviewed itinerary is ready.",
        "session_id": "session-1",
        "destination_country_code": "JP",
        "itinerary": [
            {
                "day": 1,
                "flight": [
                    {
                        "airline": "Grounded Air",
                        "flight_number": "GA101",
                        "departure_airport": {
                            "id": "KUL",
                            "name": "KLIA",
                            "time": "08:00",
                        },
                        "arrival_airport": {
                            "id": "NRT",
                            "name": "Narita",
                            "time": "16:00",
                        },
                        "departure_time": "08:00",
                        "arrival_time": "16:00",
                        "duration": 420,
                        "price": 500,
                    }
                ],
                "hotel": {
                    "hotel_name": "Sakura Inn",
                    "price_per_night": 120,
                    "location": {
                        "lat": 35.68,
                        "lng": 139.76,
                        "country_code": "JP",
                    },
                },
                "activities": [],
                "route": {
                    "ordered_stops": ["Sakura Inn", "Tokyo Tower"],
                    "profiles": {
                        "driving": {"distance_km": 10, "duration_mins": 20},
                        "walking": {"distance_km": 8, "duration_mins": 100},
                        "cycling": {"distance_km": 9, "duration_mins": 45},
                    },
                },
            }
        ],
        "daily_geojson_maps": {
            1: {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [139.76, 35.68],
                        },
                        "properties": {
                            "name": "Sakura Inn",
                            "type": "hotel",
                            "order": 0,
                        },
                    },
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "LineString",
                            "coordinates": [[139.76, 35.68], [139.75, 35.66]],
                        },
                        "properties": {
                            "type": "route",
                            "profile": "driving",
                            "distance_km": 10,
                            "duration_mins": 20,
                        },
                    },
                ],
            }
        },
    }


def _nested_set(payload, path, value):
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


@pytest.mark.unit
class TestOnboardingRequest:
    def test_valid_values_are_trimmed(self):
        model = OnboardingRequest(
            name="  Alex Tan  ",
            origin_country=" Malaysia ",
            origin_state=" Selangor ",
        )
        assert model.model_dump() == {
            "name": "Alex Tan",
            "origin_country": "Malaysia",
            "origin_state": "Selangor",
        }

    @pytest.mark.parametrize("field", ["name", "origin_country", "origin_state"])
    @pytest.mark.parametrize("value", ["", "   "])
    def test_required_text_rejects_empty_or_whitespace(self, field, value):
        payload = {
            "name": "Alex",
            "origin_country": "Malaysia",
            "origin_state": "Selangor",
        }
        payload[field] = value
        with pytest.raises(ValidationError):
            OnboardingRequest(**payload)

    def test_unknown_fields_are_ignored(self):
        model = OnboardingRequest(
            name="Alex",
            origin_country="Malaysia",
            origin_state="Selangor",
            admin=True,
        )
        assert "admin" not in model.model_dump()


@pytest.mark.unit
class TestInitialFormRequest:
    def test_valid_complete_form(self):
        model = InitialFormRequest(**VALID_FORM)
        assert model.country == "Japan"
        assert model.city == ["Tokyo", "Kyoto"]
        assert model.total_budget == 5000.0

    def test_explicit_unknown_budget_mode_allows_omitted_amount(self):
        payload = {
            **VALID_FORM,
            "total_budget": None,
            "request_budget_recommendation": True,
        }

        model = InitialFormRequest(**payload)

        assert model.total_budget is None
        assert model.request_budget_recommendation is True
        assert model.budget_assessment_id is None

    def test_confirmation_mode_accepts_positive_amount_and_assessment_id(self):
        model = InitialFormRequest(
            **{
                **VALID_FORM,
                "total_budget": 6000,
                "budget_assessment_id": "assessment-1",
            }
        )

        assert model.total_budget == 6000
        assert model.budget_assessment_id == "assessment-1"

    def test_destination_city_is_required(self):
        payload = {k: v for k, v in VALID_FORM.items() if k != "city"}
        with pytest.raises(ValidationError):
            InitialFormRequest(**payload)

    @pytest.mark.parametrize("city", [[], [""], ["   "]])
    def test_destination_city_must_include_a_named_locality(self, city):
        with pytest.raises(ValidationError):
            InitialFormRequest(**{**VALID_FORM, "city": city})

    def test_text_is_trimmed_and_extra_fields_ignored(self):
        payload = {**VALID_FORM, "country": "  Japan ", "origin_country": "Malaysia"}
        model = InitialFormRequest(**payload)
        assert model.country == "Japan"
        assert not hasattr(model, "origin_country")

    @pytest.mark.parametrize("num_people", [0, -1, -99])
    def test_people_must_be_positive(self, num_people):
        with pytest.raises(ValidationError):
            InitialFormRequest(**{**VALID_FORM, "num_people": num_people})

    @pytest.mark.parametrize("budget", [0, -0.01, -1000, float("nan"), float("inf")])
    def test_budget_must_be_positive(self, budget):
        with pytest.raises(ValidationError):
            InitialFormRequest(**{**VALID_FORM, "total_budget": budget})

    @pytest.mark.parametrize(
        "changes",
        [
            {"total_budget": None},
            {"total_budget": None, "budget_assessment_id": "assessment-1"},
            {"total_budget": 5000, "request_budget_recommendation": True},
            {
                "total_budget": None,
                "request_budget_recommendation": True,
                "budget_assessment_id": "assessment-1",
            },
        ],
    )
    def test_budget_modes_are_mutually_explicit(self, changes):
        with pytest.raises(ValidationError):
            InitialFormRequest(**{**VALID_FORM, **changes})

    def test_same_day_trip_is_accepted(self):
        request = InitialFormRequest(
            **{**VALID_FORM, "start_date": "2026-08-01", "end_date": "2026-08-01"}
        )
        assert request.end_date == request.start_date

    def test_end_must_not_precede_start(self):
        with pytest.raises(ValidationError, match="end_date must be on or after start_date"):
            InitialFormRequest(
                **{**VALID_FORM, "start_date": "2026-08-02", "end_date": "2026-08-01"}
            )

    @pytest.mark.parametrize(
        "field,value",
        [
            ("start_date", "01-08-2026"),
            ("end_date", "2026/08/05"),
            ("start_date", "2026-02-30"),
            ("end_date", "not-a-date"),
        ],
    )
    def test_dates_require_real_iso_calendar_dates(self, field, value):
        with pytest.raises(ValidationError, match="Invalid date format"):
            InitialFormRequest(**{**VALID_FORM, field: value})

    def test_one_day_trip_is_valid_when_dates_are_consecutive(self):
        model = InitialFormRequest(
            **{**VALID_FORM, "start_date": "2026-08-01", "end_date": "2026-08-02"}
        )
        assert model.end_date == "2026-08-02"

    def test_date_strings_in_the_past_are_schema_valid(self):
        yesterday = date.today() - timedelta(days=1)
        today = date.today()
        model = InitialFormRequest(
            **{
                **VALID_FORM,
                "start_date": yesterday.isoformat(),
                "end_date": today.isoformat(),
            }
        )
        assert model.start_date == yesterday.isoformat()


@pytest.mark.unit
class TestChatRequest:
    def test_valid_chat_is_trimmed(self):
        model = ChatRequest(session_id=" session-1 ", user_message=" hello ")
        assert model.session_id == "session-1"
        assert model.user_message == "hello"

    @pytest.mark.parametrize(
        "payload",
        [
            {"session_id": "", "user_message": "hello"},
            {"session_id": "s", "user_message": ""},
            {"session_id": "   ", "user_message": "hello"},
            {"session_id": "s", "user_message": "   "},
        ],
    )
    def test_empty_identifiers_and_messages_are_rejected(self, payload):
        with pytest.raises(ValidationError):
            ChatRequest(**payload)

    def test_message_boundary_8192_is_accepted(self):
        assert (
            len(ChatRequest(session_id="s", user_message="x" * 8192).user_message)
            == 8192
        )

    def test_message_over_8192_is_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(session_id="s", user_message="x" * 8193)

    def test_structured_chat_budget_confirmation_requires_both_action_and_id(self):
        """Catch requests that could invoke a confirmation without its opaque ID."""
        valid = ChatRequest(
            session_id="s",
            user_message="Use recommended budget",
            budget_action="accept_recommended",
            budget_assessment_id="assessment-1",
        )
        assert valid.budget_action == "accept_recommended"
        assert valid.budget_assessment_id == "assessment-1"

        with pytest.raises(ValidationError):
            ChatRequest(
                session_id="s",
                user_message="Use it",
                budget_action="accept_recommended",
            )
        with pytest.raises(ValidationError):
            ChatRequest(
                session_id="s",
                user_message="hello",
                budget_assessment_id="assessment-1",
            )

    @pytest.mark.parametrize(
        "payload",
        [
            {
                "budget_action": "propose_budget_change",
                "budget_assessment_id": "assessment-1",
            },
            {
                "budget_action": "accept_recommended",
                "budget_assessment_id": "   ",
            },
        ],
    )
    def test_structured_chat_budget_confirmation_rejects_invalid_action_or_id(
        self, payload
    ):
        """Catch malformed structured actions before they can reach the graph."""
        with pytest.raises(ValidationError):
            ChatRequest(session_id="s", user_message="Use it", **payload)


@pytest.mark.unit
class TestResponseAndStateModels:
    def test_activity_requires_verified_destination_location(self):
        """Catch incomplete provider data crossing the public plan boundary."""
        with pytest.raises(ValidationError):
            ActivityResult(
                name="Museum",
                type="attraction",
                address="",
                estimated_cost=10,
                order=1,
                location={"latitude": 1.3, "longitude": 103.8},
            )

    @pytest.mark.parametrize(
        "changes",
        [
            {"name": "   "},
            {"type": "shopping"},
            {"address": ""},
            {"estimated_cost": -1},
            {"estimated_cost": float("inf")},
            {"order": 0},
            {"order": True},
            {"location": {"place_name": "Museum", "country_code": "SG", "latitude": float("nan"), "longitude": 103.8}},
            {"location": {"place_name": "Museum", "country_code": "SGP", "latitude": 1.3, "longitude": 103.8}},
        ],
    )
    def test_activity_public_contract_rejects_incomplete_fields(self, changes):
        """Catch any required activity field becoming permissive."""
        payload = {
            "name": "Museum",
            "type": "attraction",
            "address": "1 Museum Road",
            "estimated_cost": 10,
            "order": 1,
            "location": {
                "place_name": "Museum",
                "country_code": "SG",
                "latitude": 1.3,
                "longitude": 103.8,
            },
        }
        with pytest.raises(ValidationError):
            ActivityResult(**{**payload, **changes})

    def test_public_activity_models_forbid_internal_fields(self):
        """Catch candidate or review internals being silently accepted publicly."""
        with pytest.raises(ValidationError):
            ActivityResult(
                name="Museum",
                type="attraction",
                address="1 Museum Road",
                estimated_cost=10,
                order=1,
                location=ActivityLocation(
                    place_name="Museum",
                    country_code="SG",
                    requested_city="Singapore",
                    verified_locality="Singapore",
                    latitude=1.3,
                    longitude=103.8,
                ),
                candidate_plan={"private": True},
            )

    def test_activity_location_rejects_requested_city_without_matching_proof(self):
        with pytest.raises(ValidationError):
            ActivityLocation(
                place_name="Penang State Museum",
                country_code="MY",
                requested_city="Kuala Lumpur",
                verified_locality="George Town",
                latitude=5.4141,
                longitude=100.3288,
            )

    def test_planning_unavailable_response_is_plan_free_and_strict(self):
        response = PlanningUnavailableResponse(
            status="planning_unavailable",
            reason="validation_failed",
            chat_reply="Planning is temporarily unavailable.",
        )

        assert response.retryable is True
        assert response.itinerary is None
        assert response.daily_geojson_maps is None
        with pytest.raises(ValidationError):
            PlanningUnavailableResponse(
                status="planning_unavailable",
                reason="validation_failed",
                chat_reply="Planning is temporarily unavailable.",
                candidate_plan={"private": True},
            )

    def test_agent_state_uses_independent_mutable_defaults(self):
        first = AgentState()
        second = AgentState()
        first.city.append("Tokyo")
        first.budget_allocation["food"] = 10
        assert second.city == []
        assert second.budget_allocation == {}

    def test_agent_state_rejects_zero_people(self):
        with pytest.raises(ValidationError):
            AgentState(num_people=0)

    def test_flight_result_defaults_are_stable(self):
        result = FlightResult()
        assert result.flight_number == ""
        assert result.price == 0.0

    def test_flight_result_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            FlightResult(tool_result={"secret": "must-not-serialize"})

    def test_flight_result_preserves_over_budget_status(self):
        result = FlightResult(airline="Test Air", price=700, over_budget=True)

        assert result.model_dump()["over_budget"] is True

    @pytest.mark.parametrize(
        "field,value", [("price", -1), ("duration", -1), ("stops", -1)]
    )
    def test_flight_numeric_fields_reject_negative_values(self, field, value):
        with pytest.raises(ValidationError):
            FlightResult(**{field: value})

    @pytest.mark.parametrize("rating", [-1, 5.1, 99])
    def test_hotel_class_is_bounded(self, rating):
        with pytest.raises(ValidationError):
            HotelResult(hotel_class=rating)

    def test_daily_itinerary_accepts_normalised_nested_data(self):
        day = DailyItinerary(
            day=1,
            flight=[{"airline": "Test Air", "price": 99}],
            hotel={"hotel_name": "Test Hotel", "price_per_night": 50},
            activities=[
                {
                    "name": "Museum",
                    "type": "attraction",
                    "address": "1 Museum Road",
                    "estimated_cost": 10,
                    "order": 1,
                    "location": {
                        "place_name": "Museum",
                        "country_code": "SG",
                        "requested_city": "Singapore",
                        "verified_locality": "Singapore",
                        "latitude": 1.3,
                        "longitude": 103.8,
                    },
                }
            ],
        )
        assert day.flight[0].airline == "Test Air"
        assert day.hotel.hotel_name == "Test Hotel"

    def test_daily_itinerary_day_starts_at_one(self):
        with pytest.raises(ValidationError):
            DailyItinerary(day=0)

    def test_map_location_allows_missing_coordinates(self):
        assert MapLocation(place_name="Somewhere").latitude is None

    def test_final_response_requires_session_id(self):
        with pytest.raises(ValidationError):
            FinalResponse(status="success")

    def test_final_response_status_is_success_only(self):
        with pytest.raises(ValidationError):
            FinalResponse(
                status="budget_confirmation_required",
                session_id="must-not-exist",
            )

    def test_budget_confirmation_response_is_explicitly_itinerary_free(self):
        response = BudgetConfirmationResponse(
            status="budget_confirmation_required",
            reason="insufficient_budget",
            chat_reply="The entered budget is below the grounded minimum.",
            budget_assessment_id="assessment-1",
            stated_budget=5000,
            recommended_minimum_budget=6000,
            base_currency="MYR",
            destination_currency="JPY",
            expires_at="2026-08-16T13:00:00+00:00",
            evidence={
                "outbound_flight_price": 32000,
                "return_flight_price": 16000,
                "hotel_price_per_night": 14000,
                "hotel_nights": 4,
            },
        )

        assert response.itinerary is None
        assert response.daily_geojson_maps is None
        assert "session_id" not in response.model_dump()

    def test_budget_unavailable_response_is_explicitly_itinerary_free(self):
        response = BudgetCheckUnavailableResponse(
            status="budget_check_unavailable",
            reason="assessment_expired_or_invalid",
            chat_reply="Refresh the budget check before planning.",
        )

        assert response.itinerary is None
        assert response.daily_geojson_maps is None
        assert "session_id" not in response.model_dump()

    def test_final_response_serialises_integer_map_keys(self):
        response = FinalResponse(
            status="success",
            chat_reply="Your reviewed itinerary is ready.",
            session_id="session-1",
            itinerary=[],
            daily_geojson_maps={1: {"type": "FeatureCollection", "features": []}},
            destination_country_code="SG",
        )
        dumped = response.model_dump(mode="json")
        assert dumped["daily_geojson_maps"] == {
            "1": {"type": "FeatureCollection", "features": []}
        }

    def test_final_response_accepts_current_producer_shapes(self):
        response = FinalResponse.model_validate(_producer_shaped_success_payload())

        assert response.itinerary[0].route.profiles.driving.distance_km == 10
        assert response.itinerary[0].flight[0].departure_airport.id == "KUL"
        assert response.daily_geojson_maps[1].features[1].properties.type == "route"

    @pytest.mark.parametrize(
        "path",
        [
            ("daily_geojson_maps", 1, "provider_blob"),
            ("daily_geojson_maps", 1, "features", 0, "provider_blob"),
            (
                "daily_geojson_maps",
                1,
                "features",
                0,
                "geometry",
                "provider_blob",
            ),
            (
                "daily_geojson_maps",
                1,
                "features",
                0,
                "properties",
                "provider_blob",
            ),
            ("itinerary", 0, "route", "provider_blob"),
            ("itinerary", 0, "route", "profiles", "driving", "provider_blob"),
            ("itinerary", 0, "flight", 0, "provider_blob"),
            (
                "itinerary",
                0,
                "flight",
                0,
                "departure_airport",
                "provider_blob",
            ),
            ("itinerary", 0, "hotel", "provider_blob"),
            ("itinerary", 0, "hotel", "location", "provider_blob"),
        ],
        ids=[
            "feature-collection",
            "feature",
            "geometry",
            "properties",
            "route",
            "route-metric",
            "flight",
            "airport",
            "hotel",
            "hotel-location",
        ],
    )
    def test_final_response_recursively_rejects_unknown_public_fields(self, path):
        payload = _producer_shaped_success_payload()
        _nested_set(payload, path, {"secret": "must-not-serialize"})

        with pytest.raises(ValidationError):
            FinalResponse.model_validate(payload)

    def test_final_response_rejects_colliding_normalized_map_day_keys(self):
        payload = _producer_shaped_success_payload()
        payload["daily_geojson_maps"] = {
            1: payload["daily_geojson_maps"][1],
            "1": {"type": "FeatureCollection", "features": []},
        }

        with pytest.raises(ValidationError):
            FinalResponse.model_validate(payload)

    def test_final_response_normalizes_canonical_string_map_day_key(self):
        payload = _producer_shaped_success_payload()
        payload["daily_geojson_maps"] = {
            "1": payload["daily_geojson_maps"][1]
        }

        response = FinalResponse.model_validate(payload)

        assert list(response.daily_geojson_maps) == [1]

    @pytest.mark.parametrize("day_key", [0, -1, "0", "-1", "01", "1.0", True])
    def test_final_response_rejects_invalid_map_day_keys(self, day_key):
        payload = _producer_shaped_success_payload()
        payload["daily_geojson_maps"] = {
            day_key: payload["daily_geojson_maps"][1]
        }

        with pytest.raises(ValidationError):
            FinalResponse.model_validate(payload)

    def test_final_response_preserves_both_flight_legs_and_booking_metadata(self):
        response = FinalResponse(
            status="success",
            chat_reply="Your reviewed itinerary is ready.",
            session_id="session-1",
            destination_country_code="SG",
            daily_geojson_maps={},
            itinerary=[
                {
                    "day": 1,
                    "flight": [
                        {
                            "airline": "Outbound Air",
                            "price": 500,
                            "booking_url": "https://book.test/outbound",
                            "over_budget": True,
                        }
                    ],
                },
                {
                    "day": 2,
                    "flight": [
                        {
                            "airline": "Return Air",
                            "price": 350,
                            "booking_url": "https://book.test/return",
                            "over_budget": True,
                        }
                    ],
                },
            ],
        )

        flights = [day["flight"][0] for day in response.model_dump()["itinerary"]]
        assert [flight["booking_url"] for flight in flights] == [
            "https://book.test/outbound",
            "https://book.test/return",
        ]
        assert all(flight["over_budget"] is True for flight in flights)
