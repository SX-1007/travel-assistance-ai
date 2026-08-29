from __future__ import annotations

import copy
from unittest.mock import Mock, call, patch

import pytest

from app.agents.state import AgentState
from app.tools import attractions, flights_hotels, mapbox
from app.tools.attractions import PlaceIdea


def load_budget_chat_module():
    import importlib
    import importlib.util

    spec = importlib.util.find_spec("app.tools.budget_chat")
    assert spec is not None, "chat budget intent tools are not implemented"
    return importlib.import_module("app.tools.budget_chat")


def test_chat_budget_tools_expose_intent_without_protected_state_updates():
    budget_chat = load_budget_chat_module()

    amount = budget_chat.propose_budget_change.invoke(
        {"proposed_total_base_budget": 500, "request_recommendation": False}
    )
    unknown = budget_chat.propose_budget_change.invoke(
        {"proposed_total_base_budget": None, "request_recommendation": True}
    )
    confirm = budget_chat.confirm_recommended_budget.invoke({})

    assert amount == {
        "status": "budget_proposal",
        "proposal": {"mode": "amount", "total_base_budget": 500.0},
    }
    assert unknown == {
        "status": "budget_proposal",
        "proposal": {"mode": "recommendation", "total_base_budget": None},
    }
    assert confirm == {
        "status": "budget_proposal",
        "proposal": {"mode": "confirm", "total_base_budget": None},
    }
    protected = {
        "total_base_budget", "total_convert_budget", "budget_allocation",
        "draft_itinerary", "daily_map_info",
    }
    assert protected.isdisjoint(amount)
    assert protected.isdisjoint(unknown)
    assert protected.isdisjoint(confirm)


@pytest.mark.parametrize("amount", [0, -1, float("nan"), float("inf")])
def test_chat_budget_proposal_rejects_non_positive_or_non_finite_amount(amount):
    budget_chat = load_budget_chat_module()

    result = budget_chat.propose_budget_change.invoke({
        "proposed_total_base_budget": amount,
        "request_recommendation": False,
    })
    assert set(result) == {"error"}
    assert "proposal" not in result


def test_chat_budget_proposal_rejects_amount_in_recommendation_mode():
    budget_chat = load_budget_chat_module()

    result = budget_chat.propose_budget_change.invoke({
        "proposed_total_base_budget": 500,
        "request_recommendation": True,
    })
    assert result == {"error": "Recommendation mode cannot include an amount."}


def sample_flight(price=500.0):
    return {
        "price": price,
        "total_duration": 180,
        "airline_logo": "logo.png",
        "flights": [
            {
                "airline": "Test Air",
                "flight_number": "TA100",
                "travel_class": "Economy",
                "airplane": "A320",
                "departure_airport": {"id": "KUL", "name": "KLIA", "time": "08:00"},
                "arrival_airport": {"id": "SIN", "name": "Changi", "time": "09:00"},
            },
            {
                "airline": "Test Air",
                "flight_number": "TA200",
                "departure_airport": {"id": "SIN", "name": "Changi", "time": "10:00"},
                "arrival_airport": {"id": "NRT", "name": "Narita", "time": "14:00"},
            },
        ],
        "layovers": [{"name": "Changi"}],
    }


def parsed_hotel(price=100.0):
    return {
        "hotel_name": "Test Hotel",
        "price_per_night": price,
        "location": {"lat": 35.0, "lng": 139.0},
        "amenities": ["WiFi"],
    }


def parsed_direct_flight(origin, destination, price=100.0, airline="Test Air"):
    return flights_hotels._parse_flight(
        {
            "price": price,
            "total_duration": 60,
            "flights": [
                {
                    "airline": airline,
                    "flight_number": "TA100",
                    "departure_airport": {"id": origin},
                    "arrival_airport": {"id": destination},
                }
            ],
        }
    )


@pytest.mark.unit
class TestFlightAndHotelParsing:
    @pytest.mark.parametrize(
        "start,end,expected",
        [
            ("2026-01-01", "2026-01-05", 4),
            ("2026-01-01", "2026-01-02", 1),
            ("2026-01-01", "2026-01-01", 0),
            ("2026-01-05", "2026-01-01", 1),
            ("bad", "2026-01-01", 1),
            (None, None, 1),
        ],
    )
    def test_calculate_total_nights(self, start, end, expected):
        assert flights_hotels.calculate_total_nights(start, end) == expected

    def test_same_day_skeleton_places_both_boundary_flights_with_zero_hotel_nights(self):
        outbound = {"flight_number": "OUT", "price": 120.0}
        return_flight = {"flight_number": "RET", "price": 80.0}
        hotel = {"hotel_name": "Unused Same-Day Hotel", "price_per_night": 500.0}

        itinerary = flights_hotels._build_itinerary_skeleton(
            start_date="2026-08-01",
            end_date="2026-08-01",
            outbound=outbound,
            return_flight=return_flight,
            hotel=hotel,
        )

        assert len(itinerary) == 1
        assert itinerary[0]["flight"] == [outbound, return_flight]
        assert itinerary[0]["hotel"] is None
        assert itinerary[0]["day_total_cost"] == 200.0

    def test_same_day_plan_skips_hotel_provider_and_emits_no_hotel(self):
        outbound = {"flight_number": "OUT", "price": 30.0}
        returning = {"flight_number": "RET", "price": 20.0}
        state = AgentState(
            origin_country="Malaysia", country="Japan", start_date="2026-08-01",
            end_date="2026-08-01", num_people=1, dest_currency_code="JPY",
            budget_allocation={"transportation": 200, "accommodation": 0},
        )
        with (
            patch.object(
                flights_hotels, "fetch_flights_api",
                side_effect=[[outbound], [returning]],
            ),
            patch.object(
                flights_hotels, "fetch_hotels_api",
                side_effect=AssertionError("same-day planning must not query hotels"),
            ) as hotel_provider,
        ):
            day = flights_hotels.plan_flight_hotel(state)["draft_itinerary"][0]

        hotel_provider.assert_not_called()
        assert day["flight"] == [outbound, returning]
        assert day["hotel"] is None
        assert day["day_total_cost"] == 50

    def test_same_day_plan_reuses_canonical_zero_night_assessment(self):
        outbound = parsed_direct_flight("KUL", "HND", 30.0)
        returning = parsed_direct_flight("HND", "KUL", 20.0)
        state = AgentState(
            origin_country="Malaysia", country="Japan", start_date="2026-08-01",
            end_date="2026-08-01", num_people=1,
            budget_assessment={
                "calculation_version": "allocation-v1", "origin": "Malaysia",
                "destination": "Japan", "start_date": "2026-08-01",
                "end_date": "2026-08-01", "num_people": 1,
                "expires_at": "2999-08-16T13:00:00+00:00",
                "evidence": {"outbound_flight": outbound,
                    "return_flight": returning, "hotel": {},
                    "hotel_nights": 0, "hotel_price_per_night": 0},
            },
        )
        with (
            patch.object(flights_hotels, "resolve_country_code") as resolve,
            patch.object(flights_hotels, "fetch_flights_api",
                         side_effect=AssertionError("assessment should be reused")),
            patch.object(flights_hotels, "fetch_hotels_api",
                         side_effect=AssertionError("zero nights need no hotel")),
        ):
            day = flights_hotels.plan_flight_hotel(state)["draft_itinerary"][0]
        resolve.assert_not_called()
        assert day["hotel"] is None
        assert day["day_total_cost"] == 50

    def test_parse_connecting_flight_uses_first_departure_and_last_arrival(self):
        result = flights_hotels._parse_flight(sample_flight(), "https://book.example")
        assert result["departure_airport"]["id"] == "KUL"
        assert result["arrival_airport"]["id"] == "NRT"
        assert result["stops"] == 1
        assert result["layovers"] == ["Changi"]
        assert result["booking_url"] == "https://book.example"

    @pytest.mark.parametrize(
        "mutation",
        [
            {"price": None},
            {"price": "not-money"},
            {"flights": []},
        ],
    )
    def test_invalid_flight_is_dropped(self, mutation):
        payload = sample_flight()
        payload.update(mutation)
        assert flights_hotels._parse_flight(payload) is None

    @pytest.mark.parametrize("source", ["provider", "cache"])
    def test_fetch_flights_drops_wrong_route_before_cache_or_selection(self, source):
        wrong_route = {
            "price": 99,
            "total_duration": 60,
            "flights": [
                {
                    "airline": "Wrong Route Air",
                    "flight_number": "WR100",
                    "departure_airport": {"id": "KUL", "name": "KLIA"},
                    "arrival_airport": {"id": "JHB", "name": "Senai"},
                }
            ],
        }
        cached = (
            [flights_hotels._parse_flight(wrong_route)] if source == "cache" else None
        )
        provider_payload = (
            {} if source == "cache" else {"best_flights": [wrong_route]}
        )

        with (
            patch.object(flights_hotels, "get_cached_data", return_value=cached),
            patch.object(
                flights_hotels,
                "resolve_iata_code",
                side_effect=lambda country: {
                    "Malaysia": "KUL",
                    "Singapore": "SIN",
                }[country],
            ),
            patch.object(
                flights_hotels, "_serpapi_get", return_value=provider_payload
            ),
            patch.object(flights_hotels, "set_cached_data") as set_cache,
        ):
            result = flights_hotels.fetch_flights_api(
                "Malaysia", "Singapore", "2026-08-01", "SGD"
            )

        assert result == []
        set_cache.assert_not_called()

    @pytest.mark.parametrize("source", ["provider", "cache"])
    def test_fetch_flights_keeps_exact_normalized_query_endpoints(self, source):
        valid_route = {
            "price": 120,
            "total_duration": 60,
            "flights": [
                {
                    "airline": "Correct Route Air",
                    "flight_number": "CR100",
                    "departure_airport": {"code": " kul ", "name": "KLIA"},
                    "arrival_airport": {"id": "sin", "name": "Changi"},
                }
            ],
        }
        cached = (
            [flights_hotels._parse_flight(valid_route)] if source == "cache" else None
        )
        provider_payload = (
            {} if source == "cache" else {"best_flights": [valid_route]}
        )

        with (
            patch.object(flights_hotels, "get_cached_data", return_value=cached),
            patch.object(
                flights_hotels,
                "resolve_iata_code",
                side_effect=lambda country: {
                    "Malaysia": "KUL",
                    "Singapore": "SIN",
                }[country],
            ),
            patch.object(
                flights_hotels, "_serpapi_get", return_value=provider_payload
            ) as provider,
            patch.object(flights_hotels, "set_cached_data"),
        ):
            result = flights_hotels.fetch_flights_api(
                "Malaysia", "Singapore", "2026-08-01", "SGD"
            )

        assert [(flight["departure_airport"], flight["arrival_airport"]) for flight in result] == [
            (
                {"code": " kul ", "name": "KLIA"},
                {"id": "sin", "name": "Changi"},
            )
        ]
        if source == "cache":
            provider.assert_not_called()

    def test_fetch_flights_rejects_symmetric_wrong_country_routes(self):
        outbound_to_wrong_country = {
            "price": 90,
            "flights": [
                {
                    "airline": "Wrong Route Air",
                    "departure_airport": {"id": "KUL"},
                    "arrival_airport": {"id": "JHB"},
                }
            ],
        }
        return_from_wrong_country = {
            "price": 80,
            "flights": [
                {
                    "airline": "Wrong Route Air",
                    "departure_airport": {"id": "JHB"},
                    "arrival_airport": {"id": "KUL"},
                }
            ],
        }

        def provider(params):
            if params["departure_id"] == "KUL":
                return {"best_flights": [outbound_to_wrong_country]}
            return {"best_flights": [return_from_wrong_country]}

        with (
            patch.object(flights_hotels, "get_cached_data", return_value=None),
            patch.object(
                flights_hotels,
                "resolve_iata_code",
                side_effect=lambda country: {
                    "Malaysia": "KUL",
                    "Singapore": "SIN",
                }[country],
            ),
            patch.object(flights_hotels, "_serpapi_get", side_effect=provider),
            patch.object(flights_hotels, "set_cached_data") as set_cache,
        ):
            outbound = flights_hotels.fetch_flights_api(
                "Malaysia", "Singapore", "2026-08-01", "SGD"
            )
            returning = flights_hotels.fetch_flights_api(
                "Singapore", "Malaysia", "2026-08-02", "SGD"
            )

        assert outbound == []
        assert returning == []
        set_cache.assert_not_called()

    def test_fetch_flights_rejects_malformed_identity_even_with_valid_alias(self):
        malformed = {
            "price": 120,
            "flights": [
                {
                    "airline": "Malformed Route Air",
                    "departure_airport": {"id": "not-iata", "code": "KUL"},
                    "arrival_airport": {"id": "SIN"},
                }
            ],
        }
        with (
            patch.object(flights_hotels, "get_cached_data", return_value=None),
            patch.object(
                flights_hotels,
                "resolve_iata_code",
                side_effect=lambda country: {
                    "Malaysia": "KUL",
                    "Singapore": "SIN",
                }[country],
            ),
            patch.object(
                flights_hotels,
                "_serpapi_get",
                return_value={"best_flights": [malformed]},
            ),
            patch.object(flights_hotels, "set_cached_data") as set_cache,
        ):
            result = flights_hotels.fetch_flights_api(
                "Malaysia", "Singapore", "2026-08-01", "SGD"
            )

        assert result == []
        set_cache.assert_not_called()

    def test_fetch_flights_rejects_non_ascii_resolved_iata_identity(self):
        with (
            patch.object(flights_hotels, "get_cached_data") as get_cache,
            patch.object(
                flights_hotels,
                "resolve_iata_code",
                side_effect=lambda country: {
                    "Malaysia": "KÜL",
                    "Singapore": "SIN",
                }[country],
            ),
            patch.object(flights_hotels, "_serpapi_get") as provider,
        ):
            result = flights_hotels.fetch_flights_api(
                "Malaysia", "Singapore", "2026-08-01", "SGD"
            )

        assert result == []
        get_cache.assert_not_called()
        provider.assert_not_called()

    def test_assessment_evidence_rejects_symmetric_wrong_country_flight_pair(self):
        outbound = flights_hotels._parse_flight(
            {
                "price": 90,
                "flights": [
                    {
                        "airline": "Wrong Route Air",
                        "departure_airport": {"id": "KUL"},
                        "arrival_airport": {"id": "JHB"},
                    }
                ],
            }
        )
        returning = flights_hotels._parse_flight(
            {
                "price": 80,
                "flights": [
                    {
                        "airline": "Wrong Route Air",
                        "departure_airport": {"id": "JHB"},
                        "arrival_airport": {"id": "KUL"},
                    }
                ],
            }
        )
        state = AgentState(
            origin_country="Malaysia",
            country="Singapore",
            start_date="2026-08-01",
            end_date="2026-08-01",
            num_people=1,
            budget_assessment={
                "calculation_version": "allocation-v1",
                "origin": "Malaysia",
                "destination": "Singapore",
                "start_date": "2026-08-01",
                "end_date": "2026-08-01",
                "num_people": 1,
                "expires_at": "2999-08-16T13:00:00+00:00",
                "evidence": {
                    "outbound_flight": outbound,
                    "return_flight": returning,
                    "hotel": {},
                    "hotel_nights": 0,
                    "hotel_price_per_night": 0,
                },
            },
        )

        assert flights_hotels._assessment_evidence_for_state(state, "SG") is None

    @pytest.mark.parametrize(
        "rate_info,expected",
        [
            ({"extracted_lowest": 123.45, "lowest": "$999"}, 123.45),
            ({"lowest": "¥12,345"}, 12345.0),
            ({"lowest": "RM 88.50"}, 88.5),
            ({"lowest": 77}, 77.0),
            ({}, None),
            ({"lowest": "free"}, None),
        ],
    )
    def test_extract_hotel_rate(self, rate_info, expected):
        assert flights_hotels._extract_rate(rate_info) == expected

    def test_parse_hotel_normalises_images_ratings_location_and_booking_fallback(self):
        payload = {
            "name": "Sakura Inn",
            "rate_per_night": {"extracted_lowest": 120},
            "extracted_hotel_class": "4",
            "overall_rating": "4.6",
            "reviews": "321",
            "images": [{"thumbnail": "thumb.jpg"}],
            "gps_coordinates": {"latitude": "35.1", "longitude": "139.2"},
            "amenities": ["WiFi"],
        }
        result = flights_hotels._parse_hotel(payload, "Tokyo")
        assert result["hotel_class"] == 4.0
        assert result["overall_rating"] == 4.6
        assert result["reviews"] == 321
        assert result["image"] == "thumb.jpg"
        assert result["location"] == {"lat": 35.1, "lng": 139.2}
        assert "google.com/travel/hotels" in result["booking_url"]

    def test_parse_hotel_prefers_canonical_coordinate_keys_over_legacy_aliases(self):
        result = flights_hotels._parse_hotel(
            {
                "name": "Canonical Hotel",
                "rate_per_night": {"extracted_lowest": 120},
                "gps_coordinates": {
                    "latitude": "35.1",
                    "longitude": "139.2",
                    "lat": 0,
                    "lng": 0,
                },
            }
        )
        assert result["location"] == {"lat": 35.1, "lng": 139.2}

    def test_parse_hotel_requires_a_rate(self):
        assert flights_hotels._parse_hotel({"name": "No Price"}) is None

    def test_fetch_hotels_rejects_other_country_and_keys_cache_by_country_code(self):
        singapore_hotel = {
            "name": "SG Hotel",
            "rate_per_night": {"extracted_lowest": 100},
            "gps_coordinates": {"latitude": 1.30, "longitude": 103.85},
        }
        malaysia_hotel = {
            "name": "MY Hotel",
            "rate_per_night": {"extracted_lowest": 80},
            "gps_coordinates": {"latitude": 1.49, "longitude": 103.74},
        }
        with (
            patch.object(flights_hotels, "get_cached_data", return_value=None) as get_cache,
            patch.object(
                flights_hotels,
                "_serpapi_get",
                return_value={"properties": [singapore_hotel, malaysia_hotel]},
            ),
            patch.object(
                flights_hotels, "resolve_country_code", side_effect=["SG", "MY"]
            ) as resolve,
            patch.object(
                flights_hotels,
                "resolve_locality_names",
                side_effect=[("Singapore",), ("Johor Bahru",)],
            ),
            patch.object(flights_hotels, "set_cached_data") as set_cache,
        ):
            result = flights_hotels.fetch_hotels_api(
                "Singapore, Singapore",
                "2026-08-01",
                "2026-08-05",
                "SGD",
                destination_country_code="SG",
            )
        assert [hotel["hotel_name"] for hotel in result] == ["SG Hotel"]
        assert result[0]["location"]["country_code"] == "SG"
        assert result[0]["location"] == {
            "lat": 1.30,
            "lng": 103.85,
            "country_code": "SG",
        }
        assert resolve.call_args_list == [call(1.30, 103.85), call(1.49, 103.74)]
        assert get_cache.call_args.kwargs["country_code"] == "SG"
        assert set_cache.call_args.kwargs["country_code"] == "SG"

    def test_fetch_hotels_reverifies_cached_coordinates_instead_of_trusting_code(self):
        cached_hotels = [
            {
                **parsed_hotel(100),
                "hotel_name": "Singapore Hotel",
                "location": {
                    "lat": "1.30",
                    "lng": "103.85",
                    "country_code": "SG",
                },
            },
            {
                **parsed_hotel(80),
                "hotel_name": "Johor Hotel",
                "location": {
                    "lat": "1.49",
                    "lng": "103.74",
                    "country_code": "SG",
                },
            },
        ]
        with (
            patch.object(flights_hotels, "get_cached_data", return_value=cached_hotels),
            patch.object(flights_hotels, "_serpapi_get") as provider,
            patch.object(
                flights_hotels, "resolve_country_code", side_effect=["SG", "MY"]
            ) as resolve,
            patch.object(
                flights_hotels,
                "resolve_locality_names",
                side_effect=[("Singapore",), ("Johor Bahru",)],
            ),
        ):
            result = flights_hotels.fetch_hotels_api(
                "Singapore, Singapore",
                "2026-08-01",
                "2026-08-05",
                "SGD",
                destination_country_code="SG",
            )
        assert [hotel["hotel_name"] for hotel in result] == ["Singapore Hotel"]
        assert result[0]["location"] == {
            "lat": 1.30,
            "lng": 103.85,
            "country_code": "SG",
        }
        assert resolve.call_args_list == [call(1.30, 103.85), call(1.49, 103.74)]
        provider.assert_not_called()

    def test_pick_cheapest_prefers_within_budget(self):
        options = [{"price": 50}, {"price": 100}, {"price": 200}]
        assert flights_hotels._pick_cheapest(options, "price", 120) == {"price": 50}

    def test_pick_cheapest_falls_back_with_flag(self):
        options = [{"price": 150}, {"price": 200}]
        assert flights_hotels._pick_cheapest(options, "price", 100) == {
            "price": 150,
            "over_budget": True,
        }

    def test_pick_cheapest_handles_empty_list(self):
        assert flights_hotels._pick_cheapest([], "price", 100) is None

    def test_select_flight_pair_applies_combined_budget_without_mutating_options(self):
        outbound_options = [{"airline": "Out Air", "price": 400.0}]
        return_options = [{"airline": "Home Air", "price": 300.0}]

        outbound, return_flight = flights_hotels._select_flight_pair(
            outbound_options, return_options, 650.0
        )

        assert outbound == {"airline": "Out Air", "price": 400.0, "over_budget": True}
        assert return_flight == {
            "airline": "Home Air",
            "price": 300.0,
            "over_budget": True,
        }
        assert outbound_options == [{"airline": "Out Air", "price": 400.0}]
        assert return_options == [{"airline": "Home Air", "price": 300.0}]

    @pytest.mark.parametrize("budget_cap", [0.0, 700.0, 750.0])
    def test_select_flight_pair_does_not_flag_a_pair_within_budget(self, budget_cap):
        outbound, return_flight = flights_hotels._select_flight_pair(
            [{"price": 400.0}], [{"price": 300.0}], budget_cap
        )

        assert "over_budget" not in outbound
        assert "over_budget" not in return_flight

    @pytest.mark.parametrize(
        "outbound_options,return_options,expected_outbound,expected_return",
        [
            ([{"price": 400.0}], [], {"price": 400.0}, None),
            ([], [{"price": 300.0}], None, {"price": 300.0}),
        ],
    )
    def test_select_flight_pair_keeps_whichever_leg_is_available(
        self,
        outbound_options,
        return_options,
        expected_outbound,
        expected_return,
    ):
        outbound, return_flight = flights_hotels._select_flight_pair(
            outbound_options, return_options, 1000.0
        )

        assert outbound == expected_outbound
        assert return_flight == expected_return

    @pytest.mark.parametrize(
        "outbound_options,return_options,expected_leg",
        [
            ([{"price": 400.0}], [], "outbound"),
            ([], [{"price": 400.0}], "return"),
        ],
    )
    def test_select_flight_pair_flags_a_single_available_leg_over_the_trip_cap(
        self, outbound_options, return_options, expected_leg
    ):
        outbound, return_flight = flights_hotels._select_flight_pair(
            outbound_options, return_options, 350.0
        )

        selected = outbound if expected_leg == "outbound" else return_flight
        assert selected == {"price": 400.0, "over_budget": True}

    def test_plan_flight_hotel_fetches_and_charges_both_boundary_day_flights(self):
        outbound = flights_hotels._parse_flight(sample_flight(500))
        return_flight = {
            **outbound,
            "airline": "Home Air",
            "flight_number": "HA200",
            "departure_airport": {"id": "NRT", "name": "Narita"},
            "arrival_airport": {"id": "KUL", "name": "KLIA"},
            "departure_time": "2026-08-04 10:00",
            "arrival_time": "2026-08-04 18:00",
            "price": 350.0,
            "booking_url": "https://book.example/return",
        }
        hotel = parsed_hotel(100)
        state = AgentState(
            origin_country="Malaysia",
            country="Japan",
            city=["Tokyo"],
            start_date="2026-08-01",
            end_date="2026-08-04",
            num_people=2,
            dest_currency_code="JPY",
            budget_allocation={"transportation": 1000, "accommodation": 600},
        )

        def flight_results(origin, destination, date, currency, max_budget, adults):
            if (origin, destination, date) == ("Malaysia", "Japan", "2026-08-01"):
                return [outbound]
            if (origin, destination, date) == ("Japan", "Malaysia", "2026-08-04"):
                return [return_flight]
            raise AssertionError(f"Unexpected flight search: {origin}, {destination}, {date}")

        with (
            patch.object(
                flights_hotels, "fetch_flights_api", side_effect=flight_results
            ) as fetch_flights,
            patch.object(flights_hotels, "fetch_hotels_api", return_value=[hotel]),
        ):
            itinerary = flights_hotels.plan_flight_hotel(state)["draft_itinerary"]

        assert {call.args for call in fetch_flights.call_args_list} == {
            ("Malaysia", "Japan", "2026-08-01", "JPY", None, 2),
            ("Japan", "Malaysia", "2026-08-04", "JPY", None, 2),
        }
        assert fetch_flights.call_count == 2
        assert [day["date"] for day in itinerary] == [
            "2026-08-01",
            "2026-08-02",
            "2026-08-03",
            "2026-08-04",
        ]
        assert itinerary[0]["flight"] == [outbound]
        assert itinerary[1]["flight"] is None
        assert itinerary[2]["flight"] is None
        assert itinerary[3]["flight"] == [return_flight]
        assert [day["day_total_cost"] for day in itinerary] == [600.0, 100.0, 100.0, 350.0]

    def test_plan_flight_hotel_reuses_matching_assessment_without_provider_calls(
        self,
    ):
        outbound = parsed_direct_flight("KUL", "HND", 500.0)
        return_flight = parsed_direct_flight("HND", "KUL", 350.0, "Home Air")
        hotel = parsed_hotel(100)
        state = AgentState(
            origin_country="Malaysia",
            country="Japan",
            city=["Tokyo"],
            start_date="2026-08-01",
            end_date="2026-08-05",
            num_people=2,
            budget_assessment={
                "assessment_id": "assessment-1",
                "calculation_version": "allocation-v1",
                "origin": "Malaysia",
                "destination": "Japan",
                "start_date": "2026-08-01",
                "end_date": "2026-08-05",
                "num_people": 2,
                "base_currency": "MYR",
                "destination_currency": "JPY",
                "exchange_rate": 32.0,
                "minimum_destination_budget": 192000.0,
                "recommended_minimum_budget": 6000.0,
                "evidence": {
                    "outbound_flight": outbound,
                    "return_flight": return_flight,
                    "hotel": hotel,
                    "outbound_flight_price": 500.0,
                    "return_flight_price": 350.0,
                    "hotel_price_per_night": 100.0,
                    "hotel_nights": 4,
                },
                "created_at": "2026-08-16T12:00:00+00:00",
                "expires_at": "2999-08-16T13:00:00+00:00",
            },
        )

        with (
            patch.object(
                flights_hotels, "resolve_country_code", return_value="JP"
            ) as resolve,
            patch.object(
                flights_hotels,
                "resolve_locality_names",
                return_value=("Tokyo",),
            ),
            patch.object(
                flights_hotels,
                "fetch_flights_api",
                side_effect=AssertionError("assessment must avoid flight API"),
            ),
            patch.object(
                flights_hotels,
                "fetch_hotels_api",
                side_effect=AssertionError("assessment must avoid hotel API"),
            ),
        ):
            itinerary = flights_hotels.plan_flight_hotel(state)["draft_itinerary"]

        assert len(itinerary) == 5
        assert itinerary[0]["flight"] == [outbound]
        assert itinerary[-1]["flight"] == [return_flight]
        verified_hotel = {
            **hotel,
            "location": {"lat": 35.0, "lng": 139.0, "country_code": "JP"},
        }
        assert [day["hotel"] for day in itinerary] == [verified_hotel] * 4 + [None]
        resolve.assert_called_once_with(35.0, 139.0)
        assert [day["day_total_cost"] for day in itinerary] == [
            600.0,
            100.0,
            100.0,
            100.0,
            350.0,
        ]

    @pytest.mark.parametrize("resolved_country_code", ["MY", None])
    def test_plan_flight_hotel_rejects_wrong_or_unverifiable_assessment_hotel(
        self, resolved_country_code
    ):
        cached_outbound = parsed_direct_flight("KUL", "HND", 999.0)
        cached_return = parsed_direct_flight("HND", "KUL", 888.0, "Stale Air")
        cached_hotel = parsed_hotel(777)
        fresh_outbound = parsed_direct_flight("KUL", "HND", 500.0)
        fresh_return = parsed_direct_flight("HND", "KUL", 350.0, "Fresh Air")
        fresh_hotel = parsed_hotel(100)
        state = AgentState(
            origin_country="Malaysia",
            country="Japan",
            start_date="2026-08-01",
            end_date="2026-08-02",
            num_people=2,
            budget_assessment={
                "assessment_id": "assessment-country-check",
                "calculation_version": "allocation-v1",
                "origin": "Malaysia",
                "destination": "Japan",
                "start_date": "2026-08-01",
                "end_date": "2026-08-02",
                "num_people": 2,
                "evidence": {
                    "outbound_flight": cached_outbound,
                    "return_flight": cached_return,
                    "hotel": cached_hotel,
                },
                "created_at": "2026-08-16T12:00:00+00:00",
                "expires_at": "2999-08-16T13:00:00+00:00",
            },
        )

        def flight_results(origin, *_args):
            return [fresh_outbound] if origin == "Malaysia" else [fresh_return]

        with (
            patch.object(
                flights_hotels,
                "resolve_country_code",
                return_value=resolved_country_code,
            ) as resolve,
            patch.object(
                flights_hotels, "fetch_flights_api", side_effect=flight_results
            ) as fetch_flights,
            patch.object(
                flights_hotels, "fetch_hotels_api", return_value=[fresh_hotel]
            ) as fetch_hotels,
        ):
            itinerary = flights_hotels.plan_flight_hotel(state)["draft_itinerary"]

        resolve.assert_called_once_with(35.0, 139.0)
        assert fetch_flights.call_count == 2
        fetch_hotels.assert_called_once()
        assert itinerary[0]["flight"] == [fresh_outbound]
        assert itinerary[-1]["flight"] == [fresh_return]
        assert all(day["hotel"] == fresh_hotel for day in itinerary[:-1])
        assert itinerary[-1]["hotel"] is None

    def test_assessment_evidence_rejects_malformed_hotel_location(self):
        outbound = parsed_direct_flight("KUL", "HND", 500.0)
        returning = parsed_direct_flight("HND", "KUL", 350.0)
        state = AgentState(
            origin_country="Malaysia",
            country="Japan",
            start_date="2026-08-01",
            end_date="2026-08-02",
            num_people=2,
            budget_assessment={
                "calculation_version": "allocation-v1",
                "origin": "Malaysia",
                "destination": "Japan",
                "start_date": "2026-08-01",
                "end_date": "2026-08-02",
                "num_people": 2,
                "evidence": {
                    "outbound_flight": outbound,
                    "return_flight": returning,
                    "hotel": {**parsed_hotel(100), "location": "not-an-object"},
                },
                "expires_at": "2999-08-16T13:00:00+00:00",
            },
        )
        with patch.object(flights_hotels, "resolve_country_code") as resolve:
            assert flights_hotels._assessment_evidence_for_state(state, "JP") is None
        resolve.assert_not_called()

    def test_plan_flight_hotel_rejects_expired_assessment_evidence(self):
        cached_outbound = flights_hotels._parse_flight(sample_flight(999))
        cached_return = {**cached_outbound, "airline": "Stale Air", "price": 888.0}
        cached_hotel = parsed_hotel(777)
        fresh_outbound = flights_hotels._parse_flight(sample_flight(500))
        fresh_return = {**fresh_outbound, "airline": "Fresh Air", "price": 350.0}
        fresh_hotel = parsed_hotel(100)
        state = AgentState(
            origin_country="Malaysia",
            country="Japan",
            start_date="2026-08-01",
            end_date="2026-08-02",
            num_people=2,
            budget_assessment={
                "assessment_id": "expired",
                "calculation_version": "allocation-v1",
                "origin": "Malaysia",
                "destination": "Japan",
                "start_date": "2026-08-01",
                "end_date": "2026-08-02",
                "num_people": 2,
                "base_currency": "MYR",
                "destination_currency": "JPY",
                "exchange_rate": 32.0,
                "minimum_destination_budget": 192000.0,
                "recommended_minimum_budget": 6000.0,
                "evidence": {
                    "outbound_flight": cached_outbound,
                    "return_flight": cached_return,
                    "hotel": cached_hotel,
                    "outbound_flight_price": 999.0,
                    "return_flight_price": 888.0,
                    "hotel_price_per_night": 777.0,
                    "hotel_nights": 1,
                },
                "created_at": "2000-01-01T00:00:00+00:00",
                "expires_at": "2000-01-01T01:00:00+00:00",
            },
        )

        def flight_results(origin, *_args):
            return [fresh_outbound] if origin == "Malaysia" else [fresh_return]

        with (
            patch.object(
                flights_hotels,
                "fetch_flights_api",
                side_effect=flight_results,
            ),
            patch.object(
                flights_hotels,
                "fetch_hotels_api",
                return_value=[fresh_hotel],
            ),
        ):
            itinerary = flights_hotels.plan_flight_hotel(state)["draft_itinerary"]

        assert itinerary[0]["flight"] == [fresh_outbound]
        assert itinerary[-1]["flight"] == [fresh_return]
        assert all(day["hotel"] == fresh_hotel for day in itinerary[:-1])
        assert itinerary[-1]["hotel"] is None

    def test_plan_flight_hotel_starts_all_provider_searches_before_releasing_any(
        self,
    ):
        outbound = flights_hotels._parse_flight(sample_flight(500))
        return_flight = {**outbound, "airline": "Home Air", "price": 350.0}
        hotel = parsed_hotel(100)
        state = AgentState(
            origin_country="Malaysia",
            country="Japan",
            start_date="2026-08-01",
            end_date="2026-08-02",
        )
        submissions = []
        result_submission_counts = []
        configured_workers = []

        class RecordingFuture:
            def __init__(self, fn, args):
                self.fn = fn
                self.args = args

            def result(self):
                result_submission_counts.append(len(submissions))
                return self.fn(*self.args)

        class RecordingExecutor:
            def __init__(self, max_workers):
                configured_workers.append(max_workers)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def submit(self, fn, *args):
                submissions.append((fn, args))
                return RecordingFuture(fn, args)

        def flight_results(origin, destination, *_args):
            if (origin, destination) == ("Malaysia", "Japan"):
                return [outbound]
            if (origin, destination) == ("Japan", "Malaysia"):
                return [return_flight]
            raise AssertionError(f"Unexpected flight search: {origin}, {destination}")

        def hotel_results(*_args):
            return [hotel]

        with (
            patch.object(flights_hotels, "ThreadPoolExecutor", RecordingExecutor),
            patch.object(flights_hotels, "fetch_flights_api", side_effect=flight_results),
            patch.object(flights_hotels, "fetch_hotels_api", side_effect=hotel_results),
        ):
            itinerary = flights_hotels.plan_flight_hotel(state)["draft_itinerary"]

        assert configured_workers == [3]
        assert len(submissions) == 3
        assert result_submission_counts == [3, 3, 3]
        assert itinerary[0]["flight"] == [outbound]
        assert itinerary[-1]["flight"] == [return_flight]
        assert all(day["hotel"] == hotel for day in itinerary[:-1])
        assert itinerary[-1]["hotel"] is None

    @pytest.mark.parametrize(
        "failed_route,expected_outbound,expected_return",
        [
            (("Malaysia", "Japan"), None, "Home Air"),
            (("Japan", "Malaysia"), "Test Air", None),
        ],
    )
    def test_plan_flight_hotel_preserves_the_other_leg_when_one_search_raises(
        self, failed_route, expected_outbound, expected_return
    ):
        outbound = flights_hotels._parse_flight(sample_flight(500))
        return_flight = {**outbound, "airline": "Home Air", "price": 350.0}
        state = AgentState(
            origin_country="Malaysia",
            country="Japan",
            start_date="2026-08-01",
            end_date="2026-08-02",
            budget_allocation={"transportation": 1000},
        )

        def flight_results(origin, destination, *_args):
            if (origin, destination) == failed_route:
                raise RuntimeError("provider unavailable")
            return [outbound] if origin == "Malaysia" else [return_flight]

        with (
            patch.object(
                flights_hotels, "fetch_flights_api", side_effect=flight_results
            ) as fetch_flights,
            patch.object(flights_hotels, "fetch_hotels_api", return_value=[]),
        ):
            itinerary = flights_hotels.plan_flight_hotel(state)["draft_itinerary"]

        assert fetch_flights.call_count == 2
        first_airline = (
            itinerary[0]["flight"][0]["airline"] if itinerary[0]["flight"] else None
        )
        last_airline = (
            itinerary[-1]["flight"][0]["airline"] if itinerary[-1]["flight"] else None
        )
        assert first_airline == expected_outbound
        assert last_airline == expected_return

    def test_plan_flight_hotel_preserves_flights_when_hotel_search_raises(self):
        outbound = flights_hotels._parse_flight(sample_flight(500))
        return_flight = {**outbound, "airline": "Home Air", "price": 350.0}
        state = AgentState(
            origin_country="Malaysia",
            country="Japan",
            start_date="2026-08-01",
            end_date="2026-08-02",
        )

        def flight_results(origin, *_args):
            return [outbound] if origin == "Malaysia" else [return_flight]

        with (
            patch.object(
                flights_hotels, "fetch_flights_api", side_effect=flight_results
            ),
            patch.object(
                flights_hotels,
                "fetch_hotels_api",
                side_effect=RuntimeError("hotel provider unavailable"),
            ),
        ):
            itinerary = flights_hotels.plan_flight_hotel(state)["draft_itinerary"]

        assert itinerary[0]["flight"] == [outbound]
        assert itinerary[-1]["flight"] == [return_flight]
        assert all(day["hotel"] is None for day in itinerary)

    def test_plan_flight_hotel_degrades_to_day_skeleton_when_providers_are_empty(self):
        state = AgentState(start_date="2026-08-01", end_date="2026-08-02")
        with (
            patch.object(flights_hotels, "fetch_flights_api", return_value=[]),
            patch.object(flights_hotels, "fetch_hotels_api", return_value=[]),
        ):
            itinerary = flights_hotels.plan_flight_hotel(state)["draft_itinerary"]
        assert len(itinerary) == 2
        assert all(d["flight"] is None and d["hotel"] is None for d in itinerary)

    def test_alternative_search_rejects_unknown_category(self):
        result = flights_hotels.search_alternative_opt.func(
            category="train", day_num=1, state=AgentState(num_people=1)
        )
        assert "Unknown category" in result["error"]

    def test_alternative_flight_second_pass_flags_over_budget_options(self):
        state = AgentState(
            origin_country="Malaysia",
            country="Japan",
            start_date="2026-01-01",
            end_date="2026-01-02",
            dest_currency_code="JPY",
            num_people=2,
            budget_allocation={"transportation": 100},
        )
        options = [[], [{"price": 200}, {"price": 250}, {"price": 300}, {"price": 400}]]
        with patch.object(flights_hotels, "fetch_flights_api", side_effect=options):
            result = flights_hotels.search_alternative_opt.func(
                category="flight", day_num=1, state=state
            )
        assert result["count"] == 3
        assert all(o["over_budget"] for o in result["options"])
        assert "warning" in result

    def test_alternative_search_uses_trusted_singapore_state_not_model_destination(self):
        state = AgentState(
            origin_country="Malaysia",
            country="Singapore",
            city=["Singapore"],
            start_date="2026-08-01",
            end_date="2026-08-05",
            dest_currency_code="SGD",
            num_people=2,
            budget_allocation={"accommodation": 900.0},
        )
        with patch.object(
            flights_hotels,
            "fetch_hotels_api",
            return_value=[{"hotel_name": "Trusted Hotel"}],
        ) as fetch:
            result = flights_hotels.search_alternative_opt.func(
                category="hotel", day_num=2, state=state
            )
        assert result["options"] == [{"hotel_name": "Trusted Hotel"}]
        fetch.assert_called_once_with(
            "Singapore, Singapore",
            "2026-08-01",
            "2026-08-05",
            "SGD",
            225.0,
            2,
            "SG",
        )

    @pytest.mark.parametrize(
        "day_num,expected_args",
        [
            (1, ("Malaysia", "Singapore", "2026-08-01", "SGD", 800.0, 3)),
            (5, ("Singapore", "Malaysia", "2026-08-05", "SGD", 800.0, 3)),
        ],
    )
    def test_alternative_flight_uses_trusted_boundary_direction_and_date(
        self, day_num, expected_args
    ):
        state = AgentState(
            origin_country="Malaysia",
            country="Singapore",
            start_date="2026-08-01",
            end_date="2026-08-05",
            dest_currency_code="SGD",
            num_people=3,
            budget_allocation={"transportation": 800.0},
        )
        with patch.object(
            flights_hotels, "fetch_flights_api", return_value=[{"price": 500.0}]
        ) as fetch:
            result = flights_hotels.search_alternative_opt.func(
                category="flight", day_num=day_num, state=state
            )
        assert result["options"] == [{"price": 500.0}]
        fetch.assert_called_once_with(*expected_args)

    def test_alternative_flight_rejects_non_boundary_day(self):
        state = AgentState(
            origin_country="Malaysia",
            country="Singapore",
            start_date="2026-08-01",
            end_date="2026-08-05",
        )
        with patch.object(flights_hotels, "fetch_flights_api") as fetch:
            result = flights_hotels.search_alternative_opt.func(
                category="flight", day_num=2, state=state
            )
        assert "boundary" in result["error"].lower()
        fetch.assert_not_called()

    def test_model_visible_tool_schemas_hide_state_and_trip_defining_arguments(self):
        alternative_properties = set(
            flights_hotels.search_alternative_opt.tool_call_schema.model_json_schema()[
                "properties"
            ]
        )
        place_properties = set(
            attractions.search_places.tool_call_schema.model_json_schema()["properties"]
        )
        assert alternative_properties == {"category", "day_num"}
        assert place_properties == {"category", "query", "city"}
        assert alternative_properties.isdisjoint(
            {
                "state",
                "origin",
                "destination",
                "start_date",
                "end_date",
                "currency",
                "max_budget",
                "num_people",
            }
        )

    def test_edit_itinerary_serialises_multiple_edits(self):
        edits = [
            flights_hotels.ItineraryEdit(
                day=1,
                action="replace",
                category="hotel",
                new_details={"hotel_name": "B"},
            ),
            flights_hotels.ItineraryEdit(
                day=2, action="remove", category="activity", index=1
            ),
        ]
        result = flights_hotels.edit_itinerary.func(edits)
        assert result["action"] == "edit_itinerary"
        assert len(result["edits"]) == 2

    def test_edit_itinerary_activity_index_is_one_based(self):
        with pytest.raises(ValueError):
            flights_hotels.ItineraryEdit(
                day=1, action="remove", category="activity", index=0
            )

    def test_edit_itinerary_rejects_unverified_partial_place(self):
        with pytest.raises(ValueError, match="copied from search_places"):
            flights_hotels.ItineraryEdit(
                day=1,
                action="add",
                category="activity",
                new_details={
                    "name": "Nearby Attraction to Umeda Sky Building",
                    "rating": 4.2,
                    "address": "Ikeda, Osaka",
                },
            )


@pytest.mark.unit
class TestAttractionPlanning:
    def idea(self, **overrides):
        base = {
            "name": "Tokyo Tower",
            "place_type": "attraction",
            "city": "Tokyo",
            "day": 1,
            "order": 1,
            "suggested_time": "Morning",
            "estimated_cost": 50,
            "reason": "Great skyline view",
        }
        return PlaceIdea(**{**base, **overrides})

    def test_haversine_zero_and_known_distance(self):
        assert attractions._haversine_km(0, 0, 0, 0) == 0
        assert attractions._haversine_km(
            3.139, 101.687, 3.157, 101.712
        ) == pytest.approx(3.4, abs=0.5)

    @pytest.mark.parametrize(
        "payload,expected",
        [
            (
                {"thumbnail": "google.jpg", "serpapi_thumbnail": "serp.jpg"},
                "google.jpg",
            ),
            ({"serpapi_thumbnail": "serp.jpg"}, "serp.jpg"),
            ({"images": [{"thumbnail": "nested.jpg"}]}, "nested.jpg"),
            ({"images": [{"image": "original.jpg"}]}, "original.jpg"),
            ({"thumbnail": None, "images": []}, ""),
        ],
    )
    def test_place_thumbnail_supports_all_serpapi_shapes(self, payload, expected):
        assert attractions._place_thumbnail(payload) == expected

    @pytest.mark.parametrize(
        "lat,lng,anchor,expected",
        [
            (1, 2, None, True),
            (None, 2, {"lat": 1, "lng": 2, "max_km": 5}, False),
            ("bad", 2, {"lat": 1, "lng": 2, "max_km": 5}, False),
            (1, 2, {"lat": 1, "lng": 2, "max_km": 5}, True),
        ],
    )
    def test_within_anchor(self, lat, lng, anchor, expected):
        assert attractions._within_anchor(lat, lng, anchor) is expected

    def test_place_inside_old_radius_is_rejected_when_country_code_differs(self):
        anchor = {"lat": 1.3521, "lng": 103.8198, "max_km": 300.0}
        assert attractions._within_anchor(1.4927, 103.7414, anchor)
        with patch.object(
            attractions, "resolve_country_code", return_value="MY"
        ) as resolve:
            assert attractions._verified_country_code(1.4927, 103.7414, "SG") is None
        resolve.assert_called_once_with(1.4927, 103.7414)

    def test_city_anchor_radius_rejects_penang_for_kuala_lumpur(self):
        """Lock the city geofence below the reviewed 300 km failure distance."""
        anchor = {
            "lat": 3.139,
            "lng": 101.6869,
            "max_km": attractions._MAX_KM_FROM_CITY,
        }

        assert not attractions._within_anchor(5.4141, 100.3288, anchor)

    def test_build_activity_requires_verified_real_place(self):
        assert attractions._build_activity(self.idea(), None) is None

    def test_build_activity_merges_verified_fields(self):
        real = {
            "real_name": "Tokyo Tower",
            "category": "Landmark",
            "rating": 4.5,
            "address": "Minato City",
            "thumbnail": "tower.jpg",
            "lat": "35.6586",
            "lng": "139.7454",
            "country_code": "JP",
            "verified_locality": "Tokyo",
        }
        result = attractions._build_activity(self.idea(), real)
        assert result["type"] == "attraction"
        assert result["is_estimated"] is True
        assert result["location"]["latitude"] == 35.6586
        assert result["location"]["longitude"] == 139.7454
        assert result["location"]["country_code"] == "JP"
        assert result["location"]["requested_city"] == "Tokyo"
        assert result["location"]["verified_locality"] == "Tokyo"

    @pytest.mark.parametrize(
        "verified_locality",
        [None, "George Town", "Penang"],
    )
    def test_build_activity_rejects_missing_or_wrong_provider_locality(
        self,
        verified_locality,
    ):
        """Catch a requested-city query label being treated as provider proof."""
        real = {
            "real_name": "Penang State Museum",
            "address": "57 Jalan Macalister, George Town, Penang",
            "lat": 5.4141,
            "lng": 100.3288,
            "country_code": "MY",
            "verified_locality": verified_locality,
        }
        idea = self.idea(
            name="Penang State Museum",
            city="Kuala Lumpur",
        )

        assert attractions._build_activity(idea, real) is None

    def test_serpapi_place_search_normalizes_verified_numeric_strings(self):
        response = Mock(status_code=200)
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "local_results": [
                {
                    "title": "National Gallery Singapore",
                    "address": "1 St Andrew's Road",
                    "gps_coordinates": {
                        "latitude": "1.2903",
                        "longitude": "103.8519",
                    },
                }
            ]
        }
        with (
            patch.object(attractions._session, "get", return_value=response),
            patch.object(
                attractions, "resolve_country_code", return_value="SG"
            ) as resolve,
            patch.object(
                attractions,
                "resolve_locality_names",
                return_value=("Singapore",),
            ),
        ):
            result = attractions._serpapi_place_search(
                "gallery",
                destination_country_code="SG",
                requested_city="Singapore",
            )
        assert result["lat"] == 1.2903
        assert result["lng"] == 103.8519
        assert isinstance(result["lat"], float)
        assert isinstance(result["lng"], float)
        assert result["verified_locality"] == "Singapore"
        resolve.assert_called_once_with(1.2903, 103.8519)

    def test_lookup_place_reverifies_and_normalizes_cached_coordinates(self):
        cached = {
            "real_name": "National Gallery Singapore",
            "lat": "1.2903",
            "lng": "103.8519",
            "country_code": "MY",
        }
        with (
            patch.object(attractions, "get_cached_data", return_value=cached),
            patch.object(
                attractions, "resolve_country_code", return_value="SG"
            ) as resolve,
            patch.object(
                attractions,
                "resolve_locality_names",
                return_value=("Singapore",),
            ),
            patch.object(attractions, "_serpapi_place_search") as provider,
        ):
            result = attractions._lookup_place_cached(
                "gallery",
                destination_country_code="SG",
                requested_city="Singapore",
            )
        assert result["lat"] == 1.2903
        assert result["lng"] == 103.8519
        assert result["country_code"] == "SG"
        assert result["verified_locality"] == "Singapore"
        resolve.assert_called_once_with(1.2903, 103.8519)
        provider.assert_not_called()

    def test_lookup_place_rejects_cached_coordinate_in_another_country(self):
        cached = {
            "real_name": "Cross-border Gallery",
            "lat": 1.4927,
            "lng": 103.7414,
            "country_code": "SG",
        }
        with (
            patch.object(attractions, "get_cached_data", return_value=cached),
            patch.object(
                attractions, "resolve_country_code", return_value="MY"
            ) as resolve,
            patch.object(
                attractions,
                "resolve_locality_names",
                return_value=("Johor Bahru",),
            ),
            patch.object(
                attractions, "_serpapi_place_search", return_value=None
            ) as provider,
        ):
            result = attractions._lookup_place_cached(
                "gallery",
                destination_country_code="SG",
                requested_city="Singapore",
            )
        assert result is None
        resolve.assert_called_once_with(1.4927, 103.7414)
        provider.assert_called_once_with(
            "gallery",
            anchor=None,
            destination_country_code="SG",
            requested_city="Singapore",
        )

    def test_search_places_rejects_city_outside_configured_destination_cities(self):
        state = AgentState(country="Singapore", city=["Singapore"])
        with patch.object(attractions._session, "get") as request:
            result = attractions.search_places.func(
                category="attraction",
                query="museum",
                state=state,
                city="Johor Bahru",
            )
        assert "configured" in result["error"].lower()
        request.assert_not_called()

    def test_search_places_accepts_casefolded_configured_city_and_verifies_country(self):
        response = Mock(status_code=200)
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "local_results": [
                {
                    "title": "National Gallery Singapore",
                    "address": "1 St Andrew's Road",
                    "gps_coordinates": {
                        "latitude": "1.2903",
                        "longitude": "103.8519",
                    },
                }
            ]
        }
        state = AgentState(country="Singapore", city=["Singapore"])
        with (
            patch.object(attractions, "_geo_anchor", return_value=None),
            patch.object(
                attractions, "resolve_country_code", return_value="SG"
            ) as resolve,
            patch.object(
                attractions,
                "resolve_locality_names",
                return_value=("Singapore",),
            ),
            patch.object(attractions._session, "get", return_value=response),
        ):
            result = attractions.search_places.func(
                category="attraction",
                query="gallery",
                state=state,
                city="  sInGaPoRe  ",
            )
        assert result["results"][0]["location"]["country_code"] == "SG"
        assert result["results"][0]["location"]["requested_city"] == "Singapore"
        assert result["results"][0]["location"]["verified_locality"] == "Singapore"
        assert result["results"][0]["location"]["latitude"] == 1.2903
        assert result["results"][0]["location"]["longitude"] == 103.8519
        resolve.assert_called_once_with(1.2903, 103.8519)

    def test_search_places_rejects_penang_result_for_kuala_lumpur_request(self):
        """Exercise the real provider-normalization loop for the reviewed 300 km bug."""
        response = Mock(status_code=200)
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "local_results": [
                {
                    "title": "Penang State Museum",
                    "address": "57 Jalan Macalister, George Town, Penang",
                    "gps_coordinates": {
                        "latitude": 5.4141,
                        "longitude": 100.3288,
                    },
                }
            ]
        }
        state = AgentState(country="Malaysia", city=["Kuala Lumpur"])
        broad_anchor = {"lat": 3.139, "lng": 101.6869, "max_km": 300.0}
        assert attractions._within_anchor(5.4141, 100.3288, broad_anchor)
        with (
            patch.object(attractions, "_geo_anchor", return_value=broad_anchor),
            patch.object(attractions, "resolve_country_code", return_value="MY"),
            patch.object(
                attractions,
                "resolve_locality_names",
                return_value=("George Town", "Penang"),
            ),
            patch.object(attractions._session, "get", return_value=response),
        ):
            result = attractions.search_places.func(
                category="attraction",
                query="museum",
                state=state,
                city="Kuala Lumpur",
            )

        assert result["count"] == 0
        assert result["error"]

    def test_clamp_costs_changes_only_requested_type(self):
        acts = [
            {"type": "attraction", "estimated_cost": 80},
            {"type": "attraction", "estimated_cost": 120},
            {"type": "restaurant", "estimated_cost": 100},
        ]
        attractions._clamp_costs(acts, "attraction", 100)
        assert [a["estimated_cost"] for a in acts] == [40, 60, 100]

    @pytest.mark.parametrize("budget", [0, -1, 1000])
    def test_clamp_costs_noop_when_not_required(self, budget):
        acts = [{"type": "attraction", "estimated_cost": 100}]
        attractions._clamp_costs(acts, "attraction", budget)
        assert acts[0]["estimated_cost"] == 100

    def test_plan_activities_populates_verified_places_and_preserves_input(self):
        state = AgentState(
            country="Japan",
            city=["Tokyo"],
            num_people=2,
            budget_allocation={"activity": 50, "food": 100},
            draft_itinerary=[
                {
                    "day": 1,
                    "date": "2026-08-01",
                    "hotel": parsed_hotel(),
                    "activities": [],
                    "day_total_cost": 100,
                }
            ],
        )
        original = copy.deepcopy(state.draft_itinerary)
        ideas = [self.idea(estimated_cost=100)]
        real = {
            "real_name": "Tokyo Tower",
            "category": "Landmark",
            "rating": 4.5,
            "address": "Minato",
            "thumbnail": "",
            "lat": 35.6,
            "lng": 139.7,
            "country_code": "JP",
            "verified_locality": "Tokyo",
        }
        with (
            patch.object(attractions, "brainstorm_places", return_value=ideas),
            patch.object(
                attractions,
                "_geo_anchor",
                return_value={"lat": 35.6, "lng": 139.7, "max_km": 80},
            ),
            patch.object(attractions, "_lookup_place_cached", return_value=real),
        ):
            updated = attractions.plan_activities(state)["draft_itinerary"]
        assert state.draft_itinerary == original
        assert updated[0]["activities"][0]["estimated_cost"] == 50
        assert updated[0]["day_total_cost"] == 150

    def test_plan_activities_never_queries_model_supplied_unconfigured_city(self):
        """Catch a same-country model city overriding the trusted trip city list."""
        state = AgentState(
            country="Malaysia",
            city=["Kuala Lumpur"],
            num_people=1,
            budget_allocation={"activity": 100, "food": 100},
            draft_itinerary=[
                {
                    "day": 1,
                    "date": "2026-08-01",
                    "activities": [],
                    "day_total_cost": 0,
                }
            ],
        )
        penang_idea = self.idea(
            name="Penang Hill",
            city="Penang",
            estimated_cost=30,
        )

        with (
            patch.object(
                attractions,
                "brainstorm_places",
                return_value=[penang_idea],
            ),
            patch.object(attractions, "_geo_anchor") as anchor,
            patch.object(attractions, "_lookup_place_cached") as lookup,
        ):
            updated = attractions.plan_activities(state)["draft_itinerary"]

        assert updated[0]["activities"] == []
        anchor.assert_not_called()
        lookup.assert_not_called()

    def test_plan_activities_canonicalizes_casefolded_trusted_city_evidence(self):
        """Catch provider-grounded city evidence preserving model casing instead of server casing."""
        state = AgentState(
            country="Malaysia",
            city=["Kuala Lumpur"],
            num_people=1,
            budget_allocation={"activity": 100, "food": 100},
            draft_itinerary=[
                {
                    "day": 1,
                    "date": "2026-08-01",
                    "activities": [],
                    "day_total_cost": 0,
                }
            ],
        )
        idea = self.idea(
            name="Petronas Twin Towers",
            city="kUaLa LuMpUr",
            estimated_cost=30,
        )
        real = {
            "real_name": "Petronas Twin Towers",
            "category": "Landmark",
            "rating": 4.7,
            "address": "Kuala Lumpur City Centre",
            "thumbnail": "",
            "lat": 3.1579,
            "lng": 101.7116,
            "country_code": "MY",
            "verified_locality": "Kuala Lumpur",
        }

        with (
            patch.object(attractions, "brainstorm_places", return_value=[idea]),
            patch.object(
                attractions,
                "_geo_anchor",
                return_value={"lat": 3.1579, "lng": 101.7116, "max_km": 80},
            ),
            patch.object(attractions, "_lookup_place_cached", return_value=real),
        ):
            updated = attractions.plan_activities(state)["draft_itinerary"]

        assert updated[0]["activities"][0]["location"]["requested_city"] == (
            "Kuala Lumpur"
        )

    def test_plan_activities_drops_ambiguous_omitted_city_for_multi_city_trip(self):
        """Catch an omitted model city silently defaulting away required coverage."""
        state = AgentState(
            country="Malaysia",
            city=["Kuala Lumpur", "Malacca"],
            num_people=1,
            budget_allocation={"activity": 100, "food": 100},
            draft_itinerary=[
                {
                    "day": 1,
                    "date": "2026-08-01",
                    "activities": [],
                    "day_total_cost": 0,
                }
            ],
        )
        idea = self.idea(name="Heritage Walk", city="", estimated_cost=30)

        with (
            patch.object(attractions, "brainstorm_places", return_value=[idea]),
            patch.object(attractions, "_geo_anchor") as anchor,
            patch.object(attractions, "_lookup_place_cached") as lookup,
        ):
            updated = attractions.plan_activities(state)["draft_itinerary"]

        assert updated[0]["activities"] == []
        anchor.assert_not_called()
        lookup.assert_not_called()

    def test_plan_activities_empty_inputs_degrade_without_mutation(self):
        assert attractions.plan_activities(AgentState()) == {}
        state = AgentState(draft_itinerary=[{"day": 1, "activities": []}])
        with patch.object(attractions, "brainstorm_places", return_value=[]):
            assert attractions.plan_activities(state) == {}


@pytest.mark.unit
class TestMapboxUtilities:
    @pytest.fixture(autouse=True)
    def clear_cache(self):
        mapbox._geocode_cache.clear()
        mapbox._country_cache.clear()
        mapbox._locality_cache.clear()
        yield
        mapbox._geocode_cache.clear()
        mapbox._country_cache.clear()
        mapbox._locality_cache.clear()

    @pytest.mark.parametrize(
        "location,expected",
        [
            ({"lat": 3.1, "lng": 101.7}, [101.7, 3.1]),
            ({"latitude": "35", "longitude": "139"}, [139.0, 35.0]),
            ({"lat": None, "lng": 1}, None),
            ({"lat": "bad", "lng": 1}, None),
            (None, None),
        ],
    )
    def test_extract_coord_supports_both_key_styles(self, location, expected):
        assert mapbox._extract_coord(location) == expected

    def test_haversine_and_path_distance(self):
        assert mapbox._haversine_km([0, 0], [0, 0]) == 0
        assert mapbox._path_straight_km([[0, 0], [0, 1], [0, 2]]) == pytest.approx(
            222.4, abs=1
        )

    def test_build_geojson_preserves_point_order_and_profile_metadata(self):
        points = [
            {"name": "Hotel", "coordinates": [101.7, 3.1], "type": "hotel", "order": 0},
            {
                "name": "Museum",
                "coordinates": [101.71, 3.11],
                "type": "attraction",
                "order": 1,
            },
        ]
        routes = {
            "walking": {
                "distance_km": 2,
                "duration_mins": 20,
                "geometry_for_map": {
                    "type": "LineString",
                    "coordinates": [[101.7, 3.1], [101.71, 3.11]],
                },
            }
        }
        fc = mapbox.build_geojson(points, routes)
        assert fc["type"] == "FeatureCollection"
        assert [f["geometry"]["type"] for f in fc["features"]] == [
            "Point",
            "Point",
            "LineString",
        ]
        assert fc["features"][-1]["properties"]["profile"] == "walking"

    @pytest.mark.regression
    def test_negative_geocode_result_is_actually_cached(self):
        response = Mock(status_code=200)
        response.json.return_value = {"features": []}
        with patch.object(
            mapbox, "_request_with_retries", return_value=response
        ) as request:
            assert mapbox.geocode_location("Missing Place") is None
            assert mapbox.geocode_location("Missing Place") is None
        assert request.call_count == 1

    def test_positive_geocode_result_and_proximity(self):
        response = Mock(status_code=200)
        response.json.return_value = {
            "features": [
                {
                    "properties": {"name": "Museum", "place_formatted": "Kuala Lumpur"},
                    "geometry": {"coordinates": [101.7, 3.1]},
                }
            ]
        }
        with patch.object(
            mapbox, "_request_with_retries", return_value=response
        ) as request:
            result = mapbox.geocode_location("Museum", [101.6, 3.0])
        assert result == {
            "lng": 101.7,
            "lat": 3.1,
            "place_name": "Museum, Kuala Lumpur",
        }
        assert request.call_args.args[1]["proximity"] == "101.6,3.0"

    def test_resolve_country_code_parses_geocoding_v6_context(self):
        response = Mock(status_code=200)
        response.json.return_value = {
            "features": [
                {
                    "properties": {
                        "feature_type": "country",
                        "context": {"country": {"country_code": "sg"}},
                    }
                }
            ]
        }
        with patch.object(mapbox, "_request_with_retries", return_value=response) as request:
            assert mapbox.resolve_country_code(1.3521, 103.8198) == "SG"
        assert request.call_args.args == (
            "https://api.mapbox.com/search/geocode/v6/reverse",
            {
                "longitude": 103.8198,
                "latitude": 1.3521,
                "types": "country",
                "limit": 1,
                "language": "en",
                "access_token": mapbox.MAPBOX_TOKEN,
            },
        )

    def test_resolve_country_code_reuses_cached_value(self):
        response = Mock(status_code=200)
        response.json.return_value = {
            "features": [
                {"properties": {"context": {"country": {"country_code": "JP"}}}}
            ]
        }
        with patch.object(mapbox, "_request_with_retries", return_value=response) as request:
            assert mapbox.resolve_country_code(35.6762, 139.6503) == "JP"
            assert mapbox.resolve_country_code(35.6762, 139.6503) == "JP"
        assert request.call_count == 1

    def test_resolve_country_code_caches_permanent_empty_response(self):
        response = Mock(status_code=200)
        response.json.return_value = {"features": []}
        with patch.object(mapbox, "_request_with_retries", return_value=response) as request:
            assert mapbox.resolve_country_code(48.8566, 2.3522) is None
            assert mapbox.resolve_country_code(48.8566, 2.3522) is None
        assert request.call_count == 1

    def test_resolve_country_code_retries_after_transport_failure(self):
        response = Mock(status_code=200)
        response.json.return_value = {
            "features": [
                {"properties": {"context": {"country": {"country_code": "FR"}}}}
            ]
        }
        with patch.object(
            mapbox, "_request_with_retries", side_effect=[None, response]
        ) as request:
            assert mapbox.resolve_country_code(43.2965, 5.3698) is None
            assert mapbox.resolve_country_code(43.2965, 5.3698) == "FR"
        assert request.call_count == 2

    def test_resolve_country_code_retries_after_non_200_response(self):
        failed_response = Mock(status_code=503)
        response = Mock(status_code=200)
        response.json.return_value = {
            "features": [
                {"properties": {"context": {"country": {"country_code": "DE"}}}}
            ]
        }
        with patch.object(
            mapbox,
            "_request_with_retries",
            side_effect=[failed_response, response],
        ) as request:
            assert mapbox.resolve_country_code(52.52, 13.405) is None
            assert mapbox.resolve_country_code(52.52, 13.405) == "DE"
        assert request.call_count == 2

    def test_resolve_country_code_uses_top_level_code_for_country_feature(self):
        response = Mock(status_code=200)
        response.json.return_value = {
            "features": [
                {
                    "properties": {
                        "feature_type": "country",
                        "country_code": "my",
                    }
                }
            ]
        }
        with patch.object(mapbox, "_request_with_retries", return_value=response):
            assert mapbox.resolve_country_code(3.139, 101.687) == "MY"

    def test_resolve_country_code_ignores_top_level_code_for_non_country_feature(self):
        response = Mock(status_code=200)
        response.json.return_value = {
            "features": [
                {
                    "properties": {
                        "feature_type": "place",
                        "country_code": "SG",
                    }
                }
            ]
        }
        with patch.object(mapbox, "_request_with_retries", return_value=response):
            assert mapbox.resolve_country_code(1.29, 103.85) is None

    def test_resolve_locality_names_uses_only_structured_admin_context(self):
        response = Mock(status_code=200)
        response.json.return_value = {
            "features": [
                {
                    "properties": {
                        "feature_type": "address",
                        "name": "57 Jalan Macalister",
                        "place_formatted": "Kuala Lumpur-looking free text",
                        "context": {
                            "place": {"name": "George Town"},
                            "region": {"name": "Penang"},
                            "country": {"country_code": "MY", "name": "Malaysia"},
                        },
                    }
                }
            ]
        }
        with patch.object(mapbox, "_request_with_retries", return_value=response) as request:
            result = mapbox.resolve_locality_names(5.4141, 100.3288, "MY")

        assert result == ("George Town", "Penang")
        assert "Kuala Lumpur-looking free text" not in result
        assert request.call_args.args == (
            "https://api.mapbox.com/search/geocode/v6/reverse",
            {
                "longitude": 100.3288,
                "latitude": 5.4141,
                "limit": 1,
                "language": "en",
                "country": "my",
                "access_token": mapbox.MAPBOX_TOKEN,
            },
        )

    def test_resolve_locality_names_caches_by_coordinate_and_country_identity(self):
        response = Mock(status_code=200)
        response.json.return_value = {
            "features": [
                {
                    "properties": {
                        "feature_type": "place",
                        "name": "Singapore",
                        "context": {
                            "place": {"name": "Singapore"},
                            "country": {"country_code": "SG"},
                        },
                    }
                }
            ]
        }
        with patch.object(mapbox, "_request_with_retries", return_value=response) as request:
            assert mapbox.resolve_locality_names(1.2903, 103.8519, "SG") == (
                "Singapore",
            )
            assert mapbox.resolve_locality_names(1.2903, 103.8519, "SG") == (
                "Singapore",
            )
            assert mapbox.resolve_locality_names(1.2903, 103.8519, "MY") == ()

        assert request.call_count == 2

    @pytest.mark.parametrize(
        "latitude,longitude",
        [
            (None, 1),
            (True, 1),
            ("bad", 1),
            (float("nan"), 1),
            (1, float("inf")),
        ],
    )
    def test_resolve_country_code_rejects_invalid_or_non_finite_coordinates(
        self, latitude, longitude
    ):
        with patch.object(mapbox, "_request_with_retries") as request:
            assert mapbox.resolve_country_code(latitude, longitude) is None
        request.assert_not_called()

    def test_fetch_routes_skips_long_walking_and_cycling_profiles(self):
        coords = [[0, 0], [0, 2]]
        with patch.object(
            mapbox,
            "_fetch_route_single",
            return_value={
                "distance_km": 200,
                "duration_mins": 120,
                "geometry_for_map": {"type": "LineString", "coordinates": coords},
            },
        ) as fetch:
            result = mapbox.fetch_routes_multi(coords)
        assert list(result) == ["driving"]
        fetch.assert_called_once_with(coords, "driving")

    def test_fetch_route_single_parses_metric_units(self):
        response = Mock(status_code=200)
        response.__bool__ = Mock(return_value=True)
        response.json.return_value = {
            "routes": [
                {
                    "distance": 12345,
                    "duration": 900,
                    "geometry": {"type": "LineString", "coordinates": []},
                }
            ]
        }
        with patch.object(mapbox, "_request_with_retries", return_value=response):
            result = mapbox._fetch_route_single([[1, 2], [3, 4]], "driving")
        assert result["distance_km"] == 12.35
        assert result["duration_mins"] == 15

    def test_process_day_does_not_mutate_input_and_orders_hotel_then_activity(self):
        day = {
            "day": 2,
            "hotel": parsed_hotel(),
            "activities": [
                {
                    "name": "Museum",
                    "type": "attraction",
                    "order": 1,
                    "location": {"latitude": 35.1, "longitude": 139.1},
                }
            ],
        }
        original = copy.deepcopy(day)
        with patch.object(mapbox, "fetch_routes_multi", return_value={}):
            day_num, updated, geojson = mapbox._process_day(1, day)
        assert day == original
        assert day_num == 2
        assert updated["route"] is None
        assert [f["properties"]["name"] for f in geojson["features"]] == [
            "Test Hotel",
            "Museum",
        ]

    def test_generate_daily_map_empty_itinerary(self):
        assert mapbox.generate_daily_map(AgentState()) == {
            "draft_itinerary": [],
            "daily_map_info": {},
        }

    def test_nearby_search_requires_location(self):
        result = mapbox.search_nearby_amenities.func("hospital", AgentState(), None)
        assert "specific_location" in result["error"]

    def test_nearby_search_sorts_by_distance(self):
        response = Mock(status_code=200)
        response.__bool__ = Mock(return_value=True)
        response.json.return_value = {
            "features": [
                {
                    "properties": {"name": "Far", "distance": 500},
                    "geometry": {"coordinates": [2, 1]},
                },
                {
                    "properties": {"name": "Near", "distance": 50},
                    "geometry": {"coordinates": [2.1, 1.1]},
                },
            ]
        }
        with (
            patch.object(mapbox, "geocode_location", return_value={"lng": 2, "lat": 1}),
            patch.object(mapbox, "_request_with_retries", return_value=response),
        ):
            result = mapbox.search_nearby_amenities.func(
                "hospital", AgentState(), "Hotel"
            )
        assert [x["name"] for x in result["results"]] == ["Near", "Far"]
