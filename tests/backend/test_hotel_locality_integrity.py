from __future__ import annotations

from unittest.mock import call, patch

import pytest

from app.agents.state import AgentState
from app.tools import flights_hotels


def _provider_hotel(name: str, latitude: float, longitude: float) -> dict:
    return {
        "name": name,
        "rate_per_night": {"extracted_lowest": 100},
        "gps_coordinates": {
            "latitude": latitude,
            "longitude": longitude,
        },
    }


def _cached_hotel(name: str, latitude: float, longitude: float) -> dict:
    return {
        "hotel_name": name,
        "hotel_class": 4.0,
        "overall_rating": 4.5,
        "reviews": 100,
        "description": "Verified test hotel",
        "price_per_night": 100.0,
        "amenities": ["WiFi"],
        "check_in_time": "15:00",
        "check_out_time": "11:00",
        "location": {
            "lat": latitude,
            "lng": longitude,
            "country_code": "SG",
        },
        "image": "",
        "booking_url": "https://hotel.example",
    }


@pytest.mark.unit
@pytest.mark.parametrize("source", ["provider", "cache"])
def test_hotel_boundary_drops_same_country_wrong_city_before_selection_or_cache(source):
    """Removing the locality comparison would admit the cheaper Johor hotel."""
    wrong_city = _provider_hotel("Johor Hotel", 1.49, 103.74)
    valid_city = _provider_hotel("Singapore Hotel", 1.30, 103.85)
    cached = (
        [
            _cached_hotel("Johor Hotel", 1.49, 103.74),
            _cached_hotel("Singapore Hotel", 1.30, 103.85),
        ]
        if source == "cache"
        else None
    )
    provider_payload = (
        {} if source == "cache" else {"properties": [wrong_city, valid_city]}
    )

    with (
        patch.object(flights_hotels, "get_cached_data", return_value=cached),
        patch.object(
            flights_hotels,
            "_serpapi_get",
            return_value=provider_payload,
        ) as provider,
        patch.object(
            flights_hotels,
            "resolve_country_code",
            side_effect=["SG", "SG"],
        ),
        patch.object(
            flights_hotels,
            "resolve_locality_names",
            side_effect=[("Johor Bahru", "Johor"), ("  SINGAPORE  ",)],
            create=True,
        ) as resolve_locality,
        patch.object(flights_hotels, "set_cached_data") as set_cache,
    ):
        result = flights_hotels.fetch_hotels_api(
            "  singapore  , Singapore",
            "2026-08-01",
            "2026-08-05",
            "SGD",
            destination_country_code="SG",
        )

    assert [hotel["hotel_name"] for hotel in result] == ["Singapore Hotel"]
    assert resolve_locality.call_args_list == [
        call(1.49, 103.74, "SG"),
        call(1.30, 103.85, "SG"),
    ]
    if source == "provider":
        cached_payload = set_cache.call_args.kwargs["payload"]
        assert [hotel["hotel_name"] for hotel in cached_payload] == [
            "Singapore Hotel"
        ]
    else:
        provider.assert_not_called()
        set_cache.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("locality_evidence", [None, (), "Singapore", (None, 123)])
def test_hotel_cache_drops_missing_or_malformed_locality_evidence(locality_evidence):
    """Removing fail-closed evidence validation would trust an unverifiable hit."""
    with (
        patch.object(
            flights_hotels,
            "get_cached_data",
            return_value=[_cached_hotel("Unverified Hotel", 1.30, 103.85)],
        ),
        patch.object(flights_hotels, "_serpapi_get") as provider,
        patch.object(flights_hotels, "resolve_country_code", return_value="SG"),
        patch.object(
            flights_hotels,
            "resolve_locality_names",
            return_value=locality_evidence,
            create=True,
        ),
    ):
        result = flights_hotels.fetch_hotels_api(
            "Singapore, Singapore",
            "2026-08-01",
            "2026-08-05",
            "SGD",
            destination_country_code="SG",
        )

    assert result == []
    provider.assert_not_called()


@pytest.mark.unit
def test_wrong_city_assessment_hotel_cannot_enter_singapore_plan():
    """Removing assessment locality revalidation would reuse the Johor hotel."""
    outbound = {
        "flight_number": "OUT",
        "price": 30.0,
        "departure_airport": {"id": "KUL"},
        "arrival_airport": {"id": "SIN"},
    }
    returning = {
        "flight_number": "RET",
        "price": 20.0,
        "departure_airport": {"id": "SIN"},
        "arrival_airport": {"id": "KUL"},
    }
    wrong_city_hotel = _cached_hotel("Johor Hotel", 1.49, 103.74)
    valid_city_hotel = _cached_hotel("Singapore Hotel", 1.30, 103.85)
    state = AgentState(
        origin_country="Malaysia",
        country="Singapore",
        city=["Singapore"],
        start_date="2026-08-01",
        end_date="2026-08-02",
        num_people=1,
        dest_currency_code="SGD",
        budget_allocation={"transportation": 200, "accommodation": 200},
        budget_assessment={
            "calculation_version": "allocation-v1",
            "origin": "Malaysia",
            "destination": "Singapore",
            "start_date": "2026-08-01",
            "end_date": "2026-08-02",
            "num_people": 1,
            "expires_at": "2999-08-16T13:00:00+00:00",
            "evidence": {
                "outbound_flight": outbound,
                "return_flight": returning,
                "hotel": wrong_city_hotel,
                "hotel_nights": 1,
                "hotel_price_per_night": 100.0,
            },
        },
    )

    with (
        patch.object(
            flights_hotels,
            "resolve_iata_code",
            side_effect=["KUL", "SIN"],
        ),
        patch.object(flights_hotels, "resolve_country_code", return_value="SG"),
        patch.object(
            flights_hotels,
            "resolve_locality_names",
            return_value=("Johor Bahru", "Johor"),
            create=True,
        ),
        patch.object(
            flights_hotels,
            "fetch_flights_api",
            side_effect=[[outbound], [returning]],
        ),
        patch.object(
            flights_hotels,
            "fetch_hotels_api",
            return_value=[valid_city_hotel],
        ) as fetch_hotels,
    ):
        itinerary = flights_hotels.plan_flight_hotel(state)["draft_itinerary"]

    fetch_hotels.assert_called_once()
    assert itinerary[0]["hotel"]["hotel_name"] == "Singapore Hotel"
    assert all(
        not day.get("hotel") or day["hotel"]["hotel_name"] != "Johor Hotel"
        for day in itinerary
    )


@pytest.mark.unit
def test_same_day_plan_still_skips_all_hotel_locality_work():
    outbound = {"flight_number": "OUT", "price": 30.0}
    returning = {"flight_number": "RET", "price": 20.0}
    state = AgentState(
        origin_country="Malaysia",
        country="Singapore",
        city=["Singapore"],
        start_date="2026-08-01",
        end_date="2026-08-01",
        num_people=1,
        dest_currency_code="SGD",
        budget_allocation={"transportation": 200, "accommodation": 0},
    )

    with (
        patch.object(
            flights_hotels,
            "fetch_flights_api",
            side_effect=[[outbound], [returning]],
        ),
        patch.object(
            flights_hotels,
            "fetch_hotels_api",
            side_effect=AssertionError("same-day plan must not query hotels"),
        ) as fetch_hotels,
        patch.object(
            flights_hotels,
            "resolve_locality_names",
            side_effect=AssertionError("same-day plan must not verify hotels"),
            create=True,
        ) as resolve_locality,
    ):
        day = flights_hotels.plan_flight_hotel(state)["draft_itinerary"][0]

    fetch_hotels.assert_not_called()
    resolve_locality.assert_not_called()
    assert day["hotel"] is None
