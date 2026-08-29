from __future__ import annotations

import math

import pytest

from app.agents.state import AgentState
from app.services.itinerary_quality import (
    ExpectedDay,
    TripRequirements,
    validate_itinerary_candidate,
)


def _activity(
    name: str,
    *,
    country_code: str = "SG",
    requested_city: str = "Singapore",
    verified_locality: str | None = None,
    latitude: float = 1.2966,
    order: int = 1,
    estimated_cost: float = 20.0,
):
    return {
        "name": name,
        "type": "attraction",
        "address": f"{name}, Singapore",
        "estimated_cost": estimated_cost,
        "order": order,
        "location": {
            "place_name": name,
            "latitude": latitude,
            "longitude": 103.7764,
            "country_code": country_code,
            "requested_city": requested_city,
            "verified_locality": verified_locality or requested_city,
        },
    }


def _flight(
    *,
    departure_id: str,
    departure_name: str,
    arrival_id: str,
    arrival_name: str,
    date: str,
    price: float = 100.0,
):
    departure_time = f"{date} 08:00"
    arrival_time = f"{date} 09:30"
    return {
        "airline": "Grounded Air",
        "flight_number": f"GA-{departure_id}-{arrival_id}",
        "departure_airport": {
            "id": departure_id,
            "name": departure_name,
            "time": departure_time,
        },
        "arrival_airport": {
            "id": arrival_id,
            "name": arrival_name,
            "time": arrival_time,
        },
        "departure_time": departure_time,
        "arrival_time": arrival_time,
        "price": price,
    }


def _hotel(*, country_code: str = "SG", price: float = 50.0):
    return {
        "hotel_name": "Verified Singapore Hotel",
        "price_per_night": price,
        "location": {
            "lat": 1.31,
            "lng": 103.82,
            "country_code": country_code,
        },
    }


def _complete_two_day_itinerary() -> list[dict]:
    hotel = _hotel()
    outbound = _flight(
        departure_id="KUL",
        departure_name="Kuala Lumpur International Airport",
        arrival_id="SIN",
        arrival_name="Singapore Changi Airport",
        date="2026-08-01",
    )
    return_flight = _flight(
        departure_id="SIN",
        departure_name="Singapore Changi Airport",
        arrival_id="KUL",
        arrival_name="Kuala Lumpur International Airport",
        date="2026-08-02",
    )
    return [
        {
            "day": 1,
            "date": "2026-08-01",
            "flight": [outbound],
            "hotel": hotel,
            "activities": [_activity("Gardens by the Bay")],
            "day_total_cost": 170.0,
        },
        {
            "day": 2,
            "date": "2026-08-02",
            "flight": [return_flight],
            "hotel": None,
            "activities": [_activity("Merlion Park")],
            "day_total_cost": 120.0,
        },
    ]


def _complete_one_day_itinerary() -> list[dict]:
    return [
        {
            "day": 1,
            "date": "2026-08-01",
            "flight": [
                _flight(
                    departure_id="KUL",
                    departure_name="Kuala Lumpur International Airport",
                    arrival_id="SIN",
                    arrival_name="Singapore Changi Airport",
                    date="2026-08-01",
                ),
                _flight(
                    departure_id="SIN",
                    departure_name="Singapore Changi Airport",
                    arrival_id="KUL",
                    arrival_name="Kuala Lumpur International Airport",
                    date="2026-08-01",
                ),
            ],
            "hotel": None,
            "activities": [_activity("Merlion Park")],
            "day_total_cost": 220.0,
        }
    ]


def _maps_for_complete_itinerary(itinerary: list[dict]) -> dict[int, dict]:
    maps: dict[int, dict] = {}
    for day in itinerary:
        hotel = day["hotel"]
        activity = day["activities"][0]
        maps[day["day"]] = {
            "type": "FeatureCollection",
            "features": [
                *(
                    [
                        _point(
                            hotel["hotel_name"],
                            "hotel",
                            latitude=hotel["location"]["lat"],
                            longitude=hotel["location"]["lng"],
                        )
                    ]
                    if isinstance(hotel, dict)
                    else []
                ),
                _point(
                    activity["name"],
                    activity["type"],
                    latitude=activity["location"]["latitude"],
                    longitude=activity["location"]["longitude"],
                ),
            ],
        }
    return maps


def _point(
    name: str,
    point_type: str = "attraction",
    *,
    latitude: float = 1.2966,
    longitude: float = 103.7764,
    order: int | None = None,
):
    if order is None:
        order = 0 if point_type in {"hotel", "airport"} else 1
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
        "properties": {"name": name, "type": point_type, "order": order},
    }


def _map_for(*names: str):
    return {"type": "FeatureCollection", "features": [_point(name) for name in names]}


def _route_feature(
    *,
    profile: str = "driving",
    distance_km: float = 2.5,
    duration_mins: float = 8.0,
    coordinates: list[list[float]] | None = None,
):
    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coordinates
            or [[103.82, 1.31], [103.7764, 1.2966]],
        },
        "properties": {
            "type": "route",
            "profile": profile,
            "distance_km": distance_km,
            "duration_mins": duration_mins,
        },
    }


def _add_matching_route(itinerary: list[dict], maps: dict[int, dict], day: int = 1):
    target = itinerary[day - 1]
    target["route"] = {
        "ordered_stops": [
            target["hotel"]["hotel_name"],
            target["activities"][0]["name"],
        ],
        "profiles": {
            "driving": {"distance_km": 2.5, "duration_mins": 8.0},
        },
    }
    maps[day]["features"].append(_route_feature())


def _canonical_allocation(total: float = 3000.0) -> dict[str, float]:
    return {
        "transportation": total * 0.25,
        "accommodation": total * 0.35,
        "food": total * 0.15,
        "activity": total * 0.15,
        "shopping": total * 0.05,
        "emergency_fund": total * 0.05,
    }


@pytest.fixture
def singapore_requirements():
    return TripRequirements(
        origin_country="Malaysia",
        destination_country="Singapore",
        destination_country_code="SG",
        cities=("Singapore",),
        expected_days=(
            ExpectedDay(day=1, date="2026-08-01"),
            ExpectedDay(day=2, date="2026-08-02"),
        ),
        num_people=2,
        base_currency="MYR",
        destination_currency="SGD",
        total_budget=3000.0,
        budget_allocation=_canonical_allocation(),
    )


@pytest.fixture
def complete_singapore_candidate():
    requirements = TripRequirements(
        origin_country="Malaysia",
        destination_country="Singapore",
        destination_country_code="SG",
        cities=("Singapore",),
        expected_days=tuple(
            ExpectedDay(day=day, date=f"2026-08-0{day}") for day in range(1, 6)
        ),
        num_people=2,
        base_currency="MYR",
        destination_currency="SGD",
        total_budget=3000.0,
        budget_allocation=_canonical_allocation(),
    )
    itinerary = [
        {
            "day": day,
            "date": f"2026-08-0{day}",
            "flight": None,
            "hotel": _hotel() if day < 5 else None,
            "activities": [_activity(f"Singapore Place {day}", latitude=1.28 + day / 100)],
            "day_total_cost": 70.0 if day < 5 else 20.0,
        }
        for day in range(1, 6)
    ]
    itinerary[0]["flight"] = [
        _flight(
            departure_id="KUL",
            departure_name="Kuala Lumpur International Airport",
            arrival_id="SIN",
            arrival_name="Singapore Changi Airport",
            date="2026-08-01",
        )
    ]
    itinerary[0]["day_total_cost"] = 170.0
    itinerary[-1]["flight"] = [
        _flight(
            departure_id="SIN",
            departure_name="Singapore Changi Airport",
            arrival_id="KUL",
            arrival_name="Kuala Lumpur International Airport",
            date="2026-08-05",
        )
    ]
    itinerary[-1]["day_total_cost"] = 120.0
    maps = _maps_for_complete_itinerary(itinerary)
    return requirements, itinerary, maps


def test_singapore_candidate_rejects_empty_day_and_malaysia_activity():
    requirements = TripRequirements(
        origin_country="Malaysia",
        destination_country="Singapore",
        destination_country_code="SG",
        cities=("Singapore",),
        expected_days=(
            ExpectedDay(day=1, date="2026-08-01"),
            ExpectedDay(day=2, date="2026-08-02"),
        ),
        num_people=2,
        base_currency="MYR",
        destination_currency="SGD",
        total_budget=3000.0,
        budget_allocation=_canonical_allocation(),
    )
    itinerary = [
        {"day": 1, "date": "2026-08-01", "activities": []},
        {
            "day": 2,
            "date": "2026-08-02",
            "activities": [
                {
                    "name": "Johor Attraction",
                    "type": "attraction",
                    "address": "Johor Bahru, Malaysia",
                    "estimated_cost": 20.0,
                    "location": {
                        "place_name": "Johor Attraction",
                        "latitude": 1.4927,
                        "longitude": 103.7414,
                        "country_code": "MY",
                    },
                }
            ],
        },
    ]

    report = validate_itinerary_candidate(requirements, itinerary, maps={})

    assert {issue.code for issue in report.issues} >= {
        "day.activities.empty",
        "activity.country.mismatch",
        "map.day.missing",
    }


def test_rejects_invalid_cost_order_total_and_missing_boundary_travel(
    singapore_requirements,
):
    """Catch financially impossible, unordered candidates qualifying for promotion."""
    itinerary = [
        {
            "day": 1,
            "date": "2026-08-01",
            "activities": [
                _activity(
                    "Gardens by the Bay",
                    order=2,
                    estimated_cost=-50.0,
                )
            ],
            "day_total_cost": -999.0,
        },
        {
            "day": 2,
            "date": "2026-08-02",
            "activities": [_activity("Merlion Park", order=2)],
            "day_total_cost": 20.0,
        },
    ]
    maps = {
        1: _map_for("Gardens by the Bay"),
        2: _map_for("Merlion Park"),
    }

    report = validate_itinerary_candidate(singapore_requirements, itinerary, maps)

    assert {issue.code for issue in report.issues} >= {
        "activity.cost.invalid",
        "day.activities.order.invalid",
        "day.total_cost.invalid",
        "flight.outbound.missing",
        "flight.return.missing",
        "hotel.night.missing",
    }


def test_rejects_same_country_activity_outside_requested_city(
    singapore_requirements,
):
    """Catch country-valid provider results being attributed to an unrequested city."""
    itinerary = _complete_two_day_itinerary()
    itinerary[0]["activities"][0]["location"]["requested_city"] = "Sentosa"
    maps = _maps_for_complete_itinerary(itinerary)

    report = validate_itinerary_candidate(singapore_requirements, itinerary, maps)

    assert "activity.city.mismatch" in {issue.code for issue in report.issues}


def test_rejects_requested_city_label_without_matching_provider_locality_proof(
    singapore_requirements,
):
    """Catch a provider result being relabelled as the requested city after lookup."""
    itinerary = _complete_two_day_itinerary()
    location = itinerary[0]["activities"][0]["location"]
    location["requested_city"] = "Singapore"
    location["verified_locality"] = "Johor Bahru"
    maps = _maps_for_complete_itinerary(itinerary)

    report = validate_itinerary_candidate(singapore_requirements, itinerary, maps)

    assert "activity.city.proof.mismatch" in {
        issue.code for issue in report.issues
    }


def test_multi_city_coverage_counts_verified_locality_not_requested_label():
    """Catch copied query labels falsely satisfying every configured city."""
    requirements = TripRequirements(
        origin_country="Malaysia",
        destination_country="Singapore",
        destination_country_code="SG",
        cities=("Singapore", "Sentosa"),
        expected_days=(
            ExpectedDay(day=1, date="2026-08-01"),
            ExpectedDay(day=2, date="2026-08-02"),
        ),
        num_people=2,
        base_currency="MYR",
        destination_currency="SGD",
        total_budget=3000.0,
        budget_allocation=_canonical_allocation(),
    )
    itinerary = _complete_two_day_itinerary()
    itinerary[1]["activities"][0]["location"].update(
        {
            "requested_city": "Sentosa",
            "verified_locality": "Singapore",
        }
    )
    maps = _maps_for_complete_itinerary(itinerary)

    report = validate_itinerary_candidate(requirements, itinerary, maps)
    codes = {issue.code for issue in report.issues}

    assert "activity.city.proof.mismatch" in codes
    assert "itinerary.city.coverage.missing" in codes


def test_penang_provider_evidence_cannot_qualify_as_kuala_lumpur():
    """Catch the reviewed 300 km Penang-as-Kuala-Lumpur candidate mutation."""
    requirements = TripRequirements(
        origin_country="Singapore",
        destination_country="Malaysia",
        destination_country_code="MY",
        cities=("Kuala Lumpur",),
        expected_days=(ExpectedDay(day=1, date="2026-08-01"),),
        num_people=1,
        base_currency="SGD",
        destination_currency="MYR",
        total_budget=1000.0,
        budget_allocation=_canonical_allocation(1000.0),
    )
    activity = _activity(
        "Penang State Museum",
        country_code="MY",
        requested_city="Kuala Lumpur",
        verified_locality="George Town",
        latitude=5.4141,
        estimated_cost=20.0,
    )
    activity["address"] = "57 Jalan Macalister, George Town, Penang"
    activity["location"]["longitude"] = 100.3288
    itinerary = [
        {
            "day": 1,
            "date": "2026-08-01",
            "flight": [
                _flight(
                    departure_id="SIN",
                    departure_name="Singapore Changi Airport",
                    arrival_id="KUL",
                    arrival_name="Kuala Lumpur International Airport",
                    date="2026-08-01",
                ),
                _flight(
                    departure_id="KUL",
                    departure_name="Kuala Lumpur International Airport",
                    arrival_id="SIN",
                    arrival_name="Singapore Changi Airport",
                    date="2026-08-01",
                ),
            ],
            "hotel": None,
            "activities": [activity],
            "route": None,
            "day_total_cost": 220.0,
        }
    ]
    maps = {
        1: {
            "type": "FeatureCollection",
            "features": [
                _point(
                    "Penang State Museum",
                    latitude=5.4141,
                    longitude=100.3288,
                )
            ],
        }
    }

    report = validate_itinerary_candidate(requirements, itinerary, maps)

    assert "activity.city.proof.mismatch" in {
        issue.code for issue in report.issues
    }


def test_rejects_candidate_missing_required_multi_city_coverage():
    """Catch a complete-looking plan that silently omits one requested city/area."""
    requirements = TripRequirements(
        origin_country="Malaysia",
        destination_country="Singapore",
        destination_country_code="SG",
        cities=("Singapore", "Sentosa"),
        expected_days=(
            ExpectedDay(day=1, date="2026-08-01"),
            ExpectedDay(day=2, date="2026-08-02"),
        ),
        num_people=2,
        base_currency="MYR",
        destination_currency="SGD",
        total_budget=3000.0,
        budget_allocation=_canonical_allocation(),
    )
    itinerary = _complete_two_day_itinerary()
    maps = _maps_for_complete_itinerary(itinerary)

    report = validate_itinerary_candidate(requirements, itinerary, maps)

    assert "itinerary.city.coverage.missing" in {
        issue.code for issue in report.issues
    }


def test_accepts_strictly_sequential_activities_and_reconciled_day_totals(
    singapore_requirements,
):
    itinerary = _complete_two_day_itinerary()
    second = _activity(
        "ArtScience Museum",
        order=2,
        estimated_cost=10.0,
        latitude=1.2863,
    )
    itinerary[0]["activities"].append(second)
    itinerary[0]["day_total_cost"] = 180.0
    maps = _maps_for_complete_itinerary(itinerary)
    maps[1]["features"].append(
        _point(
            second["name"],
            latitude=second["location"]["latitude"],
            longitude=second["location"]["longitude"],
            order=2,
        )
    )

    report = validate_itinerary_candidate(singapore_requirements, itinerary, maps)

    assert report.qualified is True


@pytest.mark.parametrize("orders", [(1, 1), (2, 1), (1, 3)])
def test_rejects_duplicate_out_of_sequence_or_gapped_activity_orders(
    singapore_requirements,
    orders,
):
    itinerary = _complete_two_day_itinerary()
    itinerary[0]["activities"] = [
        _activity("Gardens by the Bay", order=orders[0]),
        _activity("ArtScience Museum", order=orders[1]),
    ]
    itinerary[0]["day_total_cost"] = 190.0
    maps = _maps_for_complete_itinerary(itinerary)
    maps[1]["features"].append(_point("ArtScience Museum"))

    report = validate_itinerary_candidate(singapore_requirements, itinerary, maps)

    assert "day.activities.order.invalid" in {
        issue.code for issue in report.issues
    }


@pytest.mark.parametrize("price", [-0.01, math.nan, math.inf, True])
def test_rejects_invalid_flight_prices(singapore_requirements, price):
    itinerary = _complete_two_day_itinerary()
    itinerary[0]["flight"][0]["price"] = price
    maps = _maps_for_complete_itinerary(itinerary)

    report = validate_itinerary_candidate(singapore_requirements, itinerary, maps)

    assert "flight.price.invalid" in {issue.code for issue in report.issues}


@pytest.mark.parametrize("price", [-0.01, math.nan, math.inf, True])
def test_rejects_invalid_required_hotel_prices(singapore_requirements, price):
    itinerary = _complete_two_day_itinerary()
    itinerary[0]["hotel"]["price_per_night"] = price
    maps = _maps_for_complete_itinerary(itinerary)

    report = validate_itinerary_candidate(singapore_requirements, itinerary, maps)

    assert "hotel.price.invalid" in {issue.code for issue in report.issues}


def test_day_total_uses_cent_tolerance_with_no_relative_tolerance(
    singapore_requirements,
):
    within_tolerance = _complete_two_day_itinerary()
    within_tolerance[0]["day_total_cost"] = 170.009
    outside_tolerance = _complete_two_day_itinerary()
    outside_tolerance[0]["day_total_cost"] = 170.011

    accepted = validate_itinerary_candidate(
        singapore_requirements,
        within_tolerance,
        _maps_for_complete_itinerary(within_tolerance),
    )
    rejected = validate_itinerary_candidate(
        singapore_requirements,
        outside_tolerance,
        _maps_for_complete_itinerary(outside_tolerance),
    )

    assert "day.total_cost.mismatch" not in {
        issue.code for issue in accepted.issues
    }
    assert "day.total_cost.mismatch" in {issue.code for issue in rejected.issues}


def test_rejects_boundary_flight_date_with_only_a_matching_text_prefix(
    singapore_requirements,
):
    itinerary = _complete_two_day_itinerary()
    itinerary[0]["flight"][0]["departure_time"] = "2026-08-010 08:00"
    maps = _maps_for_complete_itinerary(itinerary)

    report = validate_itinerary_candidate(singapore_requirements, itinerary, maps)

    assert "flight.date.mismatch" in {issue.code for issue in report.issues}


def test_rejects_boundary_flights_that_do_not_form_a_reversed_route(
    singapore_requirements,
):
    itinerary = _complete_two_day_itinerary()
    itinerary[1]["flight"][0] = _flight(
        departure_id="SIN",
        departure_name="Singapore Changi Airport",
        arrival_id="BKK",
        arrival_name="Suvarnabhumi Airport",
        date="2026-08-02",
    )
    maps = _maps_for_complete_itinerary(itinerary)

    report = validate_itinerary_candidate(singapore_requirements, itinerary, maps)

    assert "flight.direction.mismatch" in {
        issue.code for issue in report.issues
    }


def test_one_day_trip_requires_two_boundary_flights_and_no_hotel_night():
    requirements = TripRequirements(
        origin_country="Malaysia",
        destination_country="Singapore",
        destination_country_code="SG",
        cities=("Singapore",),
        expected_days=(ExpectedDay(day=1, date="2026-08-01"),),
        num_people=2,
        base_currency="MYR",
        destination_currency="SGD",
        total_budget=3000.0,
        budget_allocation=_canonical_allocation(),
    )
    valid_itinerary = _complete_one_day_itinerary()
    missing_return = _complete_one_day_itinerary()
    missing_return[0]["flight"].pop()
    missing_return[0]["day_total_cost"] = 120.0
    maps = {1: _map_for("Merlion Park")}

    accepted = validate_itinerary_candidate(requirements, valid_itinerary, maps)
    rejected = validate_itinerary_candidate(requirements, missing_return, maps)

    assert accepted.qualified is True
    assert not any(issue.code.startswith("hotel.") for issue in accepted.issues)
    assert "flight.return.missing" in {issue.code for issue in rejected.issues}


def test_one_day_trip_rejects_unexpected_hotel_object():
    requirements = TripRequirements(
        origin_country="Malaysia", destination_country="Singapore",
        destination_country_code="SG", cities=("Singapore",),
        expected_days=(ExpectedDay(day=1, date="2026-08-01"),), num_people=2,
        base_currency="MYR", destination_currency="SGD", total_budget=3000,
        budget_allocation=_canonical_allocation(),
    )
    itinerary = _complete_one_day_itinerary()
    itinerary[0]["hotel"] = _hotel()
    report = validate_itinerary_candidate(
        requirements, itinerary, _maps_for_complete_itinerary(itinerary)
    )
    assert "hotel.unexpected" in {issue.code for issue in report.issues}


def test_rejects_missing_expected_day(singapore_requirements):
    itinerary = [{"day": 1, "date": "2026-08-01", "activities": [_activity("Gardens by the Bay")]}]
    maps = {1: _map_for("Gardens by the Bay")}

    report = validate_itinerary_candidate(singapore_requirements, itinerary, maps)

    assert "day.missing" in {issue.code for issue in report.issues}


@pytest.mark.parametrize("coordinate", ["not-a-number", "1.2966", math.nan, math.inf])
def test_rejects_malformed_activity_coordinates(singapore_requirements, coordinate):
    bad_activity = _activity("ArtScience Museum")
    bad_activity["location"]["latitude"] = coordinate
    itinerary = [
        {"day": 1, "date": "2026-08-01", "activities": [bad_activity]},
        {"day": 2, "date": "2026-08-02", "activities": [_activity("Merlion Park")]},
    ]
    maps = {2: _map_for("Merlion Park")}

    report = validate_itinerary_candidate(singapore_requirements, itinerary, maps)

    assert "activity.location.coordinates.invalid" in {
        issue.code for issue in report.issues
    }


def test_rejects_map_that_omits_required_activity_point(singapore_requirements):
    itinerary = [
        {"day": 1, "date": "2026-08-01", "activities": [_activity("Gardens by the Bay")]},
        {"day": 2, "date": "2026-08-02", "activities": [_activity("Merlion Park")]},
    ]
    maps = {1: _map_for(), 2: _map_for("Merlion Park")}

    report = validate_itinerary_candidate(singapore_requirements, itinerary, maps)

    assert "map.points.mismatch" in {issue.code for issue in report.issues}


def test_rejects_map_point_with_matching_name_but_wrong_coordinates(
    singapore_requirements,
):
    itinerary = [
        {"day": 1, "date": "2026-08-01", "activities": [_activity("Gardens by the Bay")]},
        {"day": 2, "date": "2026-08-02", "activities": [_activity("Merlion Park")]},
    ]
    maps = {
        1: {
            "type": "FeatureCollection",
            "features": [
                _point("Gardens by the Bay", latitude=0.0, longitude=0.0)
            ],
        },
        2: _map_for("Merlion Park"),
    }

    report = validate_itinerary_candidate(singapore_requirements, itinerary, maps)

    assert "map.points.mismatch" in {issue.code for issue in report.issues}


def test_rejects_hotel_map_point_with_matching_name_but_wrong_coordinates(
    singapore_requirements,
):
    hotel = {
        "hotel_name": "Singapore Hotel",
        "price_per_night": 200.0,
        "location": {"lat": 1.31, "lng": 103.82, "country_code": "SG"},
    }
    itinerary = [
        {
            "day": 1,
            "date": "2026-08-01",
            "hotel": hotel,
            "activities": [_activity("Gardens by the Bay")],
        },
        {"day": 2, "date": "2026-08-02", "activities": [_activity("Merlion Park")]},
    ]
    maps = {
        1: {
            "type": "FeatureCollection",
            "features": [
                _point("Singapore Hotel", "hotel", latitude=0.0, longitude=0.0),
                _point("Gardens by the Bay"),
            ],
        },
        2: _map_for("Merlion Park"),
    }

    report = validate_itinerary_candidate(singapore_requirements, itinerary, maps)

    assert "map.points.mismatch" in {issue.code for issue in report.issues}


def test_rejects_non_finite_geojson_point_coordinates(singapore_requirements):
    itinerary = [
        {"day": 1, "date": "2026-08-01", "activities": [_activity("Gardens by the Bay")]},
        {"day": 2, "date": "2026-08-02", "activities": [_activity("Merlion Park")]},
    ]
    maps = {
        1: {
            "type": "FeatureCollection",
            "features": [
                _point("Gardens by the Bay", latitude=math.inf)
            ],
        },
        2: _map_for("Merlion Park"),
    }

    report = validate_itinerary_candidate(singapore_requirements, itinerary, maps)

    assert "map.feature.invalid" in {
        issue.code for issue in report.issues
    }


@pytest.mark.parametrize(
    "extra_feature",
    [
        None,
        "not-a-feature",
        {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": []},
            "properties": {"name": "Injected", "type": "attraction", "order": 2},
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [103.9, 1.4]},
            "properties": {"name": "Injected", "type": "attraction", "order": 2},
            "provider_payload": {"private": True},
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [103.9, 1.4],
                "bbox": [0, 0, 1, 1],
            },
            "properties": {"name": "Injected", "type": "attraction", "order": 2},
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [103.9, 1.4]},
            "properties": {
                "name": "Injected",
                "type": "attraction",
                "order": 2,
                "profile": "walking",
            },
        },
    ],
    ids=[
        "null",
        "non-mapping",
        "polygon",
        "feature-extra-key",
        "geometry-extra-key",
        "point-route-property",
    ],
)
def test_rejects_every_unsupported_or_malformed_geojson_feature(
    singapore_requirements,
    extra_feature,
):
    itinerary = _complete_two_day_itinerary()
    maps = _maps_for_complete_itinerary(itinerary)
    maps[1]["features"].append(extra_feature)

    report = validate_itinerary_candidate(singapore_requirements, itinerary, maps)

    assert "map.feature.invalid" in {issue.code for issue in report.issues}


@pytest.mark.parametrize(
    "properties",
    [
        {"name": "Gardens by the Bay", "type": "hotel", "order": 1},
        {"name": "Gardens by the Bay", "type": "attraction", "order": 2},
    ],
    ids=["point-type", "point-order"],
)
def test_rejects_point_metadata_that_disagrees_with_itinerary(
    singapore_requirements,
    properties,
):
    itinerary = _complete_two_day_itinerary()
    maps = _maps_for_complete_itinerary(itinerary)
    maps[1]["features"][1]["properties"] = properties

    report = validate_itinerary_candidate(singapore_requirements, itinerary, maps)

    assert "map.points.mismatch" in {issue.code for issue in report.issues}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda itinerary, maps: itinerary[0].update({"private": True}),
        lambda itinerary, maps: itinerary[0]["activities"][0].update(
            {"provider_payload": True}
        ),
        lambda itinerary, maps: itinerary[0]["activities"][0]["location"].update(
            {"raw_provider": True}
        ),
        lambda itinerary, maps: itinerary[0]["flight"][0].update({"raw": True}),
        lambda itinerary, maps: itinerary[0]["flight"][0][
            "arrival_airport"
        ].update({"raw": True}),
        lambda itinerary, maps: maps[1].update({"bbox": [0, 0, 1, 1]}),
        lambda itinerary, maps: maps[1]["features"][0].update({"id": "private"}),
        lambda itinerary, maps: maps[1]["features"][0].update(
            {"type": "PrivateFeature"}
        ),
        lambda itinerary, maps: maps[1]["features"][0]["geometry"].update(
            {"bbox": [0, 0, 1, 1]}
        ),
        lambda itinerary, maps: maps[1]["features"][0]["properties"].update(
            {"provider_payload": True}
        ),
        lambda itinerary, maps: maps[1]["features"][0]["geometry"].update(
            {"coordinates": [103.82, 1.31, 10.0]}
        ),
    ],
    ids=[
        "day-extra",
        "activity-extra",
        "location-extra",
        "flight-extra",
        "airport-extra",
        "map-extra",
        "feature-extra",
        "feature-type",
        "geometry-extra",
        "properties-extra",
        "three-coordinate-point",
    ],
)
def test_rejects_candidate_tree_that_strict_public_projection_rejects(
    singapore_requirements,
    mutation,
):
    itinerary = _complete_two_day_itinerary()
    maps = _maps_for_complete_itinerary(itinerary)
    mutation(itinerary, maps)

    report = validate_itinerary_candidate(singapore_requirements, itinerary, maps)

    assert report.qualified is False


@pytest.mark.parametrize(
    "mutate_maps,expected_code",
    [
        (lambda maps: maps.update({3: _map_for("Unexpected")}), "map.day.unexpected"),
        (lambda maps: maps.__setitem__("01", maps[1]), "map.day.key.invalid"),
        (lambda maps: maps.__setitem__(0, maps[1]), "map.day.key.invalid"),
        (lambda maps: maps.__setitem__(True, maps.pop(1)), "map.day.key.invalid"),
    ],
    ids=["extra-day", "leading-zero", "zero", "boolean"],
)
def test_rejects_noncanonical_or_unexpected_map_day_keys(
    singapore_requirements,
    mutate_maps,
    expected_code,
):
    itinerary = _complete_two_day_itinerary()
    maps = _maps_for_complete_itinerary(itinerary)
    mutate_maps(maps)

    report = validate_itinerary_candidate(singapore_requirements, itinerary, maps)

    assert expected_code in {issue.code for issue in report.issues}


def test_rejects_colliding_integer_and_string_map_day_aliases(
    singapore_requirements,
):
    itinerary = _complete_two_day_itinerary()
    maps = _maps_for_complete_itinerary(itinerary)
    maps["1"] = maps[1]

    report = validate_itinerary_candidate(singapore_requirements, itinerary, maps)

    assert "map.day.key.duplicate" in {issue.code for issue in report.issues}


@pytest.mark.parametrize(
    "mutation,expected_code",
    [
        (
            lambda itinerary, maps: maps[1]["features"][-1]["properties"].update(
                {"distance_km": 999.0}
            ),
            "map.route.metrics.mismatch",
        ),
        (
            lambda itinerary, maps: maps[1]["features"].append(
                _route_feature(profile="walking")
            ),
            "map.route.profiles.mismatch",
        ),
        (
            lambda itinerary, maps: maps[1]["features"][-1]["geometry"].update(
                {"coordinates": [[0.0, 0.0], [1.0, 1.0]]}
            ),
            "map.route.endpoints.mismatch",
        ),
        (
            lambda itinerary, maps: itinerary[0]["route"].update(
                {"ordered_stops": ["Wrong", "Stops"]}
            ),
            "map.route.stops.mismatch",
        ),
    ],
    ids=["metrics", "unexpected-profile", "endpoints", "ordered-stops"],
)
def test_rejects_route_and_map_cross_object_mismatches(
    singapore_requirements,
    mutation,
    expected_code,
):
    itinerary = _complete_two_day_itinerary()
    maps = _maps_for_complete_itinerary(itinerary)
    _add_matching_route(itinerary, maps)
    mutation(itinerary, maps)

    report = validate_itinerary_candidate(singapore_requirements, itinerary, maps)

    assert expected_code in {issue.code for issue in report.issues}


def test_rejects_unexpected_route_line_when_itinerary_has_no_route(
    singapore_requirements,
):
    itinerary = _complete_two_day_itinerary()
    maps = _maps_for_complete_itinerary(itinerary)
    maps[1]["features"].append(_route_feature())

    report = validate_itinerary_candidate(singapore_requirements, itinerary, maps)

    assert "map.route.profiles.mismatch" in {issue.code for issue in report.issues}


def test_rejects_unsupported_null_itinerary_route_profile(
    singapore_requirements,
):
    """Keep backend validation aligned with the strict public/frontend route shape."""
    itinerary = _complete_two_day_itinerary()
    maps = _maps_for_complete_itinerary(itinerary)
    _add_matching_route(itinerary, maps)
    itinerary[0]["route"]["profiles"]["flying"] = None

    report = validate_itinerary_candidate(singapore_requirements, itinerary, maps)

    assert "itinerary.route.invalid" in {issue.code for issue in report.issues}


def test_accepts_exact_route_profile_metrics_stops_and_geometry_endpoints(
    singapore_requirements,
):
    itinerary = _complete_two_day_itinerary()
    maps = _maps_for_complete_itinerary(itinerary)
    _add_matching_route(itinerary, maps)

    report = validate_itinerary_candidate(singapore_requirements, itinerary, maps)

    assert report.qualified is True


@pytest.mark.parametrize(
    "allocation",
    [
        {
            **_canonical_allocation(),
            "food": -1.0,
            "shopping": 451.0,
        },
        {
            **_canonical_allocation(),
            "activity": math.nan,
        },
        {
            **_canonical_allocation(),
            "activity": 1.0,
        },
        {
            **_canonical_allocation(),
            "flight": 1.0,
        },
    ],
)
def test_requirements_reject_invalid_or_inconsistent_canonical_allocation(allocation):
    with pytest.raises(ValueError):
        TripRequirements(
            origin_country="Malaysia",
            destination_country="Singapore",
            destination_country_code="SG",
            cities=("Singapore",),
            expected_days=(ExpectedDay(day=1, date="2026-08-01"),),
            num_people=1,
            base_currency="MYR",
            destination_currency="SGD",
            total_budget=3000.0,
            budget_allocation=allocation,
        )


def test_validates_activities_on_an_unexpected_day(singapore_requirements):
    malformed_activity = _activity("Invalid Extra Stop")
    malformed_activity["location"]["longitude"] = "invalid"
    itinerary = [
        {"day": 1, "date": "2026-08-01", "activities": [_activity("Gardens by the Bay")]},
        {"day": 2, "date": "2026-08-02", "activities": [_activity("Merlion Park")]},
        {"day": 3, "date": "2026-08-03", "activities": [malformed_activity]},
    ]
    maps = {1: _map_for("Gardens by the Bay"), 2: _map_for("Merlion Park")}

    report = validate_itinerary_candidate(singapore_requirements, itinerary, maps)

    assert "activity.location.coordinates.invalid" in {
        issue.code for issue in report.issues
    }


def test_complete_five_day_candidate_is_qualified(complete_singapore_candidate):
    requirements, itinerary, maps = complete_singapore_candidate

    report = validate_itinerary_candidate(requirements, itinerary, maps)

    assert report.qualified is True
    assert report.issues == ()


def test_requirements_are_derived_immutably_from_valid_agent_state():
    requirements = TripRequirements.from_state(
        AgentState(
            origin_country="Malaysia",
            country="Singapore",
            city=["Singapore"],
            num_people=2,
            start_date="2026-08-01",
            end_date="2026-08-03",
            base_currency_code="MYR",
            dest_currency_code="SGD",
            total_convert_budget=3000.0,
            budget_allocation=_canonical_allocation(),
        )
    )

    assert requirements.destination_country_code == "SG"
    assert requirements.expected_days == (
        ExpectedDay(day=1, date="2026-08-01"),
        ExpectedDay(day=2, date="2026-08-02"),
        ExpectedDay(day=3, date="2026-08-03"),
    )
    with pytest.raises(Exception):
        requirements.total_budget = 1.0
    with pytest.raises(TypeError):
        requirements.budget_allocation["activity"] = 1.0
    assert requirements.model_dump()["budget_allocation"] == _canonical_allocation()
    copied_requirements = requirements.model_copy(deep=True)
    with pytest.raises(TypeError):
        copied_requirements.budget_allocation["activity"] = 1.0


@pytest.mark.parametrize(
    "changes",
    [
        {"country": None},
        {"base_currency_code": None},
        {"dest_currency_code": "SG"},
        {"start_date": "2026-08-03", "end_date": "2026-08-01"},
    ],
)
def test_requirements_reject_missing_destination_currency_or_day_range(changes):
    state = AgentState(
        origin_country="Malaysia",
        country="Singapore",
        city=["Singapore"],
        num_people=2,
        start_date="2026-08-01",
        end_date="2026-08-03",
        base_currency_code="MYR",
        dest_currency_code="SGD",
        total_convert_budget=3000.0,
    )

    with pytest.raises(ValueError):
        TripRequirements.from_state(state.model_copy(update=changes))
