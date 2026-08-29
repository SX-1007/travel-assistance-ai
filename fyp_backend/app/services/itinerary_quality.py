"""Immutable trip requirements and deterministic candidate-plan validation."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
import math
import re
from typing import Any, Mapping, Sequence

import pycountry
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from app.agents.state import AgentState
from app.services.budget_assessment import DEFAULT_BUDGET_RATIOS
from app.services.activity_hybrid import find_duplicate_activity_paths

import logging

logger = logging.getLogger(__name__)


_CURRENCY_CODE = re.compile(r"^[A-Z]{3}$")
_ACTIVITY_TYPES = frozenset({"attraction", "restaurant"})
_CANONICAL_BUDGET_CATEGORIES = frozenset(DEFAULT_BUDGET_RATIOS)

MAX_ACTIVITY_CITY_DISTANCE_KM = 75.0

_BUDGET_TOTAL_TOLERANCE = 0.05
_DAY_TOTAL_TOLERANCE = 0.01
_ROUTE_PROFILES = frozenset({"driving", "walking", "cycling"})
_ExpectedPoint = tuple[str, float, float, str, int]
_RoutePoint = tuple[str, float, float]
_FlightEvidence = tuple[float, str, str]


class FrozenDict(dict[str, float]):
    """A JSON-serializable mapping that rejects all public mutations."""

    def _immutable(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("FrozenDict is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __copy__(self) -> "FrozenDict":
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> "FrozenDict":
        return self


class ExpectedDay(BaseModel):
    model_config = ConfigDict(frozen=True)

    day: int = Field(ge=1)
    date: str


class TripRequirements(BaseModel):
    model_config = ConfigDict(frozen=True)

    origin_country: str
    destination_country: str
    destination_country_code: str = Field(pattern=r"^[A-Z]{2}$")
    cities: tuple[str, ...]
    expected_days: tuple[ExpectedDay, ...]
    num_people: int = Field(ge=1)
    base_currency: str
    destination_currency: str
    total_budget: float = Field(ge=0, allow_inf_nan=False)
    budget_allocation: dict[str, float]

    @model_validator(mode="after")
    def _allocation_is_canonical_and_balanced(self) -> "TripRequirements":
        validated = _validated_budget_allocation(
            self.budget_allocation,
            self.total_budget,
        )
        object.__setattr__(self, "budget_allocation", FrozenDict(validated))
        return self

    @classmethod
    def from_state(cls, state: AgentState) -> "TripRequirements":
        """Create an immutable planning contract from the server-owned state."""
        origin_country = _required_text(state.origin_country, "origin_country")
        destination_country = _required_text(state.country, "destination country")
        destination_country_code = _country_code(destination_country)
        base_currency = _currency_code(state.base_currency_code, "base currency")
        destination_currency = _currency_code(
            state.dest_currency_code, "destination currency"
        )

        try:
            start = datetime.strptime(_required_text(state.start_date, "start date"), "%Y-%m-%d")
            end = datetime.strptime(_required_text(state.end_date, "end date"), "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("trip dates must use YYYY-MM-DD calendar dates") from exc
        if end < start:
            raise ValueError("end date must not be before start date")

        if not isinstance(state.num_people, int) or isinstance(state.num_people, bool):
            raise ValueError("num_people is required")
        if state.num_people < 1:
            raise ValueError("num_people must be at least 1")

        total_budget = _finite_number(state.total_convert_budget, "total converted budget")
        if total_budget < 0:
            raise ValueError("total converted budget must be non-negative")

        allocation = state.budget_allocation
        if not isinstance(allocation, Mapping):
            raise ValueError("budget allocation must be a mapping")

        cities = state.city
        if not isinstance(cities, list) or not all(
            isinstance(city, str) and city.strip() for city in cities
        ):
            raise ValueError("cities must be a list of non-empty strings")

        expected_days: list[ExpectedDay] = []
        current = start
        day_number = 1
        while current <= end:
            expected_days.append(
                ExpectedDay(day=day_number, date=current.strftime("%Y-%m-%d"))
            )
            current += timedelta(days=1)
            day_number += 1

        return cls(
            origin_country=origin_country,
            destination_country=destination_country,
            destination_country_code=destination_country_code,
            cities=tuple(city.strip() for city in cities),
            expected_days=tuple(expected_days),
            num_people=state.num_people,
            base_currency=base_currency,
            destination_currency=destination_currency,
            total_budget=total_budget,
            budget_allocation=dict(allocation),
        )


class ValidationIssue(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    path: str
    message: str


class ValidationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def qualified(self) -> bool:
        return not self.issues


def validate_itinerary_candidate(
    requirements: TripRequirements,
    itinerary: Any,
    maps: Any,
) -> ValidationReport:
    """Validate a candidate independently of the agent or any provider claims."""
    issues: list[ValidationIssue] = []
    if not isinstance(itinerary, list):
        _add_issue(issues, "itinerary.invalid", "itinerary", "Itinerary must be a list.")
        return ValidationReport(issues=tuple(issues))
    _validate_strict_itinerary_shape(issues, itinerary)
    for duplicate_path, first_path in find_duplicate_activity_paths(itinerary):
        _add_issue(
            issues,
            "activity.duplicate",
            duplicate_path,
            f"Activity duplicates the earlier itinerary item at {first_path}.",
        )
    if not isinstance(maps, Mapping):
        _add_issue(issues, "maps.invalid", "maps", "Daily maps must be a mapping.")
        maps = {}

    expected_by_day = {expected.day: expected for expected in requirements.expected_days}
    maps = _canonical_daily_maps(issues, maps)
    expected_map_days = set(expected_by_day)
    actual_map_days = set(maps)
    for day_number in sorted(expected_map_days - actual_map_days):
        _add_issue(
            issues,
            "map.day.missing",
            f"maps[{day_number}]",
            f"Map for day {day_number} is missing.",
        )
    for day_number in sorted(actual_map_days - expected_map_days):
        _add_issue(
            issues,
            "map.day.unexpected",
            f"maps[{day_number}]",
            f"Map day {day_number} is outside the requested trip range.",
        )
    observed: dict[int, tuple[int, Mapping[str, Any]]] = {}
    observed_day_numbers: list[int] = []
    covered_city_keys: set[str] = set()
    outbound_evidence: _FlightEvidence | None = None
    return_evidence: _FlightEvidence | None = None

    for index, raw_day in enumerate(itinerary):
        path = f"itinerary[{index}]"
        if not isinstance(raw_day, Mapping):
            _add_issue(issues, "day.invalid", path, "Each itinerary day must be an object.")
            continue
        day_number = raw_day.get("day")
        if not _is_day_number(day_number):
            _add_issue(
                issues,
                "day.number.invalid",
                f"{path}.day",
                "Day must be a positive integer.",
            )
            continue
        observed_day_numbers.append(day_number)
        if day_number in observed:
            _add_issue(
                issues,
                "day.duplicate",
                f"{path}.day",
                f"Day {day_number} appears more than once.",
            )
            continue
        observed[day_number] = (index, raw_day)
        if day_number not in expected_by_day:
            _add_issue(
                issues,
                "day.unexpected",
                f"{path}.day",
                f"Day {day_number} is outside the requested trip range.",
            )

    expected_numbers = list(expected_by_day)
    if observed_day_numbers != expected_numbers:
        _add_issue(
            issues,
            "day.sequence.invalid",
            "itinerary",
            "Days must be unique and sequential for the requested trip.",
        )
    for day_number in expected_numbers:
        if day_number not in observed:
            _add_issue(
                issues,
                "day.missing",
                f"itinerary.day[{day_number}]",
                f"Expected day {day_number} is missing.",
            )

    for day_number, (index, day) in observed.items():
        expected_day = expected_by_day.get(day_number)
        day_path = f"itinerary[{index}]"
        if expected_day is not None and day.get("date") != expected_day.date:
            _add_issue(
                issues,
                "day.date.mismatch",
                f"{day_path}.date",
                f"Day {day_number} must use date {expected_day.date}.",
            )

        expected_points, activity_cost = _validate_day_activities(
            issues,
            requirements,
            day,
            day_path,
            covered_city_keys,
        )
        is_first_day = expected_day is not None and day_number == expected_numbers[0]
        is_last_day = expected_day is not None and day_number == expected_numbers[-1]
        requires_hotel_night = expected_day is not None and not is_last_day
        hotel_points, hotel_cost = _validate_hotel(
            issues,
            requirements,
            day,
            day_path,
            required=requires_hotel_night,
        )
        expected_points.extend(hotel_points)
        flight_cost, outbound, returning = _validate_day_flights(
            issues,
            day,
            day_path,
            expected_date=expected_day.date if expected_day is not None else None,
            is_first_day=is_first_day,
            is_last_day=is_last_day,
        )
        if outbound is not None:
            outbound_evidence = outbound
        if returning is not None:
            return_evidence = returning
        _validate_day_total(
            issues,
            day,
            day_path,
            expected_total=activity_cost + flight_cost + hotel_cost,
            reconcile=expected_day is not None,
        )
        if expected_day is not None:
            _validate_map_day(
                issues,
                maps,
                day_number,
                expected_points,
                day.get("route"),
                expected_airport_name=(
                    _arrival_airport_map_name(day) if is_first_day else None
                ),
            )

    if (
        outbound_evidence is not None
        and return_evidence is not None
        and (
            outbound_evidence[1] != return_evidence[2]
            or outbound_evidence[2] != return_evidence[1]
        )
    ):
        _add_issue(
            issues,
            "flight.direction.mismatch",
            "itinerary.boundary_flights",
            "Outbound and return airport identities must form a reversed route.",
        )

    for city_index, city in enumerate(requirements.cities):
        if _city_key(city) not in covered_city_keys:
            _add_issue(
                issues,
                "itinerary.city.coverage.missing",
                f"itinerary.city_coverage[{city_index}]",
                "Every requested city or area needs a verified activity.",
            )

    return ValidationReport(issues=tuple(issues))


def _validate_day_activities(
    issues: list[ValidationIssue],
    requirements: TripRequirements,
    day: Mapping[str, Any],
    day_path: str,
    covered_city_keys: set[str],
) -> tuple[list[_ExpectedPoint], float]:
    activities = day.get("activities")
    if not isinstance(activities, list) or not activities:
        _add_issue(
            issues,
            "day.activities.empty",
            f"{day_path}.activities",
            "Each expected day must contain at least one activity.",
        )
        return [], 0.0

    verified_count = 0
    points: list[_ExpectedPoint] = []
    activity_cost = 0.0
    observed_orders: list[int | None] = []
    configured_cities = {_city_key(city): city for city in requirements.cities}
    for activity_index, activity in enumerate(activities):
        activity_path = f"{day_path}.activities[{activity_index}]"
        if not isinstance(activity, Mapping):
            _add_issue(
                issues,
                "activity.invalid",
                activity_path,
                "Each activity must be an object.",
            )
            observed_orders.append(None)
            continue

        valid = True
        name = activity.get("name")
        if not _is_nonempty_text(name):
            valid = False
            _add_issue(
                issues,
                "activity.name.missing",
                f"{activity_path}.name",
                "Activity name must be non-empty.",
            )

        if activity.get("type") not in _ACTIVITY_TYPES:
            valid = False
            _add_issue(
                issues,
                "activity.type.unsupported",
                f"{activity_path}.type",
                "Activity type must be attraction or restaurant.",
            )
        if not _is_nonempty_text(activity.get("address")):
            valid = False
            _add_issue(
                issues,
                "activity.address.missing",
                f"{activity_path}.address",
                "Activity address must be non-empty.",
            )

        raw_cost = activity.get("estimated_cost")
        if (
            not _is_json_number(raw_cost)
            or not math.isfinite(raw_cost)
            or raw_cost < 0
        ):
            valid = False
            _add_issue(
                issues,
                "activity.cost.invalid",
                f"{activity_path}.estimated_cost",
                "Activity cost must be a finite non-negative number.",
            )
        else:
            activity_cost += float(raw_cost)

        raw_order = activity.get("order")
        if not _is_day_number(raw_order):
            valid = False
            observed_orders.append(None)
            _add_issue(
                issues,
                "activity.order.invalid",
                f"{activity_path}.order",
                "Activity order must be a positive integer.",
            )
        else:
            observed_orders.append(raw_order)

        location = activity.get("location")
        if not isinstance(location, Mapping):
            valid = False
            _add_issue(
                issues,
                "activity.location.missing",
                f"{activity_path}.location",
                "Activity location is required.",
            )
            continue
        if not _coordinates_are_finite(location):
            valid = False
            _add_issue(
                issues,
                "activity.location.coordinates.invalid",
                f"{activity_path}.location",
                "Activity coordinates must be finite numeric latitude and longitude.",
            )
        elif _is_nonempty_text(name):
            points.append(
                (
                    name.strip(),
                    float(location["longitude"]),
                    float(location["latitude"]),
                    str(activity.get("type")),
                    raw_order if isinstance(raw_order, int) else -1,
                )
            )
        if location.get("country_code") != requirements.destination_country_code:
            valid = False
            _add_issue(
                issues,
                "activity.country.mismatch",
                f"{activity_path}.location.country_code",
                "Activity country code must match the destination country.",
            )
        if not _is_nonempty_text(location.get("place_name")):
            valid = False
            _add_issue(
                issues,
                "activity.location.place_name.missing",
                f"{activity_path}.location.place_name",
                "Activity location name must be non-empty.",
            )
        requested_city = location.get("requested_city")
        requested_city_key: str | None = None
        if not _is_nonempty_text(requested_city):
            valid = False
            _add_issue(
                issues,
                "activity.city.missing",
                f"{activity_path}.location.requested_city",
                "Activity must carry server-bound requested-city evidence.",
            )
        else:
            requested_city_key = _city_key(requested_city)
            if requested_city_key not in configured_cities:
                valid = False
                _add_issue(
                    issues,
                    "activity.city.mismatch",
                    f"{activity_path}.location.requested_city",
                    "Activity city evidence must match a configured trip city.",
                )

        verified_locality = location.get("verified_locality")
        verified_locality_key: str | None = None

        if not _is_nonempty_text(verified_locality):
            valid = False

            _add_issue(
                issues,
                "activity.city.proof.missing",
                f"{activity_path}.location.verified_locality",
                "Activity must carry provider-grounded locality evidence.",
            )

        else:
            verified_locality_key = _city_key(
                verified_locality
            )

            locality_matches_requested_city = (
                requested_city_key is not None
                and verified_locality_key == requested_city_key
            )

            # Nearby metropolitan/day-trip places are also acceptable,
            # but only when the provider search supplied deterministic
            # distance evidence.
            raw_distance = location.get(
                "distance_from_requested_city_km"
            )

            nearby_city_proof = (
                requested_city_key is not None
                and _is_json_number(raw_distance)
                and math.isfinite(float(raw_distance))
                and 0.0
                <= float(raw_distance)
                <= MAX_ACTIVITY_CITY_DISTANCE_KM
            )

            if not (
                locality_matches_requested_city
                or nearby_city_proof
            ):
                valid = False

                _add_issue(
                    issues,
                    "activity.city.proof.mismatch",
                    f"{activity_path}.location.verified_locality",
                    (
                        "Provider locality must match the configured "
                        "city or be inside the permitted nearby "
                        "day-trip radius."
                    ),
                )

        # Count the CONFIGURED trip city as covered.
        #
        # For example:
        # requested_city    = Tokyo
        # verified_locality = Urayasu
        #
        # This still represents a Tokyo-based itinerary day.
        if valid and requested_city_key is not None:
            verified_count += 1
            covered_city_keys.add(requested_city_key)

    if observed_orders != list(range(1, len(activities) + 1)):
        _add_issue(
            issues,
            "day.activities.order.invalid",
            f"{day_path}.activities",
            "Activities must appear in strict sequential order starting at one.",
        )

    if not verified_count:
        _add_issue(
            issues,
            "day.activities.unverified",
            f"{day_path}.activities",
            "Each expected day needs a verified attraction or restaurant.",
        )
    return points, activity_cost


def _validate_hotel(
    issues: list[ValidationIssue],
    requirements: TripRequirements,
    day: Mapping[str, Any],
    day_path: str,
    *,
    required: bool,
) -> tuple[list[_ExpectedPoint], float]:
    hotel = day.get("hotel")
    if not isinstance(hotel, Mapping):
        if required:
            _add_issue(
                issues,
                "hotel.night.missing",
                f"{day_path}.hotel",
                "Each overnight stay requires verified accommodation.",
            )
        return [], 0.0
    if not required:
        _add_issue(
            issues,
            "hotel.unexpected",
            f"{day_path}.hotel",
            "Checkout and zero-night days must not contain a hotel object.",
        )
        return [], 0.0
    hotel_cost = 0.0
    price = hotel.get("price_per_night")
    if not _is_json_number(price) or not math.isfinite(price) or price < 0:
        _add_issue(
            issues,
            "hotel.price.invalid",
            f"{day_path}.hotel.price_per_night",
            "Hotel price must be a finite non-negative number.",
        )
    elif required:
        hotel_cost = float(price)
    location = hotel.get("location")
    if not isinstance(location, Mapping):
        _add_issue(
            issues,
            "hotel.location.missing",
            f"{day_path}.hotel.location",
            "Hotel location is required.",
        )
        return [], hotel_cost
    hotel_name = hotel.get("hotel_name")
    if not _is_nonempty_text(hotel_name):
        _add_issue(
            issues,
            "hotel.name.missing",
            f"{day_path}.hotel.hotel_name",
            "Hotel name must be non-empty.",
        )
    coordinates = _coordinates_from_location(location)
    if coordinates is None:
        _add_issue(
            issues,
            "hotel.location.coordinates.invalid",
            f"{day_path}.hotel.location",
            "Hotel coordinates must be finite numeric latitude and longitude.",
        )
        return [], hotel_cost
    country_code = location.get("country_code")
    if country_code != requirements.destination_country_code:
        _add_issue(
            issues,
            "hotel.country.mismatch",
            f"{day_path}.hotel.location.country_code",
            "Hotel country code must match the destination country.",
        )
    if not _is_nonempty_text(hotel_name):
        return [], hotel_cost
    latitude, longitude = coordinates
    return [(hotel_name.strip(), longitude, latitude, "hotel", 0)], hotel_cost


def _validate_day_flights(
    issues: list[ValidationIssue],
    day: Mapping[str, Any],
    day_path: str,
    *,
    expected_date: str | None,
    is_first_day: bool,
    is_last_day: bool,
) -> tuple[float, _FlightEvidence | None, _FlightEvidence | None]:
    raw_flights = day.get("flight")
    if raw_flights is None:
        flights: list[Any] = []
    elif isinstance(raw_flights, list):
        flights = raw_flights
    else:
        flights = []
        _add_issue(
            issues,
            "flight.list.invalid",
            f"{day_path}.flight",
            "Day flights must be a list.",
        )

    required_count = int(is_first_day) + int(is_last_day)
    if len(flights) != required_count:
        if is_first_day and not flights:
            _add_issue(
                issues,
                "flight.outbound.missing",
                f"{day_path}.flight",
                "The outbound boundary flight is required.",
            )
        if is_last_day and len(flights) < required_count:
            _add_issue(
                issues,
                "flight.return.missing",
                f"{day_path}.flight",
                "The return boundary flight is required.",
            )
        if not is_first_day and not is_last_day and flights:
            _add_issue(
                issues,
                "flight.boundary.unexpected",
                f"{day_path}.flight",
                "Flights are allowed only on trip boundary days.",
            )
        elif flights:
            _add_issue(
                issues,
                "flight.boundary.count.invalid",
                f"{day_path}.flight",
                "Boundary days must contain exactly the required flight legs.",
            )

    evidence: list[_FlightEvidence] = []
    total = 0.0
    for flight_index, flight in enumerate(flights):
        item_path = f"{day_path}.flight[{flight_index}]"
        item = _validate_flight(
            issues,
            flight,
            item_path,
            expected_date=expected_date,
        )
        if item is not None:
            evidence.append(item)
            total += item[0]

    outbound = evidence[0] if is_first_day and evidence else None
    if is_first_day and is_last_day:
        returning = evidence[1] if len(evidence) > 1 else None
    else:
        returning = evidence[0] if is_last_day and evidence else None
    return total, outbound, returning


def _validate_flight(
    issues: list[ValidationIssue],
    flight: Any,
    path: str,
    *,
    expected_date: str | None,
) -> _FlightEvidence | None:
    if not isinstance(flight, Mapping):
        _add_issue(issues, "flight.invalid", path, "Each flight must be an object.")
        return None

    valid = True
    price = flight.get("price")
    if not _is_json_number(price) or not math.isfinite(price) or price < 0:
        valid = False
        _add_issue(
            issues,
            "flight.price.invalid",
            f"{path}.price",
            "Flight price must be a finite non-negative number.",
        )

    departure = flight.get("departure_airport")
    arrival = flight.get("arrival_airport")
    departure_identity = _airport_identity(departure)
    arrival_identity = _airport_identity(arrival)
    if departure_identity is None:
        valid = False
        _add_issue(
            issues,
            "flight.departure.identity.missing",
            f"{path}.departure_airport",
            "Departure airport must have a provider identity.",
        )
    if arrival_identity is None:
        valid = False
        _add_issue(
            issues,
            "flight.arrival.identity.missing",
            f"{path}.arrival_airport",
            "Arrival airport must have a provider identity.",
        )

    departure_time = flight.get("departure_time")
    if not _is_nonempty_text(departure_time) and isinstance(departure, Mapping):
        departure_time = departure.get("time")
    if (
        expected_date is None
        or not _is_nonempty_text(departure_time)
        or not _matches_expected_date(departure_time, expected_date)
    ):
        valid = False
        _add_issue(
            issues,
            "flight.date.mismatch",
            f"{path}.departure_time",
            "Flight departure date must match its trip boundary date.",
        )

    if not valid:
        return None
    return float(price), departure_identity, arrival_identity


def _validate_day_total(
    issues: list[ValidationIssue],
    day: Mapping[str, Any],
    day_path: str,
    *,
    expected_total: float,
    reconcile: bool,
) -> None:
    total = day.get("day_total_cost")
    if not _is_json_number(total) or not math.isfinite(total) or total < 0:
        _add_issue(
            issues,
            "day.total_cost.invalid",
            f"{day_path}.day_total_cost",
            "Day total must be a finite non-negative number.",
        )
        return
    if reconcile and not math.isclose(
        float(total),
        round(expected_total, 2),
        rel_tol=0.0,
        abs_tol=_DAY_TOTAL_TOLERANCE,
    ):
        _add_issue(
            issues,
            "day.total_cost.mismatch",
            f"{day_path}.day_total_cost",
            "Day total must reconcile with flight, nightly hotel, and activity costs.",
        )


def _canonical_daily_maps(
    issues: list[ValidationIssue],
    maps: Mapping[Any, Any],
) -> dict[int, Any]:
    """Return collision-free canonical map days while recording every bad key."""
    canonical: dict[int, Any] = {}
    for raw_key, daily_map in maps.items():
        day_number: int | None = None
        if isinstance(raw_key, int) and not isinstance(raw_key, bool):
            if raw_key > 0:
                day_number = raw_key
        elif (
            isinstance(raw_key, str)
            and raw_key.isascii()
            and raw_key.isdigit()
            and raw_key != ""
        ):
            parsed = int(raw_key)
            if parsed > 0 and raw_key == str(parsed):
                day_number = parsed

        if day_number is None:
            _add_issue(
                issues,
                "map.day.key.invalid",
                f"maps[{raw_key!r}]",
                "Map day keys must be canonical positive integers.",
            )
            continue
        if day_number in canonical:
            _add_issue(
                issues,
                "map.day.key.duplicate",
                f"maps[{raw_key!r}]",
                "Map day keys must not collide after canonicalization.",
            )
            continue
        canonical[day_number] = daily_map
    return canonical


def _validate_strict_itinerary_shape(
    issues: list[ValidationIssue],
    itinerary: list[Any],
) -> None:
    """Repeat the closed public itinerary shape before candidate qualification.

    The import is deliberately lazy because the public response module reuses this
    deterministic validator for accepted-plan projection.
    """
    try:
        from app.schemas.responses import DailyItinerary

        TypeAdapter(list[DailyItinerary]).validate_python(itinerary)

    except ValidationError as exc:
        logger.warning(
            "Itinerary schema validation details: %s",
            exc.errors(include_url=False),
        )

        _add_issue(
            issues,
            "itinerary.schema.invalid",
            "itinerary",
            "Itinerary contains fields or values outside the closed public contract.",
        )

    except (TypeError, ValueError) as exc:
        logger.warning(
            "Itinerary schema validation error: %s",
            exc,
        )

        _add_issue(
            issues,
            "itinerary.schema.invalid",
            "itinerary",
            "Itinerary contains fields or values outside the closed public contract.",
        )


def _is_strict_geojson_feature(value: Any) -> bool:
    """Use the public feature model without creating a module import cycle."""
    try:
        from app.schemas.responses import GeoJSONFeature

        GeoJSONFeature.model_validate(value)
    except (TypeError, ValueError, ValidationError):
        return False
    return True


def _validate_map_day(
    issues: list[ValidationIssue],
    maps: Mapping[Any, Any],
    day_number: int,
    expected_points: Sequence[_ExpectedPoint],
    itinerary_route: Any,
    *,
    expected_airport_name: str | None,
) -> None:
    daily_map = maps.get(day_number)
    path = f"maps[{day_number}]"
    if daily_map is None:
        return
    if (
        not isinstance(daily_map, Mapping)
        or set(daily_map) != {"type", "features"}
        or daily_map.get("type") != "FeatureCollection"
        or not isinstance(daily_map.get("features"), list)
    ):
        _add_issue(
            issues,
            "map.day.invalid",
            path,
            "Map must be a GeoJSON FeatureCollection with features.",
        )
        return

    actual_points: list[_ExpectedPoint] = []
    actual_airport_points: list[tuple[str, int]] = []
    ordered_route_points: list[_RoutePoint] = []
    route_features: list[tuple[int, Mapping[str, Any]]] = []
    for feature_index, feature in enumerate(daily_map["features"]):
        if not _is_strict_geojson_feature(feature):
            _add_issue(
                issues,
                "map.feature.invalid",
                f"{path}.features[{feature_index}]",
                "Every map feature must match the closed public GeoJSON contract.",
            )
            continue
        assert isinstance(feature, Mapping)
        geometry = feature.get("geometry")
        properties = feature.get("properties")
        assert isinstance(geometry, Mapping)
        assert isinstance(properties, Mapping)
        geometry_type = geometry.get("type") if isinstance(geometry, Mapping) else None
        property_type = (
            properties.get("type") if isinstance(properties, Mapping) else None
        )
        if geometry_type == "LineString" or property_type == "route":
            route_features.append((feature_index, feature))
            if geometry_type != "LineString" or property_type != "route":
                _add_issue(
                    issues,
                    "map.route.invalid",
                    f"{path}.features[{feature_index}]",
                    "Route properties and LineString geometry must appear together.",
                )
            continue
        if geometry_type != "Point":
            continue
        name = properties.get("name")
        order = properties.get("order")
        coordinates = geometry.get("coordinates")
        if (
            not isinstance(coordinates, Sequence)
            or isinstance(coordinates, (str, bytes))
            or len(coordinates) < 2
            or not _is_json_number(coordinates[0])
            or not _is_json_number(coordinates[1])
            or not math.isfinite(coordinates[0])
            or not math.isfinite(coordinates[1])
        ):
            _add_issue(
                issues,
                "map.point.coordinates.invalid",
                f"{path}.features[{feature_index}].geometry.coordinates",
                "Map point coordinates must be finite numeric longitude and latitude.",
            )
            continue
        if _is_nonempty_text(name):
            route_point = (name.strip(), float(coordinates[0]), float(coordinates[1]))
            ordered_route_points.append(route_point)
            if property_type != "airport":
                assert isinstance(property_type, str)
                assert isinstance(order, int) and not isinstance(order, bool)
                actual_points.append((*route_point, property_type, order))
            else:
                assert isinstance(order, int) and not isinstance(order, bool)
                actual_airport_points.append((name.strip(), order))

    if Counter(actual_points) != Counter(expected_points):
        _add_issue(
            issues,
            "map.points.mismatch",
            path,
            "Map point names, coordinates, and counts must match the day's hotel and activities.",
        )

    if len(actual_airport_points) > 1 or (
        actual_airport_points
        and (
            expected_airport_name is None
            or actual_airport_points[0] != (expected_airport_name, 0)
        )
    ):
        _add_issue(
            issues,
            "map.airports.mismatch",
            path,
            "Map airport points must be the optional day-one arrival waypoint only.",
        )

    _validate_map_routes(
        issues,
        path,
        itinerary_route,
        ordered_route_points,
        route_features,
    )


def _validate_map_routes(
    issues: list[ValidationIssue],
    path: str,
    itinerary_route: Any,
    ordered_route_points: Sequence[_RoutePoint],
    route_features: Sequence[tuple[int, Mapping[str, Any]]],
) -> None:
    expected_profiles: dict[str, tuple[float, float]] = {}
    expected_stops: list[str] = []
    route_shape_valid = True
    if itinerary_route is not None:
        if not isinstance(itinerary_route, Mapping):
            route_shape_valid = False
        else:
            raw_stops = itinerary_route.get("ordered_stops")
            raw_profiles = itinerary_route.get("profiles")
            if not isinstance(raw_stops, list) or not all(
                _is_nonempty_text(stop) for stop in raw_stops
            ):
                route_shape_valid = False
            else:
                expected_stops = [stop.strip() for stop in raw_stops]
            if not isinstance(raw_profiles, Mapping):
                route_shape_valid = False
            else:
                for profile, metric in raw_profiles.items():
                    if profile not in _ROUTE_PROFILES:
                        route_shape_valid = False
                        continue
                    if metric is None:
                        continue
                    if not isinstance(metric, Mapping):
                        route_shape_valid = False
                        continue
                    distance = metric.get("distance_km")
                    duration = metric.get("duration_mins")
                    if not _is_finite_nonnegative_number(distance) or not (
                        _is_finite_nonnegative_number(duration)
                    ):
                        route_shape_valid = False
                        continue
                    expected_profiles[profile] = (float(distance), float(duration))

    if not route_shape_valid:
        _add_issue(
            issues,
            "itinerary.route.invalid",
            f"{path}.route",
            "Itinerary route metadata must use supported profiles and finite metrics.",
        )

    actual_stops = [point[0] for point in ordered_route_points]
    if itinerary_route is not None and expected_stops != actual_stops:
        _add_issue(
            issues,
            "map.route.stops.mismatch",
            path,
            "Route ordered stops must exactly match ordered map waypoints.",
        )

    actual_profiles: dict[str, tuple[float, float]] = {}
    observed_profile_count = 0
    for feature_index, feature in route_features:
        geometry = feature.get("geometry")
        properties = feature.get("properties")
        if (
            not isinstance(geometry, Mapping)
            or geometry.get("type") != "LineString"
            or not isinstance(properties, Mapping)
            or properties.get("type") != "route"
        ):
            continue
        observed_profile_count += 1
        profile = properties.get("profile")
        distance = properties.get("distance_km")
        duration = properties.get("duration_mins")
        coordinates = geometry.get("coordinates")
        feature_path = f"{path}.features[{feature_index}]"
        valid = True
        if profile not in _ROUTE_PROFILES or profile in actual_profiles:
            valid = False
        if not _is_finite_nonnegative_number(distance) or not (
            _is_finite_nonnegative_number(duration)
        ):
            valid = False
        if (
            not isinstance(coordinates, Sequence)
            or isinstance(coordinates, (str, bytes))
            or len(coordinates) < 2
            or any(_route_coordinate(point) is None for point in coordinates)
        ):
            valid = False
        if not valid:
            _add_issue(
                issues,
                "map.route.invalid",
                feature_path,
                "Every route feature needs one supported profile, finite metrics, and a valid LineString.",
            )
            continue

        assert isinstance(profile, str)
        actual_profiles[profile] = (float(distance), float(duration))
        first = _route_coordinate(coordinates[0])
        last = _route_coordinate(coordinates[-1])
        if (
            len(ordered_route_points) < 2
            or first != (ordered_route_points[0][1], ordered_route_points[0][2])
            or last != (ordered_route_points[-1][1], ordered_route_points[-1][2])
        ):
            _add_issue(
                issues,
                "map.route.endpoints.mismatch",
                feature_path,
                "Route geometry must start and end at the ordered map waypoints.",
            )

    if (
        observed_profile_count != len(expected_profiles)
        or set(actual_profiles) != set(expected_profiles)
    ):
        _add_issue(
            issues,
            "map.route.profiles.mismatch",
            path,
            "Map route profile count and names must match itinerary route profiles.",
        )

    for profile in actual_profiles.keys() & expected_profiles.keys():
        if actual_profiles[profile] != expected_profiles[profile]:
            _add_issue(
                issues,
                "map.route.metrics.mismatch",
                f"{path}.route[{profile}]",
                "Route distance and duration must exactly match itinerary metrics.",
            )


def _route_coordinate(value: Any) -> tuple[float, float] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
        or not _is_json_number(value[0])
        or not _is_json_number(value[1])
        or not math.isfinite(value[0])
        or not math.isfinite(value[1])
    ):
        return None
    return float(value[0]), float(value[1])


def _is_finite_nonnegative_number(value: Any) -> bool:
    return _is_json_number(value) and math.isfinite(value) and value >= 0


def _validated_budget_allocation(
    allocation: Mapping[str, Any],
    total_budget: float,
) -> dict[str, float]:
    if not isinstance(allocation, Mapping):
        raise ValueError("budget allocation must be a mapping")
    validated: dict[str, float] = {}
    for category, amount in allocation.items():
        if category not in _CANONICAL_BUDGET_CATEGORIES:
            raise ValueError(f"budget allocation category is not canonical: {category}")
        number = _finite_number(amount, f"budget allocation for {category}")
        if number < 0:
            raise ValueError("budget allocation values must be non-negative")
        validated[category] = number
    if not math.isclose(
        sum(validated.values()),
        total_budget,
        rel_tol=0.0,
        abs_tol=_BUDGET_TOTAL_TOLERANCE,
    ):
        raise ValueError("budget allocation must equal the total converted budget")
    return validated


def _required_text(value: Any, label: str) -> str:
    if not _is_nonempty_text(value):
        raise ValueError(f"{label} is required")
    return value.strip()


def _country_code(destination_country: str) -> str:
    try:
        return pycountry.countries.lookup(destination_country).alpha_2
    except LookupError as exc:
        raise ValueError("destination country must resolve to ISO 3166-1 alpha-2") from exc


def _currency_code(value: Any, label: str) -> str:
    currency = _required_text(value, label)
    if not _CURRENCY_CODE.fullmatch(currency):
        raise ValueError(f"{label} must be a three-letter ISO 4217 code")
    return currency


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def _coordinates_are_finite(location: Mapping[str, Any]) -> bool:
    return _coordinates_from_location(location, allow_aliases=False) is not None


def _coordinates_from_location(
    location: Mapping[str, Any],
    *,
    allow_aliases: bool = True,
) -> tuple[float, float] | None:
    latitude = location.get("latitude")
    longitude = location.get("longitude")
    if allow_aliases:
        latitude = latitude if latitude is not None else location.get("lat")
        longitude = longitude if longitude is not None else location.get("lng")
    if (
        not _is_json_number(latitude)
        or not _is_json_number(longitude)
        or not math.isfinite(latitude)
        or not math.isfinite(longitude)
    ):
        return None
    return float(latitude), float(longitude)


def _airport_identity(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    airport_id = value.get("id")
    if _is_nonempty_text(airport_id):
        return f"id:{airport_id.strip().upper()}"
    name = value.get("name")
    if _is_nonempty_text(name):
        return f"name:{' '.join(name.split()).casefold()}"
    return None


def _arrival_airport_map_name(day: Mapping[str, Any]) -> str | None:
    """Return the only optional airport waypoint name emitted by the map producer."""
    flights = day.get("flight")
    if not isinstance(flights, list) or not flights:
        return None
    first_flight = flights[0]
    if not isinstance(first_flight, Mapping):
        return None
    arrival = first_flight.get("arrival_airport")
    if not isinstance(arrival, Mapping):
        return None
    name = arrival.get("name")
    if _is_nonempty_text(name):
        return name.strip()
    airport_id = arrival.get("id")
    if _is_nonempty_text(airport_id):
        return f"{airport_id.strip()} airport"
    return None


def _matches_expected_date(value: str, expected_date: str) -> bool:
    normalized = value.strip()
    return normalized == expected_date or (
        normalized.startswith(expected_date)
        and len(normalized) > len(expected_date)
        and normalized[len(expected_date)] in {" ", "T"}
    )


def _city_key(value: str) -> str:
    return " ".join(value.split()).casefold()


def _is_json_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_day_number(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _add_issue(
    issues: list[ValidationIssue], code: str, path: str, message: str
) -> None:
    issues.append(ValidationIssue(code=code, path=path, message=message))
