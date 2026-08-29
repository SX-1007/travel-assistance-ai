"""Strict public response schemas and accepted-plan projection helpers."""

from __future__ import annotations

import math
from typing import Annotated, Any, Dict, List, Literal, Mapping, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)

from app.agents.state import AgentState
from app.services.itinerary_quality import (
    MAX_ACTIVITY_CITY_DISTANCE_KM,
    TripRequirements,
    validate_itinerary_candidate,
)


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
CountryCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]


def _finite_number(value: Any, *, field_name: str) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number")
    return value


# ──────────────────────────────────────────────
# Geospatial Building Blocks
# ──────────────────────────────────────────────
class MapLocation(BaseModel):
    """Geospatial coordinate with a human-readable place name."""

    model_config = ConfigDict(extra="forbid")

    place_name: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    @field_validator("latitude", "longitude", mode="before")
    @classmethod
    def _coordinates_are_finite(cls, value: Any, info: Any) -> Any:
        if value is None:
            return value
        return _finite_number(value, field_name=info.field_name)


class RouteData(BaseModel):
    """Routing metadata for a day's travel path."""

    model_config = ConfigDict(extra="forbid")

    distance_km: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    duration_mins: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)

    @field_validator("distance_km", "duration_mins", mode="before")
    @classmethod
    def _metrics_are_finite(cls, value: Any, info: Any) -> Any:
        if value is None:
            return value
        return _finite_number(value, field_name=info.field_name)


class RouteProfiles(BaseModel):
    """Closed set of Mapbox profiles emitted by the route producer."""

    model_config = ConfigDict(extra="forbid")

    driving: Optional[RouteData] = None
    walking: Optional[RouteData] = None
    cycling: Optional[RouteData] = None


class RouteInfo(BaseModel):
    """Public per-day route summary without provider geometry/tool data."""

    model_config = ConfigDict(extra="forbid")

    ordered_stops: List[str] = Field(default_factory=list)
    profiles: RouteProfiles = Field(default_factory=RouteProfiles)


class HotelLocation(BaseModel):
    """Closed public hotel coordinate shape emitted by the hotel producer."""

    model_config = ConfigDict(extra="forbid")

    lat: Optional[float] = None
    lng: Optional[float] = None
    country_code: Optional[CountryCode] = None

    @field_validator("lat", "lng", mode="before")
    @classmethod
    def _coordinates_are_finite(cls, value: Any, info: Any) -> Any:
        if value is None:
            return value
        return _finite_number(value, field_name=info.field_name)


class FlightAirport(BaseModel):
    """Closed SerpAPI airport subset consumed by the frontend."""

    model_config = ConfigDict(extra="forbid")

    id: str = ""
    name: str = ""
    time: str = ""


# ──────────────────────────────────────────────
# Search Result Models (aligned with data layer — FIX #26)
# ──────────────────────────────────────────────
class HotelResult(BaseModel):
    """Normalised hotel search result.

    Field names match flights_hotels.py output:
      • ``hotel_class`` (not ``rating``)
      • ``location``    (dict with lat/lng/address)
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    hotel_name: str = ""
    hotel_class: float = Field(default=0.0, ge=0.0, le=5.0)
    overall_rating: Optional[float] = None  # guest rating, e.g. 4.4
    reviews: Optional[int] = None  # number of guest reviews
    description: str = ""
    price_per_night: float = Field(default=0.0, ge=0.0)
    amenities: List[str] = Field(default_factory=list)
    check_in_time: str = ""
    check_out_time: str = ""
    location: Optional[HotelLocation] = None
    image: str = ""
    booking_url: str = ""
    over_budget: bool = False


class FlightResult(BaseModel):
    """Normalised flight search result.

    Field name ``flight_number`` matches flights_hotels.py output
    (not ``flight_num``).
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    airline: str = ""
    flight_number: str = ""
    airline_logo: str = ""
    travel_class: str = ""  # e.g. "Economy"
    airplane: str = ""  # aircraft model
    departure_airport: Optional[FlightAirport] = None
    arrival_airport: Optional[FlightAirport] = None
    departure_time: str = ""
    arrival_time: str = ""
    duration: int = Field(default=0, ge=0)  # total minutes, all legs
    stops: int = Field(default=0, ge=0)  # 0 = direct
    layovers: List[str] = Field(default_factory=list)  # layover airport names
    price: float = Field(default=0.0, ge=0.0)
    booking_url: Optional[str] = None
    over_budget: bool = False


# ──────────────────────────────────────────────
# Itinerary Models
# ──────────────────────────────────────────────
class ActivityItem(BaseModel):
    """A single activity within a day's itinerary."""

    model_config = ConfigDict(extra="forbid")

    time_of_day: str
    description: str
    estimated_cost: Optional[float] = Field(default=0.0, ge=0.0)
    location: Optional[MapLocation] = None


class ActivityLocation(BaseModel):
    """A provider-verified activity location safe for public rendering."""

    model_config = ConfigDict(extra="forbid")

    place_name: NonEmptyText
    country_code: CountryCode
    requested_city: NonEmptyText
    verified_locality: NonEmptyText

    latitude: float
    longitude: float

    # Optional proof for legitimate nearby metropolitan/day-trip
    # locations whose administrative locality differs from the
    # configured planning city.
    distance_from_requested_city_km: Optional[float] = Field(
        default=None,
        ge=0,
        le=MAX_ACTIVITY_CITY_DISTANCE_KM,
        allow_inf_nan=False,
    )

    @field_validator(
        "latitude",
        "longitude",
        mode="before",
    )
    @classmethod
    def _coordinates_are_finite(
        cls,
        value: Any,
        info: Any,
    ) -> Any:
        return _finite_number(
            value,
            field_name=info.field_name,
        )

    @model_validator(mode="after")
    def _locality_proves_requested_city(
        self,
    ) -> "ActivityLocation":
        requested = " ".join(
            self.requested_city.split()
        ).casefold()

        verified = " ".join(
            self.verified_locality.split()
        ).casefold()

        # Normal in-city attraction.
        if verified == requested:
            return self

        # Different administrative locality is acceptable only when
        # the provider search proved it is within the permitted
        # metropolitan/day-trip distance.
        if self.distance_from_requested_city_km is not None:
            return self

        raise ValueError(
            "verified locality must match the requested city "
            "or carry valid nearby-city distance evidence"
        )


class ActivityResult(BaseModel):
    """A complete, destination-verified activity in a public itinerary."""

    model_config = ConfigDict(extra="forbid")

    name: NonEmptyText
    type: Literal["attraction", "restaurant"]
    address: NonEmptyText
    estimated_cost: float = Field(ge=0, allow_inf_nan=False)
    order: int = Field(ge=1, strict=True)
    location: ActivityLocation
    description: Optional[str] = None
    category: Optional[str] = None
    rating: Optional[float] = Field(default=None, ge=0, le=5, allow_inf_nan=False)
    thumbnail: Optional[str] = None
    suggested_time: Optional[str] = None
    is_estimated: Optional[bool] = None

    @field_validator("estimated_cost", "rating", mode="before")
    @classmethod
    def _numbers_are_finite(cls, value: Any, info: Any) -> Any:
        if value is None and info.field_name == "rating":
            return value
        return _finite_number(value, field_name=info.field_name)


# ──────────────────────────────────────────────
# Closed GeoJSON response models
# ──────────────────────────────────────────────
class GeoJSONPointGeometry(BaseModel):
    """A two-dimensional GeoJSON point emitted by ``build_geojson``."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["Point"]
    coordinates: List[float] = Field(min_length=2, max_length=2)

    @field_validator("coordinates", mode="before")
    @classmethod
    def _coordinates_are_finite(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("Point coordinates must contain longitude and latitude")
        return [
            _finite_number(coordinate, field_name="coordinates")
            for coordinate in value
        ]


class GeoJSONLineStringGeometry(BaseModel):
    """A public Mapbox route line with only its coordinate path."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["LineString"]
    coordinates: List[List[float]] = Field(min_length=2)

    @field_validator("coordinates", mode="before")
    @classmethod
    def _coordinates_are_finite(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            raise ValueError("LineString coordinates must contain at least two points")
        normalized: list[list[float]] = []
        for point in value:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ValueError("Each LineString point must contain two coordinates")
            normalized.append(
                [
                    _finite_number(coordinate, field_name="coordinates")
                    for coordinate in point
                ]
            )
        return normalized


GeoJSONGeometry = Annotated[
    Union[GeoJSONPointGeometry, GeoJSONLineStringGeometry],
    Field(discriminator="type"),
]


class GeoJSONProperties(BaseModel):
    """Closed properties emitted for waypoint pins and route lines."""

    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    type: Literal[
        "activity",
        "attraction",
        "restaurant",
        "hotel",
        "airport",
        "route",
    ]
    order: Optional[int] = Field(default=None, ge=0, strict=True)
    profile: Optional[Literal["driving", "walking", "cycling"]] = None
    distance_km: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    duration_mins: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)

    @field_validator("distance_km", "duration_mins", mode="before")
    @classmethod
    def _metrics_are_finite(cls, value: Any, info: Any) -> Any:
        if value is None:
            return value
        return _finite_number(value, field_name=info.field_name)

    @model_serializer(mode="wrap")
    def _preserve_producer_shape(self, handler: Any) -> dict[str, Any]:
        serialized = handler(self)
        return {
            key: value
            for key, value in serialized.items()
            if key in self.model_fields_set
        }


class GeoJSONFeature(BaseModel):
    """Closed GeoJSON feature containing only producer-supported fields."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["Feature"]
    geometry: GeoJSONGeometry
    properties: GeoJSONProperties

    @model_validator(mode="after")
    def _geometry_matches_properties(self) -> GeoJSONFeature:
        route_geometry = self.geometry.type == "LineString"
        route_properties = self.properties.type == "route"
        if route_geometry != route_properties:
            raise ValueError("route properties require LineString geometry")
        property_fields = self.properties.model_fields_set
        if route_properties:
            if property_fields != {
                "type",
                "profile",
                "distance_km",
                "duration_mins",
            }:
                raise ValueError("route properties must use the exact public shape")
            if (
                self.properties.profile is None
                or self.properties.distance_km is None
                or self.properties.duration_mins is None
            ):
                raise ValueError("route properties require profile and metrics")
        else:
            if property_fields != {"name", "type", "order"}:
                raise ValueError("point properties must use the exact public shape")
            if not isinstance(self.properties.name, str) or not self.properties.name.strip():
                raise ValueError("point properties require a non-empty name")
            if self.properties.order is None:
                raise ValueError("point properties require an integer order")
        return self


class GeoJSONFeatureCollection(BaseModel):
    """Closed daily map payload safe for public serialization."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["FeatureCollection"]
    features: List[GeoJSONFeature] = Field(default_factory=list)


def canonicalize_daily_map_keys(value: Any) -> dict[int, Any]:
    """Normalize positive canonical day keys and reject ambiguous collisions."""
    if not isinstance(value, Mapping):
        raise ValueError("daily maps must be a mapping")

    canonical: dict[int, Any] = {}
    for raw_key, daily_map in value.items():
        if isinstance(raw_key, bool):
            raise ValueError("daily map keys must be positive integers")
        if isinstance(raw_key, int):
            day = raw_key
        elif (
            isinstance(raw_key, str)
            and raw_key.isascii()
            and raw_key.isdigit()
            and raw_key == str(int(raw_key))
        ):
            day = int(raw_key)
        else:
            raise ValueError("daily map keys must be canonical positive integers")
        if day <= 0:
            raise ValueError("daily map keys must be positive integers")
        if day in canonical:
            raise ValueError("daily map keys collide after normalization")
        canonical[day] = daily_map
    return canonical


def canonicalize_public_daily_maps(value: Any) -> dict[int, Any]:
    """Canonicalize raw maps while preserving an already-projected map object."""
    if isinstance(value, dict) and all(
        isinstance(day, int)
        and not isinstance(day, bool)
        and day > 0
        and isinstance(daily_map, GeoJSONFeatureCollection)
        for day, daily_map in value.items()
    ):
        return value
    return canonicalize_daily_map_keys(value)


class DailyItinerary(BaseModel):
    """A single day's travel plan including transport, lodging, and activities.

    Contains both ``route`` and ``route_metrics`` keys to align with
    mapbox.py output (FIX #26).
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    day: int = Field(ge=1)
    date: Optional[str] = None
    flight: Optional[List[FlightResult]] = None
    hotel: Optional[HotelResult] = None
    activities: List[ActivityResult] = Field(default_factory=list)
    route: Optional[RouteInfo] = None
    day_total_cost: float = Field(default=0.0, ge=0.0)


# ──────────────────────────────────────────────
# Top-Level API Response
# ──────────────────────────────────────────────
class FinalResponse(BaseModel):
    """Top-level API response for form submission and chat endpoints.

    Used as ``response_model`` in FastAPI route decorators for
    automatic validation and OpenAPI schema generation (FIX #33).
    """

    model_config = ConfigDict(extra="forbid")

    status: Literal["success"]
    chat_reply: NonEmptyText
    itinerary: List[DailyItinerary]
    daily_geojson_maps: Dict[int, GeoJSONFeatureCollection]
    destination_country_code: CountryCode
    # Verified city/cities actually used for planning.
    resolved_cities: List[NonEmptyText] = Field(min_length=1)
    # Budget context so the frontend can label the per-category allocation
    # next to the itinerary (amounts in the destination currency).
    total_budget: float = 0.0
    currency: str = ""
    budget_allocation: Dict[str, float] = Field(default_factory=dict)
    session_id: str = Field(
        ...,
        description=(
            "Identifier of the underlying LangGraph thread. The frontend "
            "MUST echo this value back as `session_id` in subsequent "
            "`/api/chat/message` requests to maintain conversation continuity."
        ),
    )

    @field_validator("daily_geojson_maps", mode="before")
    @classmethod
    def _canonical_map_keys(cls, value: Any) -> dict[int, Any]:
        return canonicalize_public_daily_maps(value)


class BudgetEvidenceResponse(BaseModel):
    """Provider price components used in the deterministic minimum."""

    model_config = ConfigDict(extra="forbid")

    outbound_flight_price: float = Field(gt=0, allow_inf_nan=False)
    return_flight_price: float = Field(gt=0, allow_inf_nan=False)
    hotel_price_per_night: float = Field(ge=0, allow_inf_nan=False)
    hotel_nights: int = Field(ge=0)

    @model_validator(mode="after")
    def _hotel_price_matches_nights(self) -> "BudgetEvidenceResponse":
        if (self.hotel_nights == 0) != (self.hotel_price_per_night == 0):
            raise ValueError("hotel nights and price must use the same zero-night semantics")
        return self


class BudgetConfirmationResponse(BaseModel):
    """Grounded minimum that requires an explicit user decision."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["budget_confirmation_required"]
    reason: Literal["insufficient_budget", "recommendation_requested"]
    chat_reply: str
    budget_assessment_id: str

    # Server-owned verified city list. The frontend must echo this list
    # back when the user confirms the budget assessment.
    resolved_cities: List[NonEmptyText] = Field(min_length=1)

    stated_budget: Optional[float] = None
    recommended_minimum_budget: float = Field(
        gt=0,
        allow_inf_nan=False,
    )
    base_currency: str
    destination_currency: str
    expires_at: str
    evidence: BudgetEvidenceResponse
    itinerary: None = None
    daily_geojson_maps: None = None


class BudgetCheckUnavailableResponse(BaseModel):
    """Fail-closed result when no reusable grounded assessment exists."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["budget_check_unavailable"]
    reason: Literal[
        "provider_data_unavailable",
        "assessment_cache_unavailable",
        "assessment_expired_or_invalid",
        "destination_resolution_unavailable",
    ]
    chat_reply: str
    itinerary: None = None
    daily_geojson_maps: None = None


PlanningUnavailableReason = Literal[
    "validation_failed",
    "provider_data_unavailable",
    "review_unavailable",
    "deadline_exhausted",
]


class PlanningUnavailableResponse(BaseModel):
    """Fail-closed initial-planning result with no itinerary payload."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["planning_unavailable"]
    reason: PlanningUnavailableReason
    chat_reply: str
    retryable: Literal[True] = True
    itinerary: None = None
    daily_geojson_maps: None = None


class AcceptedPlanProjection(BaseModel):
    """Internal strict projection of one revalidated accepted snapshot."""

    model_config = ConfigDict(extra="forbid")

    plan_revision: int = Field(ge=1)
    total_base_budget: float = Field(ge=0, allow_inf_nan=False)
    base_currency_code: str
    dest_currency_code: str
    destination_country_code: CountryCode
    total_convert_budget: float = Field(ge=0, allow_inf_nan=False)
    budget_allocation: Dict[str, float]
    draft_itinerary: List[DailyItinerary]
    daily_map_info: Dict[int, GeoJSONFeatureCollection]


_ACCEPTED_SNAPSHOT_FIELDS = frozenset(
    {
        "plan_revision",
        "total_base_budget",
        "base_currency_code",
        "dest_currency_code",
        "total_convert_budget",
        "budget_allocation",
        "draft_itinerary",
        "daily_map_info",
    }
)
def validated_accepted_plan(
    state: Mapping[str, Any],
) -> AcceptedPlanProjection | None:
    """Return only a current accepted snapshot that passes fresh validation."""
    raw = state.get("accepted_plan_snapshot")
    if not isinstance(raw, dict) or set(raw) != _ACCEPTED_SNAPSHOT_FIELDS:
        return None
    revision = raw.get("plan_revision")
    if (
        isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision <= 0
        or revision != state.get("plan_revision")
    ):
        return None

    try:
        canonical_maps = canonicalize_daily_map_keys(raw.get("daily_map_info"))
        requirements = TripRequirements.from_state(AgentState.model_validate(state))
        report = validate_itinerary_candidate(
            requirements,
            raw.get("draft_itinerary"),
            canonical_maps,
        )
        if not report.qualified:
            return None

        if (
            raw.get("base_currency_code") != requirements.base_currency
            or raw.get("dest_currency_code") != requirements.destination_currency
            or raw.get("total_convert_budget") != requirements.total_budget
            or raw.get("budget_allocation") != dict(requirements.budget_allocation)
            or raw.get("total_base_budget") != state.get("total_base_budget")
        ):
            return None

        return AcceptedPlanProjection.model_validate(
            {
                "plan_revision": raw.get("plan_revision"),
                "total_base_budget": raw.get("total_base_budget"),
                "base_currency_code": raw.get("base_currency_code"),
                "dest_currency_code": raw.get("dest_currency_code"),
                "destination_country_code": requirements.destination_country_code,
                "total_convert_budget": raw.get("total_convert_budget"),
                "budget_allocation": raw.get("budget_allocation"),
                "draft_itinerary": raw.get("draft_itinerary"),
                "daily_map_info": canonical_maps,
            }
        )
    except (TypeError, ValueError, ValidationError):
        return None


TripSubmissionResponse = Union[
    FinalResponse,
    BudgetConfirmationResponse,
    BudgetCheckUnavailableResponse,
    PlanningUnavailableResponse,
]
