"""
Mapbox geocoding + routing utilities.

Optimisation highlights
-----------------------
* Single module-level `ThreadPoolExecutor` (cap 8) — eliminates the nested-
  executor thread explosion flagged in issue #28.
* HTTPAdapter-backed `requests.Session` with `urllib3.Retry` for transport-
  level resilience.
* TTLCache (2048 entries, 24 h) for geocodes — positive AND negative results.
* Transient (network) failures are NOT cached; permanent (empty) results are.
* Defensive `copy.deepcopy` in `_process_day` — no mutation of input state (#46).
* `atexit`-registered executor shutdown for clean process exit.
"""

from __future__ import annotations

import atexit
import copy
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Annotated, Any, Optional

import requests
from cachetools import TTLCache
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.agents.state import AgentState
from app.core.config import settings
from app.tools.provider_logging import safe_provider_log

logger = logging.getLogger(__name__)

MAPBOX_TOKEN = settings.MAPBOX_TOKEN
SERPAPI_PLACE_KEY = settings.SERPAPI_PLACE

# Bounded concurrency for ALL Mapbox workloads (single global executor)
_MAPBOX_MAX_CONCURRENT = 8

REQUEST_TIMEOUT = 10
MAX_RETRIES = 3
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}

# ─────────────────────────────────────────────────────────────────────────────
# HTTP session — connection pool + transport-level retry
# ─────────────────────────────────────────────────────────────────────────────
_RETRY_STRATEGY = Retry(
    total=3,
    # Read timeouts are retried ONCE at transport level. Long/unroutable
    # Directions requests (e.g. a 500 km walking leg) exceed the 10 s read
    # timeout every time; with read=3 the transport retried 3× and the app
    # loop retried the whole thing 3× more — up to ~90 s of stalling PER
    # profile, which froze the whole request.
    read=1,
    backoff_factor=0.5,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset({"GET"}),
    raise_on_status=False,
)
_session: requests.Session = requests.Session()
_adapter = HTTPAdapter(
    pool_connections=10, pool_maxsize=20, max_retries=_RETRY_STRATEGY
)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)

# ─────────────────────────────────────────────────────────────────────────────
# Geocode cache — thread-safe TTL cache (positive + negative)
# ─────────────────────────────────────────────────────────────────────────────
_geocode_cache: TTLCache = TTLCache(maxsize=2048, ttl=86_400)  # 24 h
_geocode_lock = Lock()

# Reverse country lookups have different keys and caching semantics from
# forward geocoding, so keep them isolated. ``None`` means a successful API
# response that permanently contained no usable country evidence.
_country_cache: TTLCache = TTLCache(maxsize=4096, ttl=86_400)  # 24 h
_country_lock = Lock()

# Locality evidence is destination-bound. A coordinate accepted for one
# country/city query must never be reused as proof for a different trip.
_locality_cache: TTLCache = TTLCache(maxsize=4096, ttl=86_400)  # 24 h
_locality_lock = Lock()

# ─────────────────────────────────────────────────────────────────────────────
# Lazy singleton executor (with atexit cleanup)
# ─────────────────────────────────────────────────────────────────────────────
_executor: Optional[ThreadPoolExecutor] = None
_executor_lock = Lock()


def _get_executor() -> ThreadPoolExecutor:
    """Double-checked-locking singleton — avoids per-call executor overhead."""
    global _executor
    if _executor is not None:
        return _executor
    with _executor_lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=_MAPBOX_MAX_CONCURRENT,
                thread_name_prefix="mapbox",
            )
            atexit.register(_executor.shutdown, wait=False)
    return _executor


# ─────────────────────────────────────────────────────────────────────────────
# HTTP request with manual exponential backoff (for non-retryable status codes
# not handled by urllib3 — e.g. parse errors after a 200)
# ─────────────────────────────────────────────────────────────────────────────
def _request_with_retries(
    url: str,
    params: dict,
) -> Optional[requests.Response]:
    """HTTP GET with manual exponential backoff. Returns Response or None."""
    for attempt in range(MAX_RETRIES):
        try:
            response = _session.get(url, params=params, timeout=REQUEST_TIMEOUT)

            if response.status_code in RETRY_STATUS_CODES and attempt < MAX_RETRIES - 1:
                wait = 2**attempt
                safe_provider_log(
                    logger,
                    "provider.mapbox.retry_status",
                    status_code=response.status_code,
                )
                time.sleep(wait)
                continue
            return response

        except (requests.Timeout, requests.ConnectionError):
            if attempt < MAX_RETRIES - 1:
                wait = 2**attempt
                safe_provider_log(logger, "provider.mapbox.retry_transport")
                time.sleep(wait)
            else:
                safe_provider_log(logger, "provider.mapbox.failed")
                return None

    return None


def _cache_result(key: str, value: Optional[dict]) -> None:
    with _geocode_lock:
        _geocode_cache[key] = value


# ─────────────────────────────────────────────────────────────────────────────
# Geocoding
# ─────────────────────────────────────────────────────────────────────────────
def geocode_location(
    location_name: str,
    proximity_bias: Optional[list[float]] = None,
) -> Optional[dict]:
    """Geocode a place name via Mapbox Search Box API. Returns dict or None."""
    if not location_name or not location_name.strip():
        return None

    clean_name = location_name.lower().strip()
    cache_key = f"{clean_name}|{proximity_bias or ''}"

    with _geocode_lock:
        # Membership is required here: `.get()` cannot distinguish a cached
        # negative result (None) from an absent key.
        if cache_key in _geocode_cache:
            return _geocode_cache[cache_key]

    url = "https://api.mapbox.com/search/searchbox/v1/forward"
    params: dict[str, Any] = {
        "q": location_name,
        "access_token": MAPBOX_TOKEN,
        "limit": 1,
    }
    if proximity_bias and len(proximity_bias) == 2:
        params["proximity"] = f"{proximity_bias[0]},{proximity_bias[1]}"

    response = _request_with_retries(url, params)

    # Transient failure — do NOT cache, allow retry on next call
    if response is None:
        return None

    if response.status_code != 200:
        safe_provider_log(
            logger,
            "provider.mapbox.geocode_status",
            status_code=response.status_code,
        )
        _cache_result(cache_key, None)  # permanent failure — cache negative
        return None

    data = response.json()
    features = data.get("features") or []
    if not features:
        safe_provider_log(logger, "provider.mapbox.geocode_empty")
        _cache_result(cache_key, None)
        return None

    feature = features[0]
    props = feature.get("properties") or {}
    geom = feature.get("geometry") or {}
    coords = geom.get("coordinates")

    if not coords or len(coords) < 2:
        _cache_result(cache_key, None)
        return None

    name = props.get("name", location_name)
    address_context = props.get("place_formatted", "")
    full_place_name = f"{name}, {address_context}" if address_context else name

    result = {"lng": coords[0], "lat": coords[1], "place_name": full_place_name}
    _cache_result(cache_key, result)
    return result


def resolve_country_code(latitude: float, longitude: float) -> str | None:
    """Return the ISO 3166-1 alpha-2 country code for coordinates.

    Mapbox Geocoding v6 is used for global reverse coverage. Successful empty
    responses are cached, while transport/provider failures remain retryable.
    """
    if isinstance(latitude, bool) or isinstance(longitude, bool):
        return None
    try:
        lat = float(latitude)
        lng = float(longitude)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(lat) or not math.isfinite(lng):
        return None

    cache_key = (round(lat, 5), round(lng, 5))
    with _country_lock:
        if cache_key in _country_cache:
            return _country_cache[cache_key]

    url = "https://api.mapbox.com/search/geocode/v6/reverse"
    params = {
        "longitude": longitude,
        "latitude": latitude,
        "types": "country",
        "limit": 1,
        "language": "en",
        "access_token": MAPBOX_TOKEN,
    }
    response = _request_with_retries(url, params)
    if response is None or response.status_code != 200:
        return None

    try:
        features = response.json().get("features") or []
    except (AttributeError, ValueError):
        return None

    country_code: str | None = None
    if features and isinstance(features[0], dict):
        properties = features[0].get("properties") or {}
        context = properties.get("context") or {}
        country = context.get("country") or {}
        raw_code = country.get("country_code")
        if raw_code is None and properties.get("feature_type") == "country":
            raw_code = properties.get("country_code")
        if isinstance(raw_code, str):
            normalized = raw_code.strip().upper()
            if len(normalized) == 2 and normalized.isalpha():
                country_code = normalized

    with _country_lock:
        _country_cache[cache_key] = country_code
    return country_code


def resolve_locality_names(
    latitude: float,
    longitude: float,
    destination_country_code: str,
) -> tuple[str, ...] | None:
    """Return structured locality/admin names that contain the coordinate.

    ``None`` means the provider boundary was unavailable and remains retryable;
    an empty tuple means Mapbox returned no usable, destination-country-bound
    locality proof. Free-form formatted addresses and preferred-name aliases are
    intentionally excluded from evidence.
    """
    if isinstance(latitude, bool) or isinstance(longitude, bool):
        return ()
    try:
        lat = float(latitude)
        lng = float(longitude)
    except (TypeError, ValueError):
        return ()
    if not math.isfinite(lat) or not math.isfinite(lng):
        return ()
    if (
        not isinstance(destination_country_code, str)
        or len(destination_country_code.strip()) != 2
        or not destination_country_code.strip().isalpha()
    ):
        return ()
    country_code = destination_country_code.strip().upper()

    cache_key = (round(lat, 5), round(lng, 5), country_code)
    with _locality_lock:
        if cache_key in _locality_cache:
            return _locality_cache[cache_key]

    url = "https://api.mapbox.com/search/geocode/v6/reverse"
    params = {
        "longitude": longitude,
        "latitude": latitude,
        "limit": 1,
        "language": "en",
        "country": country_code.lower(),
        "access_token": MAPBOX_TOKEN,
    }
    response = _request_with_retries(url, params)
    if response is None or response.status_code != 200:
        return None
    try:
        features = response.json().get("features") or []
    except (AttributeError, ValueError):
        return None

    names: list[str] = []
    if features and isinstance(features[0], dict):
        properties = features[0].get("properties") or {}
        context = properties.get("context") or {}
        country = context.get("country") or {}
        raw_country = country.get("country_code")
        if isinstance(raw_country, str) and raw_country.strip().upper() == country_code:
            feature_type = properties.get("feature_type")
            if feature_type in {"region", "district", "place", "locality", "neighborhood"}:
                own_name = properties.get("name")
                if isinstance(own_name, str) and own_name.strip():
                    names.append(own_name.strip())
            for layer in ("neighborhood", "locality", "place", "district", "region"):
                component = context.get(layer) or {}
                name = component.get("name") if isinstance(component, dict) else None
                if isinstance(name, str) and name.strip():
                    names.append(name.strip())

    unique_names: list[str] = []
    observed: set[str] = set()
    for name in names:
        key = " ".join(name.split()).casefold()
        if key not in observed:
            observed.add(key)
            unique_names.append(" ".join(name.split()))
    result = tuple(unique_names)
    with _locality_lock:
        _locality_cache[cache_key] = result
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────────────────────────────────────
# Travel profiles surfaced to the user so they can toggle walking/driving/cycling
# views of the same day's route (project requirement #6).
ROUTE_PROFILES: tuple[str, ...] = ("driving", "walking", "cycling")

# Straight-line distance caps per profile. Requesting a walking route across
# ~500 km (e.g. Osaka → Tokyo day transition) makes the Directions API grind
# past the read timeout on every attempt — skip such profiles up front.
# Driving is uncapped.
_PROFILE_MAX_STRAIGHT_KM: dict[str, float] = {"walking": 30.0, "cycling": 80.0}


def _haversine_km(a: list[float], b: list[float]) -> float:
    """Great-circle distance in km between two [lng, lat] points."""
    import math

    lng1, lat1 = a
    lng2, lat2 = b
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = rlat2 - rlat1
    dlng = math.radians(lng2 - lng1)
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlng / 2) ** 2
    )
    return 6371.0 * 2 * math.asin(math.sqrt(h))


def _path_straight_km(coordinates: list[list[float]]) -> float:
    """Sum of straight-line leg distances along the ordered waypoints."""
    return sum(
        _haversine_km(coordinates[i], coordinates[i + 1])
        for i in range(len(coordinates) - 1)
    )


def _extract_coord(loc: Optional[dict]) -> Optional[list[float]]:
    """Return ``[lng, lat]`` from a location dict, or None.

    Handles both key styles used in the codebase:
      • hotels  → {"lat": .., "lng": ..}            (flights_hotels.py)
      • places  → {"latitude": .., "longitude": ..} (attractions.py)
    """
    if not isinstance(loc, dict):
        return None
    lat = loc.get("latitude", loc.get("lat"))
    lng = loc.get("longitude", loc.get("lng"))
    if lat is None or lng is None:
        return None
    try:
        return [float(lng), float(lat)]
    except (TypeError, ValueError):
        return None


def _fetch_route_single(coordinates: list[list[float]], profile: str) -> dict:
    """Fetch one Mapbox Directions route for a single travel profile."""
    empty = {"distance_km": 0, "duration_mins": 0, "geometry_for_map": None}
    if len(coordinates) < 2:
        return empty

    str_coords = ";".join(f"{c[0]},{c[1]}" for c in coordinates)
    url = f"https://api.mapbox.com/directions/v5/mapbox/{profile}/{str_coords}"
    params = {
        "access_token": MAPBOX_TOKEN,
        "geometries": "geojson",
        "overview": "full",
    }

    response = _request_with_retries(url, params)
    if not response or response.status_code != 200:
        if response is not None:
            safe_provider_log(
                logger,
                "provider.mapbox.route_status",
                status_code=response.status_code,
            )
        return empty

    routes = response.json().get("routes") or []
    if not routes:
        return empty

    route = routes[0]

    geometry = copy.deepcopy(route.get("geometry"))

    if (
        isinstance(geometry, dict)
        and geometry.get("type") == "LineString"
        and isinstance(geometry.get("coordinates"), list)
        and len(geometry["coordinates"]) >= 2
    ):
        # Mapbox may snap route endpoints to the nearby routable road.
        # Keep the provider route but bind its public endpoints to the
        # exact server-owned ordered waypoints.
        geometry["coordinates"][0] = [
            float(coordinates[0][0]),
            float(coordinates[0][1]),
        ]
        geometry["coordinates"][-1] = [
            float(coordinates[-1][0]),
            float(coordinates[-1][1]),
        ]

    return {
        "distance_km": round(route["distance"] / 1000, 2),
        "duration_mins": round(route["duration"] / 60, 2),
        "geometry_for_map": geometry,
    }


def fetch_routes_multi(
    coordinates: list[list[float]],
    profiles: tuple[str, ...] = ROUTE_PROFILES,
) -> dict[str, dict]:
    """Fetch routes for several travel profiles over the SAME ordered waypoints.

    Called sequentially per profile (the outer executor in ``generate_daily_map``
    already parallelises across days, and nesting into the shared bounded pool
    would risk exhaustion). Returns ``{profile: {distance_km, duration_mins,
    geometry_for_map}}`` for profiles that produced a valid route.
    """
    if len(coordinates) < 2:
        return {}
    straight_km = _path_straight_km(coordinates)
    out: dict[str, dict] = {}
    for profile in profiles:
        cap = _PROFILE_MAX_STRAIGHT_KM.get(profile)
        if cap is not None and straight_km > cap:
            safe_provider_log(logger, "provider.mapbox.route_distance_capped")
            continue
        r = _fetch_route_single(coordinates, profile)
        if r.get("geometry_for_map") is not None:
            out[profile] = r
    return out


def build_geojson(
    waypoints: list[dict],
    routes_by_profile: Optional[dict[str, dict]] = None,
) -> dict:
    """Construct a GeoJSON FeatureCollection: ordered point pins + one route
    LineString per travel profile (tagged with ``profile`` so the frontend can
    toggle walking/driving/cycling)."""
    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": wp["coordinates"]},
            "properties": {
                "name": wp.get("name", "Waypoint"),
                "type": wp.get("type", "activity"),
                "order": wp.get("order"),
            },
        }
        for wp in waypoints
    ]
    for profile, route in (routes_by_profile or {}).items():
        geom = route.get("geometry_for_map")
        if geom:
            features.append(
                {
                    "type": "Feature",
                    "geometry": geom,
                    "properties": {
                        "type": "route",
                        "profile": profile,
                        "distance_km": route.get("distance_km"),
                        "duration_mins": route.get("duration_mins"),
                    },
                }
            )
    return {"type": "FeatureCollection", "features": features}


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic utilities
# ─────────────────────────────────────────────────────────────────────────────
def _day_one_airport_waypoint(day_plan: dict) -> Optional[dict]:
    """For day 1, derive an arrival-airport waypoint from the flight (best-effort).

    Continuity rule (requirement #6): the traveller arrives from the airport on
    day 1, so the day's route should start there. Airport coords aren't in the
    flight payload, so we geocode the airport name/IATA. Failure is non-fatal.
    """
    flights = day_plan.get("flight")
    flight = None
    if isinstance(flights, list) and flights:
        flight = flights[0]
    elif isinstance(flights, dict):
        flight = flights
    if not isinstance(flight, dict):
        return None

    arr = flight.get("arrival_airport") or {}
    name = arr.get("name") or (
        f"{arr.get('id', '')} airport".strip() if arr.get("id") else ""
    )
    if not name:
        return None
    geo = geocode_location(name)
    if not geo:
        return None
    return {
        "name": name,
        "coordinates": [geo["lng"], geo["lat"]],
        "type": "airport",
        "order": 0,
    }


def _process_day(day_index: int, day_plan: dict) -> tuple[int, dict, dict]:
    """Process one day: assemble ordered waypoints from coordinates, fetch
    multi-profile routes, build GeoJSON.

    Waypoint order (the logical daily route, NOT shortest-path):
      day 1 : [arrival airport?] → hotel → activities (in their given order)
      day N : hotel → activities (in their given order)

    Coordinates are taken DIRECTLY from the hotel/activity payloads (SerpAPI
    already returns them), so no per-place geocoding is needed — this fixes the
    previous bug where hotels had no ``place_name`` and never appeared on the map.
    """
    # Defensive copy — never mutate the caller's state (issue #46)
    day_plan = copy.deepcopy(day_plan)
    day_num = day_plan.get("day", day_index + 1)

    waypoints: list[dict] = []

    # ── Day-1 arrival airport (best-effort) ──
    if day_num == 1:
        airport_wp = _day_one_airport_waypoint(day_plan)
        if airport_wp:
            waypoints.append(airport_wp)

    # ── Hotel (anchor: every day starts/ends around the hotel) ──
    hotel = day_plan.get("hotel")
    if isinstance(hotel, dict):
        hcoord = _extract_coord(hotel.get("location"))
        if hcoord:
            waypoints.append(
                {
                    "name": hotel.get("hotel_name", "Hotel"),
                    "coordinates": hcoord,
                    "type": "hotel",
                    "order": 0,
                }
            )

    # ── Activities in their logical (LLM-assigned) order ──
    for act in day_plan.get("activities", []) or []:
        coord = _extract_coord(act.get("location"))
        if coord:
            waypoints.append(
                {
                    "name": act.get("name", "Place"),
                    "coordinates": coord,
                    "type": act.get("type", "activity"),
                    "order": act.get("order"),
                }
            )

    # ── Multi-profile routing (walking / driving / cycling) ──
    coords = [w["coordinates"] for w in waypoints]
    routes = fetch_routes_multi(coords) if len(coords) >= 2 else {}

    if routes:
        day_plan["route"] = {
            "ordered_stops": [w["name"] for w in waypoints],
            "profiles": {
                profile: {
                    "distance_km": r["distance_km"],
                    "duration_mins": r["duration_mins"],
                }
                for profile, r in routes.items()
            },
        }
    else:
        day_plan["route"] = None

    return day_num, day_plan, build_geojson(waypoints, routes)


def generate_daily_map(state: AgentState) -> dict:
    """LangGraph node — geocode all itinerary locations, build daily GeoJSON maps."""
    itinerary_list = state.draft_itinerary
    if not itinerary_list:
        return {"draft_itinerary": [], "daily_map_info": {}}

    executor = _get_executor()
    futures = {
        executor.submit(_process_day, idx, day_plan): idx
        for idx, day_plan in enumerate(itinerary_list)
    }
    day_results: dict[int, tuple[int, dict, dict]] = {}

    for fut in as_completed(futures):
        idx = futures[fut]
        try:
            day_num, updated_plan, geojson = fut.result()
            day_results[idx] = (day_num, updated_plan, geojson)
        except Exception:
            safe_provider_log(logger, "provider.mapbox.day_processing_failed")
            day_results[idx] = (
                idx + 1,
                copy.deepcopy(itinerary_list[idx]),
                build_geojson([], None),
            )

    # Reassemble in original order
    ordered_itinerary: list[dict] = []
    compiled_maps: dict[int, dict] = {}
    for idx in range(len(itinerary_list)):
        day_num, updated_plan, geojson = day_results[idx]
        ordered_itinerary.append(updated_plan)
        compiled_maps[day_num] = geojson

    return {
        "draft_itinerary": ordered_itinerary,
        "daily_map_info": compiled_maps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Agent tools
# ─────────────────────────────────────────────────────────────────────────────
def _normalise_nearby_name(value: Any) -> str:
    """Return a case-insensitive comparison key for place/hotel names."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).casefold()


def _itinerary_location_coordinates(
    state: AgentState,
    specific_location: str,
) -> Optional[tuple[float, float]]:
    """Resolve a named itinerary hotel without calling a POI geocoder.

    Hotel coordinates are already provider-grounded by the hotel search, so
    re-geocoding the hotel name is both unnecessary and less reliable.
    Returns ``(lng, lat)`` when an exact itinerary-hotel match is found.
    """
    target = _normalise_nearby_name(specific_location)
    if not target:
        return None

    for day in getattr(state, "draft_itinerary", []) or []:
        if not isinstance(day, dict):
            continue

        hotel = day.get("hotel")

        if not isinstance(hotel, dict):
            continue

        if _normalise_nearby_name(hotel.get("hotel_name")) != target:
            continue

        coordinates = _extract_coord(hotel.get("location"))

        if coordinates is not None:
            return float(coordinates[0]), float(coordinates[1])

    return None


def _serpapi_maps_candidates(
    query: str,
    *,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    zoom: int = 15,
) -> list[dict]:
    """Run a Google Maps search through the existing SerpAPI place key."""
    params: dict[str, Any] = {
        "engine": "google_maps",
        "type": "search",
        "q": query,
        "hl": "en",
        "api_key": SERPAPI_PLACE_KEY,
    }

    if latitude is not None and longitude is not None:
        params["ll"] = f"@{latitude},{longitude},{zoom}z"

    try:
        response = _session.get(
            "https://serpapi.com/search",
            params=params,
            timeout=15,
        )

        if response.status_code != 200:
            safe_provider_log(
                logger,
                "provider.mapbox.nearby_failed",
                status_code=response.status_code,
            )
            return []

        data = response.json()

    except (
        AttributeError,
        TypeError,
        ValueError,
        requests.RequestException,
    ):
        safe_provider_log(
            logger,
            "provider.mapbox.nearby_failed",
        )
        return []

    candidates = data.get("local_results") or []

    if not candidates:
        direct = data.get("place_results")

        if isinstance(direct, dict):
            candidates = [direct]

    return [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict)
    ]


def _resolve_specific_location(
    state: AgentState,
    specific_location: str,
) -> Optional[dict]:
    """Resolve the search origin using trusted itinerary coordinates first.

    If the named location is not an itinerary hotel, fall back to the same
    Google Maps/SerpAPI provider already used by the project for places.
    """
    itinerary_coordinates = _itinerary_location_coordinates(
        state,
        specific_location,
    )

    if itinerary_coordinates is not None:
        lng, lat = itinerary_coordinates

        return {
            "name": specific_location,
            "latitude": lat,
            "longitude": lng,
        }

    city_values = [
        str(city).strip()
        for city in (getattr(state, "city", []) or [])
        if str(city).strip()
    ]

    country = str(
        getattr(state, "country", "") or ""
    ).strip()

    query = ", ".join(
        part
        for part in (
            specific_location,
            city_values[0] if city_values else "",
            country,
        )
        if part
    )

    for candidate in _serpapi_maps_candidates(query):
        gps = candidate.get("gps_coordinates") or {}

        try:
            lat = float(gps.get("latitude"))
            lng = float(gps.get("longitude"))

        except (TypeError, ValueError):
            continue

        if not math.isfinite(lat) or not math.isfinite(lng):
            continue

        return {
            "name": (
                candidate.get("title")
                or specific_location
            ),
            "latitude": lat,
            "longitude": lng,
        }

    return None


def _nearby_search_queries(
    category: str,
    country: str,
) -> tuple[str, ...]:
    """Return provider-friendly search terms, with Japan-specific fallbacks."""
    raw = " ".join(
        (category or "").split()
    )

    key = raw.casefold()

    if (
        "police" in key
        or key in {"koban", "交番", "警察署"}
    ):
        base = [
            "police station",
        ]

        japan = [
            "交番",
            "警察署",
        ]

    elif (
        "hospital" in key
        or key == "病院"
    ):
        base = [
            "hospital",
        ]

        japan = [
            "病院",
        ]

    elif (
        "fire" in key
        or key == "消防署"
    ):
        base = [
            "fire station",
        ]

        japan = [
            "消防署",
        ]

    elif (
        "pharmacy" in key
        or "drugstore" in key
        or key == "薬局"
    ):
        base = [
            "pharmacy",
        ]

        japan = [
            "薬局",
        ]

    else:
        base = [
            raw
        ] if raw else []

        japan = []

    if (
        (country or "")
        .strip()
        .casefold()
        == "japan"
    ):
        base.extend(japan)

    return tuple(
        dict.fromkeys(
            query
            for query in base
            if query
        )
    )


def _normalise_nearby_candidate(
    candidate: dict,
    *,
    origin_lng: float,
    origin_lat: float,
) -> Optional[dict]:
    gps = candidate.get(
        "gps_coordinates"
    ) or {}

    try:
        latitude = float(
            gps.get("latitude")
        )

        longitude = float(
            gps.get("longitude")
        )

    except (TypeError, ValueError):
        return None

    if (
        not math.isfinite(latitude)
        or not math.isfinite(longitude)
    ):
        return None

    distance_meters = round(
        _haversine_km(
            [
                origin_lng,
                origin_lat,
            ],
            [
                longitude,
                latitude,
            ],
        )
        * 1000,
        1,
    )

    return {
        "name": (
            candidate.get("title")
            or "Unknown place"
        ),

        "address": (
            candidate.get("address")
            or "Address Unavailable"
        ),

        "phone": candidate.get("phone"),

        "category": (
            candidate.get("type")
            or ""
        ),

        "distance_meters":
            distance_meters,

        "longitude":
            longitude,

        "latitude":
            latitude,
    }


@tool
def search_nearby_amenities(
    category: str,
    state: Annotated[
        AgentState,
        InjectedState,
    ],
    specific_location: Optional[str] = None,
) -> dict:
    """Find the nearest amenity around a named location.

    Nearby POI discovery uses Google Maps through the project's existing
    SerpAPI place key. Mapbox remains responsible for route distance/duration.
    This is important for destinations such as Japan where Mapbox Search Box
    POI coverage is not a supported geography.
    """
    if not specific_location:
        return {
            "error": (
                "No specific_location provided. "
                "Please ask the user for a specific location to search near."
            )
        }

    # ---------------------------------------------------------
    # Step 1:
    # Resolve the requested origin.
    #
    # If this is the user's itinerary hotel, use the trusted
    # coordinates already stored in the hotel instead of
    # trying to geocode the hotel again.
    # ---------------------------------------------------------
    origin = _resolve_specific_location(
        state,
        specific_location,
    )

    if origin is None:
        return {
            "error": (
                "I could not resolve the requested "
                "search origin to a provider-grounded "
                "location."
            )
        }

    origin_lat = float(
        origin["latitude"]
    )

    origin_lng = float(
        origin["longitude"]
    )

    country = str(
        getattr(state, "country", "")
        or ""
    )

    # ---------------------------------------------------------
    # Step 2:
    # Convert generic categories into useful search terms.
    #
    # Japan gets Japanese fallback queries as well:
    #
    # police -> police station / 交番 / 警察署
    # hospital -> hospital / 病院
    # fire -> fire station / 消防署
    # pharmacy -> pharmacy / 薬局
    # ---------------------------------------------------------
    queries = _nearby_search_queries(
        category,
        country,
    )

    if not queries:
        return {
            "error":
                "No amenity category was provided.",

            "search_category":
                category,
        }

    results_by_key: dict[
        tuple[str, float, float],
        dict,
    ] = {}

    # ---------------------------------------------------------
    # Step 3:
    # Search Google Maps through SerpAPI.
    #
    # Start with a close search at zoom 15.
    #
    # If nothing exists there, automatically widen the
    # provider search to zoom 12 instead of incorrectly
    # telling the user that no place exists.
    # ---------------------------------------------------------
    for zoom in (15, 12):

        for query in queries:

            candidates = (
                _serpapi_maps_candidates(
                    query,
                    latitude=origin_lat,
                    longitude=origin_lng,
                    zoom=zoom,
                )
            )

            for candidate in candidates:

                normalised = (
                    _normalise_nearby_candidate(
                        candidate,
                        origin_lng=origin_lng,
                        origin_lat=origin_lat,
                    )
                )

                if normalised is None:
                    continue

                key = (
                    _normalise_nearby_name(
                        normalised["name"]
                    ),

                    round(
                        normalised["latitude"],
                        5,
                    ),

                    round(
                        normalised["longitude"],
                        5,
                    ),
                )

                results_by_key[key] = normalised

        # A close-range result exists.
        # No reason to broaden the search.
        if results_by_key:
            break

    # ---------------------------------------------------------
    # Step 4:
    # Calculate REAL straight-line distance from the origin
    # and sort ourselves.
    #
    # Do not trust a missing provider distance as 0 metres.
    # ---------------------------------------------------------
    results = sorted(
        results_by_key.values(),
        key=lambda item:
            item["distance_meters"],
    )

    if not results:
        return {
            "search_category":
                category,

            "origin":
                origin,

            "results":
                [],

            "error": (
                f"No provider-grounded "
                f"{category} result was found near "
                f"{specific_location}."
            ),
        }

    nearest = dict(
        results[0]
    )

    # ---------------------------------------------------------
    # Step 5:
    # Mapbox is still useful for routing.
    #
    # We are only removing Mapbox Search Box from POI
    # discovery in unsupported locations such as Japan.
    # ---------------------------------------------------------
    routes = fetch_routes_multi(
        [
            [
                origin_lng,
                origin_lat,
            ],

            [
                nearest["longitude"],
                nearest["latitude"],
            ],
        ]
    )

    nearest["route_from_origin"] = {
        "origin": (
            origin.get("name")
            or specific_location
        ),

        "destination":
            nearest["name"],

        "profiles": {
            profile: {
                "distance_km":
                    route["distance_km"],

                "duration_mins":
                    route["duration_mins"],
            }
            for profile, route
            in routes.items()
        },
    }

    return {
        "search_category":
            category,

        "origin":
            origin,

        "nearest":
            nearest,

        "results":
            results,
    }