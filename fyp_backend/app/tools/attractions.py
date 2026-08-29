"""
attractions.py — Places / tourist-attraction / restaurant planning.

Design (per project requirement #5)
-----------------------------------
Unlike flights/hotels (which can be fetched directly from structured trip data),
places need an *idea* first. So the pipeline is:

  1. **LLM brainstorm** — Gemini proposes a day-by-day, logically-ordered list of
     attractions + restaurants for the destination, using the user's interests,
     pacing and budget. It also gives a rough per-place cost estimate and keeps
     the totals within the ``activity`` (attractions) and ``food`` (restaurants)
     allocations.
  2. **SerpAPI fetch** — each proposed place name is searched on Google Maps
     (via SerpAPI) to attach REAL data: address, rating, coordinates, thumbnail.
     The search is geo-anchored to the destination city (``ll`` param) and every
     candidate's coordinates are validated against that anchor, so an ambiguous
     name can never resolve to a same-named place in another country.
     Cached in Firebase (1 h TTL), same pattern as flights/hotels.
  3. **Merge** — real data + the LLM's estimated cost (flagged ``is_estimated``
     so the frontend can show a "this is an estimate" note) are written into each
     day's ``activities`` list, preserving the logical visiting order.

Costs are *estimates*: SerpAPI rarely returns a price for attractions/food, so
the LLM's knowledge-based estimate is used and marked as such.
"""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Annotated, Any, Dict, List, Literal, Optional

import pycountry
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.agents.state import AgentState
from app.core.config import settings
from app.core.firebase_db import get_cached_data, set_cached_data
from app.services.activity_hybrid import (
    activity_identity_key,
    deduplicate_activity_buckets,
    rescue_activity_buckets,
)
from app.tools.mapbox import (
    geocode_location,
    resolve_country_code,
    resolve_locality_names,
)
from app.tools.provider_logging import safe_provider_log

logger = logging.getLogger(__name__)

# SerpAPI key for places — dedicated key if configured, else reuse hotel key.
_SERPAPI_PLACES_KEY = settings.SERPAPI_PLACE

# Bound the number of places per day and provider usage.
_MAX_PLACES_PER_DAY = 8
_MAX_SERPAPI_WORKERS = 4

# Keep rescue bounded:
# round 1 = another specific-place attempt
# round 2 = final broader provider search
_MAX_HYBRID_RESCUE_ROUNDS = 2

# SerpAPI can return many Google Maps candidates for one search.
# Every candidate may require additional Mapbox verification, so never
# inspect an unlimited candidate list.
_MAX_SERPAPI_CANDIDATES = 5

# HARD limit for SerpAPI HTTP requests during ONE plan_activities() run.
#
# Normal searches + retries + rescue searches all share this same budget.
# Once 40 physical SerpAPI HTTP requests have been made, attraction
# searching stops instead of continuing to consume API quota.
_MAX_SERPAPI_REQUESTS_PER_PLAN = 40


class _SerpApiBudget:
    """Thread-safe hard cap for physical SerpAPI requests in one itinerary plan."""

    def __init__(self, limit: int) -> None:
        self._limit = max(0, int(limit))
        self._used = 0
        self._lock = Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            if self._used >= self._limit:
                return False

            self._used += 1
            return True

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

# Geo-validation radii for SerpAPI results. A place must lie within this
# distance of its geocoded anchor or it is rejected (an ambiguous name that
# Google resolved to a same-named place in another country/region).
# 75 km contains normal metropolitan/peripheral day activities while rejecting
# the reviewed Penang-as-Kuala-Lumpur drift (~293 km). This radius is only a
# relevance gate; structured reverse-geocode locality evidence is still required.
_MAX_KM_FROM_CITY = 75.0
_MAX_KM_FROM_COUNTRY = 2500.0

# ─────────────────────────────────────────────────────────────────────────────
# Shared HTTP session (pool + transport-level retry) — mirrors flights_hotels.py
# ─────────────────────────────────────────────────────────────────────────────
_RETRY_STRATEGY = Retry(
    # Do not hide extra physical requests inside urllib3.
    #
    # SerpAPI retries are controlled explicitly inside
    # _serpapi_place_search(), where every physical request can be
    # counted against _MAX_SERPAPI_REQUESTS_PER_PLAN.
    total=0,
    backoff_factor=0.0,
    status_forcelist=(),
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
# Geo-anchoring — keep every looked-up place inside the destination
# ─────────────────────────────────────────────────────────────────────────────
def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in km between two (lat, lng) points."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = rlat2 - rlat1
    dlng = math.radians(lng2 - lng1)
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlng / 2) ** 2
    )
    return 6371.0 * 2 * math.asin(math.sqrt(h))


def _geo_anchor(city: str, country: str) -> Optional[Dict[str, float]]:
    """Geocode the destination into a ``{lat, lng, max_km}`` validation anchor.

    Prefers the city (tight radius); falls back to the country centroid with a
    coarse radius. Uses the Mapbox geocoder (24 h in-process cache), so this
    costs at most one API call per distinct city per day.
    Returns None when nothing can be geocoded — validation is then skipped.
    """
    if city:
        geo = geocode_location(f"{city}, {country}".strip(", "))
        if geo:
            return {"lat": geo["lat"], "lng": geo["lng"], "max_km": _MAX_KM_FROM_CITY}
    if country:
        geo = geocode_location(country)
        if geo:
            return {
                "lat": geo["lat"],
                "lng": geo["lng"],
                "max_km": _MAX_KM_FROM_COUNTRY,
            }
    return None


def _within_anchor(lat: Any, lng: Any, anchor: Optional[Dict[str, float]]) -> bool:
    """True when (lat, lng) lies inside the anchor radius (or no anchor given)."""
    if not anchor:
        return True
    if lat is None or lng is None:
        return False
    try:
        dist = _haversine_km(float(lat), float(lng), anchor["lat"], anchor["lng"])
    except (TypeError, ValueError):
        return False
    return dist <= anchor["max_km"]


def _destination_country_code(country: str) -> Optional[str]:
    """Resolve the trusted destination name to ISO 3166-1 alpha-2."""
    try:
        return pycountry.countries.lookup((country or "").strip()).alpha_2
    except LookupError:
        return None


def _canonical_requested_city(
    model_city: Any,
    configured_cities: List[str],
) -> Optional[str]:
    """Bind a model city to one exact server-configured city or area."""
    canonical: Dict[str, str] = {}
    for configured in configured_cities:
        if not isinstance(configured, str) or not configured.strip():
            continue
        normalized = " ".join(configured.split())
        canonical.setdefault(normalized.casefold(), normalized)
    if not canonical:
        return None
    if not isinstance(model_city, str) or not model_city.strip():
        return next(iter(canonical.values())) if len(canonical) == 1 else None
    return canonical.get(" ".join(model_city.split()).casefold())


def _city_key(value: str) -> str:
    return " ".join(value.split()).casefold()


def _normalized_coordinates(lat: Any, lng: Any) -> Optional[tuple[float, float]]:
    """Return finite numeric coordinates without preserving provider strings."""
    if (
        isinstance(lat, bool)
        or isinstance(lng, bool)
        or lat is None
        or lng is None
    ):
        return None
    try:
        latitude = float(lat)
        longitude = float(lng)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        return None
    return latitude, longitude


def _verified_country_code(lat: Any, lng: Any, expected_code: str) -> Optional[str]:
    """Return provider country evidence only when it matches the destination."""
    coordinates = _normalized_coordinates(lat, lng)
    if not expected_code or coordinates is None:
        return None
    latitude, longitude = coordinates
    actual = resolve_country_code(latitude, longitude)
    return actual if actual == expected_code else None


def _verified_locality(
    lat: Any,
    lng: Any,
    requested_city: str,
    destination_country_code: str,
) -> Optional[str]:
    """Return the provider locality that exactly proves the requested city."""
    coordinates = _normalized_coordinates(lat, lng)
    if coordinates is None or not requested_city.strip():
        return None
    latitude, longitude = coordinates
    locality_names = resolve_locality_names(
        latitude,
        longitude,
        destination_country_code,
    )
    if not locality_names:
        return None
    requested_key = _city_key(requested_city)
    return next(
        (
            " ".join(name.split())
            for name in locality_names
            if isinstance(name, str) and _city_key(name) == requested_key
        ),
        None,
    )


def _place_thumbnail(place: Dict[str, Any]) -> str:
    """Return the best usable photo URL exposed by a SerpAPI Maps result.

    Google Maps search responses are not uniform: list results may expose
    ``thumbnail`` or ``serpapi_thumbnail``, while direct place results may put
    photos inside ``images``. Treat all of those documented shapes equally.
    """
    for key in ("thumbnail", "serpapi_thumbnail"):
        value = place.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    images = place.get("images") or []
    if isinstance(images, dict):
        images = [images]
    if isinstance(images, list):
        for image in images:
            if isinstance(image, str) and image.strip():
                return image.strip()
            if not isinstance(image, dict):
                continue
            for key in ("thumbnail", "image", "original_image"):
                value = image.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# LLM brainstorm — structured day-by-day place plan
# ─────────────────────────────────────────────────────────────────────────────
class PlaceIdea(BaseModel):
    """A single proposed place (attraction or restaurant) for a specific day."""

    name: str = Field(
        description="Real, searchable place name (as it appears on Google Maps)."
    )
    place_type: Literal["attraction", "restaurant"] = Field(
        description="'attraction' for sights/activities; 'restaurant' for dining."
    )
    city: str = Field(
        default="",
        description="Destination city/area (from the trip's city list) this place is in.",
    )
    day: int = Field(
        ge=1, description="Which day of the trip this belongs to (1-based)."
    )
    order: int = Field(
        ge=1,
        description="Visiting order within the day (1 = first). Logical, not shortest-path.",
    )
    suggested_time: str = Field(
        default="",
        description="Coarse time hint, e.g. 'Morning', 'Lunch', 'Afternoon', 'Evening'.",
    )
    estimated_cost: float = Field(
        default=0.0,
        ge=0.0,
        description="Rough TOTAL cost for the whole party at this place, in the DESTINATION currency.",
    )
    reason: str = Field(
        default="",
        description="One short sentence: why this place / why here in the order.",
    )


class DayByDayPlan(BaseModel):
    """The full brainstormed plan across all trip days."""

    places: List[PlaceIdea] = Field(default_factory=list)


_BRAINSTORM_SYSTEM = """\
You are an expert local travel planner. Propose a realistic, day-by-day list of \
ATTRACTIONS and RESTAURANTS for the trip described below.

Hard rules:
1. EVERY place MUST be physically located in {country}, in or around the trip's \
listed cities/areas. NEVER propose a place in any other country or region — even \
if it shares a name or theme with the destination (e.g. for a China trip, never \
suggest a 'Chinatown' or 'Great Wall restaurant' located in the US). Set each \
place's `city` to the trip city/area it belongs to.
2. Use REAL, well-known, searchable place names (exactly as they appear on Google Maps).
3. Assign each place to a specific day (1..{num_days}) and an `order` within that day.
4. The `order` must be the MOST LOGICAL route for the day — group nearby places and \
progress in one direction. Do NOT zig-zag back and forth (it wastes time and money).
5. Each day the traveller STARTS from their hotel (they slept there). Day 1 they arrive \
from the airport, so day 1 can be lighter.
6. Fill each day with a FULL schedule (morning, afternoon AND evening). \
Pacing: 'relaxed' => ~2-3 places/day; 'moderate' => ~4-6 places/day; \
'packed' => ~6-8 places/day. Only plan fewer places when the user's pacing is \
'relaxed' (they explicitly want a slow, peaceful trip). Never exceed \
{max_per_day} places in a single day, and never leave a day with fewer than \
2 places (except a late-arrival day 1).
7. Include 1-2 restaurants per day (lunch/dinner) that fit the user's dietary needs.
8. Estimate each place's TOTAL cost for the whole party ({num_people} people) in the \
DESTINATION currency. Keep the SUM of attraction costs <= {activity_budget} and the SUM \
of restaurant costs <= {food_budget}. Free attractions should be 0.
9. Personalise to the user's interests. Prefer variety over repetition.
"""

# The trip context MUST be a human message, not part of the system message.
# langchain-google-genai maps the system message to Gemini's `system_instruction`
# field; a prompt with ONLY a system message therefore sends an empty `contents`
# array, which Gemini rejects with "400 GenerateContentRequest.contents:
# contents is not specified".
_BRAINSTORM_HUMAN = """\
Plan the places for this trip:
- Destination country: {country}
- Cities/areas: {cities}
- Number of days: {num_days} (dates: {dates})
- Party size: {num_people}
- Base hotel: {hotel_name}
- User interests: {interests}
- Dietary restrictions: {dietary}
- Travel pacing: {pacing}
- Activity budget (attractions, destination currency): {activity_budget}
- Food budget (restaurants, destination currency): {food_budget}
"""

_brainstorm_llm = ChatGoogleGenerativeAI(
    model=settings.GEMINI_CHAT_MODEL,
    temperature=0.4,
    google_api_key=settings.GEMINI_API_KEY,
    max_retries=3,
)
_BRAINSTORM_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _BRAINSTORM_SYSTEM),
        ("human", _BRAINSTORM_HUMAN),
    ]
)
_BRAINSTORM_CHAIN = _BRAINSTORM_PROMPT | _brainstorm_llm.with_structured_output(
    DayByDayPlan
)


def brainstorm_places(
    *,
    country: str,
    cities: List[str],
    dates: List[str],
    num_people: int,
    hotel_name: str,
    interests: str,
    dietary: str,
    pacing: str,
    activity_budget: float,
    food_budget: float,
) -> List[PlaceIdea]:
    """Ask the LLM for a logically-ordered, budget-aware day-by-day place plan."""
    num_days = len(dates)
    try:
        plan: DayByDayPlan = _BRAINSTORM_CHAIN.invoke(
            {
                "country": country or "the destination",
                "cities": ", ".join(cities)
                if cities
                else (country or "the destination"),
                "num_days": num_days,
                "dates": ", ".join(dates),
                "num_people": num_people,
                "hotel_name": hotel_name or "the hotel",
                "interests": interests
                or "general sightseeing, local culture, good food",
                "dietary": dietary or "none",
                "pacing": pacing or "moderate",
                "activity_budget": round(activity_budget, 2),
                "food_budget": round(food_budget, 2),
                "max_per_day": _MAX_PLACES_PER_DAY,
            }
        )
        return plan.places or []
    except Exception:
        safe_provider_log(logger, "provider.attractions.brainstorm_failed")
        return []

_RESCUE_SYSTEM = """\
You are repairing missing activity days in an existing travel itinerary.
Generate NEW real, searchable ATTRACTIONS or RESTAURANTS only for the requested
missing day numbers. Do not repeat any excluded place. Keep every suggestion in
the supplied destination country and one of the supplied trip cities/areas.

Hard rules:
1. Return places ONLY for days in {missing_days}.
2. Every place name must be a real Google-Maps-searchable venue.
3. Set `city` to exactly one value from: {cities}.
4. Prefer well-known places with unambiguous names so provider verification is
   likely to succeed.
5. Do not use any excluded place: {excluded_places}.
6. Use logical day order and do not exceed {max_per_day} places for a day.
7. Keep attraction estimates within the supplied activity budget cap and restaurant
   estimates within the supplied food budget cap.
8. Prefer variety and geographically sensible grouping.
"""

_RESCUE_HUMAN = """\
Repair these missing itinerary days:
- Destination country: {country}
- Cities/areas: {cities}
- Missing day numbers: {missing_days}
- Dates by day: {dates}
- Party size: {num_people}
- Hotel: {hotel_name}
- Interests: {interests}
- Dietary restrictions: {dietary}
- Travel pacing: {pacing}
- Activity budget cap: {activity_budget}
- Food budget cap: {food_budget}
- Excluded/previously tried places: {excluded_places}
"""

_RESCUE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _RESCUE_SYSTEM),
        ("human", _RESCUE_HUMAN),
    ]
)
_RESCUE_CHAIN = _RESCUE_PROMPT | _brainstorm_llm.with_structured_output(
    DayByDayPlan
)


def brainstorm_rescue_places(
    *,
    country: str,
    cities: List[str],
    dates: List[str],
    missing_days: tuple[int, ...],
    excluded_places: List[str],
    num_people: int,
    hotel_name: str,
    interests: str,
    dietary: str,
    pacing: str,
    activity_budget: float,
    food_budget: float,
) -> List[PlaceIdea]:
    """Ask Gemini only for alternatives needed to repair missing activity days."""
    if not missing_days:
        return []
    try:
        plan: DayByDayPlan = _RESCUE_CHAIN.invoke(
            {
                "country": country or "the destination",
                "cities": ", ".join(cities)
                if cities
                else (country or "the destination"),
                "missing_days": ", ".join(str(day) for day in missing_days),
                "dates": ", ".join(
                    f"Day {index}: {date}" for index, date in enumerate(dates, start=1)
                ),
                "num_people": num_people,
                "hotel_name": hotel_name or "the hotel",
                "interests": interests
                or "general sightseeing, local culture, good food",
                "dietary": dietary or "none",
                "pacing": pacing or "moderate",
                "activity_budget": round(max(activity_budget, 0.0), 2),
                "food_budget": round(max(food_budget, 0.0), 2),
                "excluded_places": ", ".join(excluded_places[-50:]) or "none",
                "max_per_day": _MAX_PLACES_PER_DAY,
            }
        )
        allowed_days = set(missing_days)
        return [idea for idea in (plan.places or []) if idea.day in allowed_days]
    except Exception:
        safe_provider_log(logger, "provider.attractions.rescue_brainstorm_failed")
        return []

# ─────────────────────────────────────────────────────────────────────────────
# SerpAPI (Google Maps) place lookup — attach real data to a proposed name
# ─────────────────────────────────────────────────────────────────────────────
def _serpapi_place_search(
    query: str,
    anchor: Optional[Dict[str, float]] = None,
    destination_country_code: str = "",
    requested_city: str = "",
    api_budget: Optional[_SerpApiBudget] = None,
    max_retries: int = 2,
) -> Optional[Dict[str, Any]]:
    """Search Google Maps via SerpAPI; return the first VALID normalised match.

    When ``anchor`` is given the request is geo-anchored via the ``ll`` param
    (without it SerpAPI searches from US datacenters, so Google biases results
    to the US) and every candidate outside ``anchor['max_km']`` is rejected —
    this is what prevents a "China trip" from picking up a same-named US place.
    """
    url = "https://serpapi.com/search"
    params = {
        "engine": "google_maps",
        "type": "search",
        "q": query,
        "hl": "en",
        "api_key": _SERPAPI_PLACES_KEY,
    }
    if anchor:
        params["ll"] = f"@{anchor['lat']},{anchor['lng']},11z"
    for attempt in range(max_retries):
        # Count EVERY physical SerpAPI HTTP request.
        # When the plan-wide budget is exhausted, stop immediately.
        if api_budget is not None and not api_budget.try_acquire():
            safe_provider_log(
                logger,
                "provider.attractions.api_budget_exhausted",
            )
            return None

        try:
            resp = _session.get(
                url,
                params=params,
                timeout=15,
            )

            resp.raise_for_status()
            data = resp.json()
            break
        except (requests.RequestException, ValueError):
            if attempt < max_retries - 1:
                time.sleep(1.5 * (2**attempt))
            else:
                safe_provider_log(logger, "provider.attractions.search_failed")
                return None
    else:
        return None

    # google_maps search → list under 'local_results'; a direct hit → 'place_results'
    candidates = data.get("local_results") or []
    if not candidates:
        place = data.get("place_results")
        if isinstance(place, dict):
            candidates = [place]
    if not candidates:
        return None

    # Take the first candidate whose coordinates lie inside the destination
    # anchor — NOT blindly candidates[0], which may be a same-named place on
    # another continent.
    for top in candidates[:_MAX_SERPAPI_CANDIDATES]:
        if not isinstance(top, dict):
            continue
        gps = top.get("gps_coordinates") or {}
        lat, lng = gps.get("latitude"), gps.get("longitude")
        coordinates = _normalized_coordinates(lat, lng)
        if coordinates is None:
            continue
        lat, lng = coordinates
        if not _within_anchor(lat, lng, anchor):
            safe_provider_log(logger, "provider.attractions.out_of_area")
            continue
        verified_country_code = _verified_country_code(
            lat, lng, destination_country_code
        )
        if verified_country_code is None:
            safe_provider_log(logger, "provider.attractions.unverified")
            continue
        verified_locality = _verified_locality(
            lat,
            lng,
            requested_city,
            destination_country_code,
        )
        if verified_locality is None:
            safe_provider_log(logger, "provider.attractions.locality_unverified")
            continue
        return {
            "real_name": top.get("title", query),
            "address": top.get("address", ""),
            "rating": top.get("rating"),
            "reviews": top.get("reviews"),
            "category": top.get("type", ""),
            "thumbnail": _place_thumbnail(top),
            "serp_price": top.get("price"),  # usually a "$$"-style symbol, may be None
            "lat": lat,
            "lng": lng,
            "country_code": verified_country_code,
            "verified_locality": verified_locality,
        }
    return None


def _lookup_place_cached(
    query: str,
    anchor: Optional[Dict[str, float]] = None,
    destination_country_code: str = "",
    requested_city: str = "",
    api_budget: Optional[_SerpApiBudget] = None,
) -> Optional[Dict[str, Any]]:
    """Firebase-cached wrapper around :func:`_serpapi_place_search` (1 h TTL).

    Cache hits are re-validated against ``anchor`` so entries written before
    geo-validation existed (or under a different anchor) cannot poison the
    itinerary with an out-of-country place.
    """
    cache_key = {
        "query": query,
        "country_code": destination_country_code,
        "requested_city": _city_key(requested_city) if requested_city else "",
    }
    cached = get_cached_data(collection="api_cache", prefix="place", **cache_key)
    cached_coordinates = (
        _normalized_coordinates(cached.get("lat"), cached.get("lng"))
        if cached
        else None
    )
    if cached_coordinates and _within_anchor(*cached_coordinates, anchor):
        latitude, longitude = cached_coordinates
        verified = _verified_country_code(
            latitude, longitude, destination_country_code
        )
        verified_locality = _verified_locality(
            latitude,
            longitude,
            requested_city,
            destination_country_code,
        )
        if verified and verified_locality:
            return {
                **cached,
                "lat": latitude,
                "lng": longitude,
                "country_code": verified,
                "verified_locality": verified_locality,
            }
    result = _serpapi_place_search(
        query,
        anchor=anchor,
        destination_country_code=destination_country_code,
        requested_city=requested_city,
        api_budget=api_budget,
    )
    if result:
        set_cached_data(
            collection="api_cache",
            prefix="place",
            payload=result,
            ttl_hours=1,
            **cache_key,
        )
    return result


def _build_activity(
    idea: PlaceIdea, real: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Merge an LLM idea with real SerpAPI data into an activity dict for the itinerary."""
    if not real:
        # No verified location — skip so the map/route stays accurate.
        return None

    country_code = real.get("country_code")
    verified_locality = real.get("verified_locality")
    coordinates = _normalized_coordinates(real.get("lat"), real.get("lng"))

    if (
        not isinstance(country_code, str)
        or len(country_code) != 2
        or coordinates is None
        or not isinstance(verified_locality, str)
        or not verified_locality.strip()
        or _city_key(verified_locality) != _city_key(idea.city)
    ):
        return None

    latitude, longitude = coordinates

    # SerpAPI may return either a single category string or
    # multiple categories. The public ActivityResult contract
    # exposes category as one display string.
    raw_category = real.get("category", "")

    if isinstance(raw_category, list):
        category = ", ".join(
            str(item).strip()
            for item in raw_category
            if str(item).strip()
        )
    elif isinstance(raw_category, str):
        category = raw_category.strip()
    else:
        category = ""

    return {
        "name": real["real_name"] or idea.name,
        "type": idea.place_type,
        "description": idea.reason,
        "category": category,
        "rating": real.get("rating"),
        "address": real.get("address", ""),
        "thumbnail": real.get("thumbnail", ""),
        "suggested_time": idea.suggested_time,
        "estimated_cost": round(float(idea.estimated_cost or 0.0), 2),
        "is_estimated": True,
        "order": idea.order,
        "location": {
            "place_name": real["real_name"] or idea.name,
            "latitude": latitude,
            "longitude": longitude,
            "country_code": country_code,
            "requested_city": idea.city,
            "verified_locality": " ".join(verified_locality.split()),
        },
    }


def _clamp_costs(
    activities: List[Dict[str, Any]], place_type: str, budget: float
) -> None:
    """Proportionally scale down estimated costs of one type so their sum ≤ budget.

    No-op when the budget is 0/undefined (nothing to enforce against) or the
    total is already within budget.
    """
    if budget <= 0:
        return
    subset = [a for a in activities if a.get("type") == place_type]
    total = sum(float(a.get("estimated_cost", 0.0) or 0.0) for a in subset)
    if total <= budget or total <= 0:
        return
    factor = budget / total
    for a in subset:
        a["estimated_cost"] = round(
            float(a.get("estimated_cost", 0.0) or 0.0) * factor, 2
        )


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph node — populate each day's activities
# ─────────────────────────────────────────────────────────────────────────────
def plan_activities(
    state: AgentState, user_profile: Optional[dict] = None
) -> Dict[str, Any]:
    """Brainstorm, provider-verify, rescue, and deduplicate daily activities.

    Runs AFTER ``plan_flight_hotel`` and BEFORE ``generate_daily_map``. The
    normal Gemini brainstorm remains the first source of ideas, but any day that
    becomes empty after provider/locality verification is repaired independently
    for at most ``_MAX_HYBRID_RESCUE_ROUNDS`` rounds. Existing good days are
    preserved. The final rescue round broadens an unverified AI idea into a
    generic provider search instead of publishing an unverified place.
    """
    itinerary = state.draft_itinerary or []
    if not itinerary:
        return {}

    profile = user_profile or {}
    interests = (
        ", ".join(profile.get("interests", []) or [])
        if isinstance(profile.get("interests"), list)
        else str(profile.get("interests") or "")
    )
    dietary = (
        ", ".join(profile.get("dietary_restrictions", []) or [])
        if isinstance(profile.get("dietary_restrictions"), list)
        else str(profile.get("dietary_restrictions") or "")
    )
    pacing = profile.get("travel_pacing") or "moderate"

    hotel = next((d.get("hotel") for d in itinerary if d.get("hotel")), None)
    hotel_name = hotel.get("hotel_name", "") if isinstance(hotel, dict) else ""

    activity_budget = float((state.budget_allocation or {}).get("activity", 0.0) or 0.0)
    food_budget = float((state.budget_allocation or {}).get("food", 0.0) or 0.0)
    dates = [str(d.get("date", "")) for d in itinerary]
    expected_days = [
        int(day.get("day"))
        for day in itinerary
        if isinstance(day, dict)
        and isinstance(day.get("day"), int)
        and not isinstance(day.get("day"), bool)
        and day.get("day") > 0
    ]

    configured_cities = [
        city.strip()
        for city in (state.city or [])
        if isinstance(city, str) and city.strip()
    ]
    expected_day_set = set(expected_days)

    def _trusted_ideas(
        raw_ideas: List[PlaceIdea],
        *,
        allowed_days: set[int] | None = None,
    ) -> List[PlaceIdea]:
        """Bind model output to server-owned cities/days and enforce day caps."""
        trusted: List[PlaceIdea] = []
        per_day_count: Dict[int, int] = {}
        allowed = expected_day_set if allowed_days is None else allowed_days
        for idea in sorted(raw_ideas, key=lambda item: (item.day, item.order)):
            if idea.day not in allowed:
                continue
            requested_city = _canonical_requested_city(idea.city, configured_cities)
            if requested_city is None:
                continue
            if per_day_count.get(idea.day, 0) >= _MAX_PLACES_PER_DAY:
                continue
            per_day_count[idea.day] = per_day_count.get(idea.day, 0) + 1
            trusted.append(idea.model_copy(update={"city": requested_city}))
        return trusted

    country = state.country or ""
    destination_country_code = _destination_country_code(country)
    if destination_country_code is None:
        safe_provider_log(logger, "provider.attractions.destination_unresolved")
        return {}
    default_city = configured_cities[0] if configured_cities else ""

    # Resolve anchors for every configured city once so rescue rounds do not
    # repeatedly geocode the same place.
    anchor_cities = configured_cities or ([default_city] if default_city else [])
    anchors = {city: _geo_anchor(city, country) for city in anchor_cities}

    # ONE shared hard SerpAPI budget for this complete itinerary-generation run.
    #
    # Do NOT create a new budget inside each rescue round.
    # Normal searches and every rescue attempt must share the SAME counter.
    serpapi_budget = _SerpApiBudget(
        _MAX_SERPAPI_REQUESTS_PER_PLAN
    )

    # Failed queries are remembered for the duration of this plan.
    #
    # This is especially useful for broad rescue searches such as
    # "popular tourist attraction in Singapore". Previously the same failed
    # search could be repeated for many different AI-generated places.
    failed_lookup_keys: set[tuple[str, str]] = set()
    failed_lookup_lock = Lock()


    def _provider_lookup(
        query: str,
        city: str,
    ) -> Optional[Dict[str, Any]]:
        """Perform one cached/provider lookup with quota and negative caching."""

        normalized_key = (
            " ".join(query.split()).casefold(),
            _city_key(city),
        )

        # If exactly this query already failed during this itinerary plan,
        # do not spend API quota trying it repeatedly.
        with failed_lookup_lock:
            if normalized_key in failed_lookup_keys:
                return None

        result = _lookup_place_cached(
            query,
            anchor=anchors.get(city),
            destination_country_code=destination_country_code,
            requested_city=city,
            api_budget=serpapi_budget,
        )

        if result is None:
            with failed_lookup_lock:
                failed_lookup_keys.add(normalized_key)

        return result


    def _lookup_exact(
        idea: PlaceIdea,
    ) -> Optional[Dict[str, Any]]:
        city = idea.city or default_city

        query = ", ".join(
            part
            for part in (
                idea.name,
                city,
                country,
            )
            if part
        )

        return _provider_lookup(
            query,
            city,
        )

    def _lookup_rescue(
        idea: PlaceIdea,
        *,
        broaden: bool,
    ) -> Optional[Dict[str, Any]]:
        real = _lookup_exact(idea)
        if real is not None or not broaden:
            return real

        city = idea.city or default_city
        if idea.place_type == "restaurant":
            broad_queries = [
                f"popular restaurant in {city}",
                f"local restaurant in {city}",
            ]
        else:
            broad_queries = [
                f"popular tourist attraction in {city}",
                f"museum or landmark in {city}",
            ]

        for query in broad_queries:
            real = _provider_lookup(
                ", ".join(
                    part
                    for part in (
                        query,
                        country,
                    )
                    if part
                ),
                city,
            )

            if real is not None:
                return real
        return None

    def _lookup_batch(
        trusted_ideas: List[PlaceIdea],
        *,
        broaden: bool = False,
    ) -> List[Optional[Dict[str, Any]]]:
        if not trusted_ideas:
            return []
        with ThreadPoolExecutor(max_workers=_MAX_SERPAPI_WORKERS) as ex:
            if broaden:
                return list(
                    ex.map(
                        lambda idea: _lookup_rescue(idea, broaden=True),
                        trusted_ideas,
                    )
                )
            return list(ex.map(_lookup_exact, trusted_ideas))

    def _merge_verified(
        trusted_ideas: List[PlaceIdea],
        reals: List[Optional[Dict[str, Any]]],
    ) -> tuple[Dict[int, List[Dict[str, Any]]], int]:
        merged: Dict[int, List[Dict[str, Any]]] = {}
        dropped = 0
        for idea, real in zip(trusted_ideas, reals):
            activity = _build_activity(idea, real)
            if activity is None:
                dropped += 1
                continue
            merged.setdefault(idea.day, []).append(activity)
        return merged, dropped

    # ── Step 1: normal AI brainstorm + strict provider verification ──
    ideas = brainstorm_places(
        country=country,
        cities=list(configured_cities),
        dates=dates,
        num_people=state.num_people or 1,
        hotel_name=hotel_name,
        interests=interests,
        dietary=dietary,
        pacing=pacing,
        activity_budget=activity_budget,
        food_budget=food_budget,
    )
    if not ideas:
        safe_provider_log(logger, "provider.attractions.no_ideas")

    capped = _trusted_ideas(ideas)
    reals = _lookup_batch(capped)
    by_day, dropped = _merge_verified(capped, reals)
    if dropped:
        safe_provider_log(logger, "provider.attractions.unmatched_dropped")

    tried_names = {
        idea.name.strip()
        for idea in ideas
        if isinstance(idea.name, str) and idea.name.strip()
    }

    # ── Step 2: hybrid rescue only for days that are still empty ──
    def _rescue_round(
        missing_days: tuple[int, ...],
        round_number: int,
    ) -> Dict[int, List[Dict[str, Any]]]:
        accepted_names: list[str] = []
        for activities in by_day.values():
            for activity in activities:
                key = activity_identity_key(activity)
                if key:
                    accepted_names.append(str(activity.get("name") or ""))

        rescue_ideas = brainstorm_rescue_places(
            country=country,
            cities=list(configured_cities),
            dates=dates,
            missing_days=missing_days,
            excluded_places=sorted(
                name for name in {*tried_names, *accepted_names} if name
            ),
            num_people=state.num_people or 1,
            hotel_name=hotel_name,
            interests=interests,
            dietary=dietary,
            pacing=pacing,
            activity_budget=activity_budget,
            food_budget=food_budget,
        )

        for idea in rescue_ideas:
            if isinstance(idea.name, str) and idea.name.strip():
                tried_names.add(idea.name.strip())

        trusted_rescue = _trusted_ideas(
            rescue_ideas,
            allowed_days=set(missing_days),
        )

        # Build the last-resort provider hint only AFTER model output has passed
        # the server-owned city/day trust boundary. Otherwise an invalid Gemini
        # suggestion (for example Osaka during a Tokyo-only trip) could suppress
        # the fallback despite being discarded immediately afterwards.
        if round_number == _MAX_HYBRID_RESCUE_ROUNDS:
            present_days = {idea.day for idea in trusted_rescue}
            generic_ideas: List[PlaceIdea] = []
            for day_number in missing_days:
                if day_number in present_days or not configured_cities:
                    continue
                city_index = (day_number - 1) % len(configured_cities)
                city = configured_cities[city_index]
                generic_ideas.append(
                    PlaceIdea(
                        name=f"popular tourist attraction in {city}",
                        place_type="attraction",
                        city=city,
                        day=day_number,
                        order=1,
                        suggested_time="Flexible",
                        estimated_cost=0.0,
                        reason=(
                            "Provider-grounded fallback for a day whose exact "
                            "AI suggestions could not be verified."
                        ),
                    )
                )
            for idea in generic_ideas:
                tried_names.add(idea.name.strip())
            trusted_rescue.extend(
                _trusted_ideas(
                    generic_ideas,
                    allowed_days=set(missing_days),
                )
            )
        rescue_reals = _lookup_batch(
            trusted_rescue,
            broaden=round_number == _MAX_HYBRID_RESCUE_ROUNDS,
        )
        rescued, rescue_dropped = _merge_verified(trusted_rescue, rescue_reals)
        if rescue_dropped:
            safe_provider_log(logger, "provider.attractions.rescue_unmatched")
        return rescued

    rescue_result = rescue_activity_buckets(
        expected_days,
        by_day,
        rescue_round=_rescue_round,
        max_rounds=_MAX_HYBRID_RESCUE_ROUNDS,
    )
    by_day = rescue_result.by_day

    if rescue_result.removed_duplicates:
        safe_provider_log(logger, "provider.attractions.duplicates_removed")
    if rescue_result.rounds_used:
        safe_provider_log(logger, "provider.attractions.rescue_used")
    if rescue_result.missing_days:
        safe_provider_log(logger, "provider.attractions.rescue_exhausted")
    logger.info(
        "provider.attractions.serpapi_budget used=%d limit=%d",
        serpapi_budget.used,
        _MAX_SERPAPI_REQUESTS_PER_PLAN,
    )

    # A final deterministic deduplication is intentionally repeated immediately
    # before cost/map projection so later refactors cannot accidentally bypass it.
    by_day, final_removed = deduplicate_activity_buckets(by_day)
    if final_removed:
        safe_provider_log(logger, "provider.attractions.duplicates_removed")

    # ── Step 3: clamp estimated costs to the allocated budgets ──
    all_acts = [activity for activities in by_day.values() for activity in activities]
    _clamp_costs(all_acts, "attraction", activity_budget)
    _clamp_costs(all_acts, "restaurant", food_budget)

    # ── Step 4: project the clean activities back into the day skeleton ──
    updated = [dict(day) for day in itinerary]
    for day in updated:
        day_num = day.get("day")
        acts = [
            dict(activity)
            for activity in sorted(
                by_day.get(day_num, []),
                key=lambda activity: activity.get("order", 999),
            )
        ]
        for index, activity in enumerate(acts, start=1):
            activity["order"] = index

        day["activities"] = acts
        act_cost = sum(float(a.get("estimated_cost", 0.0) or 0.0) for a in acts)
        day["day_total_cost"] = round(
            float(day.get("day_total_cost", 0.0) or 0.0) + act_cost,
            2,
        )

    safe_provider_log(logger, "provider.attractions.populated")
    return {"draft_itinerary": updated}


# ─────────────────────────────────────────────────────────────────────────────
# Agent tool — search places during chat (for modifications / add requests)
# ─────────────────────────────────────────────────────────────────────────────
from langchain_core.tools import tool  # noqa: E402  (kept near the tool for clarity)

def search_places_provider(
    category: str,
    query: str,
    state: AgentState,
    city: str = "",
) -> Dict[str, Any]:
    """Search for a tourist attraction or restaurant by name/keyword.

    The configured trip city remains the trusted geographic anchor.
    ``city`` is only an optional search hint, allowing nearby day-trip
    locations such as Urayasu for a Tokyo itinerary.

    Results must:
      1. be inside the destination country; and
      2. be within the configured city's allowed metropolitan radius.

    Returns up to 3 provider-grounded candidates.
    """
    country = state.country or ""

    configured_cities = [
        c.strip()
        for c in (state.city or [])
        if isinstance(c, str) and c.strip()
    ]

    default_city = next(
        (c for c in configured_cities),
        "",
    )

    matched_city = None

    if isinstance(city, str) and city.strip():
        matched_city = _canonical_requested_city(
            city,
            configured_cities,
        )

    # Only trust a city when it is one of the cities that already belongs
    # to this trip. A model-invented city cannot change the geographic anchor.
    target_city = matched_city or default_city

    # Keep an unmatched model locality only as a search hint.
    if matched_city is not None:
        search_city = matched_city
    elif isinstance(city, str) and city.strip():
        search_city = " ".join(city.split())
    else:
        search_city = default_city

    destination_country_code = _destination_country_code(country)

    if destination_country_code is None:
        return {
            "error": "Destination country could not be verified."
        }

    # Anchor geographic validation to the selected SERVER-CONFIGURED trip city.
    anchor = _geo_anchor(
        target_city,
        country,
    )

    q = ", ".join(
        part
        for part in (query, search_city, country)
        if part
    )

    url = "https://serpapi.com/search"

    params = {
        "engine": "google_maps",
        "type": "search",
        "q": q,
        "hl": "en",
        "api_key": _SERPAPI_PLACES_KEY,
    }

    if anchor:
        params["ll"] = (
            f"@{anchor['lat']},{anchor['lng']},11z"
        )

    try:
        resp = _session.get(
            url,
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

    except (requests.RequestException, ValueError):
        return {
            "error": "Place search failed."
        }

    # ---------------------------------------------------------
    # FIX 1:
    # SerpAPI may return either:
    #
    #   local_results: [...]
    #
    # OR, for a very specific place:
    #
    #   place_results: {...}
    #
    # Tokyo Disneyland is exactly the type of specific query
    # that may be returned as place_results.
    # ---------------------------------------------------------
    candidates = data.get("local_results") or []

    if not candidates:
        direct_place = data.get("place_results")

        if isinstance(direct_place, dict):
            candidates = [direct_place]

    results: List[Dict[str, Any]] = []

    for top in candidates:
        if len(results) >= 3:
            break

        if not isinstance(top, dict):
            continue

        gps = top.get("gps_coordinates") or {}

        coordinates = _normalized_coordinates(
            gps.get("latitude"),
            gps.get("longitude"),
        )

        if coordinates is None:
            continue

        latitude, longitude = coordinates

        # -----------------------------------------------------
        # Keep the strong distance validation.
        #
        # Tokyo Disneyland is near Tokyo, so it passes.
        # A place hundreds of kilometres away does not.
        # -----------------------------------------------------
        if not _within_anchor(
            latitude,
            longitude,
            anchor,
        ):
            safe_provider_log(
                logger,
                "provider.attractions.out_of_area",
            )
            continue

        # -----------------------------------------------------
        # Still require the correct COUNTRY.
        # -----------------------------------------------------
        verified_country_code = _verified_country_code(
            latitude,
            longitude,
            destination_country_code,
        )

        if verified_country_code is None:
            continue

        # -----------------------------------------------------
        # FIX 2:
        # Get the actual provider locality.
        #
        # Do NOT require it to literally equal dest_city.
        #
        # Example:
        #   requested_city    = Tokyo
        #   verified_locality = Urayasu
        #
        # This is valid because the coordinates already passed
        # the Tokyo metropolitan distance gate above.
        # -----------------------------------------------------
        locality_names = resolve_locality_names(
            latitude,
            longitude,
            destination_country_code,
        )

        if not locality_names:
            continue

        requested_key = _city_key(target_city)

        # Prefer an exact locality match when available.
        verified_locality = next(
            (
                " ".join(name.split())
                for name in locality_names
                if (
                    isinstance(name, str)
                    and name.strip()
                    and _city_key(name) == requested_key
                )
            ),
            None,
        )

        # Otherwise preserve the provider's actual locality.
        if verified_locality is None:
            verified_locality = next(
                (
                    " ".join(name.split())
                    for name in locality_names
                    if isinstance(name, str)
                    and name.strip()
                ),
                None,
            )

        if verified_locality is None:
            continue

        # -----------------------------------------------------
        # Produce deterministic proof for nearby-city results.
        #
        # Only a real CITY anchor is allowed to generate this
        # evidence. A country-centroid fallback must not make
        # distant places acceptable.
        # -----------------------------------------------------
        distance_from_requested_city_km = None

        if (
            anchor
            and anchor.get("max_km") == _MAX_KM_FROM_CITY
        ):
            distance_from_requested_city_km = round(
                _haversine_km(
                    latitude,
                    longitude,
                    anchor["lat"],
                    anchor["lng"],
                ),
                2,
            )

        # If the actual locality is different from the configured
        # city, nearby-distance proof is mandatory.
        if (
            _city_key(verified_locality) != requested_key
            and distance_from_requested_city_km is None
        ):
            continue

        raw_description = top.get("type", "")

        if isinstance(raw_description, str):
            place_description = raw_description.strip()

        elif isinstance(raw_description, (list, tuple, set)):
            place_description = ", ".join(
                str(item).strip()
                for item in raw_description
                if str(item).strip()
            )

        elif raw_description is None:
            place_description = ""

        else:
            place_description = str(raw_description).strip()


        results.append(
            {
                "name": top.get("title", ""),
                "type": category,
                "description": place_description,
                "rating": top.get("rating"),
                "address": top.get("address", ""),
                "thumbnail": _place_thumbnail(top),

                # SerpAPI does not normally provide a reliable numeric
                # admission/meal price for place-search results.
                #
                # Keep the value explicitly estimated so the replacement
                # remains compatible with the itinerary schema.
                "estimated_cost": 0.0,
                "is_estimated": True,

                "location": {
                    "place_name": top.get("title", ""),
                    "latitude": latitude,
                    "longitude": longitude,
                    "country_code": verified_country_code,
                    "requested_city": target_city,
                    "verified_locality": verified_locality,
                    "distance_from_requested_city_km":
                        distance_from_requested_city_km,
                },
            }
        )

    if not results:
        return {
            "error": f"No places found for '{q}'.",
            "count": 0,
        }

    return {
        "category": category,
        "results": results,
        "count": len(results),
    }

@tool
def search_places(
    category: str,
    query: str,
    state: Annotated[AgentState, InjectedState],
    city: str = "",
) -> Dict[str, Any]:
    """Search for a provider-grounded attraction or restaurant during chat.

    This LangGraph-facing wrapper delegates to the same deterministic
    provider implementation used by server-side grounding recovery.
    """
    return search_places_provider(
        category=category,
        query=query,
        state=state,
        city=city,
    )