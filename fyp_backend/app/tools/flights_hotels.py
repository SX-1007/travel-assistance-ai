"""
Flight & hotel search via SerpAPI; IATA resolution via Gemini.

Optimisation highlights
-----------------------
* Module-level `requests.Session` with `HTTPAdapter` pool + `urllib3.Retry`.
* LLM chain (`prompt | llm.with_structured_output`) compiled once at import.
* Aggressive Firebase-backed caching for both flights and hotels (1 h TTL).
* Defensive float coercion on every SerpAPI numeric field (issue #44/#45).
* Parallel IATA resolution + parallel flight/hotel search.
* Transient LLM failures are NOT cached, so retry is possible next call.
"""

from __future__ import annotations

import logging
import math
import time
from cachetools import TTLCache
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Dict, List, Optional
from urllib.parse import quote_plus

from typing import Literal

import pycountry
import requests
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field, model_validator
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.agents.state import AgentState
from app.core.config import settings
from app.core.firebase_db import get_cached_data, set_cached_data
from app.tools.mapbox import resolve_country_code, resolve_locality_names
from app.tools.provider_logging import safe_provider_log

logger = logging.getLogger(__name__)


def _is_finite_number(value: Any) -> bool:
    """True for finite numeric values, including numeric API strings."""
    if isinstance(value, bool) or value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


SERPAPI_FLIGHT = settings.SERPAPI_FLIGHT
SERPAPI_HOTEL = settings.SERPAPI_HOTEL

# Static IATA map — skips LLM call entirely for top destinations
TOP_DESTINATIONS: Dict[str, str] = {
    "malaysia": "KUL",
    "singapore": "SIN",
    "thailand": "BKK",
    "japan": "HND",
    "south korea": "ICN",
    "indonesia": "CGK",
    "australia": "SYD",
    "united kingdom": "LHR",
    "usa": "JFK",
    "united states": "JFK",
}

# In-memory IATA cache (country_lower -> IATA | None)
_iata_cache: TTLCache = TTLCache(maxsize=1024, ttl=86400)

# Shared HTTP session with pooling + transport-level retry
_RETRY_STRATEGY = Retry(
    total=3,
    backoff_factor=1.0,
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
# IATA resolution via LLM (with static fast-path)
# ─────────────────────────────────────────────────────────────────────────────
class IataCodeResponse(BaseModel):
    iata_code: str = Field(
        description="The 3-letter IATA code of the primary international airport"
    )


_llm_router = ChatGoogleGenerativeAI(
    model=settings.GEMINI_UTILITY_MODEL,
    temperature=0,
    max_retries=3,
    google_api_key=settings.GEMINI_FLIGHTS_HOTELS,
)

# Compile prompt + chain once at module load
_IATA_PROMPT = PromptTemplate.from_template(
    "You are an aviation system. The user wants to fly to: '{country}'.\n"
    "If this is a real country, return the 3-letter IATA code of its busiest "
    "international airport.\n"
    "If this is NOT a real country (e.g., 'Narnia', 'xyz'), return 'INVALID'.\n"
    "Return ONLY the IATA code or 'INVALID'. No explanation. No quotes."
)
_IATA_CHAIN = _IATA_PROMPT | _llm_router.with_structured_output(IataCodeResponse)


def resolve_iata_code(country_name: str) -> Optional[str]:
    """Map a country name → primary international gateway IATA code."""
    clean_country = (country_name or "").lower().strip()
    if not clean_country:
        return None

    if clean_country in _iata_cache:
        return _iata_cache[clean_country]

    if clean_country in TOP_DESTINATIONS:
        iata = TOP_DESTINATIONS[clean_country]
        _iata_cache[clean_country] = iata
        return iata

    try:
        result = _IATA_CHAIN.invoke({"country": clean_country})
        iata_code = (result.iata_code or "").strip().upper()

        if iata_code == "INVALID" or len(iata_code) != 3 or not iata_code.isalpha():
            safe_provider_log(logger, "provider.iata.invalid_country")
            _iata_cache[clean_country] = None  # negative result, cache permanently
            return None

        _iata_cache[clean_country] = iata_code
        return iata_code

    except Exception:
        # Transient failure — do NOT cache, allow retry on next call
        safe_provider_log(logger, "provider.iata.failed")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Date utility
# ─────────────────────────────────────────────────────────────────────────────
def calculate_total_nights(start_date: str, end_date: str) -> int:
    """Number of hotel nights between two YYYY-MM-DD dates; 0 for a same-day trip."""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        elapsed_days = (end - start).days
        return 0 if elapsed_days == 0 else max(1, elapsed_days)
    except (ValueError, TypeError):
        return 1


# ─────────────────────────────────────────────────────────────────────────────
# SerpAPI wrapper — manual retry for HTTP/JSON failures
# ─────────────────────────────────────────────────────────────────────────────
def _serpapi_get(
    params: Dict[str, Any],
    max_retries: int = 3,
    initial_wait: float = 1.5,
) -> Dict[str, Any]:
    """SerpAPI GET with JSON-safety + exponential backoff (HTTP-level retry
    is already handled by urllib3 on the shared session; this catches JSON
    parse errors and 4xx that urllib3 won't retry)."""
    url = "https://serpapi.com/search"

    for attempt in range(max_retries):
        try:
            response = _session.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt < max_retries - 1:
                wait = initial_wait * (2**attempt)
                safe_provider_log(logger, "provider.serpapi.retry")
                time.sleep(wait)
            else:
                safe_provider_log(logger, "provider.serpapi.failed")
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Flight search
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_iata_code(value: Any) -> Optional[str]:
    """Return one canonical three-letter IATA identity, or ``None``."""
    if not isinstance(value, str):
        return None
    normalized = value.strip().upper()
    if len(normalized) != 3 or not normalized.isascii() or not normalized.isalpha():
        return None
    return normalized


def _airport_iata_identity(airport: Any) -> Optional[str]:
    """Read a single unambiguous IATA identity from provider airport fields."""
    if not isinstance(airport, dict):
        return None
    raw_identities = [
        airport.get(field)
        for field in ("id", "code")
        if airport.get(field) not in (None, "")
    ]
    if not raw_identities:
        return None
    normalized_identities = [_normalize_iata_code(value) for value in raw_identities]
    if any(identity is None for identity in normalized_identities):
        return None
    identities = set(normalized_identities)
    return next(iter(identities)) if len(identities) == 1 else None


def _flight_matches_query_endpoints(
    flight: Any, origin_iata: str, destination_iata: str
) -> bool:
    """Require parsed provider/cache endpoints to exactly match the query."""
    if not isinstance(flight, dict):
        return False
    expected_origin = _normalize_iata_code(origin_iata)
    expected_destination = _normalize_iata_code(destination_iata)
    if expected_origin is None or expected_destination is None:
        return False
    return (
        _airport_iata_identity(flight.get("departure_airport")) == expected_origin
        and _airport_iata_identity(flight.get("arrival_airport"))
        == expected_destination
    )


def _parse_flight(
    flight: Dict[str, Any], booking_url: str = ""
) -> Optional[Dict[str, Any]]:
    """Parse a single SerpAPI flight object. Returns None if invalid.

    Connecting flights: departure comes from the FIRST leg and arrival from
    the LAST leg — the previous code read both from leg[0], so the "arrival"
    shown to the user was actually the layover airport. Stops/layovers,
    travel class, aircraft and the airline logo are surfaced so the frontend
    can render a complete flight card.

    ``booking_url``: SerpAPI's ``booking_token`` is NOT a URL (it is an opaque
    token for SerpAPI's own booking-options endpoint), so the caller passes a
    Google Flights deep link for the route/date instead.

    NO budget filtering here — the full list is cached and callers filter
    afterwards, so a too-tight budget can fall back to the cheapest real
    option instead of yielding an empty itinerary.
    """
    price_raw = flight.get("price")
    try:
        price = float(price_raw) if price_raw is not None else None
    except (TypeError, ValueError):
        return None
    if price is None:
        return None

    segments = flight.get("flights", [])
    if not segments:
        return None

    first, last = segments[0], segments[-1]
    dep_airport = first.get("departure_airport") or {}
    arr_airport = last.get("arrival_airport") or {}
    layovers = [
        layover.get("name") or layover.get("id") or ""
        for layover in (flight.get("layovers") or [])
        if isinstance(layover, dict)
    ]

    return {
        "airline": first.get("airline", "unknown"),
        "flight_number": first.get("flight_number", "N/A"),
        "airline_logo": flight.get("airline_logo") or first.get("airline_logo", ""),
        "travel_class": first.get("travel_class", ""),
        "airplane": first.get("airplane", ""),
        "departure_airport": dep_airport,
        "arrival_airport": arr_airport,
        "departure_time": first.get("departure_time") or dep_airport.get("time", ""),
        "arrival_time": last.get("arrival_time") or arr_airport.get("time", ""),
        "duration": flight.get("total_duration", first.get("duration", 0)),
        "stops": max(0, len(segments) - 1),
        "layovers": [layover for layover in layovers if layover],
        "price": price,
        "booking_url": booking_url or None,
    }


def fetch_flights_api(
    origin: str,
    destination: str,
    date: str,
    currency: str,
    max_budget: Optional[float] = None,
    adults: int = 1,
) -> List[Dict[str, Any]]:
    """Fetch route-verified flights from SerpAPI, cheapest first. Cache 1 h.

    ``max_budget`` (when given and > 0) filters the returned list; the cache
    always stores the UNFILTERED list so different budgets share one fetch.
    Fresh and cached entries must exactly match the resolved query IATA pair.
    """
    adults = max(1, int(adults or 1))
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_origin = ex.submit(resolve_iata_code, origin)
        f_dest = ex.submit(resolve_iata_code, destination)
        origin_iata = _normalize_iata_code(f_origin.result())
        dest_iata = _normalize_iata_code(f_dest.result())

    if origin_iata is None or dest_iata is None:
        return []

    cache_key = dict(
        origin=origin,
        destination=destination,
        date=date,
        currency=currency,
        adults=adults,
    )
    cached_flights = get_cached_data(
        collection="api_cache", prefix="flight", **cache_key
    )
    valid_flights = [
        flight
        for flight in cached_flights or []
        if _flight_matches_query_endpoints(flight, origin_iata, dest_iata)
    ]

    if not valid_flights:
        params = {
            "engine": "google_flights",
            "departure_id": origin_iata,
            "arrival_id": dest_iata,
            "outbound_date": date,
            # type=2 => one-way. Without this SerpAPI defaults to round-trip
            # (type=1), which requires a return_date we don't supply -> HTTP 400.
            "type": "2",
            "adults": str(adults),
            "currency": currency,
            "hl": "en",
            "api_key": SERPAPI_FLIGHT,
        }

        # Google Flights deep link for this route/date — every flight gets a
        # working "Book flight" URL (SerpAPI's booking_token is not a URL).
        booking_url = (
            "https://www.google.com/travel/flights?q="
            + quote_plus(f"flights from {origin_iata} to {dest_iata} on {date}")
            + f"&curr={currency}&hl=en"
        )

        try:
            data = _serpapi_get(params)
            all_flights = data.get("best_flights", []) + data.get("other_flights", [])
            # Walrus comprehension — single pass, no append overhead
            valid_flights = [
                parsed
                for f in all_flights
                if (parsed := _parse_flight(f, booking_url)) is not None
                and _flight_matches_query_endpoints(
                    parsed, origin_iata, dest_iata
                )
            ]
        except Exception:
            safe_provider_log(logger, "provider.flight.parse_failed")
            valid_flights = []

        valid_flights.sort(key=lambda x: x["price"])

        if valid_flights:
            set_cached_data(
                collection="api_cache",
                prefix="flight",
                payload=valid_flights,
                ttl_hours=1,
                **cache_key,
            )

    if max_budget is not None and max_budget > 0:
        return [f for f in valid_flights if float(f.get("price", 1e18)) <= max_budget]
    return valid_flights


# ─────────────────────────────────────────────────────────────────────────────
# Hotel search
# ─────────────────────────────────────────────────────────────────────────────
def _extract_rate(rate_info: Dict[str, Any]) -> Optional[float]:
    """Extract a numeric nightly rate from SerpAPI's ``rate_per_night``.

    ``extracted_lowest`` is the numeric field; ``lowest`` is a display string
    with currency symbol / thousands separators (e.g. "¥12,345") that
    ``float()`` rejects — parsing only ``lowest`` silently dropped EVERY hotel
    for symbol-formatted currencies (root cause of itineraries missing
    hotels). Prefer the numeric field, then fall back to sanitising the string.
    """
    extracted = rate_info.get("extracted_lowest")
    try:
        if extracted is not None:
            return float(extracted)
    except (TypeError, ValueError):
        pass

    raw = rate_info.get("lowest")
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    cleaned = "".join(ch for ch in str(raw) if ch.isdigit() or ch == ".")
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def _hotel_coordinates(location: Any) -> Optional[tuple[float, float]]:
    """Normalize canonical or legacy hotel coordinates to finite floats."""
    if not isinstance(location, dict):
        return None
    raw_latitude = location.get("latitude")
    raw_longitude = location.get("longitude")
    if raw_latitude is None:
        raw_latitude = location.get("lat")
    if raw_longitude is None:
        raw_longitude = location.get("lng")
    if not _is_finite_number(raw_latitude) or not _is_finite_number(raw_longitude):
        return None
    return float(raw_latitude), float(raw_longitude)


def _normalized_city(value: Any) -> Optional[str]:
    """Return one whitespace-normalized city label, or ``None`` if unusable."""
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split())
    return normalized or None


def _requested_hotel_city(destination: Any) -> Optional[str]:
    """Extract the requested city from the ``city, country`` hotel query."""
    normalized_destination = _normalized_city(destination)
    if normalized_destination is None:
        return None
    city, _separator, _country = normalized_destination.partition(",")
    return _normalized_city(city)


def _verified_hotel_location(
    location: Any,
    *,
    requested_city: Any,
    destination_country_code: Any,
) -> Optional[Dict[str, Any]]:
    """Return canonical coordinates only with exact country and city proof.

    Country and locality are independently reverse-resolved from coordinates.
    The locality proof must be the structured tuple returned by Mapbox and must
    contain an exact whitespace-normalized, case-insensitive requested-city
    match. Provider failures and malformed evidence fail closed.
    """
    coordinates = _hotel_coordinates(location)
    normalized_city = _normalized_city(requested_city)
    if (
        coordinates is None
        or normalized_city is None
        or not isinstance(destination_country_code, str)
    ):
        return None
    country_code = destination_country_code.strip().upper()
    if len(country_code) != 2 or not country_code.isalpha():
        return None

    latitude, longitude = coordinates
    try:
        verified_country_code = resolve_country_code(latitude, longitude)
    except Exception:
        return None
    if verified_country_code != country_code:
        return None
    try:
        locality_names = resolve_locality_names(
            latitude,
            longitude,
            country_code,
        )
    except Exception:
        return None
    if not isinstance(locality_names, tuple):
        return None

    requested_key = normalized_city.casefold()
    locality_matches = any(
        isinstance(name, str)
        and bool(normalized_name := _normalized_city(name))
        and normalized_name.casefold() == requested_key
        for name in locality_names
    )
    if not locality_matches:
        return None
    return {
        "lat": latitude,
        "lng": longitude,
        "country_code": verified_country_code,
    }


def _parse_hotel(
    prop: Dict[str, Any], destination: str = ""
) -> Optional[Dict[str, Any]]:
    """Parse a single SerpAPI hotel property. Returns None if invalid.

    Every hotel gets a booking URL: SerpAPI's ``link`` is missing for many
    properties, so those fall back to a Google Hotels search for the hotel
    name + destination. Guest rating, review count and check-in/out times are
    surfaced so the frontend can render a complete hotel card.

    NO budget filtering here — see ``_parse_flight``.
    """
    rate = _extract_rate(prop.get("rate_per_night") or {})
    if rate is None:
        return None

    images = prop.get("images") or []
    gps = prop.get("gps_coordinates") or {}
    coordinates = _hotel_coordinates(gps)
    latitude, longitude = coordinates if coordinates else (None, None)
    # SerpAPI returns hotel_class as a string ("4-star hotel"); the numeric value
    # is in extracted_hotel_class. Use the numeric one so the response schema
    # (HotelResult.hotel_class: float) validates.
    try:
        hotel_class = float(prop.get("extracted_hotel_class") or 0)
    except (TypeError, ValueError):
        hotel_class = 0.0
    # image may be a plain URL string or an object with a thumbnail field
    first_image = images[0] if images else ""
    if isinstance(first_image, dict):
        first_image = (
            first_image.get("thumbnail") or first_image.get("original_image") or ""
        )

    name = prop.get("name", "No hotel name")
    booking_url = prop.get("link") or (
        "https://www.google.com/travel/hotels?q="
        + quote_plus(f"{name} {destination}".strip())
    )

    try:
        overall_rating = (
            float(prop["overall_rating"])
            if prop.get("overall_rating") is not None
            else None
        )
    except (TypeError, ValueError):
        overall_rating = None
    try:
        reviews = int(prop["reviews"]) if prop.get("reviews") is not None else None
    except (TypeError, ValueError):
        reviews = None

    return {
        "image": first_image,
        "hotel_name": name,
        "hotel_class": hotel_class,
        "overall_rating": overall_rating,
        "reviews": reviews,
        "description": prop.get("description", "No description"),
        "price_per_night": rate,
        "amenities": prop.get("amenities", []),
        "check_in_time": prop.get("check_in_time", ""),
        "check_out_time": prop.get("check_out_time", ""),
        "location": {"lat": latitude, "lng": longitude},
        "booking_url": booking_url,
    }


def fetch_hotels_api(
    destination: str,
    check_in: str,
    check_out: str,
    currency: str,
    max_price_per_night: Optional[float] = None,
    adults: int = 1,
    destination_country_code: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch hotels from SerpAPI, cheapest first. Cached 1 h (full list).

    ``max_price_per_night`` (when given and > 0) filters the returned list;
    the cache always stores the UNFILTERED list.
    """
    adults = max(1, int(adults or 1))
    if destination_country_code is None:
        country_name = destination.rsplit(",", 1)[-1].strip()
        try:
            destination_country_code = pycountry.countries.lookup(country_name).alpha_2
        except LookupError:
            return []
    destination_country_code = destination_country_code.strip().upper()
    if len(destination_country_code) != 2 or not destination_country_code.isalpha():
        return []
    requested_city = _requested_hotel_city(destination)
    if requested_city is None:
        return []
    cache_key = dict(
        destination=destination,
        check_in=check_in,
        check_out=check_out,
        currency=currency,
        adults=adults,
        country_code=destination_country_code,
    )
    valid_hotels = get_cached_data(collection="api_cache", prefix="hotel", **cache_key)
    cache_hit = bool(valid_hotels)

    if not valid_hotels:
        params = {
            "engine": "google_hotels",
            "q": destination,
            "check_in_date": check_in,
            "check_out_date": check_out,
            "adults": str(adults),
            "currency": currency,
            "hl": "en",
            "api_key": SERPAPI_HOTEL,
        }

        try:
            data = _serpapi_get(params)

            provider_error = data.get("error")
            if provider_error:
                logger.warning(
                    "Hotel provider returned an error response."
                )
                return []

            properties = data.get("properties") or []

            logger.info(
                "Hotel provider returned %d raw properties.",
                len(properties),
            )

            valid_hotels = []
            parsed_count = 0
            location_rejected_count = 0

            for prop in properties:
                parsed = _parse_hotel(prop, destination)

                if parsed is None:
                    continue

                parsed_count += 1

                location = parsed.get("location") or {}

                verified_location = _verified_hotel_location(
                    location,
                    requested_city=requested_city,
                    destination_country_code=destination_country_code,
                )

                if verified_location is None:
                    location_rejected_count += 1
                    continue

                parsed["location"] = verified_location
                valid_hotels.append(parsed)

            logger.info(
                "Hotel processing: raw=%d parsed=%d location_rejected=%d verified=%d",
                len(properties),
                parsed_count,
                location_rejected_count,
                len(valid_hotels),
            )
        except Exception:
            safe_provider_log(logger, "provider.hotel.parse_failed")
            valid_hotels = []

        # Cheapest first — consistent with flights
        valid_hotels.sort(key=lambda x: x["price_per_night"])

        if valid_hotels:
            set_cached_data(
                collection="api_cache",
                prefix="hotel",
                payload=valid_hotels,
                ttl_hours=1,
                **cache_key,
            )

    if cache_hit:
        verified_cached_hotels = []
        for hotel in valid_hotels:
            if not isinstance(hotel, dict):
                continue
            location = hotel.get("location") or {}
            verified_location = _verified_hotel_location(
                location,
                requested_city=requested_city,
                destination_country_code=destination_country_code,
            )
            if verified_location is None:
                continue
            verified_cached_hotels.append(
                {
                    **hotel,
                    "location": verified_location,
                }
            )
        valid_hotels = verified_cached_hotels

    if max_price_per_night is not None and max_price_per_night > 0:
        return [
            h
            for h in valid_hotels
            if float(h.get("price_per_night", 1e18)) <= max_price_per_night
        ]
    return valid_hotels


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph pipeline node
# ─────────────────────────────────────────────────────────────────────────────
def _pick_cheapest(
    options: List[Dict[str, Any]],
    price_key: str,
    budget_cap: float,
) -> Optional[Dict[str, Any]]:
    """Pick the cheapest option within *budget_cap*; if none fits, fall back
    to the cheapest option overall and flag it ``over_budget`` so the agent
    (and the UI) can surface the shortfall instead of showing an empty slot.
    """
    if not options:
        return None
    if budget_cap > 0:
        within = [
            o for o in options if float(o.get(price_key, 1e18) or 1e18) <= budget_cap
        ]
        if within:
            return dict(within[0])
    fallback = dict(options[0])
    if budget_cap > 0:
        fallback["over_budget"] = True
    return fallback


def _select_flight_pair(
    outbound_options: List[Dict[str, Any]],
    return_options: List[Dict[str, Any]],
    budget_cap: float,
) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Copy the cheapest available legs and assess their combined fare."""
    selected_outbound = dict(outbound_options[0]) if outbound_options else None
    selected_return = dict(return_options[0]) if return_options else None
    selected = [
        flight
        for flight in (selected_outbound, selected_return)
        if flight is not None
    ]
    total_price = sum(float(flight.get("price", 0.0) or 0.0) for flight in selected)
    if budget_cap > 0 and total_price > budget_cap:
        for flight in selected:
            flight["over_budget"] = True
    return selected_outbound, selected_return


def _future_options_or_empty(future: Any, label: str) -> List[Dict[str, Any]]:
    """Return one provider result list without letting its failure erase peers."""
    try:
        result = future.result()
    except Exception:
        safe_provider_log(logger, "provider.flight_hotel.search_failed")
        return []
    return result if isinstance(result, list) else []


def _assessment_evidence_for_state(
    state: AgentState,
    destination_country_code: Optional[str],
    hotel_destination: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return evidence only when its inputs, flight route, and hotel match."""
    assessment = state.budget_assessment
    if not isinstance(assessment, dict):
        return None
    try:
        expires_at = datetime.fromisoformat(str(assessment.get("expires_at") or ""))
    except ValueError:
        return None
    if expires_at.tzinfo is None or expires_at <= datetime.now(timezone.utc):
        return None
    if (
        str(assessment.get("calculation_version") or "") != "allocation-v1"
        or str(assessment.get("origin") or "").strip().lower()
        != str(state.origin_country or "").strip().lower()
        or str(assessment.get("destination") or "").strip().lower()
        != str(state.country or "").strip().lower()
        or str(assessment.get("start_date") or "") != str(state.start_date or "")
        or str(assessment.get("end_date") or "") != str(state.end_date or "")
    ):
        return None
    try:
        if int(assessment.get("num_people") or 0) != max(1, int(state.num_people or 1)):
            return None
    except (TypeError, ValueError):
        return None
    evidence = assessment.get("evidence")
    if not isinstance(evidence, dict):
        return None
    required = ("outbound_flight", "return_flight", "hotel")
    if not all(isinstance(evidence.get(key), dict) for key in required):
        return None
    origin_iata = resolve_iata_code(str(state.origin_country or ""))
    destination_iata = resolve_iata_code(str(state.country or ""))
    if (
        not _flight_matches_query_endpoints(
            evidence["outbound_flight"], origin_iata or "", destination_iata or ""
        )
        or not _flight_matches_query_endpoints(
            evidence["return_flight"], destination_iata or "", origin_iata or ""
        )
    ):
        return None
    total_nights = calculate_total_nights(
        str(state.start_date or ""), str(state.end_date or "")
    )
    hotel = evidence["hotel"]
    if total_nights == 0:
        if (
            hotel
            or evidence.get("hotel_nights") != 0
            or evidence.get("hotel_price_per_night") != 0
        ):
            return None
        return {**evidence, "hotel": {}}
    location = hotel.get("location") or {}
    if hotel_destination is None:
        primary_city = next((city for city in (state.city or []) if city), "")
        hotel_destination = ", ".join(
            part for part in (primary_city, state.country or "") if part
        )
    requested_city = _requested_hotel_city(hotel_destination)
    verified_location = _verified_hotel_location(
        location,
        requested_city=requested_city,
        destination_country_code=destination_country_code,
    )
    if verified_location is None:
        return None
    return {
        **evidence,
        "hotel": {
            **hotel,
            "location": verified_location,
        },
    }


def _build_itinerary_skeleton(
    *,
    start_date: str,
    end_date: str,
    outbound: Optional[Dict[str, Any]],
    return_flight: Optional[Dict[str, Any]],
    hotel: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build boundary flights, nightly hotel charges, and empty day slots."""
    try:
        base_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_value = datetime.strptime(end_date, "%Y-%m-%d")
    except (ValueError, TypeError):
        base_date = None
        total_nights = calculate_total_nights(start_date, end_date)
    else:
        total_nights = 0 if base_date == end_date_value else calculate_total_nights(
            start_date, end_date
        )

    hotel_price = hotel["price_per_night"] if hotel else 0.0
    outbound_price = outbound["price"] if outbound else 0.0
    return_price = return_flight["price"] if return_flight else 0.0
    last_day_num = total_nights + 1
    itinerary_data: List[Dict[str, Any]] = []
    for day_idx in range(total_nights + 1):
        day_num = day_idx + 1
        day_date = (
            (base_date + timedelta(days=day_idx)).strftime("%Y-%m-%d")
            if base_date
            else ""
        )
        hotel_cost = hotel_price if day_num <= total_nights else 0.0
        if day_num == 1 and day_num == last_day_num:
            selected_day_flight = [
                flight for flight in (outbound, return_flight) if flight is not None
            ]
            flight_cost = outbound_price + return_price
        elif day_num == 1:
            selected_day_flight = [outbound] if outbound else []
            flight_cost = outbound_price
        elif day_num == last_day_num:
            selected_day_flight = [return_flight] if return_flight else []
            flight_cost = return_price
        else:
            selected_day_flight = []
            flight_cost = 0.0
        itinerary_data.append(
            {
                "day": day_num,
                "date": day_date,
                "flight": selected_day_flight or None,
                "hotel": hotel if day_num <= total_nights else None,
                "activities": [],
                "route": None,
                "day_total_cost": hotel_cost + flight_cost,
            }
        )
    return itinerary_data


def plan_flight_hotel(state: AgentState) -> Dict[str, Any]:
    """Automated LangGraph node — search Google Flights & Hotels in parallel."""
    origin = state.origin_country or ""
    destination = state.country or ""
    hotel_destination = ", ".join(
        part
        for part in (next((city for city in (state.city or []) if city), ""), destination)
        if part
    )
    start_date = state.start_date or ""
    end_date = state.end_date or ""
    currency = state.dest_currency_code or "USD"
    num_people = max(1, int(state.num_people or 1))
    try:
        destination_country_code = pycountry.countries.lookup(destination).alpha_2
    except LookupError:
        destination_country_code = None

    assessment_evidence = _assessment_evidence_for_state(
        state,
        destination_country_code,
        hotel_destination,
    )
    if assessment_evidence is not None:
        return {
            "draft_itinerary": _build_itinerary_skeleton(
                start_date=start_date,
                end_date=end_date,
                outbound=dict(assessment_evidence["outbound_flight"]),
                return_flight=dict(assessment_evidence["return_flight"]),
                hotel=dict(assessment_evidence["hotel"]),
            )
        }

    allocation = state.budget_allocation or {}
    # "transportation" is canonical; "flight" is a legacy alias kept for old
    # checkpoints created before the alias was removed.
    flight_budget = float(
        allocation.get("transportation") or allocation.get("flight") or 0.0
    )
    hotel_budget = float(allocation.get("accommodation", 0.0) or 0.0)

    total_nights = calculate_total_nights(start_date, end_date)
    max_hotel_rate = (hotel_budget / total_nights) if total_nights > 0 else hotel_budget

    # Fetch UNFILTERED lists — budget preference is applied in _pick_cheapest
    # so a too-tight budget degrades to "cheapest real option, flagged
    # over_budget" instead of an empty itinerary (reported issue: itineraries
    # sometimes missing flight/hotel entirely).
    with ThreadPoolExecutor(max_workers=2 if total_nights == 0 else 3) as ex:
        f_outbound = ex.submit(
            fetch_flights_api,
            origin,
            destination,
            start_date,
            currency,
            None,
            num_people,
        )
        f_return = ex.submit(
            fetch_flights_api,
            destination,
            origin,
            end_date,
            currency,
            None,
            num_people,
        )
        f_hotel = None
        if total_nights > 0:
            f_hotel = ex.submit(
                fetch_hotels_api,
                hotel_destination,
                start_date,
                end_date,
                currency,
                None,
                num_people,
                destination_country_code,
            )
        outbound_options = _future_options_or_empty(f_outbound, "outbound flight")
        return_options = _future_options_or_empty(f_return, "return flight")
        hotel_options = (
            _future_options_or_empty(f_hotel, "hotel") if f_hotel is not None else []
        )

    selected_outbound, selected_return = _select_flight_pair(
        outbound_options, return_options, flight_budget
    )
    selected_hotel = _pick_cheapest(hotel_options, "price_per_night", max_hotel_rate)

    if selected_outbound is None:
        safe_provider_log(logger, "provider.flight_hotel.no_outbound")
    if selected_return is None:
        safe_provider_log(logger, "provider.flight_hotel.no_return")
    if total_nights > 0 and selected_hotel is None:
        safe_provider_log(logger, "provider.flight_hotel.no_hotel")

    return {
        "draft_itinerary": _build_itinerary_skeleton(
            start_date=start_date,
            end_date=end_date,
            outbound=selected_outbound,
            return_flight=selected_return,
            hotel=selected_hotel,
        )
    }


# ─────────────────────────────────────────────────────────────────────────────
# Agent tools
# ─────────────────────────────────────────────────────────────────────────────
@tool
def search_alternative_opt(
    category: Literal["flight", "hotel", "accommodation"],
    day_num: int,
    state: Annotated[AgentState, InjectedState],
) -> Dict[str, Any]:
    """Search trusted-state flight or hotel alternatives for one itinerary day."""
    cat = (category or "").lower()
    adults = max(1, int(state.num_people or 1))
    origin = state.origin_country or ""
    country = state.country or ""
    start_date = state.start_date or ""
    end_date = state.end_date or ""
    currency = state.dest_currency_code or "USD"
    allocation = state.budget_allocation or {}

    if cat == "flight":
        configured_days = [
            day.get("day")
            for day in (state.draft_itinerary or [])
            if isinstance(day, dict) and isinstance(day.get("day"), int)
        ]
        first_day = min(configured_days) if configured_days else 1
        last_day = (
            max(configured_days)
            if configured_days
            else calculate_total_nights(start_date, end_date) + 1
        )
        if day_num == first_day:
            flight_origin, flight_destination, flight_date = (
                origin,
                country,
                start_date,
            )
        elif day_num == last_day:
            flight_origin, flight_destination, flight_date = (
                country,
                origin,
                end_date,
            )
        else:
            return {"error": "Flight alternatives are only valid on a boundary day."}
        max_budget = float(
            allocation.get("transportation") or allocation.get("flight") or 0.0
        )
        options = fetch_flights_api(
            flight_origin,
            flight_destination,
            flight_date,
            currency,
            max_budget,
            adults,
        )
        if not options:
            # Second pass without the budget cap — return the cheapest real
            # options (flagged) so the agent can present concrete alternatives.
            unfiltered = fetch_flights_api(
                flight_origin,
                flight_destination,
                flight_date,
                currency,
                None,
                adults,
            )
            if unfiltered:
                top3 = [dict(o, over_budget=True) for o in unfiltered[:3]]
                return {
                    "options": top3,
                    "count": len(top3),
                    "warning": (
                        f"No flights found within {max_budget} {currency}. "
                        "These are the CHEAPEST real options — each exceeds the "
                        "budget, so warn the user and suggest a budget "
                        "reallocation or date change."
                    ),
                }
            return {
                "error": "No alternative flights found at all for these dates.",
                "suggestion": "Consider adjusting dates or the destination.",
            }

        top3 = options[:3]
        return {"options": top3, "count": len(top3)}

    if cat in ("hotel", "accommodation"):
        max_budget = float(allocation.get("accommodation", 0.0) or 0.0)
        total_nights = calculate_total_nights(start_date, end_date)
        max_rate = (max_budget / total_nights) if total_nights > 0 else max_budget
        configured_city = next((city for city in (state.city or []) if city), "")
        destination = ", ".join(
            part for part in (configured_city, country) if part
        )
        try:
            destination_country_code = pycountry.countries.lookup(country).alpha_2
        except LookupError:
            return {"error": "Destination country could not be verified."}
        options = fetch_hotels_api(
            destination,
            start_date,
            end_date,
            currency,
            max_rate,
            adults,
            destination_country_code,
        )
        if not options:
            unfiltered = fetch_hotels_api(
                destination,
                start_date,
                end_date,
                currency,
                None,
                adults,
                destination_country_code,
            )
            if unfiltered:
                top3 = [dict(o, over_budget=True) for o in unfiltered[:3]]
                return {
                    "options": top3,
                    "count": len(top3),
                    "warning": (
                        f"No hotels found within {max_rate:.2f} {currency}/night. "
                        "These are the CHEAPEST real options — each exceeds the "
                        "budget, so warn the user and suggest a budget "
                        "reallocation or date change."
                    ),
                }
            return {
                "error": "No alternative hotels found at all for these dates.",
                "suggestion": "Consider adjusting dates or the destination.",
            }

        top3 = options[:3]
        return {"options": top3, "count": len(top3)}

    return {
        "error": f"Unknown category: '{category}'. Please specify 'flight' or 'hotel'."
    }


@tool
def modify_existing_booking(
    day_num: int,
    category: str,
    new_details: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Signals the graph to replace a flight or hotel in the draft itinerary.
    The actual state update is applied by the post_tool_processing node
    after this tool returns. Do NOT call this tool unless the user has
    explicitly confirmed the new selection.

    Args:
        day_num: The day number in the itinerary to update (e.g. 1).
        category: Either 'flight' or 'hotel'.
        new_details: Dict matching FlightResult or HotelResult schema.
    """
    return {
        "status": "success",
        "action": "update_draft",
        "target_day": day_num,
        "target_category": category,
        "new_data": new_details,
        "message": f"Draft updated successfully. Day {day_num} {category} has been replaced.",
    }


class ItineraryEdit(BaseModel):
    """One change to apply to the itinerary."""

    day: int = Field(
        ge=1,
        description=(
            "Day number to edit (1-based). For activity/restaurant remove or "
            "replace, this day MUST be explicitly stated in the user's current "
            "message; never infer it from conversation context."
        ),
    )
    action: Literal["add", "remove", "replace"] = Field(
        description="'add' a new item, 'remove' an existing one, or 'replace' one."
    )
    category: Literal["flight", "hotel", "activity", "restaurant"] = Field(
        description="What to edit. 'activity' and 'restaurant' live in the day's activities list."
    )
    index: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "For remove/replace of an activity or restaurant: the 1-based number "
            "shown in that day's activities list (e.g. '[1] ...'). Ignored for flight/hotel."
        ),
    )
    new_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "The full item dict for add/replace. For flight/hotel use the object from "
            "search_alternative_opt; for activity/restaurant use an object from search_places "
            "(must include a location with latitude/longitude so the map/route can update)."
        ),
    )

    @model_validator(mode="after")
    def validate_place_details(self) -> "ItineraryEdit":
        """Prevent the agent from inventing incomplete place cards.

        Activity/restaurant additions and replacements must copy a verified
        ``search_places`` result, including its coordinates. Without this
        check a plausible-looking name/rating/address dict can enter the saved
        itinerary even though it cannot be mapped or reliably identified.
        """
        if self.action not in {"add", "replace"}:
            return self
        if self.category not in {"activity", "restaurant"}:
            return self

        details = self.new_details or {}
        name = details.get("name")
        location = details.get("location")
        lat = location.get("latitude") if isinstance(location, dict) else None
        lng = location.get("longitude") if isinstance(location, dict) else None
        if (
            not isinstance(name, str)
            or not name.strip()
            or not _is_finite_number(lat)
            or not _is_finite_number(lng)
        ):
            raise ValueError(
                "Activity/restaurant new_details must be copied from search_places "
                "and include a name plus location.latitude/location.longitude."
            )
        return self


@tool
def edit_itinerary(edits: List[ItineraryEdit]) -> Dict[str, Any]:
    """Apply one or more edits to the draft itinerary in a single call.

    Use this for ANY itinerary change the user confirms: swapping a flight/hotel,
    or adding / removing / replacing an attraction or restaurant. You may pass
    multiple edits at once (e.g. change two places on day 2, plus the hotel on
    day 3). The graph applies them, recomputes costs, and regenerates the maps
    for affected days.

        Guidance:
      • flight/hotel  → action='replace', new_details from search_alternative_opt.
      • activity/restaurant add     → action='add', new_details from search_places.
      • activity/restaurant remove  → action='remove', index=<the 1-based [n] shown in the day>.
      • activity/restaurant replace → action='replace', index=<the 1-based [n]>, new_details from search_places.
      • For activity/restaurant remove or replace, normally require the target
      day in the current user message. A server-owned clarification reply or
      a bounded affirmative confirmation may also supply the day when the
      server has verified that the user explicitly stated that same day before
      the immediately preceding proposal. Never trust an assistant-only guess.
      Only call this after the user has confirmed the change and target day.
    """
    return {
        "status": "pending_validation",
        "action": "edit_itinerary",
        "edits": [e.model_dump() for e in edits],
        "message": (
            f"Submitted {len(edits)} itinerary edit(s) "
            "for validation."
        ),
    }
