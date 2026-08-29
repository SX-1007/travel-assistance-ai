"""Resolve optional user destination input into verified planning cities."""

from __future__ import annotations

import logging
from typing import Any

import pycountry
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from app.core.config import settings
from app.tools.mapbox import geocode_location, resolve_country_code


logger = logging.getLogger(__name__)

# Countries where the country name itself is the practical city/locality.
# The mapping is keyed by an already-verified ISO country code, so these
# destinations do not need an AI call to choose their primary planning city.
_CITY_STATE_PRIMARY_CITY: dict[str, str] = {
    "SG": "Singapore",
    "MC": "Monaco",
    "VA": "Vatican City",
}


class DestinationChoice(BaseModel):
    """Structured city selected when the user supplies only a country."""

    city: str = Field(min_length=1)


class DestinationResolutionInvalid(ValueError):
    """A user-provided country/city combination is invalid."""


class DestinationResolutionUnavailable(RuntimeError):
    """A safe planning city could not be resolved or verified."""


_destination_llm = ChatGoogleGenerativeAI(
    model=settings.GEMINI_UTILITY_MODEL,
    temperature=0,
    google_api_key=settings.GEMINI_API_KEY,
    max_retries=2,
    thinking_budget=0,
)


_DESTINATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You select one practical primary travel city for a country-only trip. "
            "The city must be physically inside the requested destination country. "
            "Prefer a major tourism base with broad transport, hotel, and attraction "
            "coverage. For a city-state, return the city-state name. Do not return a "
            "country, airport, attraction, or city from another country. Return only "
            "the required structured city field.",
        ),
        (
            "human",
            "Destination country: {country}\n"
            "Trip dates: {start_date} to {end_date}\n"
            "Travellers: {num_people}\n"
            "Previously rejected candidates: {rejected}",
        ),
    ]
)


_DESTINATION_CHAIN = (
    _DESTINATION_PROMPT
    | _destination_llm.with_structured_output(DestinationChoice)
)


def _country_alpha_2(country: str) -> str:
    """Resolve a destination country to ISO 3166-1 alpha-2."""

    try:
        return pycountry.countries.lookup(country.strip()).alpha_2
    except (AttributeError, LookupError) as exc:
        raise DestinationResolutionInvalid(
            "Destination country could not be verified."
        ) from exc


def _clean_city(value: Any) -> str:
    """Normalize whitespace in one city/locality value."""

    if not isinstance(value, str):
        return ""

    return " ".join(value.split())


def _verified_city(
    *,
    city: str,
    country: str,
    country_code: str,
    user_supplied: bool,
) -> str | None:
    """Return a canonical verified locality or reject a wrong-country locality."""

    clean_city = _clean_city(city)

    if not clean_city:
        if user_supplied:
            raise DestinationResolutionInvalid(
                "Every provided destination state/city must be non-empty."
            )

        return None

    # Forward geocode the locality together with the trusted destination country.
    resolved = geocode_location(
        f"{clean_city}, {country.strip()}"
    )

    if not isinstance(resolved, dict):
        raise DestinationResolutionUnavailable(
            "Destination locality verification is currently unavailable."
        )

    lat = resolved.get("lat")
    lng = resolved.get("lng")

    if lat is None or lng is None:
        raise DestinationResolutionUnavailable(
            "Destination locality verification returned no coordinates."
        )

    # Reverse-check the coordinates. This prevents an AI-selected or
    # user-entered locality in another country from entering planning state.
    verified_country_code = resolve_country_code(lat, lng)

    if verified_country_code is None:
        raise DestinationResolutionUnavailable(
            "Destination country verification is currently unavailable."
        )

    if verified_country_code != country_code:
        if user_supplied:
            raise DestinationResolutionInvalid(
                f"'{clean_city}' could not be verified inside "
                f"{country.strip()}."
            )

        return None

    # The candidate has already been geocoded and its coordinates have been
    # verified to belong to the requested destination country.
    #
    # Keep the practical city/locality label supplied by the user or selected
    # by the AI. Mapbox may canonicalise a travel city into a wider
    # administrative region (for example, "Tokyo" -> "Tokyo Prefecture"),
    # which can break downstream hotel locality matching.


    return clean_city


def resolve_destination_cities(
    *,
    country: str,
    requested_cities: list[str],
    start_date: str,
    end_date: str,
    num_people: int,
) -> list[str]:
    """Return verified server-owned planning cities.

    The public form may omit city information. This function guarantees that
    downstream budget, provider, itinerary, and map operations receive at
    least one verified locality.
    """

    country_code = _country_alpha_2(country)

    # ── User supplied one or more cities ──────────────────────────────
    if requested_cities:
        resolved_cities: list[str] = []
        seen: set[str] = set()

        for requested_city in requested_cities:
            verified = _verified_city(
                city=requested_city,
                country=country,
                country_code=country_code,
                user_supplied=True,
            )

            if not verified:
                continue

            key = verified.casefold()

            if key not in seen:
                seen.add(key)
                resolved_cities.append(verified)

        if not resolved_cities:
            raise DestinationResolutionInvalid(
                "At least one valid destination state/city is required "
                "when city data is supplied."
            )

        return resolved_cities

    # ── Country-only city-state request: resolve deterministically ────
    # Singapore, Monaco and Vatican City do not need an LLM to invent a
    # primary city. The mapping is keyed by the already-verified ISO country
    # code, so this server-owned result is deterministic and needs no provider
    # call before budget assessment.
    city_state_candidate = _CITY_STATE_PRIMARY_CITY.get(country_code)

    if city_state_candidate:
        return [city_state_candidate]

    # ── Other country-only requests: let AI propose the primary city ──
    rejected: list[str] = []

    for _ in range(3):
        try:
            proposal = _DESTINATION_CHAIN.invoke(
                {
                    "country": country.strip(),
                    "start_date": start_date,
                    "end_date": end_date,
                    "num_people": num_people,
                    "rejected": (
                        ", ".join(rejected)
                        if rejected
                        else "None"
                    ),
                }
            )

        except Exception as exc:
            # Keep the public response generic, but preserve the provider/error
            # class in server logs so validation failures are diagnosable.
            logger.warning(
                "AI destination selection failed for country_code=%s: %s",
                country_code,
                type(exc).__name__,
            )

            raise DestinationResolutionUnavailable(
                "AI destination selection is currently unavailable."
            ) from exc

        candidate = _clean_city(
            getattr(proposal, "city", "")
        )

        if not candidate:
            rejected.append("empty candidate")
            continue

        verified = _verified_city(
            city=candidate,
            country=country,
            country_code=country_code,
            user_supplied=False,
        )

        if verified:
            return [verified]

        rejected.append(candidate)

    raise DestinationResolutionUnavailable(
        "A verified planning city could not be selected for this country."
    )