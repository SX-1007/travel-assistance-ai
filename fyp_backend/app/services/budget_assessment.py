"""Provider-grounded minimum-budget assessment for new trips."""

from __future__ import annotations

import math
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

from app.core.firebase_db import (
    claim_cached_data,
    complete_cached_data_claim,
    get_cached_data,
    invalidate_cached_data,
    release_cached_data_claim,
    wait_for_cache_claim_change,
)
from app.tools.currency import get_currency_code, get_currency_rate
from app.tools.flights_hotels import (
    calculate_total_nights,
    fetch_flights_api,
    fetch_hotels_api,
)
from app.tools.provider_logging import safe_provider_log

CALCULATION_VERSION = "allocation-v1"
ASSESSMENT_TTL_HOURS = 1
ASSESSMENT_CLAIM_LEASE_SECONDS = 120
ASSESSMENT_CLAIM_WAIT_SECONDS = 10.0
ASSESSMENT_CLAIM_POLL_SECONDS = 0.25
logger = logging.getLogger(__name__)
DEFAULT_BUDGET_RATIOS: dict[str, float] = {
    "transportation": 0.25,
    "accommodation": 0.35,
    "food": 0.15,
    "activity": 0.15,
    "shopping": 0.05,
    "emergency_fund": 0.05,
}


class BudgetEvidence(BaseModel):
    """Cheapest real provider objects and their calculation inputs."""

    outbound_flight: dict[str, Any]
    return_flight: dict[str, Any]
    hotel: dict[str, Any]
    outbound_flight_price: float = Field(gt=0, allow_inf_nan=False)
    return_flight_price: float = Field(gt=0, allow_inf_nan=False)
    hotel_price_per_night: float = Field(ge=0, allow_inf_nan=False)
    hotel_nights: int = Field(ge=0)

    @model_validator(mode="after")
    def _hotel_evidence_matches_overnight_requirement(self) -> "BudgetEvidence":
        if self.hotel_nights == 0:
            if self.hotel_price_per_night != 0 or self.hotel:
                raise ValueError("zero-night trips require canonical empty hotel evidence")
        elif self.hotel_price_per_night <= 0:
            raise ValueError("overnight trips require a positive hotel price")
        return self


class BudgetAssessment(BaseModel):
    """A reusable, provider-grounded minimum budget for one exact trip."""

    assessment_id: str = Field(min_length=1)
    calculation_version: str = Field(default=CALCULATION_VERSION, min_length=1)
    origin: str = Field(min_length=1)
    destination: str = Field(min_length=1)
    destination_city: str = Field(min_length=1)
    start_date: str = Field(min_length=1)
    end_date: str = Field(min_length=1)
    num_people: int = Field(ge=1)
    base_currency: str = Field(min_length=3, max_length=3)
    destination_currency: str = Field(min_length=3, max_length=3)
    exchange_rate: float = Field(gt=0, allow_inf_nan=False)
    minimum_destination_budget: float = Field(gt=0, allow_inf_nan=False)
    recommended_minimum_budget: float = Field(gt=0, allow_inf_nan=False)
    evidence: BudgetEvidence
    created_at: str
    expires_at: str


class BudgetAssessmentResult(BaseModel):
    """Assessment plus whether it is safe to confirm in a later request."""

    assessment: BudgetAssessment
    persisted: bool


class BudgetAssessmentUnavailable(RuntimeError):
    """Required real pricing or exchange-rate evidence is unavailable."""


def primary_destination_city(cities: Any) -> str:
    """Return the exact first requested city used by the hotel budget floor."""
    if not isinstance(cities, (list, tuple)) or not cities:
        return ""
    city = cities[0]
    return city.strip() if isinstance(city, str) else ""


def assessment_cache_key_kwargs(
    *,
    origin: str,
    destination: str,
    destination_city: str,
    start_date: str,
    end_date: str,
    num_people: int,
    base_currency: str,
    destination_currency: str,
) -> dict[str, Any]:
    """Return the budget-independent identity of a trip assessment."""
    return {
        "origin": origin.strip().lower(),
        "destination": destination.strip().lower(),
        "destination_city": destination_city.strip().lower(),
        "start_date": start_date.strip(),
        "end_date": end_date.strip(),
        "num_people": int(num_people),
        "base_currency": base_currency.strip().upper(),
        "destination_currency": destination_currency.strip().upper(),
        "calculation_version": CALCULATION_VERSION,
    }


def _finite_positive(value: Any, label: str) -> float:
    """Return a required positive number or fail the grounded assessment."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise BudgetAssessmentUnavailable(f"Invalid {label}.") from exc
    if not math.isfinite(number) or number <= 0:
        raise BudgetAssessmentUnavailable(f"Invalid {label}.")
    return number


def _assessment_arithmetic_is_valid(assessment: BudgetAssessment) -> bool:
    """Reject cached evidence whose objects and derived amounts disagree."""
    evidence = assessment.evidence
    try:
        object_outbound = _finite_positive(
            evidence.outbound_flight.get("price"),
            "cached outbound price",
        )
        object_return = _finite_positive(
            evidence.return_flight.get("price"),
            "cached return price",
        )
    except BudgetAssessmentUnavailable:
        return False

    price_pairs = [
        (object_outbound, evidence.outbound_flight_price),
        (object_return, evidence.return_flight_price),
    ]
    if evidence.hotel_nights:
        try:
            object_hotel = _finite_positive(
                evidence.hotel.get("price_per_night"), "cached hotel price"
            )
        except BudgetAssessmentUnavailable:
            return False
        price_pairs.append((object_hotel, evidence.hotel_price_per_night))
    elif evidence.hotel or evidence.hotel_price_per_night != 0:
        return False
    if not all(math.isclose(left, right, rel_tol=1e-12) for left, right in price_pairs):
        return False
    if evidence.hotel_nights != calculate_total_nights(
        assessment.start_date,
        assessment.end_date,
    ):
        return False

    expected_minimum = max(
        (evidence.outbound_flight_price + evidence.return_flight_price)
        / DEFAULT_BUDGET_RATIOS["transportation"],
        (evidence.hotel_price_per_night * evidence.hotel_nights)
        / DEFAULT_BUDGET_RATIOS["accommodation"],
    )
    expected_recommendation = (
        math.ceil((expected_minimum / assessment.exchange_rate) * 100.0) / 100.0
    )
    return math.isclose(
        assessment.minimum_destination_budget,
        expected_minimum,
        rel_tol=1e-12,
    ) and math.isclose(
        assessment.recommended_minimum_budget,
        expected_recommendation,
        rel_tol=1e-12,
    )


def assessment_matches_trip(
    assessment: BudgetAssessment | dict[str, Any],
    *,
    origin: str,
    destination: str,
    destination_city: str,
    start_date: str,
    end_date: str,
    num_people: int,
) -> bool:
    """Return whether evidence belongs to the exact current trip inputs."""
    try:
        parsed = (
            assessment
            if isinstance(assessment, BudgetAssessment)
            else BudgetAssessment.model_validate(assessment)
        )
    except ValidationError:
        return False
    try:
        expires_at = datetime.fromisoformat(parsed.expires_at)
    except ValueError:
        return False
    if expires_at.tzinfo is None or expires_at <= datetime.now(timezone.utc):
        return False
    return (
        parsed.calculation_version == CALCULATION_VERSION
        and _assessment_arithmetic_is_valid(parsed)
        and parsed.origin.strip().lower() == origin.strip().lower()
        and parsed.destination.strip().lower() == destination.strip().lower()
        and parsed.destination_city.strip().lower()
        == destination_city.strip().lower()
        and parsed.start_date == start_date
        and parsed.end_date == end_date
        and parsed.num_people == int(num_people)
    )


def _unexpired_cached_assessment(
    payload: Any,
    *,
    origin: str,
    destination: str,
    destination_city: str,
    start_date: str,
    end_date: str,
    num_people: int,
) -> BudgetAssessment | None:
    if not payload:
        return None
    try:
        assessment = BudgetAssessment.model_validate(payload)
        expires_at = datetime.fromisoformat(assessment.expires_at)
    except (TypeError, ValueError, ValidationError):
        return None
    if expires_at.tzinfo is None or expires_at <= datetime.now(timezone.utc):
        return None
    if not assessment_matches_trip(
        assessment,
        origin=origin,
        destination=destination,
        destination_city=destination_city,
        start_date=start_date,
        end_date=end_date,
        num_people=num_people,
    ):
        return None
    return assessment


def _first_provider_result(future: Any, label: str) -> dict[str, Any]:
    """Read one cheapest result without accepting missing provider evidence."""
    try:
        results = future.result()
    except Exception as exc:
        raise BudgetAssessmentUnavailable(f"{label} search failed.") from exc
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        raise BudgetAssessmentUnavailable(f"No {label} result is available.")
    return dict(results[0])


def get_or_create_budget_assessment(
    *,
    origin: str,
    destination: str,
    destination_city: str,
    start_date: str,
    end_date: str,
    num_people: int,
) -> BudgetAssessmentResult:
    """Create a deterministic minimum from cheapest flight and hotel results."""
    try:
        base_currency = get_currency_code(origin)
        destination_currency = get_currency_code(destination)
    except Exception as exc:
        raise BudgetAssessmentUnavailable("Currency resolution failed.") from exc
    cache_key = assessment_cache_key_kwargs(
        origin=origin,
        destination=destination,
        destination_city=destination_city,
        start_date=start_date,
        end_date=end_date,
        num_people=num_people,
        base_currency=base_currency,
        destination_currency=destination_currency,
    )
    try:
        cached = get_cached_data(
            collection="api_cache",
            prefix="budget_assessment",
            **cache_key,
        )
    except Exception:
        safe_provider_log(logger, "budget.cache_read_failed")
        cached = None
    cached_assessment = _unexpired_cached_assessment(
        cached,
        origin=origin,
        destination=destination,
        destination_city=destination_city,
        start_date=start_date,
        end_date=end_date,
        num_people=num_people,
    )
    if cached_assessment is not None:
        return BudgetAssessmentResult(
            assessment=cached_assessment,
            persisted=True,
        )

    claim_id: str | None = None
    deadline = time.monotonic() + ASSESSMENT_CLAIM_WAIT_SECONDS
    while claim_id is None:
        claim_status, claim_payload, candidate_claim_id = claim_cached_data(
            collection="api_cache",
            prefix="budget_assessment",
            lease_seconds=ASSESSMENT_CLAIM_LEASE_SECONDS,
            **cache_key,
        )
        if claim_status == "cached":
            claimed_assessment = _unexpired_cached_assessment(
                claim_payload,
                origin=origin,
                destination=destination,
                destination_city=destination_city,
                start_date=start_date,
                end_date=end_date,
                num_people=num_people,
            )
            if claimed_assessment is not None:
                return BudgetAssessmentResult(assessment=claimed_assessment, persisted=True)
            invalidate_cached_data(
                collection="api_cache",
                prefix="budget_assessment",
                **cache_key,
            )
            continue
        if claim_status == "claimed" and candidate_claim_id:
            claim_id = candidate_claim_id
            break
        if claim_status != "busy" or time.monotonic() >= deadline:
            raise BudgetAssessmentUnavailable("Budget assessment creation is unavailable.")
        wait_for_cache_claim_change(
            collection="api_cache",
            prefix="budget_assessment",
            timeout_seconds=min(
                ASSESSMENT_CLAIM_POLL_SECONDS,
                max(0.0, deadline - time.monotonic()),
            ),
            **cache_key,
        )

    try:
        try:
            raw_exchange_rate = get_currency_rate(base_currency, destination_currency)
        except Exception as exc:
            raise BudgetAssessmentUnavailable("Exchange-rate lookup failed.") from exc
        exchange_rate = _finite_positive(raw_exchange_rate, "exchange rate")
        nights = calculate_total_nights(start_date, end_date)
        with ThreadPoolExecutor(max_workers=2 if nights == 0 else 3) as executor:
            outbound_future = executor.submit(
                fetch_flights_api,
                origin,
                destination,
                start_date,
                destination_currency,
                None,
                num_people,
            )
            return_future = executor.submit(
                fetch_flights_api,
                destination,
                origin,
                end_date,
                destination_currency,
                None,
                num_people,
            )
            hotel_future = None
            if nights > 0:
                hotel_future = executor.submit(
                    fetch_hotels_api,
                    f"{destination_city.strip()}, {destination.strip()}",
                    start_date,
                    end_date,
                    destination_currency,
                    None,
                    num_people,
                )
            outbound = _first_provider_result(outbound_future, "outbound flight")
            return_flight = _first_provider_result(return_future, "return flight")
            hotel = (
                _first_provider_result(hotel_future, "hotel")
                if hotel_future is not None
                else {}
            )

        outbound_price = _finite_positive(outbound.get("price"), "outbound price")
        return_price = _finite_positive(return_flight.get("price"), "return price")
        hotel_price = (
            _finite_positive(hotel.get("price_per_night"), "hotel price")
            if nights > 0
            else 0.0
        )
        minimum_destination = max(
            (outbound_price + return_price)
            / DEFAULT_BUDGET_RATIOS["transportation"],
            (hotel_price * nights) / DEFAULT_BUDGET_RATIOS["accommodation"],
        )
        recommended = math.ceil((minimum_destination / exchange_rate) * 100.0) / 100.0
        now = datetime.now(timezone.utc)
        assessment = BudgetAssessment(
            assessment_id=str(uuid.uuid4()),
            origin=origin.strip(),
            destination=destination.strip(),
            destination_city=destination_city.strip(),
            start_date=start_date,
            end_date=end_date,
            num_people=num_people,
            base_currency=base_currency,
            destination_currency=destination_currency,
            exchange_rate=exchange_rate,
            minimum_destination_budget=minimum_destination,
            recommended_minimum_budget=recommended,
            evidence=BudgetEvidence(
                outbound_flight=outbound,
                return_flight=return_flight,
                hotel=hotel,
                outbound_flight_price=outbound_price,
                return_flight_price=return_price,
                hotel_price_per_night=hotel_price,
                hotel_nights=nights,
            ),
            created_at=now.isoformat(),
            expires_at=(now + timedelta(hours=ASSESSMENT_TTL_HOURS)).isoformat(),
        )
        try:
            persisted = complete_cached_data_claim(
                collection="api_cache",
                prefix="budget_assessment",
                payload=assessment.model_dump(),
                claim_id=claim_id,
                ttl_hours=ASSESSMENT_TTL_HOURS,
                **cache_key,
            )
        except Exception:
            safe_provider_log(logger, "budget.cache_publish_failed")
            persisted = False
        if persisted:
            return BudgetAssessmentResult(assessment=assessment, persisted=True)
        release_cached_data_claim(
            collection="api_cache",
            prefix="budget_assessment",
            claim_id=claim_id,
            **cache_key,
        )
        return BudgetAssessmentResult(assessment=assessment, persisted=False)
    except Exception:
        release_cached_data_claim(
            collection="api_cache",
            prefix="budget_assessment",
            claim_id=claim_id,
            **cache_key,
        )
        raise


def load_confirmed_budget_assessment(
    *,
    assessment_id: str,
    origin: str,
    destination: str,
    destination_city: str,
    start_date: str,
    end_date: str,
    num_people: int,
) -> BudgetAssessment | None:
    """Load confirmation evidence without falling back to any paid provider."""
    try:
        base_currency = get_currency_code(origin)
        destination_currency = get_currency_code(destination)
    except Exception:
        return None
    cache_key = assessment_cache_key_kwargs(
        origin=origin,
        destination=destination,
        destination_city=destination_city,
        start_date=start_date,
        end_date=end_date,
        num_people=num_people,
        base_currency=base_currency,
        destination_currency=destination_currency,
    )
    try:
        cached = get_cached_data(
            collection="api_cache",
            prefix="budget_assessment",
            **cache_key,
        )
    except Exception:
        safe_provider_log(logger, "budget.confirmed_cache_read_failed")
        return None
    assessment = _unexpired_cached_assessment(
        cached,
        origin=origin,
        destination=destination,
        destination_city=destination_city,
        start_date=start_date,
        end_date=end_date,
        num_people=num_people,
    )
    if assessment is None or assessment.assessment_id != assessment_id:
        return None
    return assessment


def is_budget_sufficient(
    total_base_budget: float,
    assessment: BudgetAssessment,
) -> bool:
    """Compare the entered amount at full destination-currency precision."""
    try:
        entered_budget = float(total_base_budget)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(entered_budget)
        and entered_budget > 0
        and entered_budget * assessment.exchange_rate
        >= assessment.minimum_destination_budget
    )
