from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from threading import Barrier, Event, Lock
from unittest.mock import patch

import pytest

from app.services import budget_assessment as service


TRIP = {
    "origin": "Malaysia",
    "destination": "Japan",
    "destination_city": "Tokyo",
    "start_date": "2026-08-01",
    "end_date": "2026-08-05",
    "num_people": 2,
}


def _currency_code(country: str) -> str:
    return {"Malaysia": "MYR", "Japan": "JPY"}[country]


def _flight_results(
    origin: str,
    destination: str,
    date: str,
    currency: str,
    max_budget: float | None,
    adults: int,
) -> list[dict]:
    del date, currency, max_budget, adults
    if (origin, destination) == ("Malaysia", "Japan"):
        return [{"price": 32000.0, "flight_number": "OUT"}]
    if (origin, destination) == ("Japan", "Malaysia"):
        return [{"price": 16000.0, "flight_number": "BACK"}]
    return []


def _assessment_payload(**changes: object) -> dict:
    now = datetime.now(timezone.utc)
    payload = {
        "assessment_id": "assessment-1",
        "calculation_version": "allocation-v1",
        "origin": "Malaysia",
        "destination": "Japan",
        "destination_city": "Tokyo",
        "start_date": "2026-08-01",
        "end_date": "2026-08-05",
        "num_people": 2,
        "base_currency": "MYR",
        "destination_currency": "JPY",
        "exchange_rate": 32.0,
        "minimum_destination_budget": 192000.0,
        "recommended_minimum_budget": 6000.0,
        "evidence": {
            "outbound_flight": {"price": 32000.0, "flight_number": "OUT"},
            "return_flight": {"price": 16000.0, "flight_number": "BACK"},
            "hotel": {
                "price_per_night": 14000.0,
                "hotel_name": "Grounded Hotel",
            },
            "outbound_flight_price": 32000.0,
            "return_flight_price": 16000.0,
            "hotel_price_per_night": 14000.0,
            "hotel_nights": 4,
        },
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(hours=1)).isoformat(),
    }
    payload.update(changes)
    return payload


@pytest.fixture(autouse=True)
def _isolated_assessment_cache_claim():
    """Keep provider unit tests independent of the Firestore transaction client."""
    with (
        patch.object(
            service,
            "claim_cached_data",
            return_value=("claimed", None, "test-claim"),
        ),
        patch.object(service, "complete_cached_data_claim", return_value=True),
        patch.object(service, "release_cached_data_claim", return_value=True),
    ):
        yield


@pytest.mark.unit
def test_calculates_grounded_minimum_and_rounds_base_up():
    with (
        patch.object(service, "get_cached_data", return_value=None),
        patch.object(service, "get_currency_code", side_effect=_currency_code),
        patch.object(service, "get_currency_rate", return_value=32.0),
        patch.object(service, "fetch_flights_api", side_effect=_flight_results),
        patch.object(
            service,
            "fetch_hotels_api",
            return_value=[
                {
                    "price_per_night": 14000.0,
                    "hotel_name": "Grounded Hotel",
                }
            ],
        ) as hotel_provider,
    ):
        result = service.get_or_create_budget_assessment(**TRIP)

    # max((32,000 + 16,000) / .25, (14,000 * 4) / .35) = 192,000 JPY.
    assert result.assessment.minimum_destination_budget == 192000.0
    assert result.assessment.recommended_minimum_budget == 6000.0
    assert result.assessment.evidence.hotel_nights == 4
    assert result.assessment.evidence.outbound_flight["flight_number"] == "OUT"
    assert result.assessment.evidence.return_flight["flight_number"] == "BACK"
    assert result.assessment.destination_city == "Tokyo"
    hotel_provider.assert_called_once_with(
        "Tokyo, Japan",
        TRIP["start_date"],
        TRIP["end_date"],
        "JPY",
        None,
        TRIP["num_people"],
    )
    assert result.persisted is True


@pytest.mark.unit
def test_same_day_assessment_skips_hotel_and_uses_flight_only_floor():
    same_day = {**TRIP, "end_date": TRIP["start_date"], "num_people": 1}

    def same_day_flights(*args, **kwargs):
        del args, kwargs
        return [{"price": 25.0, "flight_number": "BOUNDARY"}]

    with (
        patch.object(service, "get_cached_data", return_value=None),
        patch.object(service, "get_currency_code", side_effect=_currency_code),
        patch.object(service, "get_currency_rate", return_value=1.0),
        patch.object(service, "fetch_flights_api", side_effect=same_day_flights),
        patch.object(
            service,
            "fetch_hotels_api",
            side_effect=AssertionError("same-day assessment must not query hotels"),
        ) as hotel_provider,
    ):
        result = service.get_or_create_budget_assessment(**same_day)

    evidence = result.assessment.evidence
    hotel_provider.assert_not_called()
    assert evidence.hotel_nights == 0
    assert evidence.hotel_price_per_night == 0
    assert evidence.hotel == {}
    assert result.assessment.minimum_destination_budget == 200
    assert result.assessment.recommended_minimum_budget == 200


@pytest.mark.unit
def test_cache_key_normalizes_trip_and_excludes_budget_and_user_identity():
    key = service.assessment_cache_key_kwargs(
        origin=" Malaysia ",
        destination="JAPAN",
        destination_city=" Tokyo ",
        start_date=TRIP["start_date"],
        end_date=TRIP["end_date"],
        num_people=TRIP["num_people"],
        base_currency="myr",
        destination_currency="jpy",
    )

    assert key == {
        "origin": "malaysia",
        "destination": "japan",
        "destination_city": "tokyo",
        "start_date": "2026-08-01",
        "end_date": "2026-08-05",
        "num_people": 2,
        "base_currency": "MYR",
        "destination_currency": "JPY",
        "calculation_version": "allocation-v1",
    }
    assert "total_budget" not in key
    assert "user_id" not in key


@pytest.mark.unit
def test_cached_assessment_for_another_city_is_not_reused():
    kyoto_trip = {**TRIP, "destination_city": "Kyoto"}
    with (
        patch.object(service, "get_currency_code", side_effect=_currency_code),
        patch.object(service, "get_cached_data", return_value=_assessment_payload()),
        patch.object(service, "get_currency_rate", return_value=32.0),
        patch.object(service, "fetch_flights_api", side_effect=_flight_results),
        patch.object(
            service,
            "fetch_hotels_api",
            return_value=[{"price_per_night": 14000.0, "hotel_name": "Kyoto Hotel"}],
        ) as hotel_provider,
    ):
        result = service.get_or_create_budget_assessment(**kyoto_trip)

    assert result.assessment.destination_city == "Kyoto"
    hotel_provider.assert_called_once_with(
        "Kyoto, Japan",
        TRIP["start_date"],
        TRIP["end_date"],
        "JPY",
        None,
        TRIP["num_people"],
    )


@pytest.mark.unit
def test_confirmation_rejects_assessment_for_another_city():
    with (
        patch.object(service, "get_currency_code", side_effect=_currency_code),
        patch.object(service, "get_cached_data", return_value=_assessment_payload()),
    ):
        assessment = service.load_confirmed_budget_assessment(
            assessment_id="assessment-1",
            **{**TRIP, "destination_city": "Kyoto"},
        )

    assert assessment is None


@pytest.mark.unit
def test_cached_assessment_skips_all_paid_provider_boundaries():
    with (
        patch.object(service, "get_currency_code", side_effect=_currency_code),
        patch.object(
            service,
            "get_cached_data",
            return_value=_assessment_payload(),
        ),
        patch.object(service, "get_currency_rate") as currency_provider,
        patch.object(service, "fetch_flights_api") as flight_provider,
        patch.object(service, "fetch_hotels_api") as hotel_provider,
    ):
        result = service.get_or_create_budget_assessment(**TRIP)

    assert result.assessment.assessment_id == "assessment-1"
    assert result.persisted is True
    currency_provider.assert_not_called()
    flight_provider.assert_not_called()
    hotel_provider.assert_not_called()


@pytest.mark.unit
def test_confirmation_loads_matching_cached_assessment_without_provider_calls():
    with (
        patch.object(service, "get_currency_code", side_effect=_currency_code),
        patch.object(
            service,
            "get_cached_data",
            return_value=_assessment_payload(),
        ),
        patch.object(service, "get_currency_rate") as currency_provider,
        patch.object(service, "fetch_flights_api") as flight_provider,
        patch.object(service, "fetch_hotels_api") as hotel_provider,
    ):
        assessment = service.load_confirmed_budget_assessment(
            assessment_id="assessment-1",
            **TRIP,
        )

    assert assessment is not None
    assert assessment.recommended_minimum_budget == 6000.0
    currency_provider.assert_not_called()
    flight_provider.assert_not_called()
    hotel_provider.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload,assessment_id",
    [
        (_assessment_payload(assessment_id="different"), "assessment-1"),
        (
            _assessment_payload(
                expires_at=(datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
            ),
            "assessment-1",
        ),
        (_assessment_payload(destination="Taiwan"), "assessment-1"),
        (None, "assessment-1"),
    ],
)
def test_confirmation_rejects_missing_expired_or_mismatched_cache(
    payload: dict | None,
    assessment_id: str,
):
    with (
        patch.object(service, "get_currency_code", side_effect=_currency_code),
        patch.object(service, "get_cached_data", return_value=payload),
    ):
        assessment = service.load_confirmed_budget_assessment(
            assessment_id=assessment_id,
            **TRIP,
        )

    assert assessment is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "changes",
    [
        {"exchange_rate": float("inf")},
        {"minimum_destination_budget": -1.0},
        {"recommended_minimum_budget": float("nan")},
        {"minimum_destination_budget": 1.0},
    ],
)
def test_confirmation_rejects_corrupt_or_internally_inconsistent_evidence(changes):
    with (
        patch.object(service, "get_currency_code", side_effect=_currency_code),
        patch.object(
            service,
            "get_cached_data",
            return_value=_assessment_payload(**changes),
        ),
        patch.object(service, "get_currency_rate") as currency_provider,
        patch.object(service, "fetch_flights_api") as flight_provider,
        patch.object(service, "fetch_hotels_api") as hotel_provider,
    ):
        assessment = service.load_confirmed_budget_assessment(
            assessment_id="assessment-1",
            **TRIP,
        )

    assert assessment is None
    currency_provider.assert_not_called()
    flight_provider.assert_not_called()
    hotel_provider.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize(
    "rate,outbound_price,return_price,hotel_price",
    [
        (0.0, 32000.0, 16000.0, 14000.0),
        (float("nan"), 32000.0, 16000.0, 14000.0),
        (32.0, None, 16000.0, 14000.0),
        (32.0, 32000.0, None, 14000.0),
        (32.0, 32000.0, 16000.0, None),
        (32.0, float("inf"), 16000.0, 14000.0),
        (32.0, 32000.0, 16000.0, float("nan")),
    ],
)
def test_missing_or_invalid_required_evidence_fails_closed(
    rate: float,
    outbound_price: float | None,
    return_price: float | None,
    hotel_price: float | None,
):
    def flight_results(
        origin: str,
        destination: str,
        date: str,
        currency: str,
        max_budget: float | None,
        adults: int,
    ) -> list[dict]:
        del date, currency, max_budget, adults
        price = (
            outbound_price
            if (origin, destination) == ("Malaysia", "Japan")
            else return_price
        )
        return [] if price is None else [{"price": price}]

    hotels = [] if hotel_price is None else [{"price_per_night": hotel_price}]

    with (
        patch.object(service, "get_cached_data", return_value=None),
        patch.object(service, "get_currency_code", side_effect=_currency_code),
        patch.object(service, "get_currency_rate", return_value=rate),
        patch.object(service, "fetch_flights_api", side_effect=flight_results),
        patch.object(service, "fetch_hotels_api", return_value=hotels),
    ):
        with pytest.raises(service.BudgetAssessmentUnavailable):
            service.get_or_create_budget_assessment(**TRIP)


@pytest.mark.unit
def test_cache_write_failure_keeps_immediate_evidence_but_disallows_confirmation():
    with (
        patch.object(service, "get_cached_data", return_value=None),
        patch.object(service, "get_currency_code", side_effect=_currency_code),
        patch.object(service, "get_currency_rate", return_value=32.0),
        patch.object(service, "fetch_flights_api", side_effect=_flight_results),
        patch.object(
            service,
            "fetch_hotels_api",
            return_value=[{"price_per_night": 14000.0, "hotel_name": "Hotel"}],
        ),
        patch.object(service, "complete_cached_data_claim", return_value=False),
    ):
        result = service.get_or_create_budget_assessment(**TRIP)

    assert result.assessment.evidence.outbound_flight_price == 32000.0
    assert result.persisted is False


@pytest.mark.unit
def test_cache_write_exception_keeps_immediate_evidence_but_disallows_confirmation():
    with (
        patch.object(service, "get_cached_data", return_value=None),
        patch.object(service, "get_currency_code", side_effect=_currency_code),
        patch.object(service, "get_currency_rate", return_value=32.0),
        patch.object(service, "fetch_flights_api", side_effect=_flight_results),
        patch.object(
            service,
            "fetch_hotels_api",
            return_value=[{"price_per_night": 14000.0, "hotel_name": "Hotel"}],
        ),
        patch.object(
            service,
            "complete_cached_data_claim",
            side_effect=RuntimeError("cache offline"),
        ),
    ):
        result = service.get_or_create_budget_assessment(**TRIP)

    assert result.assessment.evidence.hotel_price_per_night == 14000.0
    assert result.persisted is False


@pytest.mark.unit
def test_unexpected_exchange_provider_error_fails_as_unavailable():
    with (
        patch.object(service, "get_cached_data", return_value=None),
        patch.object(service, "get_currency_code", side_effect=_currency_code),
        patch.object(
            service,
            "get_currency_rate",
            side_effect=RuntimeError("provider crashed"),
        ),
        patch.object(service, "fetch_flights_api") as flight_provider,
        patch.object(service, "fetch_hotels_api") as hotel_provider,
    ):
        with pytest.raises(service.BudgetAssessmentUnavailable):
            service.get_or_create_budget_assessment(**TRIP)

    flight_provider.assert_not_called()
    hotel_provider.assert_not_called()


@pytest.mark.unit
def test_minimum_keeps_full_precision_for_the_sufficiency_boundary():
    def tiny_flights(*args, **kwargs):
        del args, kwargs
        return [{"price": 1.0}]

    with (
        patch.object(service, "get_cached_data", return_value=None),
        patch.object(service, "get_currency_code", side_effect=_currency_code),
        patch.object(service, "get_currency_rate", return_value=3.0),
        patch.object(service, "fetch_flights_api", side_effect=tiny_flights),
        patch.object(
            service,
            "fetch_hotels_api",
            return_value=[{"price_per_night": 1.0}],
        ),
    ):
        result = service.get_or_create_budget_assessment(**TRIP)

    exact_minimum = 4.0 / service.DEFAULT_BUDGET_RATIOS["accommodation"]
    assert result.assessment.minimum_destination_budget == exact_minimum
    assert result.assessment.minimum_destination_budget != round(exact_minimum, 2)


@pytest.mark.unit
def test_sufficiency_uses_unrounded_destination_currency_boundary():
    assessment = service.BudgetAssessment.model_validate(_assessment_payload())

    assert service.is_budget_sufficient(6000.0, assessment) is True
    assert service.is_budget_sufficient(5999.99, assessment) is False


@pytest.mark.unit
def test_assessment_matches_only_the_exact_normalized_trip():
    assessment = service.BudgetAssessment.model_validate(_assessment_payload())

    assert service.assessment_matches_trip(assessment, **TRIP) is True
    assert (
        service.assessment_matches_trip(
            assessment,
            **{**TRIP, "end_date": "2026-08-06"},
        )
        is False
    )


@pytest.mark.unit
def test_synchronised_cold_assessment_requests_share_one_loadable_winner():
    """A waiter must receive the winner's persisted ID, never an overwritten ID."""
    initial_reads = Barrier(2)
    provider_started = Event()
    waiter_waiting = Event()
    allow_winner = Event()
    published = Event()
    cache_lock = Lock()
    cache: dict[str, object] = {"payload": None, "claim_id": None}
    provider_calls = {"currency": 0, "flights": 0, "hotels": 0}
    cache_reads = {"count": 0}

    def get_cached_data(**kwargs):
        del kwargs
        with cache_lock:
            cache_reads["count"] += 1
            is_initial_read = cache_reads["count"] <= 2
        if is_initial_read:
            initial_reads.wait(timeout=5)
        with cache_lock:
            return cache["payload"]

    def claim_cached_data(**kwargs):
        del kwargs
        with cache_lock:
            if cache["payload"] is not None:
                return "cached", cache["payload"], None
            if cache["claim_id"] is None:
                cache["claim_id"] = "winner-claim"
                return "claimed", None, "winner-claim"
            return "busy", None, None

    def wait_for_cache_claim_change(**kwargs):
        del kwargs
        waiter_waiting.set()
        assert published.wait(timeout=5)

    def complete_cached_data_claim(*, claim_id, payload, **kwargs):
        del kwargs
        with cache_lock:
            assert claim_id == cache["claim_id"]
            cache["payload"] = payload
            cache["claim_id"] = None
        published.set()
        return True

    def delayed_currency_rate(*args):
        del args
        provider_calls["currency"] += 1
        provider_started.set()
        assert allow_winner.wait(timeout=5)
        return 32.0

    def flights(*args, **kwargs):
        del kwargs
        provider_calls["flights"] += 1
        return _flight_results(*args)

    def hotels(*args, **kwargs):
        del args, kwargs
        provider_calls["hotels"] += 1
        return [{"price_per_night": 14000.0, "hotel_name": "Grounded Hotel"}]

    with (
        patch.object(service, "get_cached_data", side_effect=get_cached_data),
        patch.object(service, "get_currency_code", side_effect=_currency_code),
        patch.object(service, "claim_cached_data", side_effect=claim_cached_data, create=True),
        patch.object(
            service,
            "wait_for_cache_claim_change",
            side_effect=wait_for_cache_claim_change,
            create=True,
        ),
        patch.object(
            service,
            "complete_cached_data_claim",
            side_effect=complete_cached_data_claim,
            create=True,
        ),
        patch.object(service, "get_currency_rate", side_effect=delayed_currency_rate),
        patch.object(service, "fetch_flights_api", side_effect=flights),
        patch.object(service, "fetch_hotels_api", side_effect=hotels),
    ):
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(service.get_or_create_budget_assessment, **TRIP) for _ in range(2)]
            assert provider_started.wait(timeout=5)
            waiter_waiting.wait(timeout=1)
            allow_winner.set()
            results = [future.result(timeout=5) for future in futures]
        loaded = service.load_confirmed_budget_assessment(
            assessment_id=results[0].assessment.assessment_id,
            **TRIP,
        )

    assessment_ids = {result.assessment.assessment_id for result in results}
    assert len(assessment_ids) == 1
    assert all(result.persisted for result in results)
    assert loaded is not None
    assert loaded.assessment_id == next(iter(assessment_ids))
    assert cache["payload"] is not None
    assert provider_calls == {"currency": 1, "flights": 2, "hotels": 1}


@pytest.mark.unit
def test_failed_claim_is_released_so_a_retry_can_publish_an_assessment():
    claims = iter([("claimed", None, "failed-claim"), ("claimed", None, "retry-claim")])
    released: list[str] = []

    with (
        patch.object(service, "get_cached_data", return_value=None),
        patch.object(service, "get_currency_code", side_effect=_currency_code),
        patch.object(service, "claim_cached_data", side_effect=lambda **_: next(claims), create=True),
        patch.object(
            service,
            "release_cached_data_claim",
            side_effect=lambda *, claim_id, **_: released.append(claim_id),
            create=True,
        ),
        patch.object(service, "get_currency_rate", side_effect=[RuntimeError("offline"), 32.0]),
        patch.object(service, "fetch_flights_api", side_effect=_flight_results),
        patch.object(
            service,
            "fetch_hotels_api",
            return_value=[{"price_per_night": 14000.0, "hotel_name": "Grounded Hotel"}],
        ),
        patch.object(service, "complete_cached_data_claim", return_value=True, create=True),
    ):
        with pytest.raises(service.BudgetAssessmentUnavailable, match="Exchange-rate"):
            service.get_or_create_budget_assessment(**TRIP)
        retry = service.get_or_create_budget_assessment(**TRIP)

    assert released == ["failed-claim"]
    assert retry.persisted is True
