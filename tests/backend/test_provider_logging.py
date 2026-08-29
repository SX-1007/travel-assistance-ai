from __future__ import annotations

import ast
import logging
from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from app.agents.state import AgentState
from app.api import dependencies
from app.core import firebase_db
from app.core import supabase_db
from app.memory import extractor
from app.api.routers import geocode
from app.services import budget_assessment
from app.tools import (
    attractions,
    budget_chat,
    currency,
    flights_hotels,
    mapbox,
    trip_details,
)


SENTINEL_SECRET = "api-key-DO-NOT-LOG-7f9363"
SENTINEL_QUERY = "raw-user-query-DO-NOT-LOG-58c1"
SENTINEL_EXCEPTION = "provider-exception-DO-NOT-LOG-a2d4"
_PLANNING_LOG_PATHS = (
    Path(currency.__file__),
    Path(flights_hotels.__file__),
    Path(attractions.__file__),
    Path(mapbox.__file__),
    Path(firebase_db.__file__),
    Path(supabase_db.__file__),
    Path(budget_assessment.__file__),
    Path(geocode.__file__),
    Path(dependencies.__file__),
    Path(extractor.__file__),
    Path(budget_chat.__file__),
    Path(trip_details.__file__),
    Path(flights_hotels.__file__).parents[1] / "services" / "planning_transaction.py",
    Path(flights_hotels.__file__).parents[1] / "agents" / "graph.py",
    Path(flights_hotels.__file__).parents[1] / "api" / "routers" / "form.py",
    Path(flights_hotels.__file__).parents[1] / "api" / "routers" / "chat.py",
)


class _CapturingHandler(logging.Handler):
    def __init__(self, *, raise_on_emit: bool = False) -> None:
        super().__init__()
        self.messages: list[str] = []
        self.records: list[logging.LogRecord] = []
        self._raise_on_emit = raise_on_emit

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)
        self.messages.append(record.getMessage())
        if self._raise_on_emit:
            raise RuntimeError("hostile logging handler")


@contextmanager
def _provider_log_handler(module, *, raise_on_emit: bool = False):
    target_logger = module.logger
    handler = _CapturingHandler(raise_on_emit=raise_on_emit)
    original_level = target_logger.level
    original_propagate = target_logger.propagate
    target_logger.addHandler(handler)
    target_logger.setLevel(logging.DEBUG)
    target_logger.propagate = False
    try:
        yield handler
    finally:
        target_logger.removeHandler(handler)
        target_logger.setLevel(original_level)
        target_logger.propagate = original_propagate


def _assert_sentinels_absent(messages: list[str]) -> None:
    rendered = "\n".join(messages)
    for sentinel in (SENTINEL_SECRET, SENTINEL_QUERY, SENTINEL_EXCEPTION):
        assert sentinel not in rendered


def _assert_no_exception_details(handler: _CapturingHandler) -> None:
    assert all(record.exc_info is None for record in handler.records)


class _ExplosiveProviderPayload:
    def get(self, *_args, **_kwargs):
        raise RuntimeError(SENTINEL_EXCEPTION)


class _FailedDayExecutor:
    def submit(self, *_args, **_kwargs) -> Future:
        failed = Future()
        failed.set_exception(RuntimeError(SENTINEL_EXCEPTION))
        return failed


@pytest.fixture(autouse=True)
def _clear_mapbox_caches():
    mapbox._geocode_cache.clear()
    mapbox._country_cache.clear()
    yield
    mapbox._geocode_cache.clear()
    mapbox._country_cache.clear()


def test_serpapi_terminal_failure_does_not_log_secret_query_or_exception():
    params = {"api_key": SENTINEL_SECRET, "q": SENTINEL_QUERY}
    with (
        patch.object(
            flights_hotels._session,
            "get",
            side_effect=requests.ConnectionError(SENTINEL_EXCEPTION),
        ),
        _provider_log_handler(flights_hotels) as handler,
    ):
        result = flights_hotels._serpapi_get(params, max_retries=1, initial_wait=0)

    assert result == {}
    _assert_sentinels_absent(handler.messages)


def test_serpapi_terminal_failure_is_unchanged_by_a_hostile_log_handler():
    params = {"api_key": SENTINEL_SECRET, "q": SENTINEL_QUERY}
    with (
        patch.object(
            flights_hotels._session,
            "get",
            side_effect=requests.ConnectionError(SENTINEL_EXCEPTION),
        ),
        _provider_log_handler(flights_hotels, raise_on_emit=True),
    ):
        assert flights_hotels._serpapi_get(params, max_retries=1, initial_wait=0) == {}


def test_flight_and_hotel_parse_failures_do_not_log_provider_exceptions():
    with (
        patch.object(flights_hotels, "get_cached_data", return_value=None),
        patch.object(flights_hotels, "resolve_iata_code", return_value="KUL"),
        patch.object(
            flights_hotels,
            "_serpapi_get",
            return_value=_ExplosiveProviderPayload(),
        ),
        _provider_log_handler(flights_hotels) as handler,
    ):
        flights = flights_hotels.fetch_flights_api(
            SENTINEL_QUERY, "Japan", "2026-08-01", "JPY"
        )
        hotels = flights_hotels.fetch_hotels_api(
            SENTINEL_QUERY,
            "2026-08-01",
            "2026-08-02",
            "JPY",
            destination_country_code="JP",
        )

    assert flights == []
    assert hotels == []
    _assert_sentinels_absent(handler.messages)


@pytest.mark.parametrize("provider", ["flight", "hotel"])
def test_flight_and_hotel_parse_failures_ignore_hostile_log_handlers(provider):
    with (
        patch.object(flights_hotels, "get_cached_data", return_value=None),
        patch.object(flights_hotels, "resolve_iata_code", return_value="KUL"),
        patch.object(
            flights_hotels,
            "_serpapi_get",
            return_value=_ExplosiveProviderPayload(),
        ),
        _provider_log_handler(flights_hotels, raise_on_emit=True),
    ):
        if provider == "flight":
            result = flights_hotels.fetch_flights_api(
                SENTINEL_QUERY, "Japan", "2026-08-01", "JPY"
            )
        else:
            result = flights_hotels.fetch_hotels_api(
                SENTINEL_QUERY,
                "2026-08-01",
                "2026-08-02",
                "JPY",
                destination_country_code="JP",
            )

    assert result == []


def test_attraction_brainstorm_failure_does_not_log_provider_exception():
    with (
        patch.object(
            attractions,
            "_BRAINSTORM_CHAIN",
            Mock(invoke=Mock(side_effect=RuntimeError(SENTINEL_EXCEPTION))),
        ),
        _provider_log_handler(attractions) as handler,
    ):
        result = attractions.brainstorm_places(
            country=SENTINEL_QUERY,
            cities=[SENTINEL_QUERY],
            dates=["2026-08-01"],
            num_people=1,
            hotel_name=SENTINEL_QUERY,
            interests=SENTINEL_QUERY,
            dietary=SENTINEL_QUERY,
            pacing="moderate",
            activity_budget=1,
            food_budget=1,
        )

    assert result == []
    _assert_sentinels_absent(handler.messages)


def test_attraction_brainstorm_failure_ignores_hostile_log_handler():
    with (
        patch.object(
            attractions,
            "_BRAINSTORM_CHAIN",
            Mock(invoke=Mock(side_effect=RuntimeError(SENTINEL_EXCEPTION))),
        ),
        _provider_log_handler(attractions, raise_on_emit=True),
    ):
        result = attractions.brainstorm_places(
            country=SENTINEL_QUERY,
            cities=[],
            dates=["2026-08-01"],
            num_people=1,
            hotel_name="",
            interests="",
            dietary="",
            pacing="moderate",
            activity_budget=1,
            food_budget=1,
        )

    assert result == []


def test_attraction_search_failure_does_not_log_query_or_exception():
    with (
        patch.object(
            attractions._session,
            "get",
            side_effect=requests.ConnectionError(SENTINEL_EXCEPTION),
        ),
        _provider_log_handler(attractions) as handler,
    ):
        result = attractions._serpapi_place_search(SENTINEL_QUERY, max_retries=1)

    assert result is None
    _assert_sentinels_absent(handler.messages)


def test_attraction_search_failure_ignores_hostile_log_handler():
    with (
        patch.object(
            attractions._session,
            "get",
            side_effect=requests.ConnectionError(SENTINEL_EXCEPTION),
        ),
        _provider_log_handler(attractions, raise_on_emit=True),
    ):
        assert attractions._serpapi_place_search(SENTINEL_QUERY, max_retries=1) is None


def test_attraction_tool_search_failure_does_not_return_provider_exception_details():
    state = AgentState(country="Singapore", city=["Singapore"])
    failure = requests.ConnectionError(
        f"{SENTINEL_SECRET} {SENTINEL_QUERY} {SENTINEL_EXCEPTION}"
    )
    with (
        patch.object(attractions, "_destination_country_code", return_value="SG"),
        patch.object(attractions, "_geo_anchor", return_value=None),
        patch.object(attractions._session, "get", side_effect=failure),
    ):
        result = attractions.search_places.func(
            "attraction", SENTINEL_QUERY, state, city="Singapore"
        )

    assert result == {"error": "Place search failed."}
    _assert_sentinels_absent([str(result)])


def test_planning_geocode_failure_does_not_log_secret_query_or_exception():
    failure = requests.ConnectionError(
        f"{SENTINEL_SECRET} {SENTINEL_QUERY} {SENTINEL_EXCEPTION}"
    )
    with (
        patch.object(geocode.settings, "SERPAPI_PLACE", SENTINEL_SECRET),
        patch.object(geocode._session, "get", side_effect=failure),
        _provider_log_handler(geocode) as handler,
    ):
        result = geocode.geocode(
            user_id="user",
            request_id=SENTINEL_QUERY,
            q=SENTINEL_QUERY,
            lat=None,
            lng=None,
        )

    assert result == {"status": "error", "found": False}
    _assert_sentinels_absent(handler.messages)
    _assert_no_exception_details(handler)


def test_planning_geocode_failure_ignores_hostile_log_handlers():
    failure = requests.ConnectionError(
        f"{SENTINEL_SECRET} {SENTINEL_QUERY} {SENTINEL_EXCEPTION}"
    )
    with (
        patch.object(geocode.settings, "SERPAPI_PLACE", SENTINEL_SECRET),
        patch.object(geocode._session, "get", side_effect=failure),
        _provider_log_handler(geocode, raise_on_emit=True),
    ):
        assert geocode.geocode(
            user_id="user",
            request_id=SENTINEL_QUERY,
            q=SENTINEL_QUERY,
            lat=None,
            lng=None,
        ) == {"status": "error", "found": False}


def _failing_profile_client(failure: Exception) -> Mock:
    client = Mock()
    client.table.return_value.select.return_value.eq.return_value.execute.side_effect = failure
    return client


def test_planning_profile_fetch_failure_does_not_log_exception_or_change_fallback():
    failure = RuntimeError(
        f"{SENTINEL_SECRET} {SENTINEL_QUERY} {SENTINEL_EXCEPTION}"
    )
    with (
        patch.object(supabase_db, "supabase_client", _failing_profile_client(failure)),
        _provider_log_handler(supabase_db) as handler,
    ):
        assert supabase_db.fetch_user_profile(SENTINEL_QUERY) == {}

    _assert_sentinels_absent(handler.messages)
    _assert_no_exception_details(handler)


def test_planning_profile_fetch_failure_ignores_hostile_log_handlers():
    with (
        patch.object(
            supabase_db,
            "supabase_client",
            _failing_profile_client(RuntimeError(SENTINEL_EXCEPTION)),
        ),
        _provider_log_handler(supabase_db, raise_on_emit=True),
    ):
        assert supabase_db.fetch_user_profile(SENTINEL_QUERY) == {}


def test_memory_persistence_failure_does_not_log_profile_payload_or_exception():
    failure = RuntimeError(
        f"{SENTINEL_SECRET} {SENTINEL_QUERY} {SENTINEL_EXCEPTION}"
    )
    with (
        patch.object(extractor, "update_user_profile", side_effect=failure),
        _provider_log_handler(extractor) as handler,
    ):
        assert not extractor._update_relational_db(
            SENTINEL_QUERY, {"interests": [SENTINEL_QUERY]}
        )

    _assert_sentinels_absent(handler.messages)
    _assert_no_exception_details(handler)


def test_memory_persistence_failure_ignores_hostile_log_handlers():
    with (
        patch.object(
            extractor,
            "update_user_profile",
            side_effect=RuntimeError(SENTINEL_EXCEPTION),
        ),
        _provider_log_handler(extractor, raise_on_emit=True),
    ):
        assert not extractor._update_relational_db("user", {"interests": ["food"]})


@pytest.mark.parametrize("failure_source", ["geocode", "response_json", "schema"])
def test_nearby_amenities_provider_failures_return_fixed_tool_error(failure_source):
    failure = ValueError(
        f"{SENTINEL_SECRET} {SENTINEL_QUERY} {SENTINEL_EXCEPTION}"
    )
    response = Mock(status_code=200)
    response.__bool__ = Mock(return_value=True)
    if failure_source == "geocode":
        geocode_patch = patch.object(mapbox, "geocode_location", side_effect=failure)
        request_patch = patch.object(mapbox, "_request_with_retries")
    else:
        geocode_patch = patch.object(
            mapbox, "geocode_location", return_value={"lng": 1, "lat": 2}
        )
        if failure_source == "response_json":
            response.json.side_effect = failure
        else:
            response.json.return_value = {"features": [SENTINEL_QUERY]}
        request_patch = patch.object(mapbox, "_request_with_retries", return_value=response)
    with geocode_patch, request_patch, _provider_log_handler(mapbox) as handler:
        result = mapbox.search_nearby_amenities.func(
            SENTINEL_QUERY, AgentState(), SENTINEL_QUERY
        )

    assert result == {"error": "Nearby amenity search is unavailable."}
    _assert_sentinels_absent(handler.messages)
    _assert_no_exception_details(handler)


@pytest.mark.parametrize("failure_source", ["geocode", "response_json", "schema"])
def test_nearby_amenities_failures_ignore_hostile_log_handlers(failure_source):
    failure = ValueError(SENTINEL_EXCEPTION)
    response = Mock(status_code=200)
    response.__bool__ = Mock(return_value=True)
    if failure_source == "geocode":
        geocode_patch = patch.object(mapbox, "geocode_location", side_effect=failure)
        request_patch = patch.object(mapbox, "_request_with_retries")
    else:
        geocode_patch = patch.object(
            mapbox, "geocode_location", return_value={"lng": 1, "lat": 2}
        )
        if failure_source == "response_json":
            response.json.side_effect = failure
        else:
            response.json.return_value = {"features": ["malformed"]}
        request_patch = patch.object(mapbox, "_request_with_retries", return_value=response)
    with geocode_patch, request_patch, _provider_log_handler(mapbox, raise_on_emit=True):
        assert mapbox.search_nearby_amenities.func(
            "hospital", AgentState(), "hotel"
        ) == {"error": "Nearby amenity search is unavailable."}


@pytest.mark.parametrize(
    "operation,expected",
    [
        pytest.param(
            lambda: mapbox.geocode_location(SENTINEL_QUERY), None, id="geocode"
        ),
        pytest.param(
            lambda: mapbox.resolve_country_code(3.139, 101.6869), None, id="reverse"
        ),
        pytest.param(
            lambda: mapbox._fetch_route_single(
                [[101.6869, 3.139], [101.7, 3.15]], "driving"
            ),
            {"distance_km": 0, "duration_mins": 0, "geometry_for_map": None},
            id="route",
        ),
    ],
)
def test_mapbox_provider_failures_do_not_log_exception_details(operation, expected):
    with (
        patch.object(mapbox, "MAX_RETRIES", 1),
        patch.object(
            mapbox._session,
            "get",
            side_effect=requests.ConnectionError(SENTINEL_EXCEPTION),
        ),
        _provider_log_handler(mapbox) as handler,
    ):
        result = operation()

    assert result == expected
    _assert_sentinels_absent(handler.messages)


@pytest.mark.parametrize(
    "operation,expected",
    [
        pytest.param(
            lambda: mapbox.geocode_location(SENTINEL_QUERY), None, id="geocode"
        ),
        pytest.param(
            lambda: mapbox.resolve_country_code(3.139, 101.6869), None, id="reverse"
        ),
        pytest.param(
            lambda: mapbox._fetch_route_single(
                [[101.6869, 3.139], [101.7, 3.15]], "driving"
            ),
            {"distance_km": 0, "duration_mins": 0, "geometry_for_map": None},
            id="route",
        ),
    ],
)
def test_mapbox_provider_failures_ignore_hostile_log_handlers(operation, expected):
    with (
        patch.object(mapbox, "MAX_RETRIES", 1),
        patch.object(
            mapbox._session,
            "get",
            side_effect=requests.ConnectionError(SENTINEL_EXCEPTION),
        ),
        _provider_log_handler(mapbox, raise_on_emit=True),
    ):
        assert operation() == expected


def test_map_generation_failure_does_not_log_provider_exception():
    state = AgentState(draft_itinerary=[{"day": 1, "date": "2026-08-01"}])
    with (
        patch.object(mapbox, "_get_executor", return_value=_FailedDayExecutor()),
        _provider_log_handler(mapbox) as handler,
    ):
        result = mapbox.generate_daily_map(state)

    assert result["draft_itinerary"] == [{"day": 1, "date": "2026-08-01"}]
    assert result["daily_map_info"][1]["type"] == "FeatureCollection"
    _assert_sentinels_absent(handler.messages)


def test_map_generation_failure_ignores_hostile_log_handler():
    state = AgentState(draft_itinerary=[{"day": 1, "date": "2026-08-01"}])
    with (
        patch.object(mapbox, "_get_executor", return_value=_FailedDayExecutor()),
        _provider_log_handler(mapbox, raise_on_emit=True),
    ):
        result = mapbox.generate_daily_map(state)

    assert result["daily_map_info"][1]["type"] == "FeatureCollection"


def test_currency_provider_failures_do_not_log_secret_query_or_exception():
    failure = requests.ConnectionError(
        f"{SENTINEL_SECRET} {SENTINEL_QUERY} {SENTINEL_EXCEPTION}"
    )
    with (
        patch.object(currency._session, "get", side_effect=failure),
        patch.object(currency, "_fetch_rate_open_erapi", return_value=None),
        _provider_log_handler(currency) as handler,
    ):
        result = currency.get_currency_rate(SENTINEL_QUERY, "JPY", max_retries=1)

    assert result is None
    _assert_sentinels_absent(handler.messages)
    _assert_no_exception_details(handler)


def test_currency_provider_failures_ignore_hostile_log_handlers():
    failure = requests.ConnectionError(
        f"{SENTINEL_SECRET} {SENTINEL_QUERY} {SENTINEL_EXCEPTION}"
    )
    with (
        patch.object(currency._session, "get", side_effect=failure),
        patch.object(currency, "_fetch_rate_open_erapi", return_value=None),
        _provider_log_handler(currency, raise_on_emit=True),
    ):
        assert currency.get_currency_rate(SENTINEL_QUERY, "JPY", max_retries=1) is None


def test_currency_fallback_failure_does_not_log_provider_details():
    failure = requests.ConnectionError(
        f"{SENTINEL_SECRET} {SENTINEL_QUERY} {SENTINEL_EXCEPTION}"
    )
    with (
        patch.object(currency._session, "get", side_effect=failure),
        _provider_log_handler(currency) as handler,
    ):
        assert currency._fetch_rate_open_erapi(SENTINEL_QUERY, "JPY") is None

    _assert_sentinels_absent(handler.messages)
    _assert_no_exception_details(handler)


def test_currency_fallback_failure_ignores_hostile_log_handler():
    failure = requests.ConnectionError(
        f"{SENTINEL_SECRET} {SENTINEL_QUERY} {SENTINEL_EXCEPTION}"
    )
    with (
        patch.object(currency._session, "get", side_effect=failure),
        _provider_log_handler(currency, raise_on_emit=True),
    ):
        assert currency._fetch_rate_open_erapi(SENTINEL_QUERY, "JPY") is None


def test_planning_cache_read_failure_does_not_log_exception_or_change_fallback():
    document = Mock()
    document.get.side_effect = RuntimeError(
        f"{SENTINEL_SECRET} {SENTINEL_QUERY} {SENTINEL_EXCEPTION}"
    )
    database = Mock()
    database.collection.return_value.document.return_value = document
    with (
        patch.object(firebase_db, "_get_db", return_value=database),
        _provider_log_handler(firebase_db) as handler,
    ):
        assert (
            firebase_db.get_cached_data(
                "api_cache", "flight", query=SENTINEL_QUERY
            )
            is None
        )

    _assert_sentinels_absent(handler.messages)
    _assert_no_exception_details(handler)


def test_planning_cache_read_failure_ignores_hostile_log_handler():
    document = Mock()
    document.get.side_effect = RuntimeError(SENTINEL_EXCEPTION)
    database = Mock()
    database.collection.return_value.document.return_value = document
    with (
        patch.object(firebase_db, "_get_db", return_value=database),
        _provider_log_handler(firebase_db, raise_on_emit=True),
    ):
        assert firebase_db.get_cached_data("api_cache", "flight", query=SENTINEL_QUERY) is None


def test_planning_cache_write_failure_does_not_log_exception_or_change_fallback():
    document = Mock()
    document.set.side_effect = RuntimeError(
        f"{SENTINEL_SECRET} {SENTINEL_QUERY} {SENTINEL_EXCEPTION}"
    )
    database = Mock()
    database.collection.return_value.document.return_value = document
    with (
        patch.object(firebase_db, "_get_db", return_value=database),
        _provider_log_handler(firebase_db) as handler,
    ):
        assert not firebase_db.set_cached_data(
            "api_cache", "flight", {"query": SENTINEL_QUERY}
        )

    _assert_sentinels_absent(handler.messages)
    _assert_no_exception_details(handler)


def test_planning_cache_write_failure_ignores_hostile_log_handler():
    document = Mock()
    document.set.side_effect = RuntimeError(SENTINEL_EXCEPTION)
    database = Mock()
    database.collection.return_value.document.return_value = document
    with (
        patch.object(firebase_db, "_get_db", return_value=database),
        _provider_log_handler(firebase_db, raise_on_emit=True),
    ):
        assert not firebase_db.set_cached_data("api_cache", "flight", {"query": "x"})


@pytest.mark.parametrize(
    "operation,expected",
    [
        pytest.param(
            lambda: firebase_db.claim_cached_data(
                "api_cache", "flight", lease_seconds=1, query=SENTINEL_QUERY
            ),
            ("error", None, None),
            id="claim",
        ),
        pytest.param(
            lambda: firebase_db.complete_cached_data_claim(
                "api_cache",
                "flight",
                {"query": SENTINEL_QUERY},
                claim_id="claim",
                query=SENTINEL_QUERY,
            ),
            False,
            id="publish",
        ),
        pytest.param(
            lambda: firebase_db.release_cached_data_claim(
                "api_cache", "flight", claim_id="claim", query=SENTINEL_QUERY
            ),
            False,
            id="release",
        ),
        pytest.param(
            lambda: firebase_db.invalidate_cached_data(
                "api_cache", "flight", query=SENTINEL_QUERY
            ),
            False,
            id="invalidate",
        ),
    ],
)
def test_planning_cache_claim_lifecycle_failures_are_safe(operation, expected):
    failure = RuntimeError(
        f"{SENTINEL_SECRET} {SENTINEL_QUERY} {SENTINEL_EXCEPTION}"
    )
    with (
        patch.object(firebase_db, "_get_db", side_effect=failure),
        _provider_log_handler(firebase_db) as handler,
    ):
        assert operation() == expected

    _assert_sentinels_absent(handler.messages)
    _assert_no_exception_details(handler)


@pytest.mark.parametrize(
    "operation,expected",
    [
        pytest.param(
            lambda: firebase_db.claim_cached_data(
                "api_cache", "flight", lease_seconds=1, query="x"
            ),
            ("error", None, None),
            id="claim",
        ),
        pytest.param(
            lambda: firebase_db.complete_cached_data_claim(
                "api_cache", "flight", {"query": "x"}, claim_id="claim", query="x"
            ),
            False,
            id="publish",
        ),
        pytest.param(
            lambda: firebase_db.release_cached_data_claim(
                "api_cache", "flight", claim_id="claim", query="x"
            ),
            False,
            id="release",
        ),
        pytest.param(
            lambda: firebase_db.invalidate_cached_data("api_cache", "flight", query="x"),
            False,
            id="invalidate",
        ),
    ],
)
def test_planning_cache_claim_lifecycle_failures_ignore_hostile_log_handlers(
    operation, expected
):
    with (
        patch.object(firebase_db, "_get_db", side_effect=RuntimeError(SENTINEL_EXCEPTION)),
        _provider_log_handler(firebase_db, raise_on_emit=True),
    ):
        assert operation() == expected


def test_planning_cache_retry_failure_ignores_hostile_log_handler():
    attempts = 0

    def _flaky_operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError(SENTINEL_EXCEPTION)
        return "recovered"

    with (
        patch.object(firebase_db.time, "sleep"),
        _provider_log_handler(firebase_db, raise_on_emit=True),
    ):
        retrying_operation = firebase_db._with_retry(max_retries=1, base_delay=0)(
            _flaky_operation
        )
        assert retrying_operation() == "recovered"

    assert attempts == 2


def test_budget_cache_failure_does_not_attach_traceback_or_change_fallback():
    with (
        patch.object(budget_assessment, "get_currency_code", side_effect=["MYR", "JPY"]),
        patch.object(
            budget_assessment,
            "get_cached_data",
            side_effect=RuntimeError(
                f"{SENTINEL_SECRET} {SENTINEL_QUERY} {SENTINEL_EXCEPTION}"
            ),
        ),
        _provider_log_handler(budget_assessment) as handler,
    ):
        result = budget_assessment.load_confirmed_budget_assessment(
            assessment_id="assessment",
            origin=SENTINEL_QUERY,
            destination="Japan",
            destination_city="Tokyo",
            start_date="2026-08-01",
            end_date="2026-08-02",
            num_people=1,
        )

    assert result is None
    _assert_sentinels_absent(handler.messages)
    _assert_no_exception_details(handler)


def test_budget_cache_failure_ignores_hostile_log_handler():
    with (
        patch.object(budget_assessment, "get_currency_code", side_effect=["MYR", "JPY"]),
        patch.object(
            budget_assessment,
            "get_cached_data",
            side_effect=RuntimeError(SENTINEL_EXCEPTION),
        ),
        _provider_log_handler(budget_assessment, raise_on_emit=True),
    ):
        assert (
            budget_assessment.load_confirmed_budget_assessment(
                assessment_id="assessment",
                origin=SENTINEL_QUERY,
                destination="Japan",
                destination_city="Tokyo",
                start_date="2026-08-01",
                end_date="2026-08-02",
                num_people=1,
            )
            is None
        )


def test_provider_source_audit_requires_fixed_safe_logging_only():
    helper_path = Path(flights_hotels.__file__).with_name("provider_logging.py")
    assert helper_path.is_file()
    helper_tree = ast.parse(helper_path.read_text(encoding="utf-8"), filename=str(helper_path))
    allowlisted_events = {
        key.value
        for node in ast.walk(helper_tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        if isinstance(target, ast.Name) and target.id == "_EVENT_LEVELS"
        and isinstance(node.value, ast.Dict)
        for key in node.value.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    direct_logger_calls: list[str] = []
    exc_info_calls: list[str] = []
    unsafe_wrapper_calls: list[str] = []
    unknown_wrapper_events: list[str] = []
    unsafe_exception_returns: list[str] = []
    for path in _PLANNING_LOG_PATHS:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for handler in (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ExceptHandler) and node.name
        ):
                for returned in (
                    node for node in ast.walk(handler) if isinstance(node, ast.Return)
                ):
                    raw_exception_returned = returned.value is not None and any(
                        (
                            isinstance(value, ast.FormattedValue)
                            and isinstance(value.value, ast.Name)
                            and value.value.id == handler.name
                        )
                        or (
                            isinstance(value, ast.Call)
                            and isinstance(value.func, ast.Name)
                            and value.func.id in {"str", "repr"}
                            and len(value.args) == 1
                            and isinstance(value.args[0], ast.Name)
                            and value.args[0].id == handler.name
                        )
                        for value in ast.walk(returned.value)
                    )
                    if raw_exception_returned:
                        unsafe_exception_returns.append(
                            f"{path.name}:{returned.lineno}"
                        )
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if any(keyword.arg == "exc_info" for keyword in node.keywords):
                exc_info_calls.append(f"{path.name}:{node.lineno}")
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "logger"
            ):
                direct_logger_calls.append(f"{path.name}:{node.lineno}")
            if isinstance(node.func, ast.Name) and node.func.id == "safe_provider_log":
                allowed_keywords = {"status_code"}
                safe_event = (
                    len(node.args) == 2
                    and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)
                    and all(keyword.arg in allowed_keywords for keyword in node.keywords)
                )
                if not safe_event:
                    unsafe_wrapper_calls.append(f"{path.name}:{node.lineno}")

                elif node.args[1].value not in allowlisted_events:
                    unknown_wrapper_events.append(f"{path.name}:{node.lineno}")

    helper_logger_calls = [
        node
        for node in ast.walk(helper_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "target_logger"
    ]

    assert direct_logger_calls == []
    assert exc_info_calls == []
    assert unsafe_wrapper_calls == []
    assert unknown_wrapper_events == []
    assert unsafe_exception_returns == []
    assert helper_logger_calls
    assert all(call.func.attr == "log" for call in helper_logger_calls)
    assert all(
        all(keyword.arg != "exc_info" for keyword in call.keywords)
        for call in helper_logger_calls
    )
