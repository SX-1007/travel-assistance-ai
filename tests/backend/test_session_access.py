"""Direct tests for the durable trip-session ownership SQL boundary."""

from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

from app.core import session_access


class _RecordingCursor:
    def __init__(self, fetch_result=None):
        self.fetch_result = fetch_result
        self.executions: list[tuple[str, object | None]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute(self, sql, params=None):
        self.executions.append((" ".join(sql.split()), params))

    async def fetchone(self):
        return self.fetch_result


class _RecordingConnection:
    def __init__(self, fetch_result=None):
        self.cursor_instance = _RecordingCursor(fetch_result)
        self.transactions = 0

    def cursor(self):
        return self.cursor_instance

    @asynccontextmanager
    async def transaction(self):
        self.transactions += 1
        yield


class _Pool:
    def __init__(self, connection):
        self._connection = connection

    @asynccontextmanager
    async def connection(self):
        yield self._connection


class _Saver:
    def __init__(self, connection):
        self.conn = _Pool(connection)


@pytest.mark.asyncio
async def test_reservation_table_creation_is_serialized_across_processes():
    connection = _RecordingConnection()

    await session_access._ensure_reservation_table(connection)

    statements = [sql for sql, _ in connection.cursor_instance.executions]
    assert statements[0].startswith("SELECT pg_advisory_xact_lock")
    assert statements[1].startswith("CREATE TABLE IF NOT EXISTS trip_session_owners")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fetch_result", "expected"),
    [({"session_id": "trip-1"}, True), (None, False)],
)
async def test_reserve_new_session_reports_the_atomic_insert_winner(
    fetch_result,
    expected,
):
    connection = _RecordingConnection(fetch_result)

    result = await session_access.reserve_new_session(
        _Saver(connection),
        "trip-1",
        "user-1",
    )

    assert result is expected
    assert connection.transactions == 1
    sql, params = connection.cursor_instance.executions[-1]
    assert "ON CONFLICT (session_id) DO NOTHING" in sql
    assert "RETURNING session_id" in sql
    assert params == ("trip-1", "user-1")


@pytest.mark.asyncio
async def test_reservation_release_is_owner_and_state_scoped():
    connection = _RecordingConnection()

    await session_access.release_new_session_reservation(
        _Saver(connection),
        "trip-1",
        "user-1",
    )

    sql, params = connection.cursor_instance.executions[-1]
    assert sql.startswith("DELETE")
    assert "session_id = %s AND user_id = %s AND state = 'initializing'" in sql
    assert params == ("trip-1", "user-1")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fetch_result", "expected"),
    [({"session_id": "trip-1"}, True), (None, False)],
)
async def test_reservation_completion_reports_matching_owner_update(
    fetch_result,
    expected,
):
    connection = _RecordingConnection(fetch_result)

    result = await session_access.complete_new_session_reservation(
        _Saver(connection),
        "trip-1",
        "user-1",
    )

    assert result is expected
    sql, params = connection.cursor_instance.executions[-1]
    assert sql.startswith("UPDATE trip_session_owners SET state = 'ready'")
    assert "session_id = %s AND user_id = %s AND state = 'initializing'" in sql
    assert "RETURNING session_id" in sql
    assert params == ("trip-1", "user-1")


@pytest.mark.asyncio
async def test_reservation_completion_cannot_finalize_a_different_owner():
    connection = _RecordingConnection(None)

    result = await session_access.complete_new_session_reservation(
        _Saver(connection),
        "shared-trip",
        "attacker",
    )

    assert result is False
    sql, params = connection.cursor_instance.executions[-1]
    assert "user_id = %s AND state = 'initializing'" in sql
    assert "RETURNING session_id" in sql
    assert params == ("shared-trip", "attacker")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fetch_result", "expected"),
    [({"session_id": "trip-1"}, True), (None, False)],
)
async def test_only_expired_same_owner_initialization_can_be_reclaimed(
    fetch_result,
    expected,
):
    connection = _RecordingConnection(fetch_result)

    result = await session_access.reclaim_expired_new_session_reservation(
        _Saver(connection),
        "trip-1",
        "user-1",
    )

    assert result is expected
    sql, params = connection.cursor_instance.executions[-1]
    assert sql.startswith("UPDATE trip_session_owners SET created_at = NOW()")
    assert "session_id = %s AND user_id = %s AND state = 'initializing'" in sql
    assert "created_at <= NOW() - (%s * INTERVAL '1 second')" in sql
    assert "RETURNING session_id" in sql
    assert params == ("trip-1", "user-1", 960)


@pytest.mark.asyncio
async def test_cross_user_or_unexpired_initialization_is_not_reclaimed():
    connection = _RecordingConnection(None)

    result = await session_access.reclaim_expired_new_session_reservation(
        _Saver(connection),
        "shared-trip",
        "attacker",
    )

    assert result is False
    sql, params = connection.cursor_instance.executions[-1]
    assert "user_id = %s AND state = 'initializing'" in sql
    assert "created_at <= NOW() - (%s * INTERVAL '1 second')" in sql
    assert params == ("shared-trip", "attacker", 960)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fetch_result", "expected"),
    [({"session_id": "trip-1"}, True), (None, False)],
)
async def test_checkpoint_present_recovery_finalizes_only_expired_same_owner(
    fetch_result,
    expected,
):
    connection = _RecordingConnection(fetch_result)

    result = await session_access.finalize_expired_new_session_reservation(
        _Saver(connection),
        "trip-1",
        "user-1",
    )

    assert result is expected
    sql, params = connection.cursor_instance.executions[-1]
    assert sql.startswith("UPDATE trip_session_owners SET state = 'ready'")
    assert "session_id = %s AND user_id = %s AND state = 'initializing'" in sql
    assert "created_at <= NOW() - (%s * INTERVAL '1 second')" in sql
    assert "RETURNING session_id" in sql
    assert params == ("trip-1", "user-1", 960)


@pytest.mark.asyncio
async def test_cross_user_expired_initialization_cannot_be_finalized():
    connection = _RecordingConnection(None)

    result = await session_access.finalize_expired_new_session_reservation(
        _Saver(connection),
        "shared-trip",
        "attacker",
    )

    assert result is False
    sql, params = connection.cursor_instance.executions[-1]
    assert "user_id = %s AND state = 'initializing'" in sql
    assert "created_at <= NOW() - (%s * INTERVAL '1 second')" in sql
    assert params == ("shared-trip", "attacker", 960)


@pytest.mark.asyncio
async def test_legacy_backfill_cannot_overwrite_a_different_owner():
    connection = _RecordingConnection(None)

    result = await session_access.backfill_existing_session_owner(
        _Saver(connection),
        "trip-1",
        "user-1",
    )

    assert result is False
    sql, params = connection.cursor_instance.executions[-1]
    assert "WHERE trip_session_owners.user_id = EXCLUDED.user_id" in sql
    assert params == ("trip-1", "user-1")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fetch_result", "expected"),
    [({"session_id": "trip-1"}, True), (None, False)],
)
async def test_ready_session_owner_verification_is_user_scoped(
    fetch_result,
    expected,
):
    connection = _RecordingConnection(fetch_result)

    result = await session_access.verify_session_owner(
        _Saver(connection),
        "trip-1",
        "user-1",
    )

    assert result is expected
    assert connection.transactions == 1
    sql, params = connection.cursor_instance.executions[-1]
    assert sql.startswith("SELECT session_id FROM trip_session_owners")
    assert "session_id = %s AND user_id = %s AND state = 'ready'" in sql
    assert params == ("trip-1", "user-1")
