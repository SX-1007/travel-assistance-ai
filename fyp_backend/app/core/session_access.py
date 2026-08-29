"""Authenticated ownership and atomic creation reservations for trip sessions."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator


class SessionAccessDenied(PermissionError):
    """The requested thread is missing or is not owned by the current user."""


_RESERVATION_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS trip_session_owners (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('initializing', 'ready')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

# PostgreSQL can still race in its system catalogs when separate processes
# execute the first CREATE TABLE IF NOT EXISTS concurrently.  The
# transaction-scoped lock serializes only these short ownership transactions
# and is released automatically on commit/rollback or connection loss.
_RESERVATION_SCHEMA_LOCK_SQL = "SELECT pg_advisory_xact_lock(794523017)"

# Twice the hard 480-second new-trip execution ceiling. A current initializer
# keeps exclusive ownership throughout every normal invocation; only a later
# request from the same owner can recover the row after this server-owned lease.
NEW_SESSION_INITIALIZATION_LEASE_SECONDS = 960


@asynccontextmanager
async def _reservation_connection(checkpointer: Any) -> AsyncIterator[Any]:
    """Yield a transaction-capable connection from the saver pool or connection."""
    source = getattr(checkpointer, "conn", None)
    if source is None:
        raise RuntimeError("Checkpointer does not expose a durable connection.")
    connection_factory = getattr(source, "connection", None)
    if connection_factory is None:
        yield source
        return
    async with connection_factory() as connection:
        yield connection


async def _ensure_reservation_table(connection: Any) -> None:
    async with connection.cursor() as cursor:
        await cursor.execute(_RESERVATION_SCHEMA_LOCK_SQL)
        await cursor.execute(_RESERVATION_TABLE_SQL)


async def reserve_new_session(
    checkpointer: Any,
    session_id: str,
    user_id: str,
) -> bool:
    """Atomically reserve a public session ID for one initial graph invocation."""
    async with _reservation_connection(checkpointer) as connection:
        async with connection.transaction():
            await _ensure_reservation_table(connection)
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    INSERT INTO trip_session_owners (session_id, user_id, state)
                    VALUES (%s, %s, 'initializing')
                    ON CONFLICT (session_id) DO NOTHING
                    RETURNING session_id
                    """,
                    (session_id, user_id),
                )
                return await cursor.fetchone() is not None


async def complete_new_session_reservation(
    checkpointer: Any,
    session_id: str,
    user_id: str,
) -> bool:
    """Mark the matching reservation ready and report whether it still existed."""
    async with _reservation_connection(checkpointer) as connection:
        async with connection.transaction():
            await _ensure_reservation_table(connection)
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    UPDATE trip_session_owners
                    SET state = 'ready'
                    WHERE session_id = %s AND user_id = %s AND state = 'initializing'
                    RETURNING session_id
                    """,
                    (session_id, user_id),
                )
                return await cursor.fetchone() is not None


async def reclaim_expired_new_session_reservation(
    checkpointer: Any,
    session_id: str,
    user_id: str,
) -> bool:
    """Refresh an expired same-owner lease after checkpoint absence is confirmed."""
    async with _reservation_connection(checkpointer) as connection:
        async with connection.transaction():
            await _ensure_reservation_table(connection)
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    UPDATE trip_session_owners
                    SET created_at = NOW()
                    WHERE session_id = %s AND user_id = %s
                      AND state = 'initializing'
                      AND created_at <= NOW() - (%s * INTERVAL '1 second')
                    RETURNING session_id
                    """,
                    (
                        session_id,
                        user_id,
                        NEW_SESSION_INITIALIZATION_LEASE_SECONDS,
                    ),
                )
                return await cursor.fetchone() is not None


async def finalize_expired_new_session_reservation(
    checkpointer: Any,
    session_id: str,
    user_id: str,
) -> bool:
    """Finalize an expired same-owner lease when its checkpoint is present."""
    async with _reservation_connection(checkpointer) as connection:
        async with connection.transaction():
            await _ensure_reservation_table(connection)
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    UPDATE trip_session_owners
                    SET state = 'ready'
                    WHERE session_id = %s AND user_id = %s
                      AND state = 'initializing'
                      AND created_at <= NOW() - (%s * INTERVAL '1 second')
                    RETURNING session_id
                    """,
                    (
                        session_id,
                        user_id,
                        NEW_SESSION_INITIALIZATION_LEASE_SECONDS,
                    ),
                )
                return await cursor.fetchone() is not None


async def release_new_session_reservation(
    checkpointer: Any,
    session_id: str,
    user_id: str,
) -> None:
    """Release only a failed initializer's reservation so a later retry can win."""
    async with _reservation_connection(checkpointer) as connection:
        async with connection.transaction():
            await _ensure_reservation_table(connection)
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    DELETE FROM trip_session_owners
                    WHERE session_id = %s AND user_id = %s AND state = 'initializing'
                    """,
                    (session_id, user_id),
                )


async def backfill_existing_session_owner(
    checkpointer: Any,
    session_id: str,
    user_id: str,
) -> bool:
    """Atomically migrate a verified legacy checkpoint without changing its owner."""
    async with _reservation_connection(checkpointer) as connection:
        async with connection.transaction():
            await _ensure_reservation_table(connection)
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    INSERT INTO trip_session_owners (session_id, user_id, state)
                    VALUES (%s, %s, 'ready')
                    ON CONFLICT (session_id) DO UPDATE SET state = 'ready'
                    WHERE trip_session_owners.user_id = EXCLUDED.user_id
                    RETURNING session_id
                    """,
                    (session_id, user_id),
                )
                return await cursor.fetchone() is not None


async def verify_session_owner(
    checkpointer: Any,
    session_id: str,
    user_id: str,
) -> bool:
    """Verify a ready session against the dedicated durable owner table."""
    async with _reservation_connection(checkpointer) as connection:
        async with connection.transaction():
            await _ensure_reservation_table(connection)
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT session_id FROM trip_session_owners
                    WHERE session_id = %s AND user_id = %s AND state = 'ready'
                    """,
                    (session_id, user_id),
                )
                return await cursor.fetchone() is not None
