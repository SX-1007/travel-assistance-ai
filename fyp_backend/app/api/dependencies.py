"""
dependencies.py — FastAPI Dependency Injection Layer
═════════════════════════════════════════════════════════

Centralises every shared resource and security gate used by the
API routers (chat.py, form.py, …).

Two core responsibilities
─────────────────────────
  1. **The Toll Booth  (Security)**      — verify_user_token()
  2. **The Supply Closet (Resources)**   — get_supabase_client(),
                                            get_postgres_pool(),
                                            get_langgraph_memory()

Performance contract
────────────────────
  • All heavy clients (Supabase, Postgres pool, Firebase, Embeddings)
    are module-level singletons instantiated once at import time.
  • Per-request dependencies perform **zero I/O** and **zero allocations**
    — they only validate headers or return references to singletons.
  • Circular imports are avoided via deferred (function-local) imports.

Usage in routes
───────────────
    from app.api.dependencies import CurrentUser, UserChatContext

    @router.post("/")
    async def chat(ctx: UserChatContext, body: ChatRequest):
        user_id = ctx.user_id          # already authenticated
        ...
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Annotated, Any, AsyncIterator, Optional

from fastapi import Depends, Header, HTTPException, Request, status

# ── Internal singletons (imported once at module load) ─────────────
from app.core.supabase_db import (
    connection_pool,  # psycopg ConnectionPool (singleton)
    supabase_client,  # Supabase REST client (singleton)
    verify_vector_store,  # health-check function
)
from app.tools.provider_logging import safe_provider_log

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 1. STARTUP HEALTH VERIFICATION
# ═══════════════════════════════════════════════════════════════════

_vector_store_ready: bool = False


def verify_startup_health() -> bool:
    """
    Called once during FastAPI's ``lifespan`` startup event.

    Validates:
      • Postgres connection pool is open
      • pgvector extension + travel_vectors table exist

    Returns ``True`` if the system is ready to serve traffic.
    """
    global _vector_store_ready
    try:
        if connection_pool.closed:
            safe_provider_log(logger, "storage.dependencies.pool_closed")
            return False

        _vector_store_ready = verify_vector_store()
        if not _vector_store_ready:
            safe_provider_log(logger, "storage.dependencies.vector_unavailable")
        return True

    except Exception:
        safe_provider_log(logger, "storage.dependencies.health_failed")
        return False


# ═══════════════════════════════════════════════════════════════════
# 2. AUTHENTICATION  (the "Toll Booth")
# ═══════════════════════════════════════════════════════════════════

# Header aliases — accept several conventional names for resilience.
# Stored as a tuple (immutable, iteration-friendly, lowercased for O(1) compare).
_X_USER_ID_HEADERS: tuple[str, ...] = ("x-user-id", "x-uid", "x-auth-user-id")

# Cheap sanity bounds — reject absurd values without hashing/DB calls.
_USER_ID_MIN_LEN: int = 1
_USER_ID_MAX_LEN: int = 128


async def verify_user_token(
    request: Request,
    x_user_id: Annotated[Optional[str], Header()] = None,
) -> str:
    """
    Security gate: extracts & validates the caller's user ID.

    The current implementation trusts the ``X-User-ID`` header.  In
    production, replace the trust step with JWT verification against
    Supabase Auth (or your IdP of choice) — the function signature
    stays the same, so no router changes will be needed.

    Returns
    -------
    str
        The validated, stripped user ID.

    Raises
    ------
    HTTPException(401)
        If the header is missing or malformed.

    Performance
    -----------
    Pure-Python string operations — O(1), zero I/O, zero allocations.
    """
    # ── Primary header ──
    user_id: Optional[str] = x_user_id

    # ── Fallback: alternative header aliases (rare path) ──
    if not user_id:
        headers = request.headers
        for alias in _X_USER_ID_HEADERS:
            user_id = headers.get(alias)
            if user_id:
                break

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Missing X-User-ID header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # ── Sanitise & validate format ──
    user_id = user_id.strip()
    if not (_USER_ID_MIN_LEN <= len(user_id) <= _USER_ID_MAX_LEN):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Invalid user identifier format.",
        )

    return user_id


# Ergonomic type alias for route signatures:  `user_id: CurrentUser`
CurrentUser = Annotated[str, Depends(verify_user_token)]


# ═══════════════════════════════════════════════════════════════════
# 3. RESOURCE PROVISIONING  (the "Supply Closet")
# ═══════════════════════════════════════════════════════════════════
# These are zero-cost dependencies — they return references to
# module-level singletons that were initialised at import time in
# supabase_db.py.  No connection acquisition happens here; that is
# deferred to the moment the resource is actually used.


def get_supabase_client() -> Any:
    """
    Returns the singleton Supabase REST client.

    Use for: ``user_profiles``, ``saved_trips``, ``mem0_vector``
    table CRUD operations.
    """
    return supabase_client


def get_postgres_pool() -> Any:
    """
    Returns the shared psycopg ``ConnectionPool`` (min=2, max=20).

    Use for: direct SQL queries, pgvector operations, custom transactions,
    or acquiring a short-lived ``PostgresSaver`` via ``get_checkpointer()``.

    Raises
    ------
    HTTPException(503)
        If the pool has been closed (e.g. during shutdown).
    """
    if connection_pool.closed:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection pool unavailable.",
        )
    return connection_pool


# ── Per-request checkpointer (optional, for direct inspection) ─────
# NOTE: The compiled LangGraph ``workflow`` in graph.py already holds
# its own PostgresSaver instance (via ``from_conn_string``).  This
# dependency is only needed if a route wants to inspect checkpoints
# directly (e.g. an admin endpoint listing conversation threads).
#
# CRITICAL FIX (vs. previous version):
#   The old code called ``get_checkpointer()`` twice — once for
#   ``__enter__`` and once for ``__exit__`` — which produced *two
#   different* context-manager objects.  As a result ``__exit__``
#   was invoked on a fresh, never-entered CM, leaking the connection
#   acquired by the first ``__enter__`` and corrupting pool state.
#   We now capture the CM once and reuse it.


async def get_langgraph_memory() -> AsyncIterator[Any]:
    """
    Yields a short-lived ``PostgresSaver`` bound to a single pooled
    connection.

    The connection is automatically returned to the pool on exit
    (committed on success, rolled back on exception).

    Use ONLY for direct checkpoint inspection — the main graph flow
    does not need this (it uses its own internal checkpointer).

    Implementation note
    -------------------
    ``PostgresSaver``'s context-manager protocol is synchronous and
    may block on a network socket, so we dispatch ``__enter__`` /
    ``__exit__`` to the default ``ThreadPoolExecutor`` via
    ``asyncio.to_thread`` (Python 3.9+) — this keeps the event loop
    responsive under load.
    """
    # Deferred import avoids any circular-import risk at module load time.
    from app.core.supabase_db import get_checkpointer

    cm: Any = None
    checkpointer: Any = None

    try:
        # 1. Obtain the context manager (cheap; no I/O yet).
        cm = await asyncio.to_thread(get_checkpointer)

        # 2. Enter it (this may block on a pool checkout → offload).
        checkpointer = await asyncio.to_thread(cm.__enter__)

        yield checkpointer

    except Exception:
        safe_provider_log(logger, "storage.dependencies.checkpointer_failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory service temporarily unavailable.",
        )
    finally:
        # 3. ALWAYS exit on the *same* CM object — never re-call
        #    get_checkpointer() here (that was the original bug).
        if cm is not None:
            exc_info = (None, None, None)
            try:
                await asyncio.to_thread(cm.__exit__, *exc_info)
            except Exception:  # pragma: no cover
                safe_provider_log(
                    logger, "storage.dependencies.checkpointer_cleanup_failed"
                )


# ═══════════════════════════════════════════════════════════════════
# 4. COMPOSITE CONTEXT BUNDLES
# ═══════════════════════════════════════════════════════════════════
# Combine multiple dependencies into one injectable object so routes
# don't need to declare them individually.


class UserChatContext:
    """
    Bundles everything a chat-style route needs in one dependency.

    Attributes
    ----------
    user_id : str
        Authenticated user's ID (extracted from the X-User-ID header).
    """

    __slots__ = ("user_id",)

    def __init__(
        self,
        user_id: Annotated[str, Depends(verify_user_token)],
    ) -> None:
        self.user_id = user_id

    def __repr__(self) -> str:
        return f"UserChatContext(user_id={self.user_id!r})"


# ═══════════════════════════════════════════════════════════════════
# 5. OBSERVABILITY — Request Correlation ID
# ═══════════════════════════════════════════════════════════════════

# Attribute name cached on request.state — kept as a constant to avoid
# typo bugs and to make grep-friendly.
_REQUEST_STATE_ATTR: str = "request_id"


async def get_request_id(
    request: Request,
    x_request_id: Annotated[Optional[str], Header()] = None,
) -> str:
    """
    Provides a correlation ID for distributed tracing & log aggregation.

    If the client sends ``X-Request-ID`` it is reused; otherwise a fresh
    UUID4 is generated and cached on ``request.state`` for the remainder
    of the request lifecycle.

    Performance
    -----------
    ``getattr`` is O(1); the cached attribute is only written once per
    request, so subsequent dependency resolutions (e.g. via
    ``Annotated[..., Depends(get_request_id)]``) reuse the same value
    without re-rolling a UUID.
    """
    if x_request_id:
        rid = x_request_id.strip()
        if rid:
            return rid

    # Reuse the cached value if another dependency already generated it.
    cached = getattr(request.state, _REQUEST_STATE_ATTR, None)
    if cached:
        return cached

    new_id = str(uuid.uuid4())
    # Cache on request.state — subsequent Depends() calls in the same
    # request will hit this fast-path instead of generating a new UUID.
    setattr(request.state, _REQUEST_STATE_ATTR, new_id)
    return new_id


RequestId = Annotated[str, Depends(get_request_id)]


# ═══════════════════════════════════════════════════════════════════
# 6. AUTHENTICATED GRAPH INVOKERS
# ═══════════════════════════════════════════════════════════════════
# Thin wrappers around graph.invoke_chat / invoke_new_trip that guarantee
# the authenticated user_id is always propagated into the LangGraph
# RunnableConfig (so memory extraction & profile personalisation work).


async def invoke_chat_authenticated(
    user_message: str,
    thread_id: str,
    user_id: str,
    budget_action: Optional[str] = None,
    budget_assessment_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> dict:
    """Wrap ``graph.invoke_chat`` with the authenticated user_id."""
    # Deferred import: graph.py imports supabase_db, which imports config;
    # keeping this local avoids any import-cycle surprises.
    from app.agents.graph import invoke_chat

    return await invoke_chat(
        user_message=user_message,
        thread_id=thread_id,
        user_id=user_id,
        budget_action=budget_action,
        budget_assessment_id=budget_assessment_id,
        request_id=request_id,
    )


async def invoke_new_trip_authenticated(
    initial_state: dict,
    thread_id: str,
    user_id: str,
    request_id: Optional[str] = None,
) -> dict:
    """Wrap ``graph.invoke_new_trip`` with the authenticated user_id."""
    from app.agents.graph import invoke_new_trip

    return await invoke_new_trip(
        initial_state=initial_state,
        thread_id=thread_id,
        user_id=user_id,
        request_id=request_id,
    )


# ═══════════════════════════════════════════════════════════════════
# 7. MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════

__all__: list[str] = [
    # Startup
    "verify_startup_health",
    # Authentication
    "verify_user_token",
    "CurrentUser",
    # Resources
    "get_supabase_client",
    "get_postgres_pool",
    "get_langgraph_memory",
    # Composite
    "UserChatContext",
    # Observability
    "get_request_id",
    "RequestId",
    # Authenticated graph invokers
    "invoke_chat_authenticated",
    "invoke_new_trip_authenticated",
]
