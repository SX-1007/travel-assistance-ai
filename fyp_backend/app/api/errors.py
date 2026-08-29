"""
errors.py — Centralised exception → HTTPException translation
════════════════════════════════════════════════════════════

Shared by every router that calls into the LangGraph workflow.
Keeps the mapping in ONE place so chat.py / form.py / future
routers all behave identically when the upstream LLM or
Postgres checkpointer misbehaves.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Final, Mapping

import httpx
from fastapi import HTTPException, status

from app.tools.currency import CurrencyRateUnavailableError
from app.core.session_access import SessionAccessDenied
from app.schemas.responses import PlanningUnavailableReason

logger = logging.getLogger(__name__)

PLANNING_UNAVAILABLE_MESSAGE: Final[str] = (
    "Planning is temporarily unavailable. Please retry."
)


def planning_unavailable_reason(
    state: Mapping[str, Any],
) -> PlanningUnavailableReason:
    """Collapse private diagnostics into one bounded public reason."""
    raw_codes = (
        list(state.get("planning_issue_codes") or [])
        + list(state.get("output_review_issue_codes") or [])
    )
    codes = [code.lower() for code in raw_codes if isinstance(code, str)]
    if any("deadline" in code or "timeout" in code for code in codes):
        return "deadline_exhausted"
    if any("provider" in code or "data.unavailable" in code for code in codes):
        return "provider_data_unavailable"
    if any(code.startswith("review.") or "review.unavailable" in code for code in codes):
        return "review_unavailable"
    return "validation_failed"

# ── Optional SDK imports ────────────────────────────────────────────
# These packages may be absent in lightweight test environments; guard
# them so the module imports cleanly even when the SDKs aren't installed.
try:  # pragma: no cover - import guard
    # Google Gemini / Vertex AI exceptions via google-api-core
    from google.api_core.exceptions import ResourceExhausted as _GeminiRateLimitError
    from google.api_core.exceptions import DeadlineExceeded as _GeminiTimeoutError
    from google.api_core.exceptions import (
        ServiceUnavailable as _GeminiServiceUnavailable,
    )
    from google.api_core.exceptions import GoogleAPIError as _GeminiAPIError
except Exception:  # pragma: no cover
    _GeminiRateLimitError = _GeminiTimeoutError = _GeminiServiceUnavailable = (
        _GeminiAPIError
    ) = None  # type: ignore[assignment]

try:  # pragma: no cover - import guard
    from langgraph.errors import GraphRecursionError as _GraphRecursionError
except Exception:  # pragma: no cover
    _GraphRecursionError = None  # type: ignore[assignment]


# ── Exception → (HTTP status, safe message) ─────────────────────────
# ORDER IS SIGNIFICANT: subclasses MUST appear before their parents,
# otherwise isinstance() short-circuits to the wrong row.
_EXCEPTION_MAP: Final[list[tuple[type[BaseException] | None, int, str]]] = [
    (
        SessionAccessDenied,
        status.HTTP_404_NOT_FOUND,
        "Trip session not found",
    ),
    (
        CurrencyRateUnavailableError,
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "Exchange-rate service temporarily unavailable; please retry shortly",
    ),
    # ── LangGraph runtime ──
    (
        _GraphRecursionError,
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "Agent exceeded its maximum iteration limit while planning",
    ),
    # ── Google Gemini: rate limit (subclass of GoogleAPIError → must come first) ──
    (
        _GeminiRateLimitError,
        status.HTTP_429_TOO_MANY_REQUESTS,
        "Upstream LLM rate limit reached; please retry shortly",
    ),
    # ── Google Gemini: timeouts ──
    (
        _GeminiTimeoutError,
        status.HTTP_504_GATEWAY_TIMEOUT,
        "Upstream LLM call timed out",
    ),
    # ── Google Gemini: service unavailable (e.g. 503 from Google) ──
    (
        _GeminiServiceUnavailable,
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "Upstream LLM service temporarily unavailable",
    ),
    # ── Standard async/http timeouts (often raised by LangChain wrappers) ──
    (
        asyncio.TimeoutError,
        status.HTTP_504_GATEWAY_TIMEOUT,
        "Workflow execution timed out",
    ),
    (
        httpx.TimeoutException,
        status.HTTP_504_GATEWAY_TIMEOUT,
        "Upstream HTTP call timed out",
    ),
    # ── Google Gemini: generic API errors (parent class) ──
    (
        _GeminiAPIError,
        status.HTTP_502_BAD_GATEWAY,
        "Upstream LLM service returned an error",
    ),
    # ── HTTP transport errors (connection reset, DNS, …) ──
    (httpx.HTTPError, status.HTTP_502_BAD_GATEWAY, "Upstream HTTP service error"),
]


def _load_db_exception_types() -> list[type[BaseException]]:
    """Best-effort import of DB exception types we want to map to 503."""
    types: list[type[BaseException]] = []
    try:  # pragma: no cover - env-dependent
        import psycopg  # type: ignore

        types.append(psycopg.OperationalError)
    except Exception:  # pragma: no cover
        pass
    try:  # pragma: no cover - env-dependent
        from sqlalchemy.exc import DBAPIError  # type: ignore

        types.append(DBAPIError)
    except Exception:  # pragma: no cover
        pass
    return types


# Final lazily-resolved mapping (DB types added once at first use)
_DB_TYPES: list[type[BaseException]] | None = None


def map_exception_to_http(exc: BaseException) -> HTTPException:
    """
    Translate any exception raised by the graph/LLM/DB stack into a
    client-safe ``HTTPException`` with an appropriate status code.

    * Server-side: the full traceback is logged by the caller via
      ``logger.exception(...)``.
    * Client-side: only a generic, static ``detail`` string is returned —
      never ``str(exc)``, which may leak URLs / request IDs / SQL.
    """
    global _DB_TYPES
    if _DB_TYPES is None:
        _DB_TYPES = _load_db_exception_types()

    # 1) DB / checkpointer errors → 503 (transient infra issue)
    for db_type in _DB_TYPES:
        if isinstance(exc, db_type):
            return HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Trip-state store temporarily unavailable",
            )

    # 2) Walk the ordered static map
    for exc_type, code, message in _EXCEPTION_MAP:
        if exc_type is None:
            continue
        if isinstance(exc, exc_type):
            return HTTPException(status_code=code, detail=message)

    # 3) Catch-all — never leak the raw message
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="An unexpected error occurred while processing your request",
    )
