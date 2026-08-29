"""
Travel Assistant AI — Application Entry Point

High-performance FastAPI backend for the self-adaptive travel-planning agent.

Key design decisions
--------------------
• **Lifespan-managed resources** — DB pool, checkpoint tables, and vector
  store are initialised at startup (not import time) so the process can boot
  even if a backend is temporarily unavailable.
• **orjson** — default JSON serializer (~2-3× faster than stdlib json).
• **GZip middleware** — compresses responses > 1 KB (itineraries, GeoJSON).
• **Explicit CORS origins** — browsers reject `*` together with credentials.
• **Graceful shutdown** — connection pool is closed cleanly on exit.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, List

# ── Windows event-loop compatibility ────────────────────────────────
# psycopg's async driver (used by AsyncPostgresSaver) cannot run on
# Windows' default ProactorEventLoop. Select the SelectorEventLoop policy
# *before* uvicorn creates the loop. No-op on Linux/macOS.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import ORJSONResponse

from app.api.routers import chat, form, geocode, profile
from app.api.dependencies import verify_startup_health
from app.core.supabase_db import (
    close_connection_pool,
    init_connection_pool,
    init_vector_store,
)
from app.agents.graph import init_graph, shutdown_graph

# ──────────────────────────────────────────────────────────────────── Logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("travel_assistant")


# ──────────────────────────────────────────────────────────────────── Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise resources at startup; tear them down at shutdown."""
    t0 = time.perf_counter()
    logger.info("Starting Travel Assistant AI …")

    # ── Startup (order matters: pool → vector store → graph) ──
    try:
        init_connection_pool()
        logger.debug("Postgres connection pool opened.")

        init_vector_store()
        logger.debug("Vector store ready.")

        await init_graph()
        logger.debug("LangGraph workflow compiled.")
    except Exception:
        # Fail fast — a half-initialised app is worse than no app.
        logger.critical(
            "Resource initialisation failed — aborting startup.",
            exc_info=True,
        )
        raise

    # Health check — non-fatal; app continues in degraded mode
    try:
        healthy = verify_startup_health()
    except Exception:
        logger.warning("Health check raised an exception.", exc_info=True)
        healthy = False

    if not healthy:
        logger.warning("Startup health check FAILED — degraded mode active.")
    else:
        logger.debug("Startup health check passed.")

    logger.info("Startup finished in %.1f ms.", (time.perf_counter() - t0) * 1000)

    yield  # ── App is running ──

    # ── Shutdown ──
    logger.info("Shutting down Travel Assistant AI …")
    try:
        await shutdown_graph()
        logger.debug("Graph resources released.")
    except Exception:
        logger.warning("Error shutting down graph.", exc_info=True)
    try:
        close_connection_pool()
        logger.debug("Postgres connection pool closed.")
    except Exception:
        logger.warning("Error closing connection pool.", exc_info=True)
    logger.info("Shutdown complete.")


# ──────────────────────────────────────────────────────────── App factory
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Travel Assistant AI",
        description="Backend for self-adaptive travel planning agent",
        version=os.getenv("APP_VERSION", "1.0.0"),
        lifespan=lifespan,
        default_response_class=ORJSONResponse,  # orjson ≈ 2-3× faster JSON
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware (registered outermost-first) ──

    # GZip: compress large payloads (itineraries, route GeoJSON, hotel lists)
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    # CORS — explicit origins (browsers reject "*" + allow_credentials=True)
    _raw_origins = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173",
    )
    allowed_origins: List[str] = [
        origin.strip() for origin in _raw_origins.split(",") if origin.strip()
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        # Accept ANY localhost/127.0.0.1 port in addition to the explicit
        # list — Vite picks the next free port (5174, 5175, …) when 5173 is
        # busy, and a mismatched port must not silently break the app.
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time"],
    )

    # Per-request timing + structured access log
    @app.middleware("http")
    async def _timing_and_log(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Process-Time"] = f"{elapsed_ms:.2f}ms"

        path = request.url.path
        if path not in ("/", "/health", "/healthz", "/favicon.ico"):
            logger.info(
                "%s %s → %d (%.1f ms)",
                request.method,
                path,
                response.status_code,
                elapsed_ms,
            )
        return response

    # ── Routers ──
    # Tags are declared once on each APIRouter (tags=["chat"] / ["form"]).
    # Do NOT re-declare tags here — include_router *appends* them, which would
    # make every route appear under two Swagger groups (e.g. "chat" and "Chat").
    app.include_router(chat.router, prefix="/api/chat")
    app.include_router(form.router, prefix="/api/form")
    app.include_router(profile.router, prefix="/api/profile")
    app.include_router(geocode.router, prefix="/api/geocode")

    # ── Root + health endpoints ──
    @app.get("/", include_in_schema=False)
    async def root():
        return {"message": "Travel Assistant AI is running"}

    @app.get("/health", tags=["Health"])
    async def health():
        """Liveness / readiness probe for container orchestrators."""
        return {"status": "ok"}

    @app.get("/healthz", include_in_schema=False)
    async def healthz():
        """Alias used by some load balancers (e.g. AWS ALB)."""
        return {"status": "ok"}

    return app


# Module-level app instance (uvicorn: app.main:app)
app = create_app()
