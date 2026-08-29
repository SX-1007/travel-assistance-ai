"""
run.py — Windows-safe dev entrypoint.

Start the backend with:  python run.py

Why this exists
---------------
The app uses an async Postgres driver (AsyncPostgresSaver → psycopg async),
which cannot run on Windows' default ProactorEventLoop — it needs a
SelectorEventLoop.

Two obstacles on Windows:
  1. The event loop is created before our code runs when launched via
     ``uvicorn``/``fastapi dev``, so setting the policy in main.py is too late.
  2. Uvicorn's ``Server.run()`` explicitly forces WindowsProactorEventLoopPolicy
     in its own event-loop setup, overriding anything we set.

So we set the SelectorEventLoop policy first, then drive ``server.serve()``
ourselves under ``asyncio.run()`` — bypassing ``Server.run()`` and its
Proactor override entirely.

Port/host come from the PORT/HOST env vars (defaults: 127.0.0.1:8000).
"""

from __future__ import annotations

import asyncio
import os
import sys

# MUST run before the event loop is created. No-op off Windows.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn  # noqa: E402  (import after policy is set)


def main() -> None:
    config = uvicorn.Config(
        "main:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        # Do NOT use uvicorn's reload/loop setup here — it re-forces Proactor.
    )
    server = uvicorn.Server(config)
    # asyncio.run() builds the loop from OUR policy (SelectorEventLoop on
    # Windows) and never calls Server.run()'s Proactor override.
    asyncio.run(server.serve())


if __name__ == "__main__":
    main()
