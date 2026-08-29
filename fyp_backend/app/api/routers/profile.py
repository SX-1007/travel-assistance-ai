"""
profile.py — Onboarding profile router
═════════════════════════════════════════════════════════

Captures the one-time onboarding info (Name / Origin Country / Origin State)
and persists it to ``user_profiles`` BEFORE the user can plan a trip.

Because the origin is stored here, the initial trip form
(``/api/form/submit``) no longer asks for it again.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Response, status

from app.api.dependencies import CurrentUser, RequestId
from app.schemas.requests import OnboardingRequest

logger = logging.getLogger(__name__)
router = APIRouter(tags=["profile"])


@router.post(
    "/",
    status_code=200,
    summary="Create/update the onboarding profile (Name, Origin Country/State)",
)
async def upsert_profile(
    request: OnboardingRequest,
    user_id: CurrentUser,
    request_id: RequestId,
) -> dict:
    """Persist the onboarding profile so the trip form can omit origin fields."""
    # Deferred import avoids any import-cycle risk at module load time.
    from app.core.supabase_db import create_user_profile

    ok = create_user_profile(
        user_id,
        {
            "name": request.name,
            "home_country": request.origin_country,
            "home_state": request.origin_state,
        },
    )
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not save profile. Please try again.",
        )

    logger.info(
        "profile.upsert.done",
        extra={"request_id": request_id, "user_id": user_id},
    )
    return {"status": "success", "user_id": user_id}


@router.get(
    "/",
    status_code=200,
    summary="Fetch the current onboarding profile (to gate app usage)",
)
async def get_profile(
    user_id: CurrentUser,
    request_id: RequestId,
    response: Response,
) -> dict:
    """Return the stored profile, or ``onboarded=False`` if none exists yet."""
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Vary"] = "X-User-ID"
    from app.core.supabase_db import fetch_user_profile

    try:
        profile = fetch_user_profile(user_id, raise_on_error=True)
    except Exception:
        logger.exception(
            "profile.fetch.failed",
            extra={"request_id": request_id, "user_id": user_id},
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not load profile. Please try again.",
        )

    # All three onboarding fields are mandatory — a profile missing any of
    # them re-triggers the onboarding screen on the frontend.
    onboarded = bool(
        profile
        and profile.get("name")
        and profile.get("home_country")
        and profile.get("home_state")
    )
    return {
        "status": "success",
        "onboarded": onboarded,
        "profile": profile or {},
    }