"""
geocode.py — Address → coordinates lookup router
═════════════════════════════════════════════════════════

``GET /api/geocode?q=<place name, address>``

The chat UI pins ``[MAP: …]`` locations by RE-geocoding each place's
name + address through SerpAPI's Google Maps engine instead of trusting
the coordinates the LLM transcribed into the reply (Gemini garbles digits
when copying tool output, so pins landed in the wrong spot). Mapbox's
geocoder was evaluated first but is unusable for POIs / Malaysian
addresses — SerpAPI returns Google's own coordinates.

Results are LRU-cached in-process: place coordinates are stable, and the
cache keeps repeated chat re-renders from burning SerpAPI credits.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict, Optional

import requests
from fastapi import APIRouter, Query

from app.api.dependencies import CurrentUser, RequestId
from app.core.config import settings
from app.tools.provider_logging import safe_provider_log

logger = logging.getLogger(__name__)
router = APIRouter(tags=["geocode"])

_session = requests.Session()


@lru_cache(maxsize=512)
def _geocode(query: str, ll: Optional[str]) -> Optional[Dict[str, Any]]:
    """Resolve a free-text place query to Google Maps name/address/coords.

    Returns ``None`` when nothing is found (cached — a miss is stable);
    raises on transport errors so transient failures are NOT cached.
    """
    params: Dict[str, Any] = {
        "engine": "google_maps",
        "type": "search",
        "q": query,
        "hl": "en",
        "api_key": settings.SERPAPI_PLACE,
    }
    if ll:
        params["ll"] = ll
    resp = _session.get("https://serpapi.com/search", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    # A specific query returns one `place_results` dict; a broader one
    # returns a `local_results` list — take whichever has coordinates.
    candidates = [data.get("place_results") or {}]
    candidates += list(data.get("local_results") or [])
    for top in candidates:
        gps = top.get("gps_coordinates") or {}
        if gps.get("latitude") is not None and gps.get("longitude") is not None:
            return {
                "name": top.get("title", ""),
                "address": top.get("address", ""),
                "lat": gps["latitude"],
                "lng": gps["longitude"],
            }
    return None


@router.get(
    "/",
    status_code=200,
    summary="Resolve a place name/address to exact Google Maps coordinates",
)
def geocode(
    user_id: CurrentUser,
    request_id: RequestId,
    q: str = Query(..., min_length=3, max_length=300),
    lat: Optional[float] = Query(None, ge=-90, le=90),
    lng: Optional[float] = Query(None, ge=-180, le=180),
) -> dict:
    """Geocode ``q``; ``lat``/``lng`` (optional) bias the search area.

    Always answers 200 — ``found: false`` on a miss so the frontend can
    simply fall back to the coordinates it already has.
    """
    # Round the bias anchor so nearby variations hit the same cache slot.
    ll = f"@{lat:.2f},{lng:.2f},13z" if lat is not None and lng is not None else None
    try:
        hit = _geocode(q.strip(), ll)
    except (requests.RequestException, ValueError):
        safe_provider_log(logger, "provider.planning_geocode.failed")
        return {"status": "error", "found": False}
    if not hit:
        safe_provider_log(logger, "provider.planning_geocode.miss")
        return {"status": "success", "found": False}
    return {"status": "success", "found": True, **hit}
