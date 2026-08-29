"""
trip_details.py — Tool for changing core trip requirements mid-conversation.

Covers requirement changes that the budget/itinerary tools cannot express:
number of travellers, travel dates, destination country/cities, origin
country.

The tool itself only validates and echoes the requested changes; the graph's
``post_tool_processing`` node applies them to ``AgentState`` and
``route_after_post_tool`` then re-runs the planning pipeline automatically:

  • country / origin_country changed
      → full re-init (currency conversion + budget allocation + flights +
        hotels + activities + maps)
  • start_date / end_date / num_people / city changed
      → re-plan (flights + hotels + activities + maps; currency kept)
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from app.agents.state import AgentState


def _valid_date(value: str) -> bool:
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except (TypeError, ValueError):
        return False


@tool
def update_trip_details(
    state: Annotated[AgentState, InjectedState],
    num_people: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    country: Optional[str] = None,
    city: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Update core trip requirements (travellers, dates, destination).

    Use this whenever the user changes a REQUIREMENT of the trip, e.g.:
      • "actually we are 4 people now"      → num_people=4
      • "move the trip to 12-18 August"     → start_date + end_date
      • "let's go to Osaka instead"         → country/city

    Only pass the fields that changed. After this tool runs, the system
    AUTOMATICALLY re-plans the itinerary (flights, hotels, activities, and
    maps; currency and budget allocation are refreshed when the destination
    changes). Do NOT call edit_itinerary for these changes.

    Args:
        num_people: New number of travellers (>= 1).
        start_date: New trip start date, YYYY-MM-DD.
        end_date: New trip end date, YYYY-MM-DD.
        country: New destination country.
        city: New destination city/cities list.
    """
    updates: Dict[str, Any] = {}

    if num_people is not None:
        if num_people < 1:
            return {"error": "num_people must be at least 1."}
        updates["num_people"] = int(num_people)

    if start_date is not None:
        if not _valid_date(start_date):
            return {"error": f"start_date '{start_date}' is not YYYY-MM-DD."}
        updates["start_date"] = start_date

    if end_date is not None:
        if not _valid_date(end_date):
            return {"error": f"end_date '{end_date}' is not YYYY-MM-DD."}
        updates["end_date"] = end_date

    if {"start_date", "end_date"} & updates.keys():
        merged_start = updates.get("start_date", state.start_date)
        merged_end = updates.get("end_date", state.end_date)
        if merged_start and merged_end and merged_end < merged_start:
            return {"error": "end_date must be on or after start_date."}

    if country is not None and str(country).strip():
        updates["country"] = str(country).strip()

    if city is not None:
        cleaned = [str(c).strip() for c in city if str(c).strip()]
        updates["city"] = cleaned

    if not updates:
        return {"error": "No trip details were provided to update."}

    return {
        "status": "success",
        "action": "update_trip_details",
        "updates": updates,
        "message": (
            f"Trip details updated ({', '.join(sorted(updates))}). "
            "The itinerary is being re-planned automatically with the new "
            "requirements — summarise the refreshed plan for the user."
        ),
    }
