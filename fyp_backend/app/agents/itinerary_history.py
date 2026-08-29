"""Compact, server-owned itinerary revision history helpers.

The live itinerary in ``AgentState`` is intentionally mutable. These helpers keep
small immutable copies of previously accepted revisions so later read-only
questions (for example, "what was Day 2 activity 3 before the change?") can be
answered from trusted state instead of reconstructed conversation prose.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable

_MAX_HISTORY_REVISIONS = 20
_NO_HISTORY_MESSAGE = "No previous accepted itinerary revisions are available."


def _normalise_entry(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None

    revision = value.get("revision", value.get("plan_revision"))
    itinerary = value.get("draft_itinerary")

    if (
        isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision <= 0
        or not isinstance(itinerary, list)
        or not itinerary
    ):
        return None

    return {
        "revision": revision,
        "draft_itinerary": copy.deepcopy(itinerary),
    }


def append_previous_itinerary(
    history: Iterable[dict[str, Any]] | None,
    accepted_snapshot: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Append one previously accepted itinerary revision, without duplicates.

    The first revision is retained permanently and the newest 19 revisions are
    retained after that. This keeps the original baseline available while
    bounding checkpoint growth.
    """

    normalised_history: list[dict[str, Any]] = []
    seen_revisions: set[int] = set()

    for raw_entry in history or []:
        entry = _normalise_entry(raw_entry)

        if entry is None or entry["revision"] in seen_revisions:
            continue

        seen_revisions.add(entry["revision"])
        normalised_history.append(entry)

    new_entry = _normalise_entry(accepted_snapshot)

    if (
        new_entry is not None
        and new_entry["revision"] not in seen_revisions
    ):
        normalised_history.append(new_entry)

    normalised_history.sort(
        key=lambda item: item["revision"]
    )

    if len(normalised_history) > _MAX_HISTORY_REVISIONS:
        normalised_history = [
            normalised_history[0],
            *normalised_history[
                -(_MAX_HISTORY_REVISIONS - 1):
            ],
        ]

    return normalised_history


def recover_itinerary_history(
    checkpoint_states: Iterable[dict[str, Any]] | None,
    *,
    current_revision: int,
) -> list[dict[str, Any]]:
    """Recover previous revisions from legacy LangGraph checkpoint states.

    Older sessions created before ``itinerary_history`` existed may still have
    every historical checkpoint in Postgres. This function converts those
    checkpoint channel values into the same compact trusted history format while
    excluding the currently accepted revision.
    """

    if (
        isinstance(current_revision, bool)
        or not isinstance(current_revision, int)
        or current_revision <= 1
    ):
        return []

    history: list[dict[str, Any]] = []

    for state in checkpoint_states or []:
        if not isinstance(state, dict):
            continue

        snapshot = state.get("accepted_plan_snapshot")

        if not isinstance(snapshot, dict):
            revision = state.get("plan_revision")
            itinerary = state.get("draft_itinerary")

            snapshot = {
                "plan_revision": revision,
                "draft_itinerary": itinerary,
            }

        entry = _normalise_entry(snapshot)

        if (
            entry is None
            or entry["revision"] >= current_revision
        ):
            continue

        history = append_previous_itinerary(
            history,
            entry,
        )

    return history


def format_itinerary_history(
    history: Iterable[dict[str, Any]] | None,
) -> str:
    """Render trusted previous revisions compactly for the agent prompt."""

    entries = append_previous_itinerary(
        history,
        None,
    )

    if not entries:
        return _NO_HISTORY_MESSAGE

    # Keep prompt size bounded while always exposing the original revision.
    selected = (
        entries
        if len(entries) <= 6
        else [entries[0], *entries[-5:]]
    )

    blocks: list[str] = []

    for entry in selected:
        lines = [
            f"Revision {entry['revision']} "
            "(previous accepted itinerary):"
        ]

        for day in entry["draft_itinerary"]:
            if not isinstance(day, dict):
                continue

            day_num = day.get("day", "?")
            date = day.get("date", "TBD")

            lines.append(
                f"- Day {day_num} ({date})"
            )

            flights = day.get("flight")

            if isinstance(flights, dict):
                flights = [flights]

            if isinstance(flights, list):
                for flight in flights:
                    if not isinstance(flight, dict):
                        continue

                    airline = flight.get(
                        "airline",
                        "Unknown airline",
                    )

                    number = (
                        flight.get("flight_number")
                        or flight.get("flight_num")
                        or ""
                    )

                    lines.append(
                        f"    Flight: {airline} {number}".rstrip()
                    )

            hotel = day.get("hotel")

            if (
                isinstance(hotel, dict)
                and hotel.get("hotel_name")
            ):
                lines.append(
                    f"    Hotel: {hotel['hotel_name']}"
                )

            activities = day.get("activities") or []

            if isinstance(activities, list):
                for index, activity in enumerate(
                    activities,
                    start=1,
                ):
                    if not isinstance(activity, dict):
                        continue

                    name = activity.get(
                        "name",
                        "Place",
                    )

                    activity_type = activity.get(
                        "type",
                        "activity",
                    )

                    lines.append(
                        f"    [{index}] "
                        f"{activity_type}: {name}"
                    )

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)