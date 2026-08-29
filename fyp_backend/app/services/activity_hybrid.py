"""Pure helpers for hybrid activity rescue and cross-day deduplication."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def _normalise_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    folded = value.strip().casefold()
    unicode_safe = "".join(
        character if character.isalnum() else " " for character in folded
    )
    return " ".join(unicode_safe.split())


def activity_identity_key(activity: Any) -> str | None:
    """Return a stable provider-grounded identity for duplicate detection.

    The provider-normalised ``location.place_name`` is preferred over the
    display name. The verified/requested locality is included so identically
    named venues in different trip cities are not incorrectly collapsed.
    """
    if not isinstance(activity, Mapping):
        return None

    location = activity.get("location")
    location = location if isinstance(location, Mapping) else {}
    name = _normalise_text(location.get("place_name") or activity.get("name"))
    locality = _normalise_text(
        location.get("verified_locality") or location.get("requested_city")
    )
    if not name:
        return None
    latitude = location.get("latitude")
    longitude = location.get("longitude")
    if (
        isinstance(latitude, (int, float))
        and not isinstance(latitude, bool)
        and isinstance(longitude, (int, float))
        and not isinstance(longitude, bool)
        and math.isfinite(float(latitude))
        and math.isfinite(float(longitude))
    ):
        return f"{name}|{float(latitude):.5f}|{float(longitude):.5f}"
    return f"{name}|{locality}"


def deduplicate_activity_buckets(
    by_day: Mapping[int, Sequence[Mapping[str, Any]]],
) -> tuple[dict[int, list[dict[str, Any]]], int]:
    """Remove repeated activities across the full itinerary, keeping first use.

    Days are processed in numerical order and activities in their existing
    order. Remaining activities are re-numbered from one so later deterministic
    validation and map generation receive a canonical sequence.
    """
    seen: set[str] = set()
    cleaned: dict[int, list[dict[str, Any]]] = {}
    removed = 0

    for raw_day in sorted(by_day):
        try:
            day = int(raw_day)
        except (TypeError, ValueError):
            continue

        kept: list[dict[str, Any]] = []
        raw_activities = by_day.get(raw_day) or []
        for raw_activity in raw_activities:
            if not isinstance(raw_activity, Mapping):
                continue
            activity = copy.deepcopy(dict(raw_activity))
            key = activity_identity_key(activity)
            if key is not None and key in seen:
                removed += 1
                continue
            if key is not None:
                seen.add(key)
            kept.append(activity)

        for order, activity in enumerate(kept, start=1):
            activity["order"] = order
        cleaned[day] = kept

    return cleaned, removed


def missing_activity_days(
    expected_days: Sequence[int],
    by_day: Mapping[int, Sequence[Mapping[str, Any]]],
) -> tuple[int, ...]:
    """Return expected day numbers that currently have no usable activities."""
    missing: list[int] = []
    for raw_day in expected_days:
        try:
            day = int(raw_day)
        except (TypeError, ValueError):
            continue
        activities = by_day.get(day)
        if not isinstance(activities, Sequence) or isinstance(
            activities, (str, bytes)
        ) or not activities:
            missing.append(day)
    return tuple(missing)


@dataclass(frozen=True)
class ActivityRescueResult:
    by_day: dict[int, list[dict[str, Any]]]
    missing_days: tuple[int, ...]
    rounds_used: int
    removed_duplicates: int


def rescue_activity_buckets(
    expected_days: Sequence[int],
    initial_by_day: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    rescue_round: Callable[
        [tuple[int, ...], int], Mapping[int, Sequence[Mapping[str, Any]]]
    ],
    max_rounds: int = 3,
) -> ActivityRescueResult:
    """Repair only empty days while preserving already-valid activity days.

    Every rescue round receives only the currently missing day numbers. Any
    cross-day duplicate introduced by a rescue round is removed immediately; if
    that leaves the day empty, the next round gets another chance to repair it.
    """
    if max_rounds < 0:
        raise ValueError("max_rounds must be non-negative")

    current, removed_duplicates = deduplicate_activity_buckets(initial_by_day)
    for raw_day in expected_days:
        day = int(raw_day)
        current.setdefault(day, [])

    missing = missing_activity_days(expected_days, current)
    rounds_used = 0

    for round_number in range(1, max_rounds + 1):
        if not missing:
            break
        rounds_used = round_number
        additions = rescue_round(missing, round_number)
        if not isinstance(additions, Mapping):
            additions = {}

        merged = {
            day: [copy.deepcopy(activity) for activity in activities]
            for day, activities in current.items()
        }
        for day in missing:
            raw_additions = additions.get(day) or []
            if isinstance(raw_additions, Sequence) and not isinstance(
                raw_additions, (str, bytes)
            ):
                merged[day].extend(
                    copy.deepcopy(dict(activity))
                    for activity in raw_additions
                    if isinstance(activity, Mapping)
                )

        current, removed = deduplicate_activity_buckets(merged)
        removed_duplicates += removed
        for raw_day in expected_days:
            current.setdefault(int(raw_day), [])
        missing = missing_activity_days(expected_days, current)

    return ActivityRescueResult(
        by_day=current,
        missing_days=missing,
        rounds_used=rounds_used,
        removed_duplicates=removed_duplicates,
    )


def find_duplicate_activity_paths(
    itinerary: Sequence[Any],
) -> tuple[tuple[str, str], ...]:
    """Return ``(duplicate_path, first_path)`` pairs for repeated activities."""
    seen: dict[str, str] = {}
    duplicates: list[tuple[str, str]] = []
    for day_index, raw_day in enumerate(itinerary):
        if not isinstance(raw_day, Mapping):
            continue
        activities = raw_day.get("activities")
        if not isinstance(activities, Sequence) or isinstance(
            activities, (str, bytes)
        ):
            continue
        for activity_index, activity in enumerate(activities):
            key = activity_identity_key(activity)
            if key is None:
                continue
            path = f"itinerary[{day_index}].activities[{activity_index}]"
            first_path = seen.get(key)
            if first_path is None:
                seen[key] = path
            else:
                duplicates.append((path, first_path))
    return tuple(duplicates)