"""Build and deterministically validate isolated itinerary candidates."""

from __future__ import annotations

import asyncio
import copy
import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Literal

from app.agents.state import AgentState
from app.services.itinerary_quality import (
    TripRequirements,
    ValidationIssue,
    ValidationReport,
    validate_itinerary_candidate,
)
from app.tools.attractions import plan_activities
from app.tools.flights_hotels import plan_flight_hotel
from app.tools.mapbox import generate_daily_map


logger = logging.getLogger(__name__)

_MAX_IDENTIFIER_LENGTH = 64
_MAX_CODE_LENGTH = 64
_MAX_ISSUE_CODES = 8
_SAFE_LOG_TOKEN = re.compile(r"[^A-Za-z0-9._:-]+")
_CURRENT_LOG_CONTEXT: ContextVar[Mapping[str, Any]] = ContextVar(
    "planning_log_context",
    default={},
)


def safe_observability_log(
    target_logger: logging.Logger,
    event: str,
    *,
    request_id: Any = None,
    session_id: Any = None,
    attempt: Any = 0,
    stage: Any = "unknown",
    destination_country_code: Any = None,
    issue_codes: Sequence[Any] = (),
    elapsed_ms: Any = 0,
    outcome: Any = "unknown",
) -> None:
    """Emit one bounded planning record; observability must never affect flow."""
    record = {
        "request_id": _bounded_log_token(
            request_id,
            fallback="unknown",
            limit=_MAX_IDENTIFIER_LENGTH,
        ),
        "session_id": _bounded_log_token(
            session_id,
            fallback="unknown",
            limit=_MAX_IDENTIFIER_LENGTH,
        ),
        "attempt": _bounded_nonnegative_int(attempt, maximum=99),
        "stage": _bounded_log_token(stage, fallback="unknown", limit=48),
        "destination_country_code": _trusted_country_code(
            destination_country_code
        ),
        "issue_codes": _bounded_issue_codes(issue_codes),
        "elapsed_ms": _bounded_nonnegative_int(
            elapsed_ms,
            maximum=86_400_000,
        ),
        "outcome": _bounded_log_token(outcome, fallback="unknown", limit=32),
    }
    try:
        target_logger.info(
            _bounded_log_token(event, fallback="planning.unknown", limit=64),
            extra=record,
        )
    except Exception:
        # A broken formatter/handler cannot fail or alter the planning result.
        return


@contextmanager
def planning_observability_context(
    context: Mapping[str, Any] | None,
) -> Iterator[None]:
    """Scope correlation metadata without changing provider-call interfaces."""
    bounded_context = (
        {
            "request_id": context.get("request_id"),
            "session_id": context.get("session_id"),
        }
        if isinstance(context, Mapping)
        else {}
    )
    token = _CURRENT_LOG_CONTEXT.set(bounded_context)
    try:
        yield
    finally:
        _CURRENT_LOG_CONTEXT.reset(token)


def _bounded_log_token(value: Any, *, fallback: str, limit: int) -> str:
    if not isinstance(value, str):
        return fallback
    normalized = _SAFE_LOG_TOKEN.sub("_", value.strip()).strip("_")
    return normalized[:limit] or fallback


def _bounded_nonnegative_int(value: Any, *, maximum: int) -> int:
    if isinstance(value, bool):
        return 0
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, min(number, maximum))


def _trusted_country_code(value: Any) -> str:
    if not isinstance(value, str):
        return "unknown"
    code = value.strip().upper()
    return code if re.fullmatch(r"[A-Z]{2}", code) else "unknown"


def _bounded_issue_codes(issue_codes: Sequence[Any]) -> tuple[str, ...]:
    if isinstance(issue_codes, (str, bytes)) or not isinstance(
        issue_codes,
        Sequence,
    ):
        return ()
    bounded: list[str] = []
    for raw_code in issue_codes[:_MAX_ISSUE_CODES]:
        code = _bounded_log_token(raw_code, fallback="unknown", limit=_MAX_CODE_LENGTH)
        if code not in bounded:
            bounded.append(code)
    return tuple(bounded)


@dataclass(frozen=True)
class PlanningDependencies:
    flight_planner: Callable[[AgentState], dict[str, Any]]
    activity_planner: Callable[[AgentState, dict | None], dict[str, Any]]
    map_planner: Callable[[AgentState], dict[str, Any]]


DEFAULT_DEPENDENCIES = PlanningDependencies(
    flight_planner=plan_flight_hotel,
    activity_planner=plan_activities,
    map_planner=generate_daily_map,
)


@dataclass(frozen=True)
class PlanCandidate:
    itinerary: list[dict[str, Any]]
    maps: dict[int, dict[str, Any]]
    attempt: int
    validation: ValidationReport


@dataclass(frozen=True)
class PlanningTransactionResult:
    status: Literal["validated", "unavailable"]
    candidate: PlanCandidate | None
    attempts: int
    issues: tuple[ValidationIssue, ...]

    @classmethod
    def validated(
        cls,
        itinerary: list[dict[str, Any]],
        maps: dict[int, dict[str, Any]],
        attempt: int,
        report: ValidationReport,
        previous_reports: Sequence[ValidationReport] = (),
    ) -> PlanningTransactionResult:
        return cls(
            status="validated",
            candidate=PlanCandidate(
                itinerary=copy.deepcopy(itinerary),
                maps=copy.deepcopy(maps),
                attempt=attempt,
                validation=report,
            ),
            attempts=attempt,
            issues=_issues_from_reports((*previous_reports, report)),
        )

    @classmethod
    def unavailable(
        cls,
        attempts: int,
        reports: Sequence[ValidationReport],
    ) -> PlanningTransactionResult:
        return cls(
            status="unavailable",
            candidate=None,
            attempts=attempts,
            issues=_issues_from_reports(reports),
        )


async def build_validated_plan(
    state: AgentState,
    user_profile: dict | None,
    dependencies: PlanningDependencies = DEFAULT_DEPENDENCIES,
    max_attempts: int = 3,
    *,
    attempt_offset: int = 0,
    log_context: Mapping[str, Any] | None = None,
) -> PlanningTransactionResult:
    """Return the first qualified candidate from clean, isolated attempts."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if attempt_offset < 0:
        raise ValueError("attempt_offset must be non-negative")

    requirements = TripRequirements.from_state(state)
    context = (
        log_context
        if isinstance(log_context, Mapping)
        else _CURRENT_LOG_CONTEXT.get()
    )
    correlation = {
        "request_id": context.get("request_id"),
        "session_id": context.get("session_id"),
        "destination_country_code": requirements.destination_country_code,
    }
    transaction_started = time.perf_counter()
    reports: list[ValidationReport] = []
    for local_attempt in range(1, max_attempts + 1):
        attempt = attempt_offset + local_attempt
        safe_observability_log(
            logger,
            "planning.transaction.attempt",
            attempt=attempt,
            stage="attempt",
            outcome="started",
            **correlation,
        )
        attempt_state = state.model_copy(
            update={
                "draft_itinerary": [],
                "daily_map_info": {},
                "planning_outcome": None,
                "candidate_plan": None,
                "planning_issue_codes": [],
                "output_review_attempts": 0,
                "output_review_issue_codes": [],
            },
            deep=True,
        )
        stage = "flight_planner"
        stage_started = time.perf_counter()
        try:
            skeleton_update = await asyncio.to_thread(
                dependencies.flight_planner,
                attempt_state,
            )
            attempt_state = _apply_candidate_update(attempt_state, skeleton_update)
            safe_observability_log(
                logger,
                "planning.transaction.stage",
                attempt=attempt,
                stage=stage,
                elapsed_ms=(time.perf_counter() - stage_started) * 1000,
                outcome="completed",
                **correlation,
            )

            stage = "activity_planner"
            stage_started = time.perf_counter()
            activity_update = await asyncio.to_thread(
                dependencies.activity_planner,
                attempt_state,
                copy.deepcopy(user_profile),
            )
            attempt_state = _apply_candidate_update(attempt_state, activity_update)
            safe_observability_log(
                logger,
                "planning.transaction.stage",
                attempt=attempt,
                stage=stage,
                elapsed_ms=(time.perf_counter() - stage_started) * 1000,
                outcome="completed",
                **correlation,
            )

            stage = "map_planner"
            stage_started = time.perf_counter()
            map_update = await asyncio.to_thread(
                dependencies.map_planner,
                attempt_state,
            )
            attempt_state = _apply_candidate_update(attempt_state, map_update)
            safe_observability_log(
                logger,
                "planning.transaction.stage",
                attempt=attempt,
                stage=stage,
                elapsed_ms=(time.perf_counter() - stage_started) * 1000,
                outcome="completed",
                **correlation,
            )
        except (asyncio.CancelledError, asyncio.TimeoutError) as exc:
            safe_observability_log(
                logger,
                "planning.transaction.stage",
                attempt=attempt,
                stage=stage,
                issue_codes=(
                    "deadline.exhausted"
                    if isinstance(exc, asyncio.TimeoutError)
                    else "transaction.cancelled",
                ),
                elapsed_ms=(time.perf_counter() - stage_started) * 1000,
                outcome="interrupted",
                **correlation,
            )
            raise
        except Exception as exc:
            report = _provider_exception_report(attempt, stage, exc)
            reports.append(report)
            safe_observability_log(
                logger,
                "planning.transaction.stage",
                attempt=attempt,
                stage=stage,
                issue_codes=tuple(issue.code for issue in report.issues),
                elapsed_ms=(time.perf_counter() - stage_started) * 1000,
                outcome="failed",
                **correlation,
            )
            continue

        validation_started = time.perf_counter()
        report = validate_itinerary_candidate(
            requirements,
            attempt_state.draft_itinerary,
            attempt_state.daily_map_info,
        )
        reports.append(report)

        if not report.qualified:
            logger.warning(
                "Planning candidate rejected: attempt=%s issues=%s",
                attempt,
                [
                    {
                        "code": issue.code,
                        "path": issue.path,
                        "message": issue.message,
                    }
                    for issue in report.issues
                ],
            )

        safe_observability_log(
            logger,
            "planning.transaction.validation",
            attempt=attempt,
            stage="validation",
            issue_codes=tuple(issue.code for issue in report.issues),
            elapsed_ms=(time.perf_counter() - validation_started) * 1000,
            outcome="approved" if report.qualified else "rejected",
            **correlation,
        )
        if report.qualified:
            result = PlanningTransactionResult.validated(
                attempt_state.draft_itinerary,
                attempt_state.daily_map_info,
                attempt,
                report,
                reports[:-1],
            )
            safe_observability_log(
                logger,
                "planning.transaction.complete",
                attempt=attempt,
                stage="transaction",
                issue_codes=tuple(issue.code for issue in result.issues),
                elapsed_ms=(time.perf_counter() - transaction_started) * 1000,
                outcome="validated",
                **correlation,
            )
            return result

    total_attempts = attempt_offset + max_attempts
    result = PlanningTransactionResult.unavailable(total_attempts, reports)
    safe_observability_log(
        logger,
        "planning.transaction.complete",
        attempt=total_attempts,
        stage="transaction",
        issue_codes=tuple(issue.code for issue in result.issues),
        elapsed_ms=(time.perf_counter() - transaction_started) * 1000,
        outcome="unavailable",
        **correlation,
    )
    return result


def _apply_candidate_update(
    state: AgentState,
    update: Mapping[str, Any],
) -> AgentState:
    if not isinstance(update, Mapping):
        raise TypeError("planner update must be a mapping")
    candidate_update = {
        field: copy.deepcopy(update[field])
        for field in ("draft_itinerary", "daily_map_info")
        if field in update
    }
    return state.model_copy(update=candidate_update, deep=True)


def _provider_exception_report(
    attempt: int,
    stage: str,
    exc: Exception,
) -> ValidationReport:
    return ValidationReport(
        issues=(
            ValidationIssue(
                code=f"planning.{stage}.exception",
                path=f"attempt[{attempt}].{stage}",
                message=f"{stage} failed with {type(exc).__name__}.",
            ),
        )
    )


def _issues_from_reports(
    reports: Sequence[ValidationReport],
) -> tuple[ValidationIssue, ...]:
    return tuple(issue for report in reports for issue in report.issues)
