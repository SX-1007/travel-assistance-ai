"""
Currency conversion + budget management tools.

Optimisation highlights
-----------------------
* HTTPAdapter-backed `requests.Session` with urllib3 `Retry` — reuses TCP/TLS,
  handles 429/5xx transparently at the transport layer.
* `TTLCache` + `Lock` for both country→currency and FX-rate caches.
* Reciprocal-rate inference: fetching `A→B` also caches `B→A = 1/rate`,
  eliminating ~50% of downstream API calls in round-trip workflows.
* Parallel country-code resolution (`ThreadPoolExecutor`).
* All public APIs unchanged; safe drop-in replacement.
"""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Annotated, Any, Literal, Optional

import pycountry
import requests
from babel.numbers import get_territory_currencies
from cachetools import TTLCache
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.agents.state import AgentState
from app.core.config import settings
from app.tools.provider_logging import safe_provider_log


logger = logging.getLogger(__name__)


class CurrencyRateUnavailableError(RuntimeError):
    """Raised when no configured exchange-rate provider can return a rate."""


# ─────────────────────────────────────────────────────────────────────────────
# HTTP session — connection pool + transport-level retry
# ─────────────────────────────────────────────────────────────────────────────
_RETRY_STRATEGY = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset({"GET"}),
    raise_on_status=False,
)

_session: requests.Session = requests.Session()
_adapter = HTTPAdapter(
    pool_connections=10, pool_maxsize=20, max_retries=_RETRY_STRATEGY
)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_ALLOCATIONS: dict[str, float] = {
    "transportation": 0.0,
    "accommodation": 0.0,
    "food": 0.0,
    "activity": 0.0,
    "shopping": 0.0,
    "emergency_fund": 0.0,
}

# Vague-term modifier → percentage delta
_MODIFIER_FACTOR: dict[str, float] = {
    "small_increase": 0.10,
    "large_increase": 0.30,
    "small_decrease": -0.10,
    "large_decrease": -0.30,
}


class BudgetAdjustment(BaseModel):
    """Single category mutation requested by the user / LLM."""

    category: Literal[
        "transportation",
        "accommodation",
        "food",
        "activity",
        "shopping",
        "emergency_fund",
    ] = Field(
        description=(
            "Structural category to change. Map user words: "
            "'hotel/resort/stay' -> 'accommodation'; "
            "'meal/restaurant/drinks' -> 'food'; "
            "'train/flight/car' -> 'transportation'; "
            "'tickets/tours' -> 'activity'."
        )
    )
    add_delete_amount: Optional[float] = Field(
        default=None,
        description=(
            "Positive (add) or negative (subtract) value in destination currency. "
            "Leave null when using `modifier`."
        ),
    )
    modifier: Optional[
        Literal[
            "small_increase",
            "small_decrease",
            "large_increase",
            "large_decrease",
        ]
    ] = Field(
        default=None,
        description=(
            "Use ONLY for vague user language ('a little', 'a lot'). "
            "'small_increase' = +10% | 'large_increase' = +30% | "
            "'small_decrease' = -10% | 'large_decrease' = -30%."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Country → currency (in-process TTL cache, thread-safe)
# ─────────────────────────────────────────────────────────────────────────────
_country_cache: TTLCache = TTLCache(maxsize=256, ttl=86_400)  # 24 h
_country_lock = Lock()


def get_currency_code(country: str) -> str:
    """Resolve a country name to its primary ISO-4217 currency code."""
    if not country or not country.strip():
        return "USD"

    key = country.lower().strip()

    with _country_lock:
        cached = _country_cache.get(key)
    if cached is not None:
        return cached

    try:
        country_data = pycountry.countries.search_fuzzy(country)[0]
        currencies = get_territory_currencies(country_data.alpha_2)
        code = currencies[0].upper() if currencies else "USD"
    except (LookupError, IndexError, AttributeError):
        safe_provider_log(logger, "provider.currency.country_resolution_failed")
        code = "USD"

    with _country_lock:
        _country_cache[key] = code
    return code


# ─────────────────────────────────────────────────────────────────────────────
# FX rates — TTL cache + reciprocal inference
# ─────────────────────────────────────────────────────────────────────────────
_rate_cache: TTLCache = TTLCache(maxsize=512, ttl=3_600)  # 1 h
_rate_lock = Lock()


def _lookup_rate(origin: str, dest: str) -> Optional[float]:
    with _rate_lock:
        return _rate_cache.get((origin, dest))


def _cache_rate(origin: str, dest: str, rate: float) -> None:
    with _rate_lock:
        _rate_cache[(origin, dest)] = rate
        if rate > 0:
            _rate_cache[(dest, origin)] = 1.0 / rate  # reciprocal inference


def _fetch_rate_open_erapi(origin: str, dest: str) -> Optional[float]:
    """Fallback FX provider (keyless, broad currency coverage incl. TWD).

    Used only when freecurrencyapi fails or does not return the requested pair
    (its free tier omits some currencies). Prevents the silent ``rate = 1.0``
    fallback that corrupts budget maths (reported error #6).
    """
    try:
        resp = _session.get(f"https://open.er-api.com/v6/latest/{origin}", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data.get("result") == "success":
            rate = data.get("rates", {}).get(dest)
            if rate is not None:
                return float(rate)
    except (requests.RequestException, KeyError, ValueError):
        safe_provider_log(logger, "provider.currency.fallback_failed")
    return None


def get_currency_rate(
    origin_curr_code: str,
    des_curr_code: str,
    max_retries: int = 3,
) -> Optional[float]:
    """Fetch conversion rate, using cache or freecurrencyapi with bounded retry.

    Falls back to a keyless provider if freecurrencyapi fails or lacks the pair.
    """
    origin = (origin_curr_code or "").upper().strip() or "USD"
    dest = (des_curr_code or "").upper().strip() or "USD"

    if origin == dest:
        return 1.0

    cached = _lookup_rate(origin, dest)
    if cached is not None:
        return cached

    url = "https://api.freecurrencyapi.com/v1/latest"
    params = {
        "apikey": settings.FREECURRENCY_API,
        "base_currency": origin,
        "currencies": dest,
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = _session.get(url, params=params, timeout=5)
            response.raise_for_status()
            rate = float(response.json()["data"][dest])
            _cache_rate(origin, dest, rate)
            return rate
        except (requests.RequestException, KeyError, ValueError) as e:
            # 4xx (other than 429) is permanent — e.g. freecurrencyapi's free
            # tier returns 422 for unsupported currencies such as TWD. Retrying
            # only burns time; go straight to the fallback provider.
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                safe_provider_log(
                    logger,
                    "provider.currency.primary_client_error",
                    status_code=status,
                )
                break
            safe_provider_log(logger, "provider.currency.primary_retry")
            if attempt < max_retries:
                time.sleep(min(2 ** (attempt - 1), 5))

    # ── Secondary provider before giving up ──
    fallback = _fetch_rate_open_erapi(origin, dest)
    if fallback is not None:
        safe_provider_log(logger, "provider.currency.fallback_used")
        _cache_rate(origin, dest, fallback)
        return fallback

    safe_provider_log(logger, "provider.currency.all_failed")
    return None


def budget_calculation(amount: float, rate: Optional[float]) -> Optional[float]:
    """Convert `amount` by `rate`, rounded to 2 dp."""
    if rate is None:
        return None
    return round(amount * rate, 2)


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph pipeline node
# ─────────────────────────────────────────────────────────────────────────────
def currency_pipeline(state: AgentState) -> dict[str, Any]:
    """Pre-process & normalise all structural travel financial numbers."""
    # Parallel CPU-bound pycountry lookups
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_origin = ex.submit(get_currency_code, state.origin_country or "")
        f_dest = ex.submit(get_currency_code, state.country or "")
        origin_curr_code = f_origin.result()
        dest_curr_code = f_dest.result()

    currency_rate = get_currency_rate(origin_curr_code, dest_curr_code)
    if currency_rate is None:
        raise CurrencyRateUnavailableError(
            f"No exchange rate is currently available for "
            f"{origin_curr_code}->{dest_curr_code}."
        )

    total_convert_budget = (
        budget_calculation(state.total_base_budget or 0.0, currency_rate) or 0.0
    )

    return {
        "base_currency_code": origin_curr_code,
        "dest_currency_code": dest_curr_code,
        "exchange_rate": {
            origin_curr_code: 1.0,
            dest_curr_code: currency_rate,
        },
        "total_convert_budget": total_convert_budget,
        "currency_fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Agent tools
# ─────────────────────────────────────────────────────────────────────────────
@tool
def chat_currency_conversion(
    from_country: str,
    to_country: str,
    amount: float,
) -> dict[str, Any]:
    """Convert arbitrary amounts or check exchange rates mid-conversation."""
    from_currency = get_currency_code(from_country)
    to_currency = get_currency_code(to_country)
    rate = get_currency_rate(from_currency, to_currency)

    if rate is None:
        return {
            "error": f"Failed to fetch exchange rate from {from_currency} to {to_currency}",
            "from": from_currency,
            "to": to_currency,
        }

    return {
        "from": from_currency,
        "to": to_currency,
        "rate": rate,
        "original_amount": amount,
        "converted_amount": budget_calculation(amount, rate),
    }


@tool
def update_total_budget(
    new_total_budget: float,
    state: Annotated[AgentState, InjectedState],
) -> dict[str, Any]:
    """Update absolute total trip budget; rescale all allocations proportionally."""
    if not math.isfinite(new_total_budget) or new_total_budget <= 0:
        return {"error": "new_total_budget must be greater than 0."}

    old_total = state.total_convert_budget or 0.0
    current_allocation = dict(state.budget_allocation or {})

    if old_total == 0 or not current_allocation:
        return {
            "total_convert_budget": new_total_budget,
            "budget_allocation": current_allocation,
        }

    scale_factor = new_total_budget / old_total
    new_allocation = {
        cat: round(amt * scale_factor, 2) for cat, amt in current_allocation.items()
    }
    return {
        "total_convert_budget": new_total_budget,
        "budget_allocation": new_allocation,
    }


@tool
def update_budget_category(
    adjustment: list[BudgetAdjustment],
    state: Annotated[AgentState, InjectedState],
) -> dict[str, Any]:
    """Adjust one or more budget categories simultaneously with auto-rebalancing."""
    current_allocation: dict[str, float] = dict(
        state.budget_allocation or DEFAULT_ALLOCATIONS
    )
    # Legacy checkpoints carried a "flight" alias duplicating "transportation".
    # Treating it as an independent category double-counts transport in the
    # total and corrupts the proportional rebalance — drop it here.
    current_allocation.pop("flight", None)
    # Ensure all canonical keys exist
    for cat in DEFAULT_ALLOCATIONS:
        current_allocation.setdefault(cat, 0.0)

    total_allowed = state.total_convert_budget or 0.0
    explicit_categories: set[str] = set()
    net_added = 0.0

    for adj in adjustment:
        explicit_categories.add(adj.category)
        old_value = current_allocation.get(adj.category, 0.0)

        delta = adj.add_delete_amount
        if delta is None:
            if adj.modifier is not None:
                reference_value = old_value if old_value > 0 else (0.05 * total_allowed)
                delta = _MODIFIER_FACTOR[adj.modifier] * reference_value
            else:
                delta = 0.0

        new_value = max(0.0, round(old_value + delta, 2))
        current_allocation[adj.category] = new_value
        # Rebalance the amount that was actually applied. A large negative
        # request may clamp at zero; using the raw requested delta here causes
        # a false over-budget trim and warning.
        net_added += new_value - old_value

    # Auto-rebalance donor categories proportionally
    donor_categories = [c for c in current_allocation if c not in explicit_categories]
    sum_of_donors = sum(current_allocation[c] for c in donor_categories)

    if abs(net_added) > 1e-6 and sum_of_donors > 0:
        for cat in donor_categories:
            weight = current_allocation[cat] / sum_of_donors
            offset = net_added * weight
            current_allocation[cat] = max(
                0.0, round(current_allocation[cat] - offset, 2)
            )

    # Enforce total budget cap via proportional trimming
    total_allocation = round(sum(current_allocation.values()), 2)
    warnings: list[str] = []
    if total_allowed > 0 and total_allocation > total_allowed:
        warnings.append(
            f"Adjusted total {total_allocation} exceeded budget {total_allowed}; "
            "categories were proportionally trimmed to fit."
        )
        scale = total_allowed / total_allocation
        current_allocation = {
            cat: round(amt * scale, 2) for cat, amt in current_allocation.items()
        }

    result: dict[str, Any] = {"budget_allocation": current_allocation}
    if warnings:
        result["warnings"] = warnings
    return result
