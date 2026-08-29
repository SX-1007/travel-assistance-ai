from __future__ import annotations

import math
from typing import Any, Optional

from langchain_core.tools import tool


@tool
def propose_budget_change(
    proposed_total_base_budget: Optional[float] = None,
    request_recommendation: bool = False,
) -> dict[str, Any]:
    """Propose a home-currency total or explicitly request a recommendation."""
    if request_recommendation:
        if proposed_total_base_budget is not None:
            return {"error": "Recommendation mode cannot include an amount."}
        return {
            "status": "budget_proposal",
            "proposal": {"mode": "recommendation", "total_base_budget": None},
        }
    if proposed_total_base_budget is None:
        return {"error": "Provide a positive budget or request a recommendation."}
    amount = float(proposed_total_base_budget)
    if not math.isfinite(amount) or amount <= 0:
        return {"error": "proposed_total_base_budget must be finite and greater than 0."}
    return {
        "status": "budget_proposal",
        "proposal": {"mode": "amount", "total_base_budget": amount},
    }


@tool
def confirm_recommended_budget() -> dict[str, Any]:
    """Explicitly accept the exact server-owned pending recommendation."""
    return {
        "status": "budget_proposal",
        "proposal": {"mode": "confirm", "total_base_budget": None},
    }
