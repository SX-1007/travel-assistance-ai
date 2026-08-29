# ═══════════════════════════════════════════════════════════
# state.py — LangGraph State Schema (LLM Working Memory)
# ═══════════════════════════════════════════════════════════
from __future__ import annotations

from typing import Annotated, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(BaseModel):
    """LangGraph state schema for the AI Travel Agent.

    All fields are optional except ``messages`` (defaults to empty list).
    Fields are populated progressively as the graph executes:

    1. Form submission → origin_country, country, city, dates, budget
    2. Currency pipeline → currency codes, exchange rate, converted budget
    3. Itinerary planning → draft_itinerary
    4. Map generation → daily_map_info
    5. Chat → messages (appended via ``add_messages`` reducer)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── Conversation History ────────────────────────────
    messages: Annotated[List[BaseMessage], add_messages] = Field(
        default_factory=list,
        description="Conversation history. The add_messages reducer "
        "automatically appends new messages instead of overwriting.",
    )

    # ── User Origin Info ────────────────────────────────
    origin_country: Optional[str] = Field(
        default=None,
        description="User's home country (e.g. 'Malaysia').",
    )
    origin_state: Optional[str] = Field(
        default=None,
        description="User's home state/province (optional).",
    )

    # ── Travel Destination Info ─────────────────────────
    country: Optional[str] = Field(
        default=None,
        description="Destination country (e.g. 'Japan').",
    )
    city: List[str] = Field(
        default_factory=list,
        description=(
            "Verified server-owned destination city/cities used by "
            "budget, provider, itinerary, and map planning."
        ),
    )
    num_people: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of travelers (must be ≥ 1).",
    )
    total_base_budget: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total trip budget in the user's home currency.",
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Trip start date in YYYY-MM-DD format.",
    )
    end_date: Optional[str] = Field(
        default=None,
        description="Trip end date in YYYY-MM-DD format.",
    )

    # ── Currency Conversion ─────────────────────────────
    base_currency_code: Optional[str] = Field(
        default=None,
        description="ISO 4217 code for the user's home currency (e.g. 'USD').",
    )
    dest_currency_code: Optional[str] = Field(
        default=None,
        description="ISO 4217 code for the destination currency (e.g. 'JPY').",
    )
    total_convert_budget: Optional[float] = Field(
        default=0.0,
        ge=0,
        description="Total budget expressed in the destination currency.",
    )
    budget_allocation: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-category budget breakdown in destination currency. "
        "Canonical keys: transportation, accommodation, food, "
        "activity, shopping, emergency_fund.",
    )
    exchange_rate: Dict[str, float] = Field(
        default_factory=dict,
        description="Exchange rate mapping, e.g. {'USD': 1.0, 'JPY': 150.0}.",
    )
    currency_fetched_at: Optional[str] = Field(
        default=None,
        description="ISO 8601 UTC timestamp of the last exchange-rate fetch. "
        "Used for TTL-based refresh logic.",
    )

    # ── Provider-Grounded Budget Preflight ─────────────
    budget_assessment: Optional[dict] = Field(
        default=None,
        description=(
            "Provider-grounded prices and exchange rate verified before a new "
            "trip enters the itinerary workflow."
        ),
    )
    pending_budget_proposal: Optional[dict] = Field(
        default=None,
        description="Current-turn budget intent awaiting deterministic assessment.",
    )
    pending_budget_confirmation: Optional[dict] = Field(
        default=None,
        description="Server-owned recommendation awaiting explicit confirmation.",
    )

    pending_itinerary_mutation: Optional[dict] = Field(
        default=None,
        description=(
            "Server-owned itinerary edit intent awaiting a missing target "
            "detail such as the day number. Provider payloads are never "
            "trusted from this field; place replacements must be re-grounded "
            "against a fresh current-turn provider result before commit."
        ),
    )

    accepted_budget_decision: Optional[dict] = Field(
        default=None,
        description=(
            "Server-created accepted decision awaiting authoritative cache "
            "validation before any planning node may run."
        ),
    )
    plan_revision: int = Field(
        default=0,
        ge=0,
        description="Monotonic server-owned revision of the committed plan.",
    )
    accepted_plan_snapshot: Optional[dict] = Field(
        default=None,
        description=(
            "Last fully committed financial, itinerary, and map snapshot. "
            "Candidate budget replans never write into this object."
        ),
    )
    itinerary_history: List[dict] = Field(
        default_factory=list,
        description=(
            "Server-owned copies of previously accepted itinerary revisions. "
            "Used only for historical questions and never as the live plan."
        ),
    )
    budget_gate_outcome: Optional[
        Literal[
            "accepted",
            "budget_confirmation_required",
            "budget_check_unavailable",
        ]
    ] = None
    budget_gate_reason: Optional[str] = None
    budget_gate_message: Optional[str] = None
    chat_budget_action: Optional[Literal["accept_recommended"]] = None
    chat_budget_assessment_id: Optional[str] = None

    # ── Draft Itinerary ─────────────────────────────────
    draft_itinerary: List[dict] = Field(
        default_factory=list,
        description="Day-by-day trip plan. Each dict has keys: day, date, "
        "flight (list), hotel (dict), activities (list), route (dict), "
        "day_total_cost (float).",
    )

    # ── Per-Day GeoJSON Maps ────────────────────────────
    daily_map_info: Dict[int, dict] = Field(
        default_factory=dict,
        description="Mapping of day number → GeoJSON FeatureCollection "
        "for frontend map rendering.",
    )

    # ── Private Planning Transaction ───────────────────
    planning_outcome: Optional[Literal["validated", "unavailable"]] = None
    candidate_plan: Optional[dict] = Field(
        default=None,
        description="Private deterministically validated plan awaiting review.",
    )
    planning_attempts: int = Field(default=0, ge=0)
    planning_issue_codes: List[str] = Field(default_factory=list)
    output_review_attempts: int = Field(default=0, ge=0)
    output_review_issue_codes: List[str] = Field(default_factory=list)
