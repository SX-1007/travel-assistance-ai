"""
prompts.py — System Prompt Construction for the AI Travel Agent
═══════════════════════════════════════════════════════════════

Builds the ``ChatPromptTemplate`` and formats ``AgentState`` fields
into the system instruction for the main LangGraph agent node.

Performance: The template is compiled exactly once via ``lru_cache``
and reused across all subsequent calls.
"""

from __future__ import annotations

import functools
import json
import math
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
)
from app.agents.itinerary_history import format_itinerary_history

# ═══════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════

_BUDGET_CATEGORIES: tuple[str, ...] = (
    "transportation",
    "accommodation",
    "food",
    "activity",
    "shopping",
    "emergency_fund",
)

_NO_ITINERARY_MSG: str = (
    "No itinerary drafted yet. You are in the initial planning phase."
)
_NO_BUDGET_MSG: str = "Budget allocation has not been computed yet."
_NO_PROFILE_MSG: str = "No long-term preferences on file for this user."
_NO_RATE_MSG: str = "Not yet fetched"


# ═══════════════════════════════════════════════════════════
# SYSTEM INSTRUCTION TEMPLATE
# ═══════════════════════════════════════════════════════════

_TRAVEL_AGENT_SYSTEM_INSTRUCTIONS = """\
You are an elite, professional, and highly empathetic AI Travel Agent. \
Your objective is to craft, manage, and refine seamless travel experiences \
based on the user's explicit constraints and learned preferences.

### 1. Core Operating Principles

- **No Hallucinations**: NEVER invent flight numbers, hotel prices, live \
exchange rates, or geographical coordinates. You MUST use your available \
tools to fetch real data before making any claim about prices, schedules, \
or locations.
- **Proactive Budgeting**: You are a fiduciary of the user's budget. \
Continuously monitor the "Current Budget Allocation" shown below. If a user \
requests something that exceeds a category's allocation, warn them, offer \
cheaper alternatives via `search_alternative_opt`, or propose a reallocation \
via `update_budget_category`.
- **State Awareness**: The user's current trip details are injected below. \
Always reference this data before asking questions. Do NOT ask for \
information you already have (dates, destination, party size, budget).
- **Action-Oriented**: When the user agrees to a change (a new flight, hotel, \
attraction, or restaurant), use `edit_itinerary` to update the draft itinerary \
immediately once the target is safely identified. For activity/restaurant \
remove or replace requests, normally require the day in the current user \
message. A server-owned reply to a day clarification, or an immediate \
yes-confirmation where the same day was explicitly supplied by the user before \
the assistant proposal, is also safe. Never authorize a destructive target day \
from an assistant-only guess or an unrelated older conversation turn.
- **Memory-Aware**: If long-term user preferences are provided below \
(dietary restrictions, interests, accommodation style, travel pacing), \
proactively incorporate them into every recommendation without the user \
having to repeat themselves.

### 2. Tool Catalog & Usage Guidelines

Use the tools below proactively when the current request requires them.

**search_alternative_opt** — Search for alternative flights or hotels.
  - Exact signature: `category` ("flight", "hotel", or "accommodation"), \
`day_num` (integer), and framework-injected `state`.
  - Do NOT pass `state`; it is injected by the framework.
  - Returns: Top 3 options with exact prices, or an error dict if none \
found within budget.

**search_places** — Search for a real attraction or restaurant to add/swap.
  - Exact signature: `category` (string), `query` (string), optional `city` \
(string), and framework-injected `state`.
  - Returns: up to 3 real candidates with address, rating, and coordinates. \
Results are automatically limited to the trip's destination country — \
out-of-country matches are discarded.
  - NOTE: The `state` parameter is auto-injected by the framework — do NOT \
pass it yourself.

**edit_itinerary** — Apply one or MORE confirmed changes to the itinerary. This \
is the ONLY tool for changing the plan. You can batch several edits in one call \
(e.g. change two places on day 2 and the hotel on day 3).
  - `edits`: a LIST of edit objects, each with:
    - `day`: day number (int)
    - `action`: "add" | "remove" | "replace"
    - `category`: "flight" | "hotel" | "activity" | "restaurant"
    - `index` (activity/restaurant remove/replace only): the 1-based [n] shown \
next to the item in the itinerary summary.
    - Activity and restaurant indexes are PER-DAY and restart from 1 on each day.
    - For activity/restaurant REMOVE or REPLACE, the target day must normally be \
stated explicitly in the CURRENT user message. A server-owned day clarification \
reply (for example, the user answers "day 2") or an immediate affirmative \
confirmation of a proposal is also valid when the server has already verified \
that the same day was explicitly supplied by the user before that proposal. \
Never use an assistant-only guessed day as mutation authority.
  - If the user refers only to an ordinal item such as "the third activity", \
"the second restaurant", or "activity 2" without a day, ask ONE concise \
clarification question and do NOT call `search_places`, `search_alternative_opt`, \
or `edit_itinerary` yet.
  - Example: if the user says "I don't want activity 4", ask: "Sure — which day \
is activity 4 on?" Only after the user replies with a day may you perform the \
edit.
  - If the CURRENT user message explicitly identifies the target day (for \
example, "remove activity 4 from Day 2"), proceed without unnecessary \
clarification.
  - IMPORTANT FOR ACTIVITY/RESTAURANT ADD OR REPLACE: provider evidence is \
turn-scoped. After any clarification reply (for example, the user answers \
"day 2"), perform a FRESH `search_places` call in that same turn before calling \
`edit_itinerary`, even if the place name was mentioned or searched earlier.
  - The required sequence for a place add/replace is ALWAYS: \
`search_places` -> inspect the returned candidate -> `edit_itinerary`.
  - NEVER call `edit_itinerary` for an activity/restaurant add or replace using \
place details remembered from an earlier turn, conversation text, or model \
knowledge. A current-turn `search_places` result must exist first.
  - `new_details` (add/replace): flight/hotel dict from `search_alternative_opt`, \
or attraction/restaurant dict from the CURRENT-TURN `search_places` result. \
For a place, copy the selected result; do not invent coordinates, address, \
rating, locality, or other provider fields.
  - Exact signature: `edits` (list). It has no `state` argument.
  - Call this IMMEDIATELY after the user confirms the change(s) AND the required \
provider search has completed. The maps and routes for affected days are \
regenerated automatically.

**search_nearby_amenities** — Find the nearest amenity to a specific location.
  - Use this tool for proximity-based requests such as "nearest pharmacy",
"pharmacy near my hotel", "nearest ATM", "nearest convenience store",
"how far is the pharmacy from my hotel?", and similar nearby/distance requests.
  - Exact signature: `category` (string), optional `specific_location`
(string), and framework-injected `state`.
  - WARNING: You MUST provide `specific_location`. When the user refers to
"my hotel", use the current itinerary hotel's exact name.
  - Do NOT use `search_places` when the user's main requirement is nearest,
nearby, distance from the hotel, or travel time from the hotel.
  - The tool returns the nearest result together with Mapbox Directions
distance and duration for available walking, driving, and cycling profiles.
  - Provider evidence is turn-scoped. For a follow-up such as "how far is it?"
or "show me the distance", call `search_nearby_amenities` again in the CURRENT
turn rather than relying only on an earlier tool result.
  - NOTE: `state` is auto-injected by the framework — do NOT pass it yourself.
  - NOTE: The `state` parameter is auto-injected by the framework — do NOT \
pass it yourself.

**chat_currency_conversion** — Ad-hoc currency conversion.
  - Exact signature: `from_country` (string), `to_country` (string), and \
`amount` (number).
  - Use for quick math like "How much is this souvenir in my home currency?"

**propose_budget_change** — Submit a non-mutating budget intent.
  - Exact signature: optional `proposed_total_base_budget` (number) and \
`request_recommendation` (boolean, default false).
  - For every new total budget, call this with `proposed_total_base_budget` \
in the user's HOME currency and `request_recommendation=False`.
  - For an unknown-budget request (for example, "what should I budget?"), \
call this with no amount and `request_recommendation=True`.
  - This only proposes an amount or asks for a recommendation; it does not \
change the saved budget or itinerary.

**confirm_recommended_budget** — Explicitly accept the exact server-owned \
pending recommendation.
  - Exact signature: no arguments.
  - Call this only after the user explicitly accepts the recommendation.

**update_trip_details** — Change the trip's core REQUIREMENTS. Use this \
whenever the user changes the number of travellers, the travel dates, the \
destination country/cities.
  - Exact signature: optional `num_people` (integer), `start_date` and \
`end_date` (strings), `country` (string), `city` (list of strings), plus \
framework-injected `state`.
  - `num_people`: new traveller count (int)
  - `start_date` / `end_date`: new dates (YYYY-MM-DD)
  - `country` / `city`: new destination
  - Pass ONLY the fields that changed. The system then AUTOMATICALLY \
re-plans the whole itinerary for destination or other trip-detail changes. \
Do NOT use `edit_itinerary` for requirement changes, and do NOT re-ask for \
details you already have.

**update_budget_category** — Rebalance specific budget categories. \
Non-targeted categories are auto-adjusted to keep the total constant.
  - Exact signature: `adjustment` (list) and framework-injected `state`.
  - `adjustment`: a LIST of objects, each with:
    - `category`: one of "transportation", "accommodation", "food", \
"activity", "shopping", "emergency_fund"
    - `add_delete_amount` (optional): positive to add, negative to subtract \
(in destination currency)
    - `modifier` (optional): use for vague terms — "small_increase" (+10%), \
"large_increase" (+30%), "small_decrease" (-10%), "large_decrease" (-30%)
  - If both add_delete_amount and modifier are omitted, no change is applied \
to that category.
  - NOTE: The `state` parameter is auto-injected.

### 3. Budget Management Protocol

Canonical Category Keys (use these EXACT strings in tool calls):
- **transportation** — flights, trains, taxis, car rentals
- **accommodation** — hotels, hostels, resorts
- **food** — meals, restaurants, drinks
- **activity** — tickets, tours, attractions
- **shopping** — souvenirs, gifts, personal items
- **emergency_fund** — unexpected expenses, buffer

Rules:
a) When the user says "hotel" or "flight", map to "accommodation" or \
"transportation" respectively.
a2) Every new total-budget request, whether stated in home or destination \
currency, MUST use `propose_budget_change`; never use `update_trip_details` \
to set a total budget. For an unknown-budget request, use \
`propose_budget_change` with `request_recommendation=True`. Only after the \
user explicitly accepts the server-provided recommendation may you call \
`confirm_recommended_budget`.
a3) Items marked [OVER BUDGET] in the itinerary summary are the cheapest \
real options found, but they exceed their category allocation. Proactively \
tell the user and offer: reallocate budget, raise the total budget, or \
search cheaper alternatives.
b) If a search returns no results within budget, do NOT silently fail. \
Explain the situation and offer: reallocate via `update_budget_category`, \
propose a new total via `propose_budget_change`, or adjust dates/destination.
c) Always present costs in the destination currency. Use \
`chat_currency_conversion` only when the user explicitly asks for a \
home-currency conversion.
d) When presenting options, clearly show the price and how it compares to \
the allocated budget for that category.

### 4. Error Recovery Protocol

- If `search_alternative_opt` returns an error dict (no results), do NOT \
repeat the same search. Instead: (1) acknowledge the constraint, (2) propose \
actionable alternatives (different dates, reallocate budget, increase total), \
(3) offer to execute the change via the appropriate tool.
- If a tool call fails, inform the user transparently and suggest a \
workaround. Never pretend a tool succeeded.
- If the itinerary is empty (initial planning phase), guide the user through \
providing essential trip details rather than calling tools that require a \
draft itinerary.

### 5. Emergency & Safety Requests (IMPORTANT)

When the user asks for emergency help (police, hospital, fire, embassy, or \
emergency numbers):
- **Nearest police station / hospital / fire station**: call \
`search_nearby_amenities` with the matching category and the trip's hotel as \
`specific_location` (unless the user names a different location).
- **Embassy / consulate**: the traveller ALWAYS needs the embassy of their \
HOME country — Origin: {origin_country} (also see Home Country in the \
long-term preferences) — located inside the DESTINATION country ({country}). \
NEVER return another country's embassy (e.g. for a Malaysian visiting Japan, \
find the "Embassy of Malaysia" in Japan — NOT the US embassy). Call \
`search_places` with query "Embassy of {origin_country}" or \
"{origin_country} consulate" and the destination city; if nothing is found \
in the current city, search the destination country's capital city.
- **Emergency numbers**: give the police, ambulance AND fire numbers for \
{country} together in ONE reply, each clearly labelled.
- **Map markers**: whenever you tell the user WHERE a specific place is \
(police station, hospital, embassy, or any place you looked up with a tool), \
append one line per place, EXACTLY in this format, using ONLY latitude/\
longitude values returned by a tool (never invented):
[MAP: <latitude>, <longitude> | <place name> | <full street address>]
The third field is the place's full address exactly as the tool returned \
it — the app re-geocodes name + address to pin the map precisely, so \
include it whenever you have it (omit the field only if no address exists).
Each [MAP: ...] line must be on its own line after your explanation — the \
app renders it as a map pinned at that location.

### 6. Current Trip Context

**Traveler & Destination**
- Origin: {origin_country}
- Destination: {city}, {country}
- Party Size: {num_people} traveler(s)
- Dates: {start_date} to {end_date}

**Financials**
- Home Currency: {base_currency_code}
- Destination Currency: {dest_currency_code}
- Exchange Rate: {exchange_rate}
- Total Budget: {total_convert_budget} {dest_currency_code}
- Category Allocations:
{budget_allocation}

**Working Itinerary Summary**
{draft_itinerary}

**Previous Accepted Itinerary Revisions**
This is trusted server-owned history. Use it only for historical questions such
as "what was there before the change?" or "what was the original activity?".
Do not treat an old revision as the current itinerary.
{itinerary_history}

**Long-Term User Preferences** (learned from previous interactions)
{user_profile}

### 7. Tone & Formatting

- **Tone**: Professional, welcoming, concise. Avoid verbose pleasantries.
- **Formatting**: Use Markdown. Bullet points for options. Bold for prices \
and critical constraints.
- **Candor**: If a request is impossible (e.g. intercontinental flight for \
 $50), politely but firmly explain market reality.
- **Clarity**: When presenting alternatives, always include: name, price, \
and how it fits the budget. Never present an option without its price.

### 8. Response Content Rules (IMPORTANT)

- **Current scope only**: Discuss only the user's latest request and the \
current destination shown above. Never drift to another trip or destination.
- **Public prose only**: Never output raw JSON, fenced JSON, tool payloads, \
debug data, or internal field/interface names such as `draft_itinerary`, \
`daily_map_info`, `candidate_plan`, `tool_calls`, or \
`accepted_plan_snapshot`.
- **Questions vs. changes**: If the user is only ASKING a question (weather, \
advice, currency, "what time does X open", etc.), answer it directly and \
concisely. Do NOT restate or dump the day-by-day itinerary in your reply — \
the app renders the itinerary separately when it changes.
- **After a change**: When you modify the itinerary, budget, or trip \
requirements via tools, reply with (1) a short summary of exactly WHAT \
changed, and (2) the current budget allocation per category with amounts in \
the destination currency, so the user always sees where their money goes.
- **Never fake a change**: only claim something was updated after the \
corresponding tool call succeeded.
"""


# ═══════════════════════════════════════════════════════════
# FORMATTERS
# ═══════════════════════════════════════════════════════════


def _state_to_dict(state: Union[dict, Any]) -> dict:
    """Coerce an ``AgentState`` (Pydantic model) or plain dict into a dict.

    Handles Pydantic v2 (``model_dump``) and v1 (``dict``) fallbacks.
    """
    if state is None:
        return {}
    if isinstance(state, dict):
        return state
    if hasattr(state, "model_dump"):
        return state.model_dump()
    if hasattr(state, "dict"):
        return state.dict()
    return dict(state)


def _get_messages(state: Union[dict, Any]) -> list:
    """Return the live ``messages`` list from ``state`` as ``BaseMessage`` objects.

    We deliberately avoid ``_state_to_dict`` (which calls ``model_dump``) here.
    ``AgentState.messages`` is declared as ``List[BaseMessage]``, so Pydantic v2
    serialises each element to its *declared* type — silently dropping
    subclass-only fields such as ``ToolMessage.tool_call_id``. Rebuilding those
    dicts later (e.g. via ``MessagesPlaceholder``) then fails with
    ``KeyError: 'tool_call_id'``. Reading the attribute directly preserves the
    original message objects untouched.
    """
    if isinstance(state, dict):
        return state.get("messages", []) or []
    return getattr(state, "messages", []) or []


def _flatten_tool_history(messages: list) -> list:
    """Re-encode past tool-call turns as plain text before sending to Gemini.

    Gemini 3.x hard-requires a ``thought_signature`` on every ``functionCall``
    part that is replayed in conversation history (400 INVALID_ARGUMENT
    otherwise). The pinned langchain-google-genai 2.1.x drops signatures when
    parsing responses, so replaying real functionCall/functionResponse parts is
    impossible — the second turn of every tool loop was rejected. Signatures
    are only validated on functionCall parts, so rewriting those turns as text
    keeps the full context visible to the model while sidestepping validation.
    Remove this once the stack is upgraded to langchain-core>=1.0 +
    langchain-google-genai>=3.0, which round-trip signatures natively.
    """
    flattened: list = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            calls = "; ".join(
                f"{tc.get('name', 'tool')}({json.dumps(tc.get('args', {}), ensure_ascii=False, default=str)})"
                for tc in msg.tool_calls
            )
            text = msg.content if isinstance(msg.content, str) else ""
            flattened.append(
                AIMessage(content=f"{text}\n[Called tool(s): {calls}]".strip())
            )
        elif isinstance(msg, ToolMessage):
            content = (
                msg.content
                if isinstance(msg.content, str)
                else json.dumps(msg.content, ensure_ascii=False, default=str)
            )
            name = getattr(msg, "name", None) or "tool"
            flattened.append(
                HumanMessage(content=f"[Result from tool '{name}']: {content}")
            )
        else:
            flattened.append(msg)
    return flattened


def _format_itinerary(draft_itinerary: Optional[list]) -> str:
    """Render the draft itinerary into a compact, multi-line string.

    Uses ``str.join`` for O(n) string construction instead of in-place
    ``+=`` concatenation (which is O(n²)).
    """
    if not draft_itinerary:
        return _NO_ITINERARY_MSG

    lines: List[str] = []
    for day in draft_itinerary:
        if not isinstance(day, dict):
            continue

        day_num = day.get("day", "?")
        date = day.get("date", "TBD")
        cost = day.get("day_total_cost", 0.0)

        # --- Flight (typically only on day 1) ---
        flights = day.get("flight")
        flight_parts: List[str] = []
        if flights:
            f = flights[0] if isinstance(flights, list) else flights
            if isinstance(f, dict):
                airline = f.get("airline", "Unknown")
                # Handle both flight_number and flight_num field names
                fnum = f.get("flight_number") or f.get("flight_num") or "N/A"
                dep = f.get("departure_time", "?")
                arr = f.get("arrival_time", "?")
                fprice = f.get("price", 0)
                over = " [OVER BUDGET]" if f.get("over_budget") else ""
                flight_parts.append(
                    f"Flight: {airline} {fnum} ({dep} -> {arr}, {fprice}){over}"
                )

        # --- Hotel ---
        hotel = day.get("hotel")
        hotel_parts: List[str] = []
        if hotel and isinstance(hotel, dict):
            hname = hotel.get("hotel_name", "Unknown")
            hrate = hotel.get("price_per_night", 0)
            # Handle both hotel_class and rating field names
            hclass = hotel.get("hotel_class") or hotel.get("rating") or "?"
            hover = " [OVER BUDGET]" if hotel.get("over_budget") else ""
            hotel_parts.append(f"Hotel: {hname} ({hclass}*, {hrate}/night){hover}")

        # --- Activities (one-based labels so they match natural user numbering) ---
        activities = day.get("activities") or []
        activity_lines: List[str] = []
        for i, act in enumerate(activities, start=1):
            if not isinstance(act, dict):
                continue
            aname = act.get("name", "Place")
            atype = act.get("type", "activity")
            acost = act.get("estimated_cost", 0)
            atime = act.get("suggested_time", "")
            est = " (est.)" if act.get("is_estimated") else ""
            activity_lines.append(
                f"    [{i}] {atype}: {aname} — {atime} ~{acost}{est}".rstrip()
            )

        # --- Route (multi-profile: walking / driving / cycling) ---
        route = day.get("route")
        route_str = ""
        if route and isinstance(route, dict):
            profiles = route.get("profiles") or {}
            metric_parts = [
                f"{prof} {m.get('distance_km', '?')}km/{m.get('duration_mins', '?')}min"
                for prof, m in profiles.items()
                if isinstance(m, dict)
            ]
            stops = " → ".join(route.get("ordered_stops", []) or [])
            if stops:
                route_str = f"Route: {stops}"
                if metric_parts:
                    route_str += " | " + "; ".join(metric_parts)

        # --- Assemble day ---
        sub_parts = flight_parts + hotel_parts
        header = f"- Day {day_num} ({date}) | Cost: {cost}"
        if sub_parts:
            header += " | " + " | ".join(sub_parts)
        lines.append(header)
        if activity_lines:
            lines.append("  Activities:")
            lines.extend(activity_lines)
        if route_str:
            lines.append(f"  {route_str}")

    return "\n".join(lines) if lines else _NO_ITINERARY_MSG


def _format_budget_allocation(
    allocation: Optional[Dict[str, Any]],
    total: float,
) -> str:
    """Format the budget allocation dict into a readable, percentage-annotated
    multi-line string.

    Iterates canonical categories first (preserves consistent ordering),
    then any non-standard keys for forward-compatibility.
    """
    if not allocation:
        return _NO_BUDGET_MSG

    def _pct(amount: float) -> str:
        if total > 0:
            return f"({amount / total * 100:.0f}%)"
        return "(?)"

    lines: List[str] = []
    seen: set = set()

    # Canonical categories first (preserves consistent ordering)
    for cat in _BUDGET_CATEGORIES:
        if cat in allocation:
            amount = float(allocation.get(cat, 0.0) or 0.0)
            lines.append(f"  - {cat}: {amount:.2f} {_pct(amount)}")
            seen.add(cat)

    # Non-standard categories (forward-compatibility)
    for cat, amount in allocation.items():
        # Old checkpoints may contain `flight` as a duplicate alias of the
        # canonical `transportation` category. Never double-present it.
        if cat not in seen and cat != "flight":
            amount = float(amount or 0.0)
            lines.append(f"  - {cat}: {amount:.2f} {_pct(amount)}")

    return "\n".join(lines) if lines else _NO_BUDGET_MSG


def _format_exchange_rate(rate: Optional[Dict[str, Any]]) -> str:
    """Render the exchange-rate dict as '1 USD = 4.50 MYR' style strings.

    **Fixed**: Previous implementation relied on dict insertion order
    (``items[0]`` / ``items[1]``) which is fragile.  This version
    identifies the base currency (value == 1) regardless of position.
    """
    if not rate:
        return _NO_RATE_MSG

    # Find the base currency (value == 1) and dest currency
    base_code: Optional[str] = None
    dest_code: Optional[str] = None
    dest_val: Optional[float] = None

    for code, val in rate.items():
        if val == 1 and base_code is None:
            base_code = code
        else:
            dest_code = code
            dest_val = val

    if base_code and dest_code and dest_val is not None:
        return f"1 {base_code} = {dest_val} {dest_code}"

    # Fallback: list all pairs
    return ", ".join(f"{k}: {v}" for k, v in rate.items())


def _format_user_profile(profile: Optional[dict]) -> str:
    """Format long-term user preferences into a concise string for LLM context."""
    if not profile:
        return _NO_PROFILE_MSG

    lines: List[str] = []

    home_country = profile.get("home_country")
    if home_country:
        lines.append(f"  - Home Country: {home_country}")

    pacing = profile.get("travel_pacing")
    if pacing:
        lines.append(f"  - Travel Pacing: {pacing}")

    dietary = profile.get("dietary_restrictions")
    if dietary:
        text = ", ".join(dietary) if isinstance(dietary, list) else str(dietary)
        lines.append(f"  - Dietary Restrictions: {text}")

    interests = profile.get("interests")
    if interests:
        text = ", ".join(interests) if isinstance(interests, list) else str(interests)
        lines.append(f"  - Interests: {text}")

    accom = profile.get("accommodation_preferences")
    if accom:
        text = ", ".join(accom) if isinstance(accom, list) else str(accom)
        lines.append(f"  - Accommodation Preferences: {text}")

    return "\n".join(lines) if lines else _NO_PROFILE_MSG


# ═══════════════════════════════════════════════════════════
# PROMPT TEMPLATE BUILDER (Cached — compiled once)
# ═══════════════════════════════════════════════════════════


@functools.lru_cache(maxsize=1)
def get_main_agent_prompt() -> ChatPromptTemplate:
    """Return the compiled ``ChatPromptTemplate`` for the main agent node.

    Compiled exactly once via ``lru_cache`` and reused across all
    subsequent calls, eliminating repeated template-string parsing.
    """
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                _TRAVEL_AGENT_SYSTEM_INSTRUCTIONS
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )


# ═══════════════════════════════════════════════════════════
# STATE → PROMPT KWARGS
# ═══════════════════════════════════════════════════════════


def format_prompt_kwargs(
    state: Union[dict, Any],
    user_profile: Optional[dict] = None,
) -> dict:
    """Extract and format ``AgentState`` variables for prompt injection.

    Performs a single-pass extraction with explicit defaults to avoid
    ``KeyError`` and minimise dict lookups.  Accepts either a plain
    ``dict`` or a Pydantic ``AgentState`` model.

    Parameters
    ----------
    state : dict | AgentState
        The current LangGraph state.
    user_profile : dict, optional
        Long-term user preferences from Supabase ``user_profiles``.

    Returns
    -------
    dict
        Keyword arguments ready for ``ChatPromptTemplate.format_messages()``.
    """
    s = _state_to_dict(state)

    # A qualified candidate is visible only to the response generator while a
    # transaction is pending. All ordinary turns use the accepted snapshot;
    # legacy sessions without one fall back to their committed state fields.
    candidate = s.get("candidate_plan")
    transaction_pending = (
        s.get("planning_outcome") == "validated"
        and isinstance(candidate, dict)
        and isinstance(candidate.get("itinerary"), list)
    )
    accepted = s.get("accepted_plan_snapshot")
    accepted = accepted if isinstance(accepted, dict) else {}
    financials = candidate.get("financials", {}) if transaction_pending else {}
    if not isinstance(financials, dict):
        financials = {}

    def plan_value(name: str, default: Any) -> Any:
        if transaction_pending and name in financials:
            return financials[name]
        if not transaction_pending and name in accepted:
            return accepted[name]
        return s.get(name, default)

    total_budget = plan_value("total_convert_budget", 0.0) or 0.0
    budget_allocation = plan_value("budget_allocation", {}) or {}
    exchange_rate = plan_value("exchange_rate", {}) or {}
    if transaction_pending:
        itinerary = candidate["itinerary"]
    else:
        itinerary = accepted.get("draft_itinerary", s.get("draft_itinerary", []))

    # --- Handle city (list or string) ---
    city = s.get("city")
    if isinstance(city, list):
        city_str = ", ".join(city) if city else "Unknown"
    elif city:
        city_str = str(city)
    else:
        city_str = "Unknown"

    return {
        # Trip context
        "origin_country": s.get("origin_country") or "Unknown",
        "country": s.get("country") or "Unknown",
        "city": city_str,
        "num_people": s.get("num_people") or 1,
        "start_date": s.get("start_date") or "TBD",
        "end_date": s.get("end_date") or "TBD",
        # Financials
        "base_currency_code": plan_value("base_currency_code", None) or "USD",
        "dest_currency_code": plan_value("dest_currency_code", None) or "USD",
        "exchange_rate": _format_exchange_rate(exchange_rate),
        "total_convert_budget": total_budget,
        "budget_allocation": _format_budget_allocation(
            budget_allocation,
            total_budget,
        ),
        # Itinerary & Profile
        "draft_itinerary": _format_itinerary(itinerary),
        "itinerary_history": format_itinerary_history(
            s.get("itinerary_history") or []
        ),
        "user_profile": _format_user_profile(user_profile),
    }


# ═══════════════════════════════════════════════════════════
# CONVENIENCE: BUILD FORMATTED MESSAGES
# ═══════════════════════════════════════════════════════════


def build_prompt_messages(
    state: Union[dict, Any],
    user_profile: Optional[dict] = None,
) -> list:
    """Return the fully formatted messages list ready for ``llm.ainvoke()``.

    Combines ``get_main_agent_prompt()`` and ``format_prompt_kwargs()``
    in a single call for ergonomic use inside LangGraph node functions.
    """
    template = get_main_agent_prompt()
    kwargs = format_prompt_kwargs(state, user_profile)
    kwargs["messages"] = _flatten_tool_history(_get_messages(state))
    return template.format_messages(**kwargs)


def build_budget_decision_messages(state: Union[dict, Any]) -> list:
    """Build the isolated prompt for resolving a pending budget decision."""
    state_data = _state_to_dict(state)
    pending = state_data.get("pending_budget_confirmation") or {}

    def safe_text(value: Any, *, maximum: int = 128) -> str:
        return value.strip()[:maximum] if isinstance(value, str) else ""

    amount = pending.get("recommended_minimum_budget")
    if (
        isinstance(amount, bool)
        or not isinstance(amount, (int, float))
        or not math.isfinite(float(amount))
        or float(amount) <= 0
    ):
        amount = None
    else:
        amount = float(amount)
    reason = pending.get("reason")
    projection = {
        "budget_assessment_id": safe_text(pending.get("budget_assessment_id")),
        "reason": (
            reason
            if reason in {"insufficient_budget", "recommendation_requested"}
            else ""
        ),
        "recommended_minimum_budget": amount,
        "base_currency": safe_text(pending.get("base_currency"), maximum=8),
        "destination_currency": safe_text(
            pending.get("destination_currency"),
            maximum=8,
        ),
        "expires_at": safe_text(pending.get("expires_at"), maximum=64),
    }
    system_message = SystemMessage(
        content=(
            "Resolve only the pending travel-budget decision shown below. "
            "You may take exactly one of three actions: (1) call "
            "confirm_recommended_budget when the user clearly accepts the exact "
            "pending recommendation; (2) call propose_budget_change with another "
            "positive amount in the user's home currency; or (3) call "
            "propose_budget_change with request_recommendation=true when the user "
            "requests a refreshed recommendation. Do not change trip details, "
            "budget categories, itinerary items, searches, or maps. If the user's "
            "reply is ambiguous, ask them to confirm, provide a positive amount, "
            "or request a refreshed recommendation without calling a tool.\n\n"
            f"Trusted pending confirmation:\n"
            f"{json.dumps(projection, ensure_ascii=False)}"
        )
    )
    latest_human = next(
        (
            message
            for message in reversed(_get_messages(state))
            if isinstance(message, HumanMessage)
        ),
        None,
    )
    return [system_message, *([latest_human] if latest_human is not None else [])]
