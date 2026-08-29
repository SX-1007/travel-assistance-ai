"""
Memory Extraction Module — Mem0 Tiered Extraction Engine

Extracts long-term user traits from conversation messages and routes them
to appropriate storage tiers (relational profile + vector database).

Performance Optimisations:
  • Lazy LLM initialisation (defers failures to runtime, not import time)
  • Parallel DB writes via module-level ThreadPoolExecutor
  • Exponential backoff with jitter for LLM retries
  • Trivial-message pre-filtering to skip unnecessary LLM calls
  • Double-checked locking for thread-safe lazy init
  • Module-level prompt template (compiled once at import)
"""

from __future__ import annotations

import re
import time
import random
import logging
import threading
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed, Future

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers.openai_tools import PydanticToolsParser

from app.core.supabase_db import update_user_profile, insert_mem0_preferences
from app.tools.provider_logging import safe_provider_log
from app.core.config import settings

__all__ = [
    "LongTermMemory",
    "analyse_and_extract_traits",
    "run_memory_extraction_task",
]

# ──────────────────────────────────────────────
# Logging Setup
# ──────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
_EXTRACTION_MODEL = settings.GEMINI_UTILITY_MODEL  # FIX #3: valid model name
MAX_LLM_RETRIES = 3
LLM_RETRY_BASE_WAIT = 1.5
LLM_RETRY_MAX_WAIT = 10.0
_MAX_MESSAGE_LENGTH = 8_192  # Guard against oversized inputs

# Pre-compiled regex for trivial-message detection (O(1) match)
_TRIVIAL_RE = re.compile(
    r"^(hi|hello|hey|ok|okay|thanks|thank\s+you|yes|no|sure|cool|nice|great|bye|gotcha|alright)\s*[!.?]*$",
    re.IGNORECASE,
)


# ──────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────
class LongTermMemory(BaseModel):
    """Permanent user traits stored in the relational profile and vector DB."""

    home_country: Optional[str] = Field(
        default=None,
        description="The permanent home country of the user.",
    )
    home_state: Optional[str] = Field(
        default=None,
        description="The permanent home state or province of the user.",
    )
    travel_pacing: Optional[str] = Field(
        default=None,
        description="Permanent travel speed preference (e.g., 'relaxed', 'packed schedule', 'moderate').",
    )
    dietary_restrictions: Optional[List[str]] = Field(
        default=None,
        description="Permanent food allergies or diets (e.g., 'User is vegan', 'User has a peanut allergy').",
    )
    interests: Optional[List[str]] = Field(
        default=None,
        description="Lifelong broad interests (e.g., 'User loves history', 'User enjoys hiking').",
    )
    accommodation_preferences: Optional[List[str]] = Field(
        default=None,
        description="Permanent hotel/lodging styles (e.g., 'User prefers 5-star hotels').",
    )


# ──────────────────────────────────────────────
# Prompt Template (compiled once at module load)
# ──────────────────────────────────────────────
_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [  # FIX #2: from_messages (plural)
        (
            "system",
            """You are the Mem0 Context Management Engine for an AI Travel Agent.
Your job is to read the user's latest message and extract ONLY permanent, long-term memory points.

RULES:
1. ONLY extract permanent traits. IGNORE short-term trip requests (e.g., "I want a pool for this trip").
2. Sort extracted traits into their specific categories. Do not mix them.
3. Format traits as clear, standalone factual sentences.

EXAMPLES:
User: "I'm allergic to peanuts, I love history, and I only stay in 5-star hotels."
→ dietary_restrictions: ["User has a peanut allergy."]
→ interests: ["User loves history."]
→ accommodation_preferences: ["User only stays in 5-star hotels."]

User: "I live in Selangor, Malaysia, and I like a relaxed pace when traveling."
→ home_country: "Malaysia", home_state: "Selangor", travel_pacing: "relaxed"

User: "Can you find a hotel with a gym for next week?"
→ All fields null.
""",
        ),
        ("human", "{message}"),
    ]
)


# ──────────────────────────────────────────────
# LLM Initialisation (Lazy + Thread-Safe)
# ──────────────────────────────────────────────
_init_lock = threading.Lock()
_llm: Optional[ChatGoogleGenerativeAI] = None
_extraction_chain: Optional[Runnable] = None


def _get_extraction_chain() -> Runnable:
    """
    Lazily initialise and return the extraction chain.

    Uses double-checked locking for thread safety.
    Defers LLM initialisation to first use so import-time
    failures (missing API key, network issues) don't crash the app.
    """
    global _llm, _extraction_chain

    if _extraction_chain is not None:  # Fast path — no lock
        return _extraction_chain

    with _init_lock:  # Slow path — acquire lock
        if _extraction_chain is not None:  # Re-check inside lock
            return _extraction_chain

        _llm = ChatGoogleGenerativeAI(
            model=_EXTRACTION_MODEL,
            temperature=0.0,
            google_api_key=settings.effective_extractor_api_key,
            max_retries=2,
        )

        # Force the memory schema tool explicitly.
        #
        # The project's older langchain-google-genai integration does not
        # automatically recognise Gemini 3.x model names when deciding whether
        # structured output can force a tool call. Without an explicit tool_choice,
        # Gemini may return a normal text response and the Pydantic parser returns
        # None, causing valid preferences to be treated as "no traits".
        _forced_llm = _llm.bind_tools(
            [LongTermMemory],
            tool_choice=LongTermMemory.__name__,
        )

        _parser = PydanticToolsParser(
            tools=[LongTermMemory],
            first_tool_only=True,
        )

        _extraction_chain = (
            _EXTRACTION_PROMPT
            | _forced_llm
            | _parser
        )

        safe_provider_log(
            logger,
            "memory.extractor_chain_ready",
        )

        return _extraction_chain


# ──────────────────────────────────────────────
# Module-Level Thread Pool for Parallel DB Writes
# ──────────────────────────────────────────────
# Shared across all requests — avoids per-call executor creation overhead.
# max_workers=4 allows concurrent writes across multiple user sessions.
_db_executor = ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="mem0-db",
)


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────
def _is_trivial(message: str) -> bool:
    """
    Check if a message is trivial (greeting / acknowledgement).
    These messages are skipped to avoid unnecessary LLM calls.
    """
    stripped = message.strip().lower()
    return len(stripped) < 4 or bool(_TRIVIAL_RE.match(stripped))


def _calculate_backoff(attempt: int) -> float:
    """
    Calculate exponential backoff with jitter.
    Jitter prevents thundering-herd when the LLM service recovers.
    """
    base = min(LLM_RETRY_BASE_WAIT * (2 ** (attempt - 1)), LLM_RETRY_MAX_WAIT)
    jitter = random.uniform(0, base * 0.1)  # Up to 10% jitter
    return base + jitter


def _has_long_term_traits(
    memory: LongTermMemory,
) -> bool:
    """Return True only when at least one persistent trait was extracted."""

    return any(
        (
            bool(memory.home_country),
            bool(memory.home_state),
            bool(memory.travel_pacing),
            bool(memory.dietary_restrictions),
            bool(memory.interests),
            bool(memory.accommodation_preferences),
        )
    )


def _update_relational_db(
    user_id: str,
    updates: Dict[str, Any],
) -> bool:
    """Update the relational user profile."""

    try:
        return update_user_profile(
            user_id,
            updates,
        )

    except Exception:
        safe_provider_log(
            logger,
            "memory.extractor.relational_failed",
        )
        return False


def _update_vector_db(user_id: str, categories: Dict[str, List[str]]) -> bool:
    """Update the vector DB with categorised preferences. Returns True on success."""
    try:
        return insert_mem0_preferences(  # FIX #6: plural function name
            user_id=user_id,
            categorized_preferences=categories,
        )
    except Exception:
        safe_provider_log(logger, "memory.extractor.vector_failed")
        return False


# ──────────────────────────────────────────────
# Core Extraction Logic
# ──────────────────────────────────────────────
def analyse_and_extract_traits(user_message: str) -> Optional[LongTermMemory]:
    """
    Send the user message to the LLM and extract long-term memory traits.

    Implements exponential backoff with jitter for transient LLM failures.
    Pre-filters trivial messages to avoid unnecessary LLM calls.

    Args:
        user_message: The user's latest chat message.

    Returns:
        LongTermMemory instance if traits were found, None otherwise.
    """
    if not user_message or not user_message.strip():
        return None

    if _is_trivial(user_message):
        safe_provider_log(logger, "memory.extractor.trivial")
        return None

    # Guard against excessively long inputs (Google flash handles 1M tokens,
    # but truncating saves bandwidth and reduces latency)
    message = user_message.strip()[:_MAX_MESSAGE_LENGTH]

    chain = _get_extraction_chain()

    for attempt in range(1, MAX_LLM_RETRIES + 1):
        try:
            result = chain.invoke(
                {
                    "message": message,
                }
            )

            # Missing structured output is an extraction failure.
            # It does NOT mean the user has no long-term preferences.
            if not isinstance(result, LongTermMemory):
                raise RuntimeError(
                    "Memory extractor returned no structured "
                    "LongTermMemory output."
                )

            # A valid LongTermMemory object with every field empty really
            # means that this message contains no persistent user traits.
            if not _has_long_term_traits(result):
                safe_provider_log(
                    logger,
                    "memory.extractor.no_traits",
                )
                return None

            safe_provider_log(
                logger,
                "memory.extractor.completed",
            )

            return result

        except Exception:
            if attempt == MAX_LLM_RETRIES:
                safe_provider_log(
                    logger,
                    "memory.extractor.failed",
                )
                return None

            wait_time = _calculate_backoff(attempt)

            safe_provider_log(
                logger,
                "memory.extractor.retry",
            )

            time.sleep(wait_time)

    return None


# ──────────────────────────────────────────────
# Graph Node — Parallel DB Routing
# ──────────────────────────────────────────────
def run_memory_extraction_task(
    session_id: str,
    user_id: str,
    user_message: str,
) -> Dict[str, Any]:
    """
    LangGraph-compatible node function.

    Extracts long-term traits from the user message and routes them to
    the appropriate storage tier **in parallel**:
      • Relational DB (home_country, home_state, travel_pacing)
      • Vector DB (dietary, interests, accommodation)

    Uses a module-level ThreadPoolExecutor so relational and vector
    writes happen concurrently, reducing total latency from
    (DB1 + DB2) to max(DB1, DB2).

    Args:
        session_id: The current conversation session ID.
        user_id: The authenticated user's ID.
        user_message: The user's latest chat message.

    Returns:
        Dict with key ``"memory_updated"`` (bool) indicating whether any
        persistence operation succeeded.
    """
    safe_provider_log(logger, "memory.extractor.started")

    if not user_id:
        return {"memory_updated": False}

    # ── Step 1: Extract traits via LLM ──────────────────────
    extracted_data = analyse_and_extract_traits(user_message)

    if not extracted_data:
        return {"memory_updated": False}

    # ── Step 2: Prepare DB updates ──────────────────────────
    relational_updates: Dict[str, Any] = {}

    if extracted_data.home_country:
        relational_updates["home_country"] = extracted_data.home_country

    if extracted_data.home_state:
        relational_updates["home_state"] = extracted_data.home_state

    if extracted_data.travel_pacing:
        relational_updates["travel_pacing"] = extracted_data.travel_pacing

    if extracted_data.dietary_restrictions:
        relational_updates["dietary_restrictions"] = (
            extracted_data.dietary_restrictions
        )

    if extracted_data.interests:
        relational_updates["interests"] = extracted_data.interests

    if extracted_data.accommodation_preferences:
        relational_updates["accommodation_preferences"] = (
            extracted_data.accommodation_preferences
        )


    # Keep a semantic/vector copy as the second memory tier.
    vector_categories: Dict[str, Optional[List[str]]] = {
        "dietary": extracted_data.dietary_restrictions,
        "interest": extracted_data.interests,
        "accommodation": extracted_data.accommodation_preferences,

        # Keep a Mem0 fallback for pacing too.
        # This prevents the preference from disappearing if an older
        # user_profiles schema does not contain travel_pacing.
        "travel_pacing": (
            [extracted_data.travel_pacing]
            if extracted_data.travel_pacing
            else None
        ),
    }

    valid_categories: Dict[str, List[str]] = {
        cat: prefs
        for cat, prefs in vector_categories.items()
        if prefs
    }

    if not relational_updates and not valid_categories:
        return {"memory_updated": False}

    # ── Step 3: Parallel DB Writes ──────────────────────────
    # Submit both writes to the shared executor — they run concurrently.
    futures: Dict[Future, str] = {}

    if relational_updates:
        safe_provider_log(logger, "memory.extractor.relational_queued")
        futures[
            _db_executor.submit(_update_relational_db, user_id, relational_updates)
        ] = "relational"

    if valid_categories:
        safe_provider_log(logger, "memory.extractor.vector_queued")
        futures[_db_executor.submit(_update_vector_db, user_id, valid_categories)] = (
            "vector"
        )

    # Collect results as they complete (non-blocking on the slower one)
    memory_updated = False
    for future in as_completed(futures):
        try:
            if future.result():
                memory_updated = True
                safe_provider_log(logger, "memory.extractor.persistence_succeeded")
            else:
                safe_provider_log(logger, "memory.extractor.persistence_failed")
        except Exception:
            safe_provider_log(logger, "memory.extractor.persistence_raised")

    return {"memory_updated": memory_updated}
