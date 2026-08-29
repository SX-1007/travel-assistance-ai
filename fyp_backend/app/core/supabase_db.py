"""
Supabase / PostgreSQL data layer.

Responsibilities:
  • pgvector travel-item store (embed → upsert → semantic search)
  • LangGraph checkpointer (PostgresSaver)
  • Mem0 preference store
  • User profile & saved-itinerary persistence

Thread-safety:
  • ConnectionPool is thread-safe by design (psycopg_pool).
  • Supabase REST client (PostgREST) is stateless & thread-safe.
  • Embedding models (GoogleGenerativeAIEmbeddings) are thread-safe for calls.

Performance:
  • Batch embedding with parallel workers (configurable).
  • Multi-row INSERT with ON CONFLICT upsert.
  • HNSW index for approximate nearest-neighbour search.
  • Statement-level timeout to prevent runaway queries.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.postgres import PostgresSaver
from supabase import Client, create_client

from app.core.config import settings
from app.tools.provider_logging import safe_provider_log

# ── Optional pgvector native adapter ────────────────────────────────
try:
    from pgvector.psycopg import register_vector as _register_vector

    _HAS_PGVECTOR_PYTHON = True
except ImportError:
    _HAS_PGVECTOR_PYTHON = False

# ── Logger ──────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Configuration constants ────────────────────────────────────────
EMBEDDING_MODEL_NAME = settings.GEMINI_EMBED_MODEL
EMBEDDING_DIMENSION = 3072
EMBED_BATCH_SIZE = 64
MAX_EMBED_RETRIES = 3
RETRY_BASE_WAIT = 1.5
EMBED_PARALLEL_WORKERS = 3  # parallel batch embedding workers

UPSERT_BATCH_SIZE = 200
DEFAULT_TTL_HOURS = 1
HNSW_EF_SEARCH = 40
STATEMENT_TIMEOUT_MS = 15_000

POOL_MIN_SIZE = 2
POOL_MAX_SIZE = 20
POOL_MAX_IDLE = 300
POOL_TIMEOUT = 30.0


# ── Connection-pool configure hook ─────────────────────────────────
def _configure_conn(conn) -> None:
    """Called once per new connection — sets timeout & registers pgvector.

    Both statements run implicitly inside a transaction (the pool hands us a
    non-autocommit connection), so we must ``commit()`` before returning.
    Otherwise psycopg_pool sees the connection still ``INTRANS`` and discards
    it — starving the pool until ``open(wait=True)`` times out.
    """
    conn.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")
    if _HAS_PGVECTOR_PYTHON:
        _register_vector(conn)
    conn.commit()


# ── Supabase REST client ────────────────────────────────────────────
# FIX (Report #9): argument order is (url, key) — was previously swapped.
supabase_client: Client = create_client(
    settings.SUPABASE_URL,
    settings.SUPABASE_API_KEY,
)

# ── PostgreSQL connection pool (lazy open) ──────────────────────────
connection_pool: ConnectionPool = ConnectionPool(
    conninfo=settings.SUPABASE_POSTGRES_URI,
    min_size=POOL_MIN_SIZE,
    max_size=POOL_MAX_SIZE,
    max_idle=POOL_MAX_IDLE,
    timeout=POOL_TIMEOUT,
    configure=_configure_conn,
    # Supabase's session pooler silently drops idle TCP connections;
    # keepalive probes keep them alive and the checkout health-check
    # discards any that died anyway instead of handing them to a request.
    kwargs={
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 3,
    },
    check=ConnectionPool.check_connection,
    open=False,  # deferred to lifespan startup
)

_pool_initialized = False


def init_connection_pool() -> None:
    """Open the pool — call during FastAPI lifespan startup."""
    global _pool_initialized
    if _pool_initialized:
        return
    connection_pool.open(wait=True)
    _pool_initialized = True
    safe_provider_log(logger, "storage.supabase.pool_opened")


def close_connection_pool() -> None:
    """Close the pool — call during FastAPI lifespan shutdown."""
    global _pool_initialized
    try:
        if _pool_initialized:
            connection_pool.close()
            _pool_initialized = False
            safe_provider_log(logger, "storage.supabase.pool_closed")
    except Exception:
        safe_provider_log(logger, "storage.supabase.pool_close_failed")


def verify_connection_pool() -> bool:
    """Health-check: can we execute a trivial query?"""
    try:
        with connection_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                return cur.fetchone() is not None
    except Exception:
        safe_provider_log(logger, "storage.supabase.pool_health_failed")
        return False


# ── Embedding models ────────────────────────────────────────────────
# FIX (Report #10): pass google_api_key explicitly.
_embed_api_key = settings.GEMINI_EMBED_API

_doc_embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    task_type="RETRIEVAL_DOCUMENT",
    google_api_key=_embed_api_key,
)
_query_embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL_NAME,
    task_type="RETRIEVAL_QUERY",
    google_api_key=_embed_api_key,
)


# ══════════════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════════════


def _vector_cast() -> str:
    """SQL cast suffix for vector parameters (empty if native adapter present)."""
    return "" if _HAS_PGVECTOR_PYTHON else "::vector"


def _prepare_vector(vec: List[float]):
    """
    Prepare a vector for **psycopg** (direct Postgres) parameter binding.

    • With pgvector-python: returns the raw list (binary protocol).
    • Without:              returns a string '[v1,v2,...]' for text casting.
    """
    if _HAS_PGVECTOR_PYTHON:
        return vec
    inner = ",".join(repr(x) for x in vec)
    return f"[{inner}]"


def _format_vector_for_rest(vec: List[float]) -> str:
    """
    Format a vector as a pgvector-compatible **string** for PostgREST
    (Supabase REST API).

    PostgREST cannot handle Python lists as vectors — it always expects
    the string representation '[v1,v2,...]'.
    """
    return "[" + ",".join(str(x) for x in vec) + "]"


def _deterministic_id(session_id: str, category: str, item: Dict[str, Any]) -> str:
    """Deterministic UUID via uuid5 — prevents duplicate vectors on re-index."""
    if category == "hotel":
        identity = str(item.get("hotel_name", "")).strip().lower()
    elif category == "flight":
        identity = (
            f"{str(item.get('airline', '')).strip().lower()}_"
            f"{str(item.get('flight_number', '')).strip().lower()}"
        )
    elif category == "activity":
        identity = str(item.get("name", "")).strip().lower()
    else:
        identity = ""

    if not identity:
        identity = json.dumps(item, sort_keys=True, default=str, ensure_ascii=False)

    raw_key = f"{session_id}|{category}|{identity}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_key))


def _format_text_payload(category: str, item: Dict[str, Any]) -> str:
    """Format a travel item into a text string optimal for embedding."""
    if category == "hotel":
        amenities = ", ".join(item.get("amenities", []) or [])
        return (
            f"Hotel: {item.get('hotel_name', '')}. "
            f"Description: {item.get('description', '')}. "
            f"Amenities: {amenities}."
        )
    if category == "activity":
        return (
            f"Activity: {item.get('name', '')}. "
            f"Type: {item.get('type', '')}. "
            f"Description: {item.get('description', '')}."
        )
    if category == "flight":
        return (
            f"Flight {item.get('flight_number', '')} by {item.get('airline', '')} "
            f"departing {item.get('departure_time', '')} "
            f"arriving {item.get('arrival_time', '')} "
            f"price {item.get('price', '')}."
        )
    return json.dumps(item, default=str, ensure_ascii=False)


# ── Embedding (parallel batch) ──────────────────────────────────────


def _embed_single_batch(
    model,
    batch: List[str],
    is_query: bool,
    batch_num: int,
) -> List[List[float]]:
    """Embed one batch with exponential-backoff retry."""
    for attempt in range(1, MAX_EMBED_RETRIES + 1):
        try:
            if is_query and len(batch) == 1:
                return [model.embed_query(batch[0])]
            return model.embed_documents(batch)
        except Exception:
            if attempt == MAX_EMBED_RETRIES:
                safe_provider_log(logger, "storage.supabase.embedding_failed")
                raise
            wait = min(RETRY_BASE_WAIT * (2 ** (attempt - 1)), 10)
            safe_provider_log(logger, "storage.supabase.embedding_retry")
            time.sleep(wait)
    return []  # unreachable — satisfies type checker


def _embed_texts_batched(
    texts: List[str],
    is_query: bool = False,
) -> List[List[float]]:
    """
    Embed texts in batches.

    • Single batch  → inline call (no thread-pool overhead).
    • Multiple batches → parallel via ThreadPoolExecutor
      (capped at EMBED_PARALLEL_WORKERS to respect API rate limits).

    Returns a flat list of vectors in the **original text order**.
    """
    if not texts:
        return []

    model = _query_embeddings if is_query else _doc_embeddings

    # Partition into batches preserving order
    batches: List[tuple[int, List[str]]] = [
        (i // EMBED_BATCH_SIZE + 1, texts[i : i + EMBED_BATCH_SIZE])
        for i in range(0, len(texts), EMBED_BATCH_SIZE)
    ]

    # Fast path: single batch
    if len(batches) == 1:
        return _embed_single_batch(model, batches[0][1], is_query, 1)

    # Parallel path
    results: List[Optional[List[List[float]]]] = [None] * len(batches)

    with ThreadPoolExecutor(
        max_workers=min(EMBED_PARALLEL_WORKERS, len(batches))
    ) as ex:
        future_to_idx: Dict[Any, int] = {}
        for idx, (num, batch) in enumerate(batches):
            fut = ex.submit(_embed_single_batch, model, batch, is_query, num)
            future_to_idx[fut] = idx

        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            results[idx] = fut.result()  # propagates exceptions

    # Flatten in order
    out: List[List[float]] = []
    for r in results:
        if r:
            out.extend(r)
    return out


# ══════════════════════════════════════════════════════════════════════
#  Application Data (user profiles, saved trips)
# ══════════════════════════════════════════════════════════════════════


def fetch_user_profile(
    user_id: str,
    *,
    raise_on_error: bool = False,
) -> Dict[str, Any]:
    """Retrieve permanent user traits (home currency, allergies, etc.).

    ``raise_on_error=False`` preserves the existing fail-soft behaviour used by
    optional personalisation paths.  Request gates such as ``GET /api/profile``
    pass ``raise_on_error=True`` so a temporary Supabase failure cannot be
    mistaken for a genuinely missing profile.
    """
    try:
        response = (
            supabase_client.table("user_profiles")
            .select("*")
            .eq("id", user_id)
            .execute()
        )
        return response.data[0] if response.data else {}
    except Exception:
        safe_provider_log(logger, "storage.supabase.profile_fetch_failed")
        if raise_on_error:
            raise
        return {}


def create_user_profile(user_id: str, profile: Dict[str, Any]) -> bool:
    """Create or overwrite a user's onboarding profile.

    Upserts into ``user_profiles`` keyed by ``id`` so that repeated onboarding
    submissions are idempotent. Returns True on success.
    """
    payload = {"id": user_id, **profile}
    try:
        (
            supabase_client.table("user_profiles")
            .upsert(payload, on_conflict="id")
            .execute()
        )
        safe_provider_log(logger, "storage.supabase.profile_created")
        return True
    except Exception:
        safe_provider_log(logger, "storage.supabase.profile_create_failed")
        return False


_MEMORY_LIST_FIELDS = {
    "dietary_restrictions",
    "interests",
    "accommodation_preferences",
}


def _merge_unique_memory_values(
    existing: Any,
    incoming: Any,
) -> List[str]:
    """Merge list-style memories without duplicating learned values."""

    existing_values = (
        existing
        if isinstance(existing, list)
        else ([existing] if existing else [])
    )

    incoming_values = (
        incoming
        if isinstance(incoming, list)
        else ([incoming] if incoming else [])
    )

    merged: List[str] = []
    seen: set[str] = set()

    for value in [*existing_values, *incoming_values]:
        text = str(value).strip()

        if not text:
            continue

        key = text.casefold()

        if key in seen:
            continue

        seen.add(key)
        merged.append(text)

    return merged


def update_user_profile(
    user_id: str,
    updates: Dict[str, Any],
) -> bool:
    """Persist learned long-term traits into the user's profile."""

    if not user_id or not updates:
        return False

    try:
        current = fetch_user_profile(user_id)

        payload = dict(updates)

        # Preserve previously learned list-style memories.
        for field in _MEMORY_LIST_FIELDS:
            if field in payload:
                payload[field] = _merge_unique_memory_values(
                    current.get(field),
                    payload[field],
                )

        result = (
            supabase_client.table("user_profiles")
            .update(payload)
            .eq("id", user_id)
            .execute()
        )

        if result.data:
            safe_provider_log(
                logger,
                "storage.supabase.profile_updated",
            )
            return True

        safe_provider_log(
            logger,
            "storage.supabase.profile_update_failed",
        )
        return False

    except Exception:
        safe_provider_log(
            logger,
            "storage.supabase.profile_update_failed",
        )
        return False


def save_final_itinerary(
    user_id: str,
    trip_id: str,
    itinerary_data: dict,
    geojson_data: dict,
) -> None:
    """Persist the completed trip to the database."""
    payload = {
        "user_id": user_id,
        "trip_id": trip_id,
        "itinerary_text": itinerary_data,
        "map_geojson": geojson_data,
    }
    try:
        result = (
            supabase_client.table("saved_trips")
            .upsert(payload, on_conflict="trip_id")
            .execute()
        )
        if result.data:
            safe_provider_log(logger, "storage.supabase.itinerary_saved")
        else:
            safe_provider_log(logger, "storage.supabase.itinerary_save_empty")
    except Exception:
        safe_provider_log(logger, "storage.supabase.itinerary_save_failed")


# ══════════════════════════════════════════════════════════════════════
#  LangGraph Checkpointer
# ══════════════════════════════════════════════════════════════════════


def init_langgraph_checkpoint_tables() -> None:
    """RUN ONCE AT STARTUP: creates LangGraph tables in Supabase Postgres."""
    with connection_pool.connection() as conn:
        saver = PostgresSaver(conn=conn)
        saver.setup()
    safe_provider_log(logger, "storage.supabase.checkpointer_ready")


@contextmanager
def get_checkpointer() -> Generator[PostgresSaver, None, None]:
    """
    Yield a live PostgresSaver for the graph compiler.

    The connection is auto-returned to the pool on exit
    (committed on success, rolled back on exception).
    """
    with connection_pool.connection() as conn:
        saver = PostgresSaver(conn=conn)
        yield saver


def _load_chat_history_checkpoint(
    row: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Load one persisted checkpoint using its own pooled connection."""

    with connection_pool.connection() as conn:
        saver = PostgresSaver(conn=conn)

        saved = saver.get_tuple(
            {
                "configurable": {
                    "thread_id": row["thread_id"],
                    "checkpoint_ns": row["checkpoint_ns"],
                    "checkpoint_id": row["checkpoint_id"],
                }
            }
        )

    if saved is None:
        return None

    checkpoint = saved.checkpoint

    return {
        "session_id": row["thread_id"],
        "updated_at": checkpoint.get("ts", ""),
        "state": checkpoint.get(
            "channel_values",
            {},
        ) or {},
    }


def fetch_chat_history_checkpoints(
    user_id: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Return the newest persisted LangGraph state for each user session.

    First retrieve the latest checkpoint IDs. Then load the heavier checkpoint
    payloads concurrently through the existing thread-safe connection pool
    instead of performing every remote Postgres read sequentially.
    """

    if not user_id:
        return []

    safe_limit = max(
        1,
        min(int(limit), 20),
    )

    # Only hold this connection while retrieving the lightweight checkpoint
    # identifiers. Release it before loading the full checkpoint payloads.
    with connection_pool.connection() as conn:
        with conn.cursor(
            row_factory=dict_row,
        ) as cur:
            cur.execute(
                """
                SELECT thread_id, checkpoint_ns, checkpoint_id
                FROM (
                    SELECT DISTINCT ON (thread_id)
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id
                    FROM checkpoints
                    WHERE checkpoint_ns = ''
                      AND metadata @> %s
                    ORDER BY
                        thread_id,
                        checkpoint_id DESC
                ) AS latest
                ORDER BY checkpoint_id DESC
                LIMIT %s
                """,
                (
                    Jsonb(
                        {
                            "user_id": user_id,
                        }
                    ),
                    safe_limit,
                ),
            )

            rows = cur.fetchall()

    if not rows:
        return []

    # Four workers provides concurrency while leaving capacity in your
    # existing 20-connection pool for chat/profile requests.
    worker_count = min(
        4,
        len(rows),
    )

    with ThreadPoolExecutor(
        max_workers=worker_count,
    ) as executor:

        futures = [
            executor.submit(
                _load_chat_history_checkpoint,
                row,
            )
            for row in rows
        ]

        history = [
            record
            for future in futures
            if (
                record := future.result()
            ) is not None
        ]

    return history


# ══════════════════════════════════════════════════════════════════════
#  pgvector Store
# ══════════════════════════════════════════════════════════════════════


def init_vector_store() -> None:
    """
    RUN ONCE AT STARTUP: enables pgvector, creates table & indexes.

    Indexes:
      • B-tree on (session_id, category, expires_at) — session filtering
      • HNSW  on embedding (vector_cosine_ops)       — ANN search
    """
    with connection_pool.connection() as conn:
        try:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        except Exception:
            safe_provider_log(logger, "storage.supabase.vector_extension_unavailable")

        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS travel_vectors (
                id          UUID PRIMARY KEY,
                session_id  TEXT NOT NULL,
                category    TEXT NOT NULL,
                raw_data    JSONB NOT NULL,
                expires_at  DOUBLE PRECISION NOT NULL,
                created_at  TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                embedding   vector({EMBEDDING_DIMENSION})
            );
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_travel_vectors_session_cat_expires
            ON travel_vectors (session_id, category, expires_at);
        """)


        # ── Long-term semantic user memory ─────────────────────
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS mem0_vector (
                id              UUID PRIMARY KEY,
                user_id         TEXT NOT NULL,
                category        TEXT NOT NULL,
                preference_text TEXT NOT NULL,
                embedding       vector({EMBEDDING_DIMENSION}) NOT NULL,
                created_at      TIMESTAMP WITH TIME ZONE
                                NOT NULL DEFAULT CURRENT_TIMESTAMP,

                UNIQUE (
                    user_id,
                    category,
                    preference_text
                )
            );
        """)


        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_mem0_vector_user_category
            ON mem0_vector (
                user_id,
                category,
                created_at DESC
            );
        """)

    # HNSW ANN index. pgvector's ``vector`` type only supports HNSW indexes up
    # to 2000 dimensions, but our embeddings are 3072-dim — so we index a
    # ``halfvec`` cast instead (supported up to 4000 dims). The search query
    # (search_personalised_options) casts to the same halfvec expression so the
    # planner can use this index. Non-fatal: if index creation fails, semantic
    # search still works via an exact scan.
    try:
        with connection_pool.connection() as conn:
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_travel_vectors_embedding
                ON travel_vectors
                USING hnsw ((embedding::halfvec({EMBEDDING_DIMENSION})) halfvec_cosine_ops)
                WITH (m = 16, ef_construction = 64);
            """)
    except Exception:
        safe_provider_log(logger, "storage.supabase.vector_index_unavailable")

    safe_provider_log(logger, "storage.supabase.vector_ready")


def verify_vector_store() -> bool:
    """Health check: verify pgvector extension and travel_vectors table exist."""
    try:
        with connection_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector';")
                if not cur.fetchone():
                    safe_provider_log(
                        logger, "storage.supabase.vector_extension_missing"
                    )
                    return False

                cur.execute(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_name = 'travel_vectors';"
                )
                if not cur.fetchone():
                    safe_provider_log(logger, "storage.supabase.vector_table_missing")
                    return False
        return True
    except Exception:
        safe_provider_log(logger, "storage.supabase.vector_verify_failed")
        return False


def upsert_travel_data(
    session_id: str,
    category: str,
    items: List[Dict[str, Any]],
    ttl_hours: int = DEFAULT_TTL_HOURS,
) -> bool:
    """
    Embed + upsert travel items into the pgvector table.

    Uses deterministic UUIDs (uuid5) so re-indexing the same item updates
    it in-place instead of creating duplicates.

    Returns True on success, False on failure.
    """
    if not items:
        return False

    expires_at = time.time() + (ttl_hours * 3600)
    cast = _vector_cast()

    texts_to_embed = [_format_text_payload(category, item) for item in items]

    try:
        vectors = _embed_texts_batched(texts_to_embed, is_query=False)
        if len(vectors) != len(texts_to_embed):
            raise RuntimeError(
                f"Embedding count mismatch: {len(vectors)} != {len(texts_to_embed)}"
            )

        # Build records
        records: List[tuple] = []
        for item, vector in zip(items, vectors):
            doc_id = _deterministic_id(session_id, category, item)
            records.append(
                (
                    doc_id,
                    session_id,
                    category,
                    Jsonb(item),
                    expires_at,
                    _prepare_vector(vector),
                )
            )

        # Batch INSERT with ON CONFLICT upsert
        with connection_pool.connection() as conn:
            with conn.cursor() as cur:
                for i in range(0, len(records), UPSERT_BATCH_SIZE):
                    batch = records[i : i + UPSERT_BATCH_SIZE]
                    n = len(batch)
                    row_ph = f"(%s, %s, %s, %s, %s, %s{cast})"
                    placeholders = ", ".join([row_ph] * n)

                    query = f"""
                        INSERT INTO travel_vectors
                            (id, session_id, category, raw_data, expires_at, embedding)
                        VALUES {placeholders}
                        ON CONFLICT (id) DO UPDATE SET
                            raw_data   = EXCLUDED.raw_data,
                            expires_at = EXCLUDED.expires_at,
                            embedding  = EXCLUDED.embedding;
                    """
                    params: list = []
                    for rec in batch:
                        params.extend(rec)

                    cur.execute(query, params)

        safe_provider_log(logger, "storage.supabase.vector_upsert_succeeded")
        return True

    except Exception:
        safe_provider_log(logger, "storage.supabase.vector_upsert_failed")
        return False


def search_personalised_options(
    session_id: str,
    user_preferences: str,
    category: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Semantic search over the user's session data using cosine distance.

    Uses HNSW index with ef_search=40 for a good accuracy/speed tradeoff.

    Returns ranked list of original item dicts with a 'relevance_score' field.
    """
    if not user_preferences.strip():
        return []

    try:
        query_vectors = _embed_texts_batched([user_preferences], is_query=True)
        if not query_vectors:
            return []
        query_vector = query_vectors[0]
        query_param = _prepare_vector(query_vector)
        # Match the halfvec HNSW index built in init_vector_store() so the
        # planner can use it (vector<=> would force an exact scan on 3072-dim
        # embeddings, which cannot be HNSW-indexed as a plain vector).
        hv = f"::halfvec({EMBEDDING_DIMENSION})"
        current_time = time.time()

        with connection_pool.connection() as conn:
            with conn.transaction():
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(f"SET LOCAL hnsw.ef_search = {HNSW_EF_SEARCH}")
                    cur.execute(
                        f"""
                        SELECT
                            raw_data,
                            1 - (embedding{hv} <=> %s{hv}) AS relevance_score
                        FROM travel_vectors
                        WHERE session_id = %s
                        AND category   = %s
                        AND expires_at > %s
                        ORDER BY embedding{hv} <=> %s{hv}
                        LIMIT %s;
                        """,
                        (
                            query_param,
                            session_id,
                            category,
                            current_time,
                            query_param,
                            top_k,
                        ),
                    )
                    rows = cur.fetchall()

        ranked_items: List[Dict[str, Any]] = []
        for row in rows:
            raw = row["raw_data"]
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    continue
            elif not isinstance(raw, dict):
                continue
            raw["relevance_score"] = round(float(row["relevance_score"]), 3)
            ranked_items.append(raw)

        return ranked_items

    except Exception:
        safe_provider_log(logger, "storage.supabase.vector_search_failed")
        return []


def fetch_all_options(
    session_id: str,
    category: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Retrieve all non-expired items for a session+category (no semantic ranking)."""
    try:
        with connection_pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT raw_data
                    FROM travel_vectors
                    WHERE session_id = %s
                      AND category   = %s
                      AND expires_at > %s
                    ORDER BY created_at DESC
                    LIMIT %s;
                    """,
                    (session_id, category, time.time(), limit),
                )
                rows = cur.fetchall()

        results: List[Dict[str, Any]] = []
        for row in rows:
            raw = row["raw_data"]
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    continue
            elif not isinstance(raw, dict):
                continue
            results.append(raw)
        return results

    except Exception:
        safe_provider_log(logger, "storage.supabase.vector_fetch_failed")
        return []


def clear_session_vectors(session_id: str) -> bool:
    """Delete all vectors associated with a specific session ID."""
    try:
        with connection_pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM travel_vectors WHERE session_id = %s;",
                    (session_id,),
                )
        safe_provider_log(logger, "storage.supabase.vector_clear_succeeded")
        return True
    except Exception:
        safe_provider_log(logger, "storage.supabase.vector_clear_failed")
        return False


def clear_expired_vectors(session_id: Optional[str] = None) -> int:
    """
    Best-effort sweep of expired vectors (call periodically via cron).

    Returns number of deleted rows.
    """
    try:
        now = time.time()
        with connection_pool.connection() as conn:
            with conn.cursor() as cur:
                if session_id:
                    cur.execute(
                        "DELETE FROM travel_vectors "
                        "WHERE expires_at < %s AND session_id = %s;",
                        (now, session_id),
                    )
                else:
                    cur.execute(
                        "DELETE FROM travel_vectors WHERE expires_at < %s;",
                        (now,),
                    )
                deleted = cur.rowcount

        if deleted > 0:
            safe_provider_log(logger, "storage.supabase.vector_sweep_succeeded")
        return deleted

    except Exception:
        safe_provider_log(logger, "storage.supabase.vector_sweep_failed")
        return 0


# ══════════════════════════════════════════════════════════════════════
#  Mem0 architecture
# ══════════════════════════════════════════════════════════════════════


_MEM0_TO_PROFILE_FIELD = {
    "dietary": "dietary_restrictions",
    "interest": "interests",
    "accommodation": "accommodation_preferences",
}


def fetch_mem0_preferences(
    user_id: str,
    *,
    raise_on_error: bool = False,
) -> Dict[str, List[str]]:
    """Fetch all persisted semantic preferences for one user.

    The Mem0 table was previously write-only: preferences were inserted but
    never read back into the agent. This deterministic user-scoped read is
    intentionally used for long-term profile context; semantic ranking is not
    appropriate when the user asks for all remembered preferences.
    """
    if not user_id:
        return {}

    try:
        response = (
            supabase_client
            .table("mem0_vector")
            .select("category,preference_text,created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(200)
            .execute()
        )

        grouped: Dict[str, List[str]] = {}
        seen: Dict[str, set[str]] = {}

        for row in response.data or []:
            category = str(row.get("category") or "").strip()
            text = str(row.get("preference_text") or "").strip()

            if not category or not text:
                continue

            key = text.casefold()
            category_seen = seen.setdefault(category, set())

            if key in category_seen:
                continue

            category_seen.add(key)
            grouped.setdefault(category, []).append(text)

        safe_provider_log(
            logger,
            "storage.supabase.mem0_fetch_succeeded",
        )

        return grouped

    except Exception:
        safe_provider_log(
            logger,
            "storage.supabase.mem0_fetch_failed",
        )

        if raise_on_error:
            raise

        return {}


def fetch_user_memory_context(user_id: str) -> Dict[str, Any]:
    """Return one merged long-term-memory view for prompting and review.

    user_profiles remains the authoritative relational source for onboarding
    and scalar traits. mem0_vector is merged in as a durable fallback for
    learned list-style preferences and travel pacing.
    """
    if not user_id:
        return {}

    # First obtain relational profile information.
    merged: Dict[str, Any] = dict(
        fetch_user_profile(user_id)
    )

    # Then recover semantic long-term memories.
    vector_memory = fetch_mem0_preferences(user_id)

    for category, values in vector_memory.items():
        if not values:
            continue

        if category == "travel_pacing":
            # Mem0 rows are returned newest-first.
            # Prefer the relational value if it exists.
            if not merged.get("travel_pacing"):
                merged["travel_pacing"] = values[0]

            continue

        profile_field = _MEM0_TO_PROFILE_FIELD.get(category)

        if profile_field:
            merged[profile_field] = _merge_unique_memory_values(
                merged.get(profile_field),
                values,
            )

    return merged


def insert_mem0_preferences(
    user_id: str,
    categorized_preferences: Dict[str, List[str]],
) -> bool:
    """
    Embed categorized preferences in a single batch and insert into
    the mem0_vector table via the Supabase REST client (PostgREST).

    FIX (Report #21): PostgREST cannot handle Python lists as vectors.
    We always serialize to the pgvector string format '[v1,v2,...]'.

    Returns True on success, False on failure.
    """
    if not categorized_preferences:
        return False

    try:
        all_texts: List[str] = []
        text_categories: List[str] = []

        for category, preferences in categorized_preferences.items():
            for pref_text in preferences:
                all_texts.append(pref_text)
                text_categories.append(category)

        if not all_texts:
            return False

        # Batch embed all preferences
        vectors = _embed_texts_batched(all_texts, is_query=False)
        if len(vectors) != len(all_texts):
            raise RuntimeError("Embedding count mismatch.")

        # Build records — always use string format for REST API
        records = []

        for text, category, vector in zip(
            all_texts,
            text_categories,
            vectors,
        ):
            memory_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{user_id}|{category}|{text.strip().casefold()}",
                )
            )

            records.append(
                {
                    "id": memory_id,
                    "user_id": user_id,
                    "preference_text": text,
                    "category": category,
                    "embedding": _format_vector_for_rest(vector),
                }
            )


        result = (
            supabase_client
            .table("mem0_vector")
            .upsert(
                records,
                on_conflict="id",
            )
            .execute()
        )

        if result.data:
            safe_provider_log(logger, "storage.supabase.mem0_insert_succeeded")
            return True
        return False

    except Exception:
        safe_provider_log(logger, "storage.supabase.mem0_insert_failed")
        return False


# ── Public API ──────────────────────────────────────────────────────
__all__ = [
    # Client & pool
    "supabase_client",
    "connection_pool",
    "init_connection_pool",
    "close_connection_pool",
    "verify_connection_pool",
    # Checkpointer
    "init_langgraph_checkpoint_tables",
    "get_checkpointer",
    "fetch_chat_history_checkpoints",
    # Vector store
    "init_vector_store",
    "verify_vector_store",
    "upsert_travel_data",
    "search_personalised_options",
    "fetch_all_options",
    "clear_session_vectors",
    "clear_expired_vectors",
    # App data
    "fetch_user_profile",
    "create_user_profile",
    "update_user_profile",
    "save_final_itinerary",
    # Mem0
    "fetch_mem0_preferences",
    "fetch_user_memory_context",
    "insert_mem0_preferences",
]
