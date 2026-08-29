from __future__ import annotations
import json
import logging
import threading
import time
import uuid
from typing import Dict, Any, List, Optional
from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeException
from app.core.config import settings

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# initialise embeddings and pinecone client
EMBEDDING_MODEL = "llama-text-embed-v2"
EMBEDDING_DIMENSION = 1024
INDEX_NAME = "travel-context-index"
METRIC = "cosine"
CLOUD = "aws"
REGION = "us-east-1"

UPSERT_BATCH = 100
EMBED_BATCH = 64
DEFAULT_TTL_HOURS = 1
MAX_EMBED_RETIRES = 3
INDEX_READY_TIMEOUT = 120
RETRY_BASE_WAIT = 1.5

_pc: Optional[Pinecone] = None
_index: Any = None
_init_lock = threading.RLock()


def _get_pinecone() -> Pinecone:
    global _pc
    if _pc is None:
        with _init_lock:
            if _pc is None:
                _pc = Pinecone(api_key=settings.PINECONE_API_KEY)
                logger.info("Pinecone client initialised")

    return _pc


def _get_or_create_index() -> Any:
    """
    Returns the singleton Pinecone index object.
    Creates the index if it does not exist.
    """
    global _index
    if _index is not None:
        return _index

    with _init_lock:
        if _index is not None:
            return _index

        pc = _get_pinecone()
        try:
            existing_indexes = {idx["name"] for idx in pc.list_indexes()}

            if INDEX_NAME not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {INDEX_NAME}")
                pc.create_index(
                    name=INDEX_NAME,
                    dimension=EMBEDDING_DIMENSION,
                    metric=METRIC,
                    spec=ServerlessSpec(cloud=CLOUD, region=REGION),
                )
                deadline = time.time() + INDEX_READY_TIMEOUT
                while time.time() < deadline:
                    desc = pc.describe_index(INDEX_NAME)
                    if desc.status.ready:
                        break
                    time.sleep(2)
                else:
                    logger.error(
                        f"Index {INDEX_NAME} not ready within {INDEX_READY_TIMEOUT}s"
                    )
                    return None

            _index = pc.Index(INDEX_NAME)
            logger.info(f"Connected to Pinecone index {INDEX_NAME}")
        except Exception as e:
            logger.error(f"Failed to initialise Pinecone index: {e}", exc_info=True)
            _index = None

        return _index


def _embed_texts(
    texts: List[str],
    input_type: str = "passage",
) -> List[List[float]]:
    """
    Embed a list of strings via Pinecone's hosted llama-text-embed-v2.

    * Batches requests to respect inference API size limits.
    * Exponential-backoff retry on transient failures.
    """
    if not texts:
        return []

    pc = _get_pinecone()
    out: List[List[float]] = []

    for start in range(0, len(texts), EMBED_BATCH):
        chunk = texts[start : start + EMBED_BATCH]
        for attempt in range(1, MAX_EMBED_RETIRES + 1):
            try:
                resp = pc.inference.embed(
                    model=EMBEDDING_MODEL,
                    inputs=chunk,
                    parameters={"input_type": input_type, "truncate": "END"},
                )
                out.extend([v.values for v in resp])
                break
            except PineconeException as e:
                if attempt == MAX_EMBED_RETIRES:
                    raise
                wait = min(RETRY_BASE_WAIT * (2 ** (attempt - 1)), 10)
                logger.warning(
                    f"Embed batch {start // EMBED_BATCH + 1} attempt {attempt}"
                    f"failed: {e}: retry in {wait}s"
                )
                time.sleep(wait)
    return out


# upsert data to pinecone
def upsert_travel_data(
    session_id: str,
    category: str,
    items: List[Dict[str, Any]],
    ttl_hours: int = DEFAULT_TTL_HOURS,
) -> bool:
    """
    Embed + upsert travel items into the session-scoped Pinecone namespace.

    Args:
        session_id: user/trip session identifier (used as Pinecone namespace).
        category:   "hotel" | "activity" | "flight".
        items:      list of dicts describing the items.
        ttl_hours:  expiry window; stored as epoch metadata for filtered queries.
    Returns:
        True on success, False on failure.
    """
    index = _get_or_create_index()
    if not index or not items:
        return False

    # build text payloads, metadata and vector IDs
    texts_to_embed: List[str] = []
    metadata_list: List[Dict[str, Any]] = []
    vector_ids: List[str] = []

    expires_at = time.time() + (ttl_hours * 3600)
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for item in items:
        if category == "hotel":
            amenities = ",".join(item.get("amenities", []) or [])
            text_payload = (
                f"Hotel: {item.get('hotel_name', '')}."
                f"Description: {item.get('description', '')}."
                f"Amenities: {amenities}."
            )

        elif category == "activity":
            text_payload = (
                f"Activity: {item.get('name', '')}. "
                f"Type: {item.get('type', '')}. "
                f"Description: {item.get('description', '')}."
            )
        elif category == "flight":
            text_payload = (
                f"Flight {item.get('flight_number', '')} by "
                f"{item.get('airline', '')} departing "
                f"{item.get('departure_time', '')} arriving "
                f"{item.get('arrival_time', '')} price "
                f"{item.get('price', '')}."
            )
        else:
            text_payload = json.dumps(item, default=str, ensure_ascii=False)

        texts_to_embed.append(text_payload)

        # store original item and filtering tags
        metadata = {
            "session_id": session_id,
            "category": category,
            "raw_data": json.dumps(item, default=str, ensure_ascii=False),
            "expires_at": expires_at,
            "created_at": created_at,
        }

        metadata_list.append(metadata)
        # Stable IDs make repeated indexing idempotent instead of creating a
        # duplicate vector on every refresh.
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
            identity = json.dumps(item, sort_keys=True, default=str, ensure_ascii=False)
        raw_key = f"{session_id}|{category}|{identity}"
        vector_ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_key)))

    try:
        # embed in bulk
        vectors = _embed_texts(texts_to_embed, input_type="passage")
        if len(vectors) != len(texts_to_embed):
            raise RuntimeError(
                f"Embedding count mismatch: {len(vectors)} != {len(texts_to_embed)}"
            )

        # prepare records for Pinecone
        records = [
            {"id": vid, "values": vec, "metadata": meta}
            for vid, vec, meta in zip(vector_ids, vectors, metadata_list)
        ]

        # upsert data in batch
        total_batches = (len(records) + UPSERT_BATCH - 1) // UPSERT_BATCH
        for i in range(0, len(records), UPSERT_BATCH):
            batch = records[i : i + UPSERT_BATCH]
            batch_num = i // UPSERT_BATCH + 1
            for attempt in range(1, MAX_EMBED_RETIRES + 1):
                try:
                    index.upsert(vectors=batch, namespace=session_id)
                    break
                except PineconeException as e:
                    if attempt == MAX_EMBED_RETIRES:
                        raise
                    wait = min(RETRY_BASE_WAIT * (2 ** (attempt - 1)), 10)
                    logger.warning(
                        f"Upsert batch {batch_num}/{total_batches} attempt "
                        f"{attempt} failed: {e}; retry in {wait}s"
                    )
                    time.sleep(wait)

        logger.info(
            f"Successfully upserted {len(records)} {category} records for session {session_id}"
        )
        return True

    except Exception as e:
        logger.error(f"Vector upsert failed for {category}: {e}", exc_info=True)
        return False


# vector search (read and re-rank)
def search_personalised_options(
    session_id: str, user_preferences: str, category: str, top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Semantic search over the user's session namespace.

    Args:
        session_id:        restricts search to this namespace.
        user_preferences:  natural-language query (e.g. "quiet hotel with pool").
        category:          "hotel" | "activity" | "flight".
        top_k:             number of matches to return.
    Returns:
        Ranked list of original item dicts with a `relevance_score` field.
    """
    index = _get_or_create_index()
    if not index or not user_preferences.strip():
        return []

    try:
        # embed user's search query
        query_vector = _embed_texts([user_preferences], input_type="query")[0]

        # query with metadata filtering and session namespace
        search_results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=session_id,
            filter={"category": {"$eq": category}, "expires_at": {"$gt": time.time()}},
        )

        # reconstruct results
        ranked_items: List[Dict[str, Any]] = []
        for match in search_results.get("matches", []):
            meta = match.get("metadata") or {}
            try:
                # convert raw_data string into dictionary
                raw_dict = json.loads(meta.get("raw_data", "{}"))
            except (ValueError, SyntaxError):
                continue

            # attach similarity score
            raw_dict["relevance_score"] = round(float(match.get("score") or 0.0), 3)
            ranked_items.append(raw_dict)

        return ranked_items

    except Exception as e:
        logger.error(
            f"Vector search failed for '{user_preferences}': {e}", exc_info=True
        )
        return []


# cleanup
def clear_session_vectors(session_id: str) -> bool:
    """
    Deletes all vectors associated with a specific session ID to prevent memory bloat
    and maintain a clean context window for new trips.
    """
    index = _get_or_create_index()
    if not index:
        return False

    try:
        index.delete(delete_all=True, namespace=session_id)
        logger.info(f"Cleared vector namespace for session {session_id}")
        return True
    except Exception as e:
        logger.error(
            f"Failed to clear session vectors for {session_id}: {e}", exc_info=True
        )
        return False


def clear_expired_vectors(namespace: Optional[str] = None) -> int:
    """
    Best-effort sweep of expired vectors (call periodically, e.g. cron).
    If `namespace` is None, scans the default namespace.
    Returns number of vectors flagged for deletion (approx).
    """
    index = _get_or_create_index()
    if not index:
        return 0

    try:
        now = time.time()

        deleted = 0
        for ids in index.list(namespace=namespace or ""):
            if not ids:
                break

            # fetch metadata to inspect expires_at
            fetch_resp = index.fetch(ids=ids, namespace=namespace or "")
            expired_ids = [
                v_id
                for v_id, v in fetch_resp.vectors.items()
                if (v.metadata or {}).get("expires_at", now) < now
            ]

            # fetch metadata to check expires_at
            if expired_ids:
                index.delete(ids=expired_ids, namespace=namespace or "")
                deleted += len(expired_ids)

        if deleted:
            logger.info(f"Swept {deleted} expired vectors (ns={namespace})")
        return deleted
    except Exception as e:
        logger.error(f"Expired-vector sweep failed: {e}", exc_info=True)
        return 0
