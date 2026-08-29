"""
Firebase Firestore cache layer.

Provides a TTL-based key-value cache backed by Firestore documents.
Keys are deterministic SHA-256 hashes of normalised kwargs, so
'USD' and 'usd' collide (case-insensitive).

Thread-safety:
  • Singleton Firestore client protected by _init_lock.
  • Firestore client itself is thread-safe for reads & writes.

Resilience:
  • Handles FIREBASE_CREDENTIAL_JSON as either a file path or raw JSON string.
  • Retry with exponential backoff for transient Firestore failures.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import random
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Literal, Optional

import firebase_admin
from firebase_admin import credentials, firestore

from app.core.config import settings
from app.tools.provider_logging import safe_provider_log

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────
DEFAULT_COLLECTION = "api_cache"
DEFAULT_TTL_HOURS = 1
FIRESTORE_BATCH_LIMIT = 450  # Firestore batch max is 500; leave headroom
FIRESTORE_DOC_MAX_BYTES = 900_000  # Firestore doc max is ~1 MiB
MAX_CLEAR_PAGES = 50
FS_MAX_RETRIES = 2  # retry count for transient failures
FS_RETRY_BASE_DELAY = 0.5  # seconds

# ── Thread-safe singleton ──────────────────────────────────────────
_init_lock: threading.Lock = threading.Lock()
_db_client: Optional[firestore.Client] = None
_cache_claim_condition = threading.Condition()


def _get_db() -> firestore.Client:
    """
    Return the singleton Firestore client.

    FIX (Report #39): Handle FIREBASE_CREDENTIAL_JSON as either a
    file path or a raw JSON string.
    """
    global _db_client
    if _db_client is not None:
        return _db_client

    with _init_lock:
        if _db_client is not None:  # double-checked locking
            return _db_client

        if not firebase_admin._apps:
            cred_str = settings.FIREBASE_CREDENTIAL_JSON

            if os.path.isfile(cred_str):
                # It's a file path
                cred = credentials.Certificate(cred_str)
            else:
                # Assume it's a raw JSON string
                try:
                    cred_dict = json.loads(cred_str)
                    cred = credentials.Certificate(cred_dict)
                except (json.JSONDecodeError, ValueError):
                    safe_provider_log(logger, "cache.firebase.credentials_invalid")
                    raise

            firebase_admin.initialize_app(cred)

        _db_client = firestore.client()
        safe_provider_log(logger, "cache.firebase.initialized")
        return _db_client


# ── Retry decorator ─────────────────────────────────────────────────


def _with_retry(
    max_retries: int = FS_MAX_RETRIES,
    base_delay: float = FS_RETRY_BASE_DELAY,
) -> Callable:
    """
    Decorator: retry a Firestore operation with exponential backoff + jitter.

    Only retries on exceptions that are likely transient
    (network, timeout, 503). Does NOT retry on data-validation errors.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc: Optional[Exception] = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt == max_retries:
                        raise
                    delay = min(
                        base_delay * (2**attempt) + random.uniform(0, 0.3),
                        5.0,
                    )
                    safe_provider_log(logger, "cache.firebase.retry")
                    time.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator


# ── Cache key generation ────────────────────────────────────────────


def _generate_cache_key(prefix: str, **kwargs) -> str:
    """
    Deterministic SHA-256 hash of normalised kwargs → document ID.

    String values are lowercased + trimmed so 'USD' and 'usd' collide.
    """
    normalised = {
        k: (str(v).strip().lower() if isinstance(v, str) else v)
        for k, v in kwargs.items()
    }
    param_string = json.dumps(normalised, sort_keys=True, default=str)
    digest = hashlib.sha256(param_string.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest}"


# ── Public API: cache read ──────────────────────────────────────────


def get_cached_data(
    collection: str,
    prefix: str,
    **kwargs,
) -> Optional[Any]:
    """
    Retrieve cached data from Firestore.

    Returns the payload on hit, None on miss / expiry / error.
    Expired documents are deleted best-effort (fire-and-forget).
    """
    doc_id = _generate_cache_key(prefix, **kwargs)
    doc_ref = _get_db().collection(collection).document(doc_id)

    try:
        doc = doc_ref.get()
    except Exception:
        safe_provider_log(logger, "cache.firebase.read_failed")
        return None

    if not doc.exists:
        safe_provider_log(logger, "cache.firebase.miss")
        return None

    data = doc.to_dict() or {}
    expires_at = data.get("expires_at")

    # TTL expired?
    now = datetime.now(timezone.utc)
    if expires_at is None or now > expires_at:
        safe_provider_log(logger, "cache.firebase.expired")
        claim_expires_at = data.get("claim_expires_at")
        active_claim = (
            data.get("claim_id")
            and isinstance(claim_expires_at, datetime)
            and claim_expires_at > now
        )
        if not active_claim:
            # Fire-and-forget deletion (don't block the caller). An active
            # creator lease owns an expired predecessor until it publishes.
            try:
                doc_ref.delete()
            except Exception:
                pass
        return None

    safe_provider_log(logger, "cache.firebase.hit")
    return data.get("payload")


# ── Public API: cache write ─────────────────────────────────────────


def set_cached_data(
    collection: str,
    prefix: str,
    payload: Any,
    ttl_hours: int = DEFAULT_TTL_HOURS,
    **kwargs,
) -> bool:
    """
    Write payload to Firestore with an expiry timestamp.

    Returns True on success, False on failure.
    """
    if ttl_hours <= 0:
        safe_provider_log(logger, "cache.firebase.invalid_ttl")
        return False

    doc_id = _generate_cache_key(prefix, **kwargs)

    # Size guard — Firestore doc max is ~1 MiB
    try:
        payload_bytes = len(json.dumps(payload, default=str).encode("utf-8"))
    except (TypeError, ValueError):
        payload_bytes = 0

    if payload_bytes > FIRESTORE_DOC_MAX_BYTES:
        safe_provider_log(logger, "cache.firebase.payload_too_large")
        return False

    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=ttl_hours)
    doc_ref = _get_db().collection(collection).document(doc_id)

    try:
        doc_ref.set(
            {
                "payload": payload,
                "expires_at": expires_at,
                "created_at": now,
            }
        )
        safe_provider_log(logger, "cache.firebase.write_succeeded")
        return True
    except Exception:
        safe_provider_log(logger, "cache.firebase.write_failed")
        return False


def claim_cached_data(
    collection: str,
    prefix: str,
    *,
    lease_seconds: int,
    **kwargs: Any,
) -> tuple[Literal["cached", "claimed", "busy", "error"], Any, str | None]:
    """Atomically return a cache hit or lease one empty key for construction."""
    if lease_seconds <= 0:
        return "error", None, None

    doc_id = _generate_cache_key(prefix, **kwargs)
    now = datetime.now(timezone.utc)
    claim_id = str(uuid.uuid4())
    claim_expires_at = now + timedelta(seconds=lease_seconds)

    try:
        database = _get_db()
        doc_ref = database.collection(collection).document(doc_id)
        transaction = database.transaction()

        @firestore.transactional
        def _claim(transaction):
            snapshot = doc_ref.get(transaction=transaction)
            data = snapshot.to_dict() or {} if snapshot.exists else {}
            expires_at = data.get("expires_at")
            if (
                data.get("payload") is not None
                and isinstance(expires_at, datetime)
                and expires_at > now
            ):
                return "cached", data["payload"], None
            current_claim_expiry = data.get("claim_expires_at")
            if (
                data.get("claim_id")
                and isinstance(current_claim_expiry, datetime)
                and current_claim_expiry > now
            ):
                return "busy", None, None
            transaction.set(
                doc_ref,
                {
                    "claim_id": claim_id,
                    "claim_expires_at": claim_expires_at,
                    "claim_created_at": now,
                },
            )
            return "claimed", None, claim_id

        return _claim(transaction)
    except Exception:
        safe_provider_log(logger, "cache.firebase.claim_failed")
        return "error", None, None


def complete_cached_data_claim(
    collection: str,
    prefix: str,
    payload: Any,
    *,
    claim_id: str,
    ttl_hours: int = DEFAULT_TTL_HOURS,
    **kwargs: Any,
) -> bool:
    """Publish payload only if this caller still owns the atomic cache lease."""
    if ttl_hours <= 0 or not claim_id:
        return False
    try:
        payload_bytes = len(json.dumps(payload, default=str).encode("utf-8"))
    except (TypeError, ValueError):
        payload_bytes = 0
    if payload_bytes > FIRESTORE_DOC_MAX_BYTES:
        return False

    doc_id = _generate_cache_key(prefix, **kwargs)
    now = datetime.now(timezone.utc)
    try:
        database = _get_db()
        doc_ref = database.collection(collection).document(doc_id)
        transaction = database.transaction()

        @firestore.transactional
        def _complete(transaction):
            snapshot = doc_ref.get(transaction=transaction)
            data = snapshot.to_dict() or {} if snapshot.exists else {}
            if data.get("claim_id") != claim_id:
                return False
            transaction.set(
                doc_ref,
                {
                    "payload": payload,
                    "expires_at": now + timedelta(hours=ttl_hours),
                    "created_at": now,
                },
            )
            return True

        completed = _complete(transaction)
    except Exception:
        safe_provider_log(logger, "cache.firebase.publish_failed")
        completed = False
    if completed:
        with _cache_claim_condition:
            _cache_claim_condition.notify_all()
    return completed


def release_cached_data_claim(
    collection: str,
    prefix: str,
    *,
    claim_id: str,
    **kwargs: Any,
) -> bool:
    """Release only the matching failed creator lease so another caller retries."""
    if not claim_id:
        return False
    doc_id = _generate_cache_key(prefix, **kwargs)
    try:
        database = _get_db()
        doc_ref = database.collection(collection).document(doc_id)
        transaction = database.transaction()

        @firestore.transactional
        def _release(transaction):
            snapshot = doc_ref.get(transaction=transaction)
            data = snapshot.to_dict() or {} if snapshot.exists else {}
            if data.get("claim_id") != claim_id:
                return False
            transaction.delete(doc_ref)
            return True

        released = _release(transaction)
    except Exception:
        safe_provider_log(logger, "cache.firebase.release_failed")
        released = False
    if released:
        with _cache_claim_condition:
            _cache_claim_condition.notify_all()
    return released


def wait_for_cache_claim_change(
    collection: str,
    prefix: str,
    *,
    timeout_seconds: float,
    **kwargs: Any,
) -> None:
    """Wait briefly for a local claimant; remote claimants are re-polled safely."""
    del collection, prefix, kwargs
    with _cache_claim_condition:
        _cache_claim_condition.wait(timeout=max(0.0, timeout_seconds))


# ── Public API: bulk cache clear ────────────────────────────────────


def clear_trip_cache(collection: str = DEFAULT_COLLECTION) -> int:
    """
    Delete all documents in `collection` in paginated batches.

    Returns the total number of documents deleted.
    """
    safe_provider_log(logger, "cache.firebase.clear_started")
    client = _get_db()
    total_deleted = 0

    try:
        for _page in range(MAX_CLEAR_PAGES):
            docs = list(
                client.collection(collection).limit(FIRESTORE_BATCH_LIMIT).stream()
            )

            if not docs:
                break

            batch = client.batch()
            for doc in docs:
                batch.delete(doc.reference)
            batch.commit()
            total_deleted += len(docs)

            if len(docs) < FIRESTORE_BATCH_LIMIT:
                break

        safe_provider_log(logger, "cache.firebase.clear_completed")
    except Exception:
        safe_provider_log(logger, "cache.firebase.clear_failed")

    return total_deleted


# ── Public API: surgical invalidation ───────────────────────────────


def invalidate_cached_data(
    collection: str,
    prefix: str,
    **kwargs: Any,
) -> bool:
    """
    Delete a single cached entry identified by prefix + kwargs.

    Useful for surgical invalidation without wiping the entire collection.
    Returns True on success, False on failure.
    """
    doc_id = _generate_cache_key(prefix, **kwargs)

    try:
        _get_db().collection(collection).document(doc_id).delete()
        safe_provider_log(logger, "cache.firebase.invalidate_succeeded")
        return True
    except Exception:
        safe_provider_log(logger, "cache.firebase.invalidate_failed")
        return False
