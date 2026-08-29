"""Safe, bounded observability for external planning providers."""

from __future__ import annotations

import logging
from typing import Any, Final


_EVENT_LEVELS: Final[dict[str, int]] = {
    "provider.iata.invalid_country": logging.WARNING,
    "provider.iata.failed": logging.ERROR,
    "provider.serpapi.retry": logging.WARNING,
    "provider.serpapi.failed": logging.ERROR,
    "provider.flight.parse_failed": logging.ERROR,
    "provider.hotel.parse_failed": logging.ERROR,
    "provider.flight_hotel.search_failed": logging.ERROR,
    "provider.flight_hotel.no_outbound": logging.WARNING,
    "provider.flight_hotel.no_return": logging.WARNING,
    "provider.flight_hotel.no_hotel": logging.WARNING,
    "provider.attractions.brainstorm_failed": logging.ERROR,
    "provider.attractions.search_failed": logging.ERROR,
    "provider.attractions.api_budget_exhausted": logging.WARNING,
    "provider.attractions.out_of_area": logging.INFO,
    "provider.attractions.unverified": logging.INFO,
    "provider.attractions.locality_unverified": logging.INFO,
    "provider.attractions.no_ideas": logging.WARNING,
    "provider.attractions.destination_unresolved": logging.WARNING,
    "provider.attractions.unmatched_dropped": logging.INFO,
    "provider.attractions.rescue_brainstorm_failed": logging.WARNING,
    "provider.attractions.rescue_unmatched": logging.INFO,
    "provider.attractions.rescue_used": logging.INFO,
    "provider.attractions.rescue_exhausted": logging.WARNING,
    "provider.attractions.duplicates_removed": logging.INFO,
    "provider.attractions.populated": logging.INFO,
    "provider.mapbox.retry_status": logging.WARNING,
    "provider.mapbox.retry_transport": logging.WARNING,
    "provider.mapbox.failed": logging.ERROR,
    "provider.mapbox.geocode_status": logging.WARNING,
    "provider.mapbox.geocode_empty": logging.INFO,
    "provider.mapbox.route_status": logging.WARNING,
    "provider.mapbox.route_distance_capped": logging.INFO,
    "provider.mapbox.day_processing_failed": logging.ERROR,
    "provider.currency.country_resolution_failed": logging.WARNING,
    "provider.currency.fallback_failed": logging.WARNING,
    "provider.currency.primary_client_error": logging.WARNING,
    "provider.currency.primary_retry": logging.WARNING,
    "provider.currency.fallback_used": logging.INFO,
    "provider.currency.all_failed": logging.ERROR,
    "cache.firebase.credentials_invalid": logging.ERROR,
    "cache.firebase.initialized": logging.INFO,
    "cache.firebase.retry": logging.WARNING,
    "cache.firebase.read_failed": logging.WARNING,
    "cache.firebase.miss": logging.DEBUG,
    "cache.firebase.expired": logging.DEBUG,
    "cache.firebase.hit": logging.DEBUG,
    "cache.firebase.invalid_ttl": logging.WARNING,
    "cache.firebase.payload_too_large": logging.ERROR,
    "cache.firebase.write_succeeded": logging.DEBUG,
    "cache.firebase.write_failed": logging.ERROR,
    "cache.firebase.claim_failed": logging.ERROR,
    "cache.firebase.publish_failed": logging.ERROR,
    "cache.firebase.release_failed": logging.ERROR,
    "cache.firebase.clear_started": logging.INFO,
    "cache.firebase.clear_completed": logging.INFO,
    "cache.firebase.clear_failed": logging.ERROR,
    "cache.firebase.invalidate_succeeded": logging.DEBUG,
    "cache.firebase.invalidate_failed": logging.ERROR,
    "budget.cache_read_failed": logging.WARNING,
    "budget.cache_publish_failed": logging.WARNING,
    "budget.confirmed_cache_read_failed": logging.WARNING,
    "provider.planning_geocode.failed": logging.WARNING,
    "provider.planning_geocode.miss": logging.INFO,
    "provider.mapbox.nearby_failed": logging.WARNING,
    "storage.supabase.pool_opened": logging.INFO,
    "storage.supabase.pool_closed": logging.INFO,
    "storage.supabase.pool_close_failed": logging.ERROR,
    "storage.supabase.pool_health_failed": logging.ERROR,
    "storage.supabase.embedding_failed": logging.ERROR,
    "storage.supabase.embedding_retry": logging.WARNING,
    "storage.supabase.profile_fetch_failed": logging.ERROR,
    "storage.supabase.profile_created": logging.INFO,
    "storage.supabase.profile_create_failed": logging.ERROR,
    "storage.supabase.profile_updated": logging.INFO,
    "storage.supabase.profile_update_failed": logging.ERROR,
    "storage.supabase.itinerary_saved": logging.INFO,
    "storage.supabase.itinerary_save_empty": logging.WARNING,
    "storage.supabase.itinerary_save_failed": logging.ERROR,
    "storage.supabase.checkpointer_ready": logging.INFO,
    "storage.supabase.vector_extension_unavailable": logging.WARNING,
    "storage.supabase.vector_index_unavailable": logging.WARNING,
    "storage.supabase.vector_ready": logging.INFO,
    "storage.supabase.vector_extension_missing": logging.ERROR,
    "storage.supabase.vector_table_missing": logging.ERROR,
    "storage.supabase.vector_verify_failed": logging.ERROR,
    "storage.supabase.vector_upsert_succeeded": logging.INFO,
    "storage.supabase.vector_upsert_failed": logging.ERROR,
    "storage.supabase.vector_search_failed": logging.ERROR,
    "storage.supabase.vector_fetch_failed": logging.ERROR,
    "storage.supabase.vector_clear_succeeded": logging.INFO,
    "storage.supabase.vector_clear_failed": logging.ERROR,
    "storage.supabase.vector_sweep_succeeded": logging.INFO,
    "storage.supabase.vector_sweep_failed": logging.ERROR,
    "storage.supabase.mem0_insert_succeeded": logging.INFO,
    "storage.supabase.mem0_insert_failed": logging.ERROR,
    "storage.supabase.mem0_fetch_succeeded": logging.INFO,
    "storage.supabase.mem0_fetch_failed": logging.ERROR,
    "memory.extractor_chain_ready": logging.INFO,
    "memory.extractor.relational_failed": logging.ERROR,
    "memory.extractor.vector_failed": logging.ERROR,
    "memory.extractor.trivial": logging.DEBUG,
    "memory.extractor.completed": logging.DEBUG,
    "memory.extractor.failed": logging.ERROR,
    "memory.extractor.retry": logging.WARNING,
    "memory.extractor.started": logging.INFO,
    "memory.extractor.no_traits": logging.INFO,
    "memory.extractor.relational_queued": logging.INFO,
    "memory.extractor.vector_queued": logging.INFO,
    "memory.extractor.persistence_succeeded": logging.INFO,
    "memory.extractor.persistence_failed": logging.WARNING,
    "memory.extractor.persistence_raised": logging.ERROR,
    "storage.dependencies.pool_closed": logging.ERROR,
    "storage.dependencies.vector_unavailable": logging.WARNING,
    "storage.dependencies.health_failed": logging.ERROR,
    "storage.dependencies.checkpointer_failed": logging.ERROR,
    "storage.dependencies.checkpointer_cleanup_failed": logging.ERROR,
}


def _bounded_http_status(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        status_code = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return status_code if 100 <= status_code <= 599 else None


def safe_provider_log(
    target_logger: logging.Logger,
    event: str,
    *,
    status_code: Any = None,
) -> None:
    """Emit an allowlisted provider event without affecting provider flow."""
    level = _EVENT_LEVELS.get(event)
    if level is None:
        return
    try:
        bounded_status = _bounded_http_status(status_code)
        if bounded_status is None:
            target_logger.log(level, event)
        else:
            target_logger.log(level, "%s status_code=%d", event, bounded_status)
    except Exception:
        return
