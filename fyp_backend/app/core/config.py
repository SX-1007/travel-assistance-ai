"""
Application configuration — loaded from environment / .env file.

Uses Pydantic BaseSettings for:
  • Type validation at startup (fail-fast)
  • Automatic .env loading
  • Computed fallback properties for optional Gemini sub-keys
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings singleton — validates all required env vars at import time."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # ── Gemini API keys ──────────────────────────────────────────────
    GEMINI_API_KEY: str = Field(
        ..., min_length=1, description="Primary Gemini API key (required)"
    )
    GEMINI_IATA_API: str
    GEMINI_FLIGHTS_HOTELS: str
    GEMINI_EXTRACTOR_API: str
    GEMINI_EMBED_API: str

    # ------------------- Model Configuration ------------------
    GEMINI_CHAT_MODEL: str = "gemini-3.1-flash-lite"
    GEMINI_UTILITY_MODEL: str = "gemini-3.1-flash-lite"
    GEMINI_EMBED_MODEL: str = "gemini-embedding-2"

    # ── Supabase ─────────────────────────────────────────────────────
    SUPABASE_URL: str = Field(..., min_length=1)
    SUPABASE_API_KEY: str = Field(..., min_length=1)
    SUPABASE_POSTGRES_URI: str = Field(
        ..., min_length=1, description="PostgreSQL connection string for pgvector"
    )

    # ── Firebase ─────────────────────────────────────────────────────
    FIREBASE_CREDENTIAL_JSON: str = Field(
        ..., min_length=1, description="Path to JSON file or raw JSON string"
    )

    # ── External APIs ────────────────────────────────────────────────
    MAPBOX_TOKEN: str = Field(..., min_length=1)
    SERPAPI_FLIGHT: str = Field(..., min_length=1)
    SERPAPI_HOTEL: str = Field(..., min_length=1)
    SERPAPI_PLACE: str = Field(..., min_length=1)
    FREECURRENCY_API: str = Field(..., min_length=1)
    # Legacy/optional vector-store fallback. The primary implementation uses
    # Supabase pgvector, so absence must not prevent the application booting.
    PINECONE_API_KEY: str = ""

    # ── Computed helpers (fallback to primary Gemini key) ────────────
    @property
    def effective_embed_api_key(self) -> str:
        """Embedding API key — falls back to primary Gemini key."""
        return self.GEMINI_EMBED_API or self.GEMINI_API_KEY

    @property
    def effective_flights_hotels_api_key(self) -> str:
        return self.GEMINI_FLIGHTS_HOTELS or self.GEMINI_API_KEY

    @property
    def effective_extractor_api_key(self) -> str:
        return self.GEMINI_EXTRACTOR_API or self.GEMINI_API_KEY


settings = Settings()
