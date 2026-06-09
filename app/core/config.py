"""Centralized application settings, loaded from environment variables / .env."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- App ---
    app_name: str = "Manga Tracker API"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    api_v1_prefix: str = "/api/v1"

    # --- Security / JWT ---
    secret_key: str = "change-me-to-a-long-random-string"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 30
    jwt_algorithm: str = "HS256"

    # --- Database ---
    database_url: str = "postgresql+asyncpg://manga:manga@localhost:5432/manga_tracker"
    database_echo: bool = False

    # --- Redis ---
    redis_enabled: bool = True
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600

    # --- MyAnimeList (single-user sync) ---
    mal_client_id: str = ""
    mal_client_secret: str = ""
    mal_username: str = ""
    mal_access_token: str = ""
    mal_refresh_token: str = ""

    # --- Google Gemini (LLM provider: text generation + embeddings) ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"
    gemini_thinking_level: Literal["minimal", "low", "medium", "high"] = "high"

    # --- Embeddings provider ---
    # Gemini has a first-party embeddings endpoint, so the same API key and SDK
    # cover both generation and embeddings — no separate embeddings vendor needed.
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 1536

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    """Settings are cached for the lifetime of the process."""
    return Settings()
