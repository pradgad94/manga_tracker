"""
Thin async JSON cache over Redis, used to avoid re-running expensive LLM-backed
operations (recommendations, search, habit analysis) on every request.

Caching is explicitly optional per the spec ("Redis (optional caching)") — when
`REDIS_ENABLED=false` or Redis is unreachable, every method becomes a no-op and
callers transparently fall through to computing fresh results. Application code
should never need to branch on whether caching is enabled.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from functools import lru_cache
from typing import TypeVar

from pydantic import BaseModel
from redis import asyncio as redis_asyncio

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class CacheClient:
    def __init__(self, settings: Settings) -> None:
        self._enabled = settings.redis_enabled
        self._default_ttl = settings.cache_ttl_seconds
        self._redis: redis_asyncio.Redis | None = (
            redis_asyncio.from_url(settings.redis_url, decode_responses=True) if self._enabled else None
        )

    async def aclose(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()

    async def get_model(self, key: str, model: type[T]) -> T | None:
        raw = await self._safe_get(key)
        if raw is None:
            return None
        try:
            return model.model_validate_json(raw)
        except Exception:
            logger.warning("cache_deserialize_failed", key=key)
            return None

    async def set_model(self, key: str, value: BaseModel, *, ttl: int | None = None) -> None:
        await self._safe_set(key, value.model_dump_json(), ttl=ttl)

    async def get_or_compute(
        self, key: str, model: type[T], compute: Callable[[], Awaitable[T]], *, ttl: int | None = None
    ) -> T:
        cached = await self.get_model(key, model)
        if cached is not None:
            logger.debug("cache_hit", key=key)
            return cached

        value = await compute()
        await self.set_model(key, value, ttl=ttl)
        return value

    async def invalidate(self, key: str) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.delete(key)
        except Exception:
            logger.warning("cache_invalidate_failed", key=key, exc_info=True)

    # --- internals: every Redis call is best-effort — cache failures must never
    # surface as user-facing errors, since caching is purely an optimization. ---

    async def _safe_get(self, key: str) -> str | None:
        if self._redis is None:
            return None
        try:
            return await self._redis.get(key)
        except Exception:
            logger.warning("cache_get_failed", key=key, exc_info=True)
            return None

    async def _safe_set(self, key: str, value: str, *, ttl: int | None) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.set(key, value, ex=ttl if ttl is not None else self._default_ttl)
        except Exception:
            logger.warning("cache_set_failed", key=key, exc_info=True)


def cache_key(*parts: str) -> str:
    return ":".join(["manga_api", *parts])


@lru_cache
def get_cache_client() -> CacheClient:
    return CacheClient(get_settings())
