"""
Thin async client over the public Jikan API (https://docs.api.jikan.moe/), an
unofficial but widely-relied-upon read-only REST wrapper around MyAnimeList's
website data.

Why Jikan and not the official MAL API: MAL's official v2 API
(https://myanimelist.net/apiconfig/references/api/v2), which `MALClient` talks to
for syncing, has no reviews endpoint — user-written reviews only exist on the
website itself. Jikan scrapes that and republishes it as JSON, including
`/manga/{id}/reviews`, which is the only practical source for this feature.

Unlike `MALClient`, Jikan is public and unauthenticated — no client ID, secret, or
per-user OAuth token, so this client carries no credentials and is safe to share
process-wide. It does enforce a public rate limit (~3 req/s, ~60 req/min); calls
are paced by `_REQUEST_INTERVAL_SECONDS` so a backfill sweeping many manga doesn't
trip it (a 429 there would just stall the whole batch on retries).

Verified live against `https://api.jikan.moe/v4/manga/{1,25}/reviews` — the response
shape used here (`data[].review/score/tags/is_spoiler/user.username`) reflects what
that endpoint actually returns, not a guess from the (sparser) published schema docs.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from app.services.jikan.exceptions import JikanError

_API_BASE = "https://api.jikan.moe/v4"
_REQUEST_INTERVAL_SECONDS = 0.7


class JikanClient:
    """Stateless, unauthenticated wrapper — Jikan is a public read-only API."""

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self._client = http_client or httpx.AsyncClient(timeout=30.0)
        self._owns_client = http_client is None
        self._throttle_lock = asyncio.Lock()
        self._last_request_at = 0.0

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def get_manga_reviews(self, mal_id: int) -> list[dict[str, Any]]:
        """
        Return the first page of community reviews for a manga, in the order Jikan
        serves them (MAL's own "reviews" ordering — a mix of recency and reception).
        That's plenty of signal for a digest without paging through a series with
        hundreds of reviews; callers that want fewer can simply slice the result.
        """
        payload = await self._get(f"{_API_BASE}/manga/{mal_id}/reviews", params={"page": 1})
        reviews = payload.get("data")
        return reviews if isinstance(reviews, list) else []

    async def _get(self, url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        await self._throttle()

        try:
            response = await self._client.get(url, params=params)
        except httpx.ConnectError as exc:
            raise JikanError(f"Could not reach the Jikan API ({url})") from exc
        except httpx.TimeoutException as exc:
            raise JikanError(f"Jikan API request timed out ({url})") from exc

        if response.status_code == 404:
            return {"data": []}
        if response.status_code >= 400:
            raise JikanError(f"Jikan API error for {url}: {response.status_code} {response.text[:200]}")

        try:
            return response.json()
        except ValueError as exc:
            raise JikanError(f"Jikan API returned a non-JSON response for {url}") from exc

    async def _throttle(self) -> None:
        """Serialize requests with a minimum gap — Jikan has no per-key quota to lean on."""
        async with self._throttle_lock:
            elapsed = time.monotonic() - self._last_request_at
            remaining = _REQUEST_INTERVAL_SECONDS - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining)
            self._last_request_at = time.monotonic()
