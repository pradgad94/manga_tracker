"""
Thin async client over the MyAnimeList API v2 (https://myanimelist.net/apiconfig/references/api/v2).

Scope note: this app syncs a single, operator-owned MAL account (see MALAccount).
The client only implements what that needs — reading the authenticated user's
manga list and manga details, plus refreshing the OAuth2 token. It deliberately
does not implement the full MAL API surface.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from app.services.mal.exceptions import MALAPIError, MALAuthenticationError

_API_BASE = "https://api.myanimelist.net/v2"
_OAUTH_TOKEN_URL = "https://myanimelist.net/v1/oauth2/token"

# Fields requested for each manga list entry. `list_status` carries the user's
# personal progress/score/status; the rest is canonical catalog metadata.
_LIST_FIELDS = (
    "list_status{status,score,num_chapters_read,num_volumes_read,is_rereading,"
    "num_times_reread,start_date,finish_date,updated_at},"
    "alternative_titles,synopsis,background,media_type,status,genres,authors{first_name,last_name},"
    "num_chapters,num_volumes,mean,rank,popularity,start_date,end_date,main_picture"
)

_LIST_PAGE_SIZE = 100


class MALTokenSet:
    __slots__ = ("access_token", "refresh_token", "expires_at")

    def __init__(self, access_token: str, refresh_token: str, expires_at: datetime) -> None:
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.expires_at = expires_at


class MALClient:
    """Stateless wrapper — callers pass the current access token on each call."""

    def __init__(self, client_id: str, client_secret: str, http_client: httpx.AsyncClient | None = None) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._client = http_client or httpx.AsyncClient(timeout=30.0)
        self._owns_client = http_client is None

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # --- OAuth2 -----------------------------------------------------------------

    async def exchange_authorization_code(self, code: str, code_verifier: str) -> MALTokenSet:
        """Exchange an authorization code (from the PKCE flow) for an initial token set."""
        return await self._request_token(
            grant_type="authorization_code",
            code=code,
            code_verifier=code_verifier,
        )

    async def refresh_access_token(self, refresh_token: str) -> MALTokenSet:
        return await self._request_token(grant_type="refresh_token", refresh_token=refresh_token)

    async def _request_token(self, **fields: str) -> MALTokenSet:
        data = {"client_id": self._client_id, "client_secret": self._client_secret, **fields}
        response = await self._client.post(_OAUTH_TOKEN_URL, data=data)

        if response.status_code == 401:
            raise MALAuthenticationError("MAL rejected the OAuth client credentials or token")
        if response.status_code >= 400:
            raise MALAPIError(f"MAL token endpoint error: {response.text}", response.status_code)

        payload = response.json()
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=payload["expires_in"])
        return MALTokenSet(
            access_token=payload["access_token"],
            refresh_token=payload["refresh_token"],
            expires_at=expires_at,
        )

    # --- Manga list --------------------------------------------------------------

    async def iter_user_manga_list(self, access_token: str, username: str = "@me") -> AsyncIterator[dict[str, Any]]:
        """
        Yield every entry `{"node": {...manga...}, "list_status": {...progress/score...}}`
        in the authenticated user's manga list, following MAL's cursor-based pagination.
        """
        url: str | None = f"{_API_BASE}/users/{username}/mangalist"
        params: dict[str, Any] | None = {"fields": _LIST_FIELDS, "limit": _LIST_PAGE_SIZE, "nsfw": "true"}

        while url is not None:
            response = await self._authed_get(url, access_token, params=params)
            payload = response.json()

            for entry in payload.get("data", []):
                yield entry

            url = payload.get("paging", {}).get("next")
            params = None  # the `next` URL already carries query params

    async def _authed_get(
        self, url: str, access_token: str, *, params: dict[str, Any] | None = None
    ) -> httpx.Response:
        response = await self._client.get(url, headers={"Authorization": f"Bearer {access_token}"}, params=params)

        if response.status_code == 401:
            raise MALAuthenticationError("MAL access token was rejected (likely expired)")
        if response.status_code >= 400:
            raise MALAPIError(f"MAL API error for {url}: {response.text}", response.status_code)

        return response
