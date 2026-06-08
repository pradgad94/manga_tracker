from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class MALConnectRequest(BaseModel):
    """Body for POST /sync/mal/connect — see scripts/mal_auth.py for how to obtain these."""

    mal_username: str = Field(min_length=1, max_length=120)
    access_token: str
    refresh_token: str
    expires_in: int = Field(description="Seconds until the access token expires, as returned by MAL's token endpoint")


class SyncStatus(BaseModel):
    mal_username: str | None
    last_synced_at: datetime | None
    last_sync_status: str | None
    last_sync_error: str | None


class SyncResult(BaseModel):
    manga_created: int
    manga_updated: int
    entries_created: int
    entries_updated: int
    activities_logged: int
    started_at: datetime
    finished_at: datetime
