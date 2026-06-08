"""
Single-user MAL -> local DB sync.

Pulls the operator's manga list from MyAnimeList, upserts canonical `Manga` rows
and the operator's `LibraryEntry` rows, and appends `ReadingActivity` events for
anything that changed since the last sync — that activity log is what lets the
taste-profile pipeline see *how* preferences evolved, not just where they ended up.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.activity import ReadingActivity, ReadingActivityType
from app.models.library import LibraryEntry, ReadingStatus
from app.models.mal_account import MALAccount
from app.models.manga import Manga
from app.schemas.sync import SyncResult
from app.services.mal.client import MALClient
from app.services.mal.exceptions import MALAuthenticationError

logger = get_logger(__name__)

# MAL's list_status.status values line up 1:1 with ReadingStatus — both are MAL's vocabulary.
_VALID_STATUSES = {s.value for s in ReadingStatus}


class MALSyncService:
    def __init__(self, client: MALClient) -> None:
        self._client = client

    async def sync_user(self, db: AsyncSession, account: MALAccount) -> SyncResult:
        started_at = datetime.now(timezone.utc)
        access_token = await self._ensure_fresh_token(db, account)

        manga_created = manga_updated = entries_created = entries_updated = activities_logged = 0

        try:
            async for entry in self._client.iter_user_manga_list(access_token, username=account.mal_username):
                node = entry["node"]
                list_status = entry.get("list_status", {})

                manga, created = await self._upsert_manga(db, node)
                manga_created += int(created)
                manga_updated += int(not created)

                lib_entry, entry_created, logged = await self._upsert_library_entry(
                    db, user_id=account.user_id, manga=manga, list_status=list_status
                )
                entries_created += int(entry_created)
                entries_updated += int(not entry_created)
                activities_logged += logged

            account.last_sync_status = "success"
            account.last_sync_error = None
        except Exception as exc:
            account.last_sync_status = "error"
            account.last_sync_error = str(exc)
            raise
        finally:
            finished_at = datetime.now(timezone.utc)
            account.last_synced_at = finished_at
            await db.commit()

        logger.info(
            "mal_sync_complete",
            user_id=str(account.user_id),
            manga_created=manga_created,
            manga_updated=manga_updated,
            entries_created=entries_created,
            entries_updated=entries_updated,
            activities_logged=activities_logged,
        )

        return SyncResult(
            manga_created=manga_created,
            manga_updated=manga_updated,
            entries_created=entries_created,
            entries_updated=entries_updated,
            activities_logged=activities_logged,
            started_at=started_at,
            finished_at=finished_at,
        )

    # --- token lifecycle ---------------------------------------------------------

    async def _ensure_fresh_token(self, db: AsyncSession, account: MALAccount) -> str:
        if not account.access_token or not account.refresh_token:
            raise MALAuthenticationError(
                "No MAL tokens on file — connect the account first via POST /sync/mal/connect"
            )

        now = datetime.now(timezone.utc)
        expires_at = account.token_expires_at
        if expires_at is not None and expires_at > now:
            return account.access_token

        token_set = await self._client.refresh_access_token(account.refresh_token)
        account.access_token = token_set.access_token
        account.refresh_token = token_set.refresh_token
        account.token_expires_at = token_set.expires_at
        await db.flush()
        return account.access_token

    # --- manga upsert -------------------------------------------------------------

    async def _upsert_manga(self, db: AsyncSession, node: dict[str, Any]) -> tuple[Manga, bool]:
        mal_id = node["id"]
        result = await db.execute(select(Manga).where(Manga.mal_id == mal_id))
        manga = result.scalar_one_or_none()
        created = manga is None

        if manga is None:
            manga = Manga(mal_id=mal_id)
            db.add(manga)

        alt_titles = node.get("alternative_titles") or {}
        authors = [
            f"{a['node'].get('first_name', '')} {a['node'].get('last_name', '')}".strip()
            for a in node.get("authors", [])
            if a.get("node")
        ]

        manga.title = node.get("title", manga.title or "Untitled")
        manga.title_english = alt_titles.get("en") or None
        manga.title_japanese = alt_titles.get("ja") or None
        manga.alternative_titles = alt_titles.get("synonyms") or None
        manga.synopsis = node.get("synopsis")
        manga.background = node.get("background")
        manga.media_type = node.get("media_type")
        manga.status = node.get("status")
        manga.genres = [g["name"] for g in node.get("genres", [])] or None
        manga.authors = authors or None
        manga.num_volumes = node.get("num_volumes") or None
        manga.num_chapters = node.get("num_chapters") or None
        manga.mal_mean_score = node.get("mean")
        manga.mal_rank = node.get("rank")
        manga.mal_popularity = node.get("popularity")
        manga.start_date = _parse_date(node.get("start_date"))
        manga.end_date = _parse_date(node.get("end_date"))
        manga.main_picture_url = (node.get("main_picture") or {}).get("large") or (node.get("main_picture") or {}).get(
            "medium"
        )
        manga.mal_raw = node

        await db.flush()
        return manga, created

    # --- library entry upsert + activity log --------------------------------------

    async def _upsert_library_entry(
        self,
        db: AsyncSession,
        *,
        user_id: uuid.UUID,
        manga: Manga,
        list_status: dict[str, Any],
    ) -> tuple[LibraryEntry, bool, int]:
        result = await db.execute(
            select(LibraryEntry).where(LibraryEntry.user_id == user_id, LibraryEntry.manga_id == manga.id)
        )
        entry = result.scalar_one_or_none()
        created = entry is None

        if entry is None:
            entry = LibraryEntry(user_id=user_id, manga_id=manga.id, status=ReadingStatus.PLAN_TO_READ)
            db.add(entry)
            await db.flush()

        new_status = list_status.get("status")
        new_status = ReadingStatus(new_status) if new_status in _VALID_STATUSES else entry.status
        new_score = list_status.get("score") or None
        new_chapter = list_status.get("num_chapters_read", entry.progress_chapter)
        new_volume = list_status.get("num_volumes_read", entry.progress_volume)

        activities = list(
            _detect_activities(
                manga_id=manga.id,
                created=created,
                old_status=entry.status,
                new_status=new_status,
                old_score=entry.score,
                new_score=new_score,
                old_chapter=entry.progress_chapter,
                new_chapter=new_chapter,
            )
        )

        entry.status = new_status
        entry.score = new_score
        entry.progress_chapter = new_chapter
        entry.progress_volume = new_volume
        entry.times_reread = list_status.get("num_times_reread", entry.times_reread)
        entry.started_at = _parse_date(list_status.get("start_date"))
        entry.finished_at = _parse_date(list_status.get("finish_date"))
        entry.mal_list_status_raw = list_status
        entry.synced_with_mal_at = date.today()

        for activity_type, payload in activities:
            db.add(
                ReadingActivity(
                    user_id=user_id,
                    manga_id=manga.id,
                    activity_type=activity_type,
                    payload=payload,
                )
            )

        await db.flush()
        return entry, created, len(activities)


def _detect_activities(
    *,
    manga_id: uuid.UUID,
    created: bool,
    old_status: ReadingStatus,
    new_status: ReadingStatus,
    old_score: int | None,
    new_score: int | None,
    old_chapter: int,
    new_chapter: int,
):
    if created:
        yield (
            ReadingActivityType.SYNCED_FROM_MAL,
            {"status": new_status.value, "score": new_score, "chapter": new_chapter},
        )
        if new_status == ReadingStatus.READING:
            yield ReadingActivityType.STARTED, {"chapter": new_chapter}
        elif new_status == ReadingStatus.COMPLETED:
            yield ReadingActivityType.COMPLETED, {"chapter": new_chapter}
        return

    if old_status != new_status:
        yield ReadingActivityType.STATUS_CHANGE, {"from": old_status.value, "to": new_status.value}
        if new_status == ReadingStatus.READING and old_status == ReadingStatus.PLAN_TO_READ:
            yield ReadingActivityType.STARTED, {"chapter": new_chapter}
        elif new_status == ReadingStatus.COMPLETED:
            yield ReadingActivityType.COMPLETED, {"chapter": new_chapter}

    if old_score != new_score:
        yield ReadingActivityType.SCORE_CHANGE, {"from": old_score, "to": new_score, "manga_id": str(manga_id)}

    if new_chapter != old_chapter:
        yield ReadingActivityType.PROGRESS_UPDATE, {"from": old_chapter, "to": new_chapter}


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None
