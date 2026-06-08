from __future__ import annotations

import uuid
from datetime import date

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from app.api.deps import CurrentUser, DbSession
from app.models.activity import ReadingActivity, ReadingActivityType
from app.models.library import LibraryEntry, ReadingStatus
from app.models.manga import Manga
from app.schemas.common import Page
from app.schemas.library import LibraryEntryCreate, LibraryEntryRead, LibraryEntryUpdate

router = APIRouter(prefix="/library", tags=["library"])


async def _get_owned_entry(db: DbSession, current_user: CurrentUser, entry_id: uuid.UUID) -> LibraryEntry:
    result = await db.execute(
        select(LibraryEntry)
        .where(LibraryEntry.id == entry_id, LibraryEntry.user_id == current_user.id)
        .options(selectinload(LibraryEntry.manga))
    )
    entry = result.scalar_one_or_none()
    if entry is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Library entry not found")
    return entry


@router.get("", response_model=Page[LibraryEntryRead])
async def list_library(
    db: DbSession,
    current_user: CurrentUser,
    status_filter: ReadingStatus | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> Page[LibraryEntryRead]:
    query = (
        select(LibraryEntry)
        .where(LibraryEntry.user_id == current_user.id)
        .options(selectinload(LibraryEntry.manga))
    )
    if status_filter is not None:
        query = query.where(LibraryEntry.status == status_filter)

    count_result = await db.execute(query.with_only_columns(LibraryEntry.id))
    total = len(count_result.all())

    result = await db.execute(query.order_by(LibraryEntry.updated_at.desc()).limit(limit).offset(offset))
    items = list(result.scalars().all())

    return Page(items=[LibraryEntryRead.model_validate(e) for e in items], total=total, limit=limit, offset=offset)


@router.post("", response_model=LibraryEntryRead, status_code=status.HTTP_201_CREATED)
async def add_to_library(payload: LibraryEntryCreate, db: DbSession, current_user: CurrentUser) -> LibraryEntry:
    manga = await db.get(Manga, payload.manga_id)
    if manga is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Manga not found")

    entry = LibraryEntry(
        user_id=current_user.id,
        manga_id=manga.id,
        status=payload.status,
        score=payload.score,
        progress_chapter=payload.progress_chapter,
        progress_volume=payload.progress_volume,
        notes=payload.notes,
    )
    db.add(entry)
    try:
        await db.flush()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="This manga is already in your library"
        ) from exc

    db.add(
        ReadingActivity(
            user_id=current_user.id,
            manga_id=manga.id,
            activity_type=ReadingActivityType.STATUS_CHANGE,
            payload={"from": None, "to": entry.status.value, "source": "manual_add"},
        )
    )
    await db.commit()

    return await _get_owned_entry(db, current_user, entry.id)


@router.patch("/{entry_id}", response_model=LibraryEntryRead)
async def update_library_entry(
    entry_id: uuid.UUID, payload: LibraryEntryUpdate, db: DbSession, current_user: CurrentUser
) -> LibraryEntry:
    entry = await _get_owned_entry(db, current_user, entry_id)
    changes = payload.model_dump(exclude_unset=True)

    activities: list[tuple[ReadingActivityType, dict]] = []

    if "status" in changes and changes["status"] != entry.status:
        activities.append((ReadingActivityType.STATUS_CHANGE, {"from": entry.status.value, "to": changes["status"].value}))
        if changes["status"] == ReadingStatus.COMPLETED:
            activities.append((ReadingActivityType.COMPLETED, {"chapter": entry.progress_chapter}))
            entry.finished_at = entry.finished_at or date.today()
        elif changes["status"] == ReadingStatus.READING and entry.status == ReadingStatus.PLAN_TO_READ:
            activities.append((ReadingActivityType.STARTED, {}))
            entry.started_at = entry.started_at or date.today()

    if "score" in changes and changes["score"] != entry.score:
        activities.append((ReadingActivityType.SCORE_CHANGE, {"from": entry.score, "to": changes["score"]}))

    if "progress_chapter" in changes and changes["progress_chapter"] != entry.progress_chapter:
        activities.append(
            (ReadingActivityType.PROGRESS_UPDATE, {"from": entry.progress_chapter, "to": changes["progress_chapter"]})
        )

    for field, value in changes.items():
        setattr(entry, field, value)

    for activity_type, payload_data in activities:
        db.add(
            ReadingActivity(
                user_id=current_user.id,
                manga_id=entry.manga_id,
                activity_type=activity_type,
                payload=payload_data,
            )
        )

    await db.commit()
    return await _get_owned_entry(db, current_user, entry.id)


@router.delete("/{entry_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_from_library(entry_id: uuid.UUID, db: DbSession, current_user: CurrentUser) -> None:
    entry = await _get_owned_entry(db, current_user, entry_id)
    await db.delete(entry)
    await db.commit()
