from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import ColumnElement, any_, func, select

from app.api.deps import DbSession
from app.models.manga import Manga
from app.schemas.common import Page
from app.schemas.manga import MangaRead, MangaSummary

router = APIRouter(prefix="/manga", tags=["manga"])


@router.get("", response_model=Page[MangaSummary])
async def list_manga(
    db: DbSession,
    q: str | None = Query(default=None, description="Filter by title (case-insensitive substring)"),
    genre: str | None = Query(default=None, description="Filter by genre (exact match)"),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> Page[MangaSummary]:
    filters: list[ColumnElement[bool]] = []
    if q:
        filters.append(Manga.title.ilike(f"%{q}%"))
    if genre:
        filters.append(any_(Manga.genres) == genre)

    base = select(Manga)
    count_query = select(func.count()).select_from(Manga)
    for condition in filters:
        base = base.where(condition)
        count_query = count_query.where(condition)

    total = (await db.execute(count_query)).scalar_one()
    result = await db.execute(base.order_by(Manga.title).limit(limit).offset(offset))
    items = list(result.scalars().all())

    return Page(items=[MangaSummary.model_validate(m) for m in items], total=total, limit=limit, offset=offset)


@router.get("/{manga_id}", response_model=MangaRead)
async def get_manga(manga_id: uuid.UUID, db: DbSession) -> Manga:
    manga = await db.get(Manga, manga_id)
    if manga is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Manga not found")
    return manga
