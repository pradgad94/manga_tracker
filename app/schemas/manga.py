from __future__ import annotations

import uuid
from datetime import date

from app.schemas.common import ORMModel


class MangaRead(ORMModel):
    id: uuid.UUID
    mal_id: int
    title: str
    title_english: str | None
    synopsis: str | None
    media_type: str | None
    status: str | None
    genres: list[str] | None
    authors: list[str] | None
    num_volumes: int | None
    num_chapters: int | None
    mal_mean_score: float | None
    start_date: date | None
    end_date: date | None
    main_picture_url: str | None


class MangaSummary(ORMModel):
    """Lightweight projection used in lists, search results, and recommendations."""

    id: uuid.UUID
    mal_id: int
    title: str
    media_type: str | None
    genres: list[str] | None
    mal_mean_score: float | None
    main_picture_url: str | None
