from __future__ import annotations

import uuid
from datetime import date

from pydantic import BaseModel, Field

from app.models.library import ReadingStatus
from app.schemas.common import ORMModel
from app.schemas.manga import MangaSummary


class LibraryEntryCreate(BaseModel):
    manga_id: uuid.UUID
    status: ReadingStatus = ReadingStatus.PLAN_TO_READ
    score: int | None = Field(default=None, ge=1, le=10)
    progress_chapter: int = Field(default=0, ge=0)
    progress_volume: int = Field(default=0, ge=0)
    notes: str | None = None


class LibraryEntryUpdate(BaseModel):
    status: ReadingStatus | None = None
    score: int | None = Field(default=None, ge=1, le=10)
    progress_chapter: int | None = Field(default=None, ge=0)
    progress_volume: int | None = Field(default=None, ge=0)
    is_favorite: bool | None = None
    started_at: date | None = None
    finished_at: date | None = None
    notes: str | None = None


class LibraryEntryRead(ORMModel):
    id: uuid.UUID
    manga: MangaSummary
    status: ReadingStatus
    score: int | None
    progress_chapter: int
    progress_volume: int
    times_reread: int
    is_favorite: bool
    started_at: date | None
    finished_at: date | None
    notes: str | None
