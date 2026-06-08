from __future__ import annotations

import uuid
from datetime import date
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Date, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.manga import Manga
    from app.models.user import User


class ReadingStatus(StrEnum):
    PLAN_TO_READ = "plan_to_read"
    READING = "reading"
    COMPLETED = "completed"
    ON_HOLD = "on_hold"
    DROPPED = "dropped"


class LibraryEntry(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A user's personal tracking record for one manga: progress, score, status."""

    __tablename__ = "library_entries"
    __table_args__ = (
        UniqueConstraint("user_id", "manga_id", name="uq_library_entry_user_manga"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    manga_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("manga.id", ondelete="CASCADE"), nullable=False, index=True
    )

    status: Mapped[ReadingStatus] = mapped_column(
        String(20), default=ReadingStatus.PLAN_TO_READ, nullable=False, index=True
    )
    score: Mapped[int | None] = mapped_column(Integer, nullable=True)  # personal rating, 1-10
    progress_chapter: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    progress_volume: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    times_reread: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    is_favorite: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    started_at: Mapped[date | None] = mapped_column(Date, nullable=True)
    finished_at: Mapped[date | None] = mapped_column(Date, nullable=True)

    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Set when this row originated from / is mirrored to MAL; lets the sync
    # service detect local edits that still need to be pushed upstream.
    mal_list_status_raw: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    synced_with_mal_at: Mapped[date | None] = mapped_column(Date, nullable=True)

    user: Mapped["User"] = relationship(back_populates="library_entries")
    manga: Mapped["Manga"] = relationship(back_populates="library_entries")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"<LibraryEntry user_id={self.user_id} manga_id={self.manga_id} "
            f"status={self.status} progress={self.progress_chapter}>"
        )
