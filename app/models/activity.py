from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.manga import Manga
    from app.models.user import User


class ReadingActivityType(StrEnum):
    STATUS_CHANGE = "status_change"
    PROGRESS_UPDATE = "progress_update"
    SCORE_CHANGE = "score_change"
    REVIEW_ADDED = "review_added"
    STARTED = "started"
    COMPLETED = "completed"
    SYNCED_FROM_MAL = "synced_from_mal"


class ReadingActivity(UUIDPrimaryKeyMixin, Base):
    """
    Append-only log of everything that happens to a user's library.

    This is the raw material the taste-profile pipeline (services/ai/taste_profile.py)
    reads to understand *how* a user's preferences evolve over time — not just their
    current snapshot of ratings.
    """

    __tablename__ = "reading_activities"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    manga_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("manga.id", ondelete="SET NULL"), nullable=True, index=True
    )

    activity_type: Mapped[ReadingActivityType] = mapped_column(String(30), nullable=False, index=True)

    # Free-form details about the event, e.g. {"from": "reading", "to": "completed"}
    # or {"chapter": 42, "previous_chapter": 40}.
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    user: Mapped["User"] = relationship(back_populates="activities")
    manga: Mapped["Manga | None"] = relationship(back_populates="reading_activities")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<ReadingActivity user_id={self.user_id} type={self.activity_type}>"
