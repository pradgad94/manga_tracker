from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.manga import Manga
    from app.models.user import User


class Review(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A user-written review/note about a manga, optionally enriched by LLM analysis."""

    __tablename__ = "reviews"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    manga_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("manga.id", ondelete="CASCADE"), nullable=False, index=True
    )

    body: Mapped[str] = mapped_column(Text, nullable=False)
    score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source: Mapped[str] = mapped_column(String(20), default="local", nullable=False)  # local | mal

    # Structured output produced by the LLM analysis pipeline — see
    # services/ai/habit_analysis.py and schemas/review.py:ReviewAnalysis.
    llm_analysis: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    llm_analyzed_at: Mapped[str | None] = mapped_column(String(40), nullable=True)

    user: Mapped["User"] = relationship(back_populates="reviews")
    manga: Mapped["Manga"] = relationship(back_populates="reviews")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Review id={self.id} user_id={self.user_id} manga_id={self.manga_id}>"
