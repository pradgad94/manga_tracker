from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, Date, DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import get_settings
from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.activity import ReadingActivity
    from app.models.library import LibraryEntry
    from app.models.review import Review

_EMBEDDING_DIM = get_settings().embedding_dimensions


class Manga(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Canonical manga record, kept in sync with MyAnimeList."""

    __tablename__ = "manga"
    __table_args__ = (UniqueConstraint("mal_id", name="uq_manga_mal_id"),)

    # MyAnimeList identity — the source of truth for canonical manga data.
    mal_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)

    title: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    title_english: Mapped[str | None] = mapped_column(String(500), nullable=True)
    title_japanese: Mapped[str | None] = mapped_column(String(500), nullable=True)
    alternative_titles: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)

    synopsis: Mapped[str | None] = mapped_column(Text, nullable=True)
    background: Mapped[str | None] = mapped_column(Text, nullable=True)

    media_type: Mapped[str | None] = mapped_column(String(40), nullable=True)
    status: Mapped[str | None] = mapped_column(String(40), nullable=True)
    genres: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)
    authors: Mapped[list[str] | None] = mapped_column(ARRAY(String), nullable=True)

    num_volumes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    num_chapters: Mapped[int | None] = mapped_column(Integer, nullable=True)

    mal_mean_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    mal_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mal_popularity: Mapped[int | None] = mapped_column(Integer, nullable=True)

    start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    main_picture_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    # Raw payload from MAL, kept for fields we don't model explicitly yet.
    mal_raw: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Semantic-search embedding over title + synopsis + genres (see services/llm/gemini_provider.py).
    embedding: Mapped[list[float] | None] = mapped_column(Vector(_EMBEDDING_DIM), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(120), nullable=True)

    # LLM-summarized digest of MAL community reviews (praise/criticism/themes),
    # fetched via Jikan since MAL's official API exposes no reviews endpoint —
    # see services/ai/community_reviews.py and schemas/review.py:CommunityReviewDigest.
    # Null until backfilled; absent entirely for manga with no community reviews yet.
    community_review_digest: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    community_review_digest_generated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    library_entries: Mapped[list["LibraryEntry"]] = relationship(back_populates="manga")
    reviews: Mapped[list["Review"]] = relationship(back_populates="manga")
    reading_activities: Mapped[list["ReadingActivity"]] = relationship(back_populates="manga")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<Manga id={self.id} mal_id={self.mal_id} title={self.title!r}>"
