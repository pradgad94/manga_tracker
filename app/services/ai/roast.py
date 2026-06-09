"""
On-demand "roast my manga" — a funny, affectionate take on a single title in the
reader's library, personalized to *their* progress, score, status, and review. A
different axis again from `HabitAnalysisService` (behavior across the whole library)
and `CommunityReviewService` (what MAL's wider community thinks of one title): this
one is just for laughs, about one manga, for one reader, on demand.
"""
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.logging import get_logger
from app.models.library import LibraryEntry
from app.models.review import Review
from app.schemas.roast import MangaRoast
from app.services.llm.base import TextGenerationProvider

logger = get_logger(__name__)

_ROAST_SYSTEM_PROMPT = """\
You are a witty friend roasting a manga the reader is currently tracking — the \
fond, ribbing kind of roast you'd give someone about their own taste, not a public \
takedown review. Be funny and sharp, and stay affectionate throughout: the reader \
should finish this laughing at themselves, not feeling judged.

<source_material>
Build every joke from what you're actually given — the manga's genres, synopsis, \
and MAL reception, and (most importantly) THIS reader's own relationship to it: \
their progress, score, status, favorite/reread habits, notes, and review if they \
wrote one. The best material lives in the gap between the catalog description and \
how the reader actually responded to it — e.g. someone three volumes into a \
"feel-good slice of life" who scored it a 4, or someone who marked a 30-volume epic \
a favorite and reread it twice. Reference specifics; generic jokes that could apply \
to any manga are the least funny option available to you.
</source_material>

<constraints>
Stay completely spoiler-free — never describe plot twists, character fates, or how \
anything ends; the reader may not have gotten there yet. Aim humor at tropes, \
pacing, and the reader's own habits — never at real people (authors, artists, etc.), \
and never in a way that tips from affectionate ribbing into mean.
</constraints>

<output_format>
- `roast`: 2-3 short, punchy paragraphs addressed directly to the reader. Each \
  paragraph should land a distinct angle. Under 250 words total.
- `signature_burn`: exactly one sentence — the sharpest, most quotable line from \
  the roast, self-contained enough to make sense out of context.
- `backhanded_compliment`: one sentence that acknowledges a real strength of the \
  manga (or the reader's taste/persistence) while wrapping it in a jab.
- `verdict`: one short, tongue-in-cheek line on whether they should keep going.
</output_format>"""

_SYNOPSIS_CHARS = 600
_NOTES_CHARS = 400
_REVIEW_CHARS = 1200


class MangaNotTrackedError(Exception):
    """Raised when the reader hasn't added this manga to their library — nothing personal to roast yet."""


class RoastService:
    def __init__(self, text_provider: TextGenerationProvider) -> None:
        self._text_provider = text_provider

    async def roast(self, db: AsyncSession, user_id: uuid.UUID, manga_id: uuid.UUID) -> MangaRoast:
        entry = await self._find_entry(db, user_id, manga_id)
        if entry is None:
            raise MangaNotTrackedError("You're not tracking this manga — add it to your library before roasting it.")

        review = await self._find_review(db, user_id, manga_id)

        roast = await self._text_provider.generate_structured(
            system=_ROAST_SYSTEM_PROMPT,
            user_prompt=_render_prompt(entry, review),
            output_model=MangaRoast,
            max_tokens=2048,
        )
        logger.info("manga_roasted", user_id=str(user_id), manga_id=str(manga_id))
        return roast

    async def _find_entry(self, db: AsyncSession, user_id: uuid.UUID, manga_id: uuid.UUID) -> LibraryEntry | None:
        result = await db.execute(
            select(LibraryEntry)
            .where(LibraryEntry.user_id == user_id, LibraryEntry.manga_id == manga_id)
            .options(selectinload(LibraryEntry.manga))
        )
        return result.scalar_one_or_none()

    async def _find_review(self, db: AsyncSession, user_id: uuid.UUID, manga_id: uuid.UUID) -> Review | None:
        result = await db.execute(select(Review).where(Review.user_id == user_id, Review.manga_id == manga_id))
        return result.scalar_one_or_none()


def _render_prompt(entry: LibraryEntry, review: Review | None) -> str:
    manga = entry.manga

    catalog_lines = [f'Manga: "{manga.title}"' + (f" ({manga.media_type})" if manga.media_type else "")]
    if manga.genres:
        catalog_lines.append(f"Genres: {', '.join(manga.genres)}")
    if manga.synopsis:
        catalog_lines.append(f"Synopsis: {manga.synopsis.strip()[:_SYNOPSIS_CHARS]}")
    if manga.mal_mean_score is not None:
        catalog_lines.append(f"MAL mean score: {manga.mal_mean_score:.2f}")

    digest = manga.community_review_digest
    if digest:
        consensus = (digest.get("consensus") or "").strip()
        if consensus:
            catalog_lines.append(f"Community consensus: {consensus}")
        if digest.get("best_for"):
            catalog_lines.append(f"Who it tends to land well with: {digest['best_for']}")

    reader_lines = ["This reader's own relationship to it — the good roast material:"]
    progress = f"ch.{entry.progress_chapter}"
    if entry.progress_volume:
        progress += f" / vol.{entry.progress_volume}"
    reader_lines.append(f"  - Status: {entry.status.value}, progress: {progress}")
    if entry.score is not None:
        reader_lines.append(f"  - Their score: {entry.score}/10")
    if entry.is_favorite:
        reader_lines.append("  - Marked it as a favorite")
    if entry.times_reread:
        reader_lines.append(f"  - Reread it {entry.times_reread} time(s)")
    if entry.notes:
        reader_lines.append(f"  - Their notes: {entry.notes.strip()[:_NOTES_CHARS]}")

    if review is not None:
        score_bit = f" ({review.score}/10)" if review.score is not None else ""
        reader_lines.append(f"  - Their written review{score_bit}: {review.body.strip()[:_REVIEW_CHARS]}")
    else:
        reader_lines.append("  - No written review yet")

    return (
        "<catalog>\n" + "\n".join(catalog_lines) + "\n</catalog>\n\n"
        "<reader_context>\n" + "\n".join(reader_lines) + "\n</reader_context>\n\n"
        "Roast this manga for this specific reader. Produce a structured MangaRoast."
    )
