"""
AI-powered recommendations: pgvector retrieval against the user's current taste-
profile embedding narrows the catalog to a plausible shortlist (excluding what
they've already logged), then a structured-output LLM call picks and justifies
the final list against the *narrative* taste profile — which captures nuance
(contradictions, recent shifts, rating tendencies) that raw vector similarity can't.

The LLM only ever selects from the shortlist by `mal_id` — it never invents catalog
entries — and `_resolve_candidates` re-attaches real `Manga` rows after the fact.
"""
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.library import LibraryEntry
from app.models.manga import Manga
from app.schemas.manga import MangaSummary
from app.schemas.recommendation import RecommendationBatch, RecommendationItem, RecommendationResponse
from app.services.ai.taste_profile import TasteProfileService
from app.services.llm.base import TextGenerationProvider

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You recommend manga to a specific reader based on a narrative description of their \
taste (built from their actual library, ratings, and reviews) plus a shortlist of \
candidates a similarity search has already retrieved.

<critical_constraint>
Recommend ONLY titles present in the provided <candidates> list. Use each \
candidate's `mal_id` exactly as given and copy its title verbatim. Never recommend \
a title that is not in the list, and never invent or assume details about a \
candidate beyond what is explicitly provided. If the list has fewer candidates than \
the requested count, return only what the list contains — do not pad with invented \
titles.
</critical_constraint>

<task>
For each pick, write a short reason addressed directly to the reader that connects \
the pick to specific aspects of their taste profile — do not just restate the \
synopsis or community take verbatim. Some candidates include a "community take" \
(a digest of what MAL readers say); treat it as a real signal: e.g. slow-pacing \
warnings matter for a reader whose profile shows they bounce off slow starts, while \
praised found-family dynamics may be a strong match for a reader who gravitates \
toward that. Aim for a mix: a few confident "safe bets" that closely match \
established preferences, and a couple of well-reasoned "stretch" picks that extend \
from a genuine signal in their profile (e.g. an emerging interest in `recent_shifts`). \
Explain that balance briefly in `overall_rationale`.
</task>"""

_CANDIDATE_POOL_SIZE = 60
_SYNOPSIS_CHARS = 350


class NoTasteProfileError(Exception):
    """Raised when a user has no taste profile yet — generate one first."""


class RecommendationService:
    def __init__(self, text_provider: TextGenerationProvider, taste_profiles: TasteProfileService) -> None:
        self._text_provider = text_provider
        self._taste_profiles = taste_profiles

    async def recommend(self, db: AsyncSession, user_id: uuid.UUID, count: int = 8) -> RecommendationResponse:
        profile = await self._taste_profiles.get_current(db, user_id)
        if profile is None or profile.embedding is None:
            raise NoTasteProfileError(
                "No taste profile yet — call POST /ai/taste-profile to generate one before requesting recommendations."
            )

        candidates = await self._fetch_candidates(db, user_id, profile.embedding)
        if not candidates:
            return RecommendationResponse(items=[], overall_rationale="Not enough unread catalog data to recommend from yet — sync more manga first.", based_on_taste_profile_version=profile.version)

        batch = await self._select_recommendations(profile.summary, candidates, count)
        items = _resolve_candidates(batch, candidates)

        logger.info("recommendations_generated", user_id=str(user_id), count=len(items), profile_version=profile.version)
        return RecommendationResponse(
            items=items,
            overall_rationale=batch.overall_rationale,
            based_on_taste_profile_version=profile.version,
        )

    async def _fetch_candidates(self, db: AsyncSession, user_id: uuid.UUID, profile_embedding) -> list[Manga]:
        already_tracked = select(LibraryEntry.manga_id).where(LibraryEntry.user_id == user_id)
        distance = Manga.embedding.cosine_distance(profile_embedding)

        result = await db.execute(
            select(Manga)
            .where(Manga.embedding.is_not(None), Manga.id.notin_(already_tracked))
            .order_by(distance)
            .limit(_CANDIDATE_POOL_SIZE)
        )
        return list(result.scalars().all())

    async def _select_recommendations(self, taste_summary: str, candidates: list[Manga], count: int) -> RecommendationBatch:
        candidate_lines = []
        for manga in candidates:
            bits = [f"mal_id={manga.mal_id}", f'"{manga.title}"']
            if manga.genres:
                bits.append(f"genres: {', '.join(manga.genres)}")
            if manga.mal_mean_score is not None:
                bits.append(f"MAL mean: {manga.mal_mean_score:.2f}")
            if manga.synopsis:
                bits.append(f"synopsis: {manga.synopsis.strip()[:_SYNOPSIS_CHARS]}")
            community_take = _community_take(manga)
            if community_take:
                bits.append(f"community take: {community_take}")
            candidate_lines.append("- " + " | ".join(bits))

        user_prompt = (
            f"<taste_profile>\n{taste_summary}\n</taste_profile>\n\n"
            f"<candidates count=\"{len(candidates)}\" note=\"none of these are in the reader's library yet\">\n"
            + "\n".join(candidate_lines)
            + f"\n</candidates>\n\n"
            f"Recommend {count} of these candidates."
        )

        return await self._text_provider.generate_structured(
            system=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            output_model=RecommendationBatch,
            max_tokens=8192,
        )


def _resolve_candidates(batch: RecommendationBatch, candidates: list[Manga]) -> list[RecommendationItem]:
    by_mal_id = {manga.mal_id: manga for manga in candidates}

    items = []
    for picked in batch.candidates:
        manga = by_mal_id.get(picked.mal_id)
        if manga is None:
            logger.warning("recommendation_unknown_mal_id", mal_id=picked.mal_id, title=picked.title)
            continue
        items.append(
            RecommendationItem(manga=MangaSummary.model_validate(manga), reason=picked.reason, confidence=picked.confidence)
        )
    return items


_COMMUNITY_ASPECT_LIMIT = 2


def _community_take(manga: Manga) -> str | None:
    """
    Compresses `Manga.community_review_digest` (a `CommunityReviewDigest` dict
    persisted as JSONB by `CommunityReviewService` — `None` until backfilled, or
    if MAL has no reviews for this title) into one line for the candidate prompt.
    Using `.get()` rather than indexing: this is persisted JSON from a prior LLM
    run, not a live Pydantic model, so it's read defensively at this boundary.
    """
    digest = manga.community_review_digest
    if not digest:
        return None

    bits = [digest.get("consensus", "").strip()]
    praised = digest.get("aspects_praised") or []
    criticized = digest.get("aspects_criticized") or []
    if praised:
        bits.append(f"praised for {', '.join(praised[:_COMMUNITY_ASPECT_LIMIT])}")
    if criticized:
        bits.append(f"criticized for {', '.join(criticized[:_COMMUNITY_ASPECT_LIMIT])}")
    return "; ".join(bit for bit in bits if bit)
