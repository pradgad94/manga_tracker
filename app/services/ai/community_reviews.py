"""
Ingests MyAnimeList's *community* reviews for synced manga and distills them into
a structured digest — a different axis from `HabitAnalysisService.analyze_review`,
which dissects a review the user themselves wrote.

Pipeline: fetch a sample of community reviews via Jikan (MAL's official API has no
reviews endpoint — see services/jikan/client.py), summarize them with a single
structured-output LLM call into a `CommunityReviewDigest`, and persist it on
`Manga`. `RecommendationService` then folds the digest into candidate descriptions
so picks can be grounded in what readers actually report experiencing — not just
synopsis/genre similarity to the user's taste-profile embedding.

`backfill_community_review_digests` mirrors `manga_index.backfill_missing_embeddings`'s
shape (find what's missing, process in bounded batches, commit incrementally), but
is throttled far more conservatively: each manga costs one rate-limited Jikan HTTP
round trip *and* one LLM call, versus one batched embeddings call for many titles
at once. It's wired into the post-sync background pipeline (see api/routes/sync.py)
with a small per-run cap so a large catalog gets backfilled gradually across runs
rather than stalling a single sync for a long time.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.manga import Manga
from app.schemas.review import CommunityReviewDigest
from app.services.jikan.client import JikanClient
from app.services.jikan.exceptions import JikanError
from app.services.llm.base import TextGenerationProvider
from app.services.llm.exceptions import LLMError

logger = get_logger(__name__)

_DIGEST_SYSTEM_PROMPT = """\
You are summarizing what MyAnimeList's community says about a manga, based on a \
sample of user-written reviews (shown ranked by how much engagement each one got). \
Distill genuine consensus — points multiple reviewers independently make — rather \
than amplifying any single reviewer's idiosyncratic take. Where reviewers disagree, \
reflect that by keeping claims modest and specific rather than overstating agreement, \
and prefer concrete, checkable observations ("pacing slows in the middle arcs") over \
vague praise or complaint ("it's good"/"it's bad").

Some source reviews may describe plot events, twists, or endings. Your digest must \
NOT repeat any of that — summarize at the level of themes, tone, craft, and reader \
reception only, so it stays spoiler-free for someone who hasn't read this yet."""

# Bounds the prompt: a handful of the most-engaged-with reviews carry most of the
# consensus signal, and capping per-review length keeps any one wall-of-text from
# crowding out the rest of the sample.
_MAX_REVIEWS = 12
_REVIEW_CHARS = 1200

# Each manga costs one throttled Jikan round trip *and* one LLM call — keep each
# sync's backfill slice small so the background pipeline stays responsive; the
# remaining backlog gets picked up incrementally on subsequent syncs.
_DEFAULT_BACKFILL_LIMIT = 15


class CommunityReviewService:
    def __init__(self, jikan_client: JikanClient, text_provider: TextGenerationProvider) -> None:
        self._jikan = jikan_client
        self._text_provider = text_provider

    async def generate_digest(self, manga: Manga) -> CommunityReviewDigest | None:
        """
        Fetch and summarize `manga`'s community reviews. Returns `None` when MAL
        has no reviews on file for it yet — a normal outcome for niche or very new
        titles, not an error; callers persist that as "checked, nothing found".
        """
        reviews = await self._jikan.get_manga_reviews(manga.mal_id)
        if not reviews:
            return None

        sample = sorted(reviews, key=_engagement, reverse=True)[:_MAX_REVIEWS]
        user_prompt = (
            f'Manga: "{manga.title}"'
            + (f" ({manga.media_type})" if manga.media_type else "")
            + "\n"
            + (f"Genres: {', '.join(manga.genres)}\n" if manga.genres else "")
            + f"\n{len(sample)} community reviews, ranked by reader engagement (most-engaged first):\n"
            + "\n".join(_render_review(review) for review in sample)
            + f"\n\nSummarize the community's reception as a structured CommunityReviewDigest. "
            f"Set review_count_considered to exactly {len(sample)}."
        )

        return await self._text_provider.generate_structured(
            system=_DIGEST_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            output_model=CommunityReviewDigest,
            max_tokens=2048,
        )


async def backfill_community_review_digests(
    db: AsyncSession, service: CommunityReviewService, *, limit: int | None = _DEFAULT_BACKFILL_LIMIT
) -> int:
    """
    Generate digests for manga that have never been checked against Jikan.

    `community_review_digest_generated_at` (not the digest itself) is the
    "have we looked?" marker — a manga with no community reviews yet still gets
    stamped with `digest=None` so it isn't re-queried every single sync.
    """
    query = select(Manga).where(Manga.community_review_digest_generated_at.is_(None))
    if limit is not None:
        query = query.limit(limit)

    result = await db.execute(query)
    pending = list(result.scalars().all())
    if not pending:
        return 0

    processed = 0
    for manga in pending:
        try:
            digest = await service.generate_digest(manga)
        except JikanError as exc:
            logger.warning("community_review_fetch_failed", manga_id=str(manga.id), mal_id=manga.mal_id, error=str(exc))
            continue
        except LLMError as exc:
            logger.warning("community_review_digest_failed", manga_id=str(manga.id), mal_id=manga.mal_id, error=str(exc))
            continue

        manga.community_review_digest = digest.model_dump(mode="json") if digest is not None else None
        manga.community_review_digest_generated_at = datetime.now(timezone.utc)
        await db.commit()
        processed += 1

    logger.info("community_review_digests_backfilled", processed=processed, found_reviews=sum(
        1 for m in pending if m.community_review_digest is not None
    ))
    return processed


def _engagement(review: dict[str, Any]) -> int:
    return (review.get("reactions") or {}).get("overall") or 0


def _render_review(review: dict[str, Any]) -> str:
    bits = [f"score: {review['score']}/10" if review.get("score") else "score: n/a"]
    tags = review.get("tags") or []
    if tags:
        bits.append(f"tags: {', '.join(tags)}")

    spoiler_flag = " [CONTAINS SPOILERS — do not echo plot details from this one]" if review.get("is_spoiler") else ""
    text = (review.get("review") or "").strip().replace("\n", " ")[:_REVIEW_CHARS]
    return f"- ({' | '.join(bits)}){spoiler_flag} {text}"
