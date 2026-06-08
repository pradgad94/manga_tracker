"""
LLM-driven analysis of *behavior* (reading habits) and *reviews* — the two
"analyze reading habits and reviews" pieces of the spec, kept distinct from
TasteProfileAnalysis (which is about content preferences, not behavior).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.review import Review
from app.schemas.habits import ReadingHabitAnalysis
from app.schemas.review import ReviewAnalysis
from app.services.ai.context import build_library_snapshot, render_activity_context, render_library_context
from app.services.llm.base import TextGenerationProvider

logger = get_logger(__name__)

_HABITS_SYSTEM_PROMPT = """\
You are analyzing one reader's *behavior* — how they read, not what they like.

Work only from the library snapshot and activity log you're given. Focus on \
patterns: pace, follow-through, how they use the rating scale, and what kinds of \
series they gravitate toward by length/format. Ground every insight in specific, \
checkable evidence (counts, titles, statuses) — never speculate beyond the data. \
Keep `suggestions` genuinely optional and low-pressure; omit it if nothing useful \
comes to mind."""

_REVIEW_SYSTEM_PROMPT = """\
You are analyzing a single manga review written by the reader themselves (not a \
public/critic review). Read it closely and extract sentiment, themes, and what \
specifically they praised or criticized. Stay grounded in what the text actually \
says — don't infer opinions the reviewer didn't express."""

_MIN_ENTRIES_FOR_HABITS = 5


class InsufficientDataError(Exception):
    pass


class HabitAnalysisService:
    def __init__(self, text_provider: TextGenerationProvider) -> None:
        self._text_provider = text_provider

    async def analyze_reading_habits(self, db: AsyncSession, user_id: uuid.UUID) -> ReadingHabitAnalysis:
        snapshot = await build_library_snapshot(db, user_id)
        if snapshot["total_entries"] < _MIN_ENTRIES_FOR_HABITS:
            raise InsufficientDataError(
                f"Need at least {_MIN_ENTRIES_FOR_HABITS} library entries to analyze reading habits "
                f"(have {snapshot['total_entries']})."
            )

        user_prompt = "\n".join(
            [
                render_library_context(snapshot),
                "",
                await render_activity_context(db, user_id),
                "",
                "Analyze this reader's habits and produce a structured ReadingHabitAnalysis.",
            ]
        )

        analysis = await self._text_provider.generate_structured(
            system=_HABITS_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            output_model=ReadingHabitAnalysis,
            max_tokens=4096,
        )
        logger.info("habit_analysis_generated", user_id=str(user_id))
        return analysis

    async def analyze_review(self, db: AsyncSession, review: Review) -> ReviewAnalysis:
        user_prompt = (
            f'Manga: "{review.manga.title}"\n'
            f"Reader's score: {review.score}/10\n"
            f"Review text:\n{review.body}\n\n"
            "Analyze this review and produce a structured ReviewAnalysis."
        )

        analysis = await self._text_provider.generate_structured(
            system=_REVIEW_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            output_model=ReviewAnalysis,
            max_tokens=2048,
        )

        review.llm_analysis = analysis.model_dump(mode="json")
        review.llm_analyzed_at = datetime.now(timezone.utc).isoformat()
        await db.commit()

        logger.info("review_analyzed", review_id=str(review.id))
        return analysis
