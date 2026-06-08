"""
Generates versioned "taste profile" snapshots from a user's library, ratings,
reviews, and activity history — the evolving-over-time piece of the spec.

Each call appends a new, immutable TasteProfile version rather than overwriting
the previous one, so drift is visible and recommendations can always cite "as of
version N". The profile's narrative summary is embedded so it can be matched
against manga embeddings for recommendations (services/ai/recommendations.py).
"""
from __future__ import annotations

import uuid

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.taste_profile import TasteProfile
from app.schemas.taste_profile import TasteProfileAnalysis
from app.services.ai.context import (
    build_library_snapshot,
    render_activity_context,
    render_library_context,
    render_reviews_context,
    summarize_source_stats,
)
from app.services.llm.base import EmbeddingProvider, TextGenerationProvider

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You are a perceptive reading-taste analyst working from one reader's actual manga \
library, ratings, written reviews, and activity history.

Ground every claim in the data you're given — cite specific titles, genres, or \
patterns as evidence. Do not invent series the reader hasn't engaged with. Write \
the narrative summary directly to the reader, in second person, in a warm but \
honest tone (call out contradictions or blind spots if you see them — that's more \
useful than flattery).

If a previous taste-profile summary is provided, pay particular attention to what \
has changed since then and reflect that in `recent_shifts`."""

_MIN_ENTRIES_FOR_ANALYSIS = 5


class InsufficientDataError(Exception):
    """Raised when a user's library is too small to produce a meaningful profile."""


class TasteProfileService:
    def __init__(self, text_provider: TextGenerationProvider, embedding_provider: EmbeddingProvider) -> None:
        self._text_provider = text_provider
        self._embeddings = embedding_provider

    async def get_current(self, db: AsyncSession, user_id: uuid.UUID) -> TasteProfile | None:
        result = await db.execute(
            select(TasteProfile).where(TasteProfile.user_id == user_id, TasteProfile.is_current.is_(True))
        )
        return result.scalar_one_or_none()

    async def generate_new_version(self, db: AsyncSession, user_id: uuid.UUID) -> TasteProfile:
        snapshot = await build_library_snapshot(db, user_id)
        if snapshot["total_entries"] < _MIN_ENTRIES_FOR_ANALYSIS:
            raise InsufficientDataError(
                f"Need at least {_MIN_ENTRIES_FOR_ANALYSIS} library entries to build a taste profile "
                f"(have {snapshot['total_entries']}). Sync your MAL list or add entries first."
            )

        previous = await self.get_current(db, user_id)

        user_prompt_parts = [
            render_library_context(snapshot),
            "",
            await render_reviews_context(db, user_id),
            "",
            await render_activity_context(db, user_id),
        ]
        if previous is not None:
            user_prompt_parts += ["", f"Previous taste-profile summary (version {previous.version}):", previous.summary]

        user_prompt_parts += [
            "",
            "Analyze this reader's taste and produce a structured TasteProfileAnalysis.",
        ]

        analysis = await self._text_provider.generate_structured(
            system=_SYSTEM_PROMPT,
            user_prompt="\n".join(user_prompt_parts),
            output_model=TasteProfileAnalysis,
            max_tokens=8192,
        )

        [embedding] = await self._embeddings.embed_documents([analysis.summary])

        next_version = (previous.version + 1) if previous is not None else 1

        if previous is not None:
            await db.execute(
                update(TasteProfile)
                .where(TasteProfile.id == previous.id)
                .values(is_current=False)
            )

        profile = TasteProfile(
            user_id=user_id,
            version=next_version,
            is_current=True,
            summary=analysis.summary,
            analysis=analysis.model_dump(mode="json"),
            embedding=embedding,
            embedding_model=self._embeddings.model_name,
            source_stats=summarize_source_stats(snapshot),
        )
        db.add(profile)
        await db.commit()
        await db.refresh(profile)

        logger.info("taste_profile_generated", user_id=str(user_id), version=next_version)
        return profile

    async def list_versions(self, db: AsyncSession, user_id: uuid.UUID) -> list[TasteProfile]:
        result = await db.execute(
            select(TasteProfile).where(TasteProfile.user_id == user_id).order_by(TasteProfile.version.desc())
        )
        return list(result.scalars().all())
