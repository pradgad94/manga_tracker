"""
Natural-language manga search: pgvector cosine similarity over `Manga.embedding`
for candidate retrieval, then a single structured-output LLM call that
interprets the query and explains *why* each candidate matches it.

Splitting the work this way keeps it fast and cheap — embeddings do the heavy
lifting of finding plausible matches across the whole catalog, and the LLM is
only asked to reason over a small shortlist rather than the entire library.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.manga import Manga
from app.schemas.manga import MangaSummary
from app.schemas.search import (
    NaturalLanguageSearchResponse,
    SearchExplanationBatch,
    SearchResultItem,
)
from app.services.llm.base import EmbeddingProvider, TextGenerationProvider

logger = get_logger(__name__)

_SYSTEM_PROMPT = """\
You help a reader find manga that match what they're describing in their own words.

<task>
You'll receive the reader's search phrase in <query> and a pre-filtered shortlist \
of candidates in <candidates>. For each candidate, write exactly ONE sentence — \
addressed directly to the reader — explaining concretely why it fits what they \
asked for, grounded in its actual genres and synopsis. If a candidate is only a \
loose match, say so honestly rather than overselling it. Also fill `interpreted_query` \
with a short restatement of what you think they're really looking for (e.g. \
"character-driven historical drama with moral ambiguity") — this helps the reader \
refine their next search if these results miss the mark.
</task>"""

# Over-fetch from the vector index so the LLM has enough real signal to rank/explain,
# then truncate to the user-requested limit after explanations are attached.
_CANDIDATE_MULTIPLIER = 3
_MAX_CANDIDATES = 30
_SYNOPSIS_CHARS = 400


class SearchService:
    def __init__(self, text_provider: TextGenerationProvider, embedding_provider: EmbeddingProvider) -> None:
        self._text_provider = text_provider
        self._embeddings = embedding_provider

    async def search(self, db: AsyncSession, query: str, limit: int = 10) -> NaturalLanguageSearchResponse:
        query_vector = await self._embeddings.embed_query(query)

        candidate_count = min(limit * _CANDIDATE_MULTIPLIER, _MAX_CANDIDATES)
        distance = Manga.embedding.cosine_distance(query_vector)
        result = await db.execute(
            select(Manga, distance.label("distance"))
            .where(Manga.embedding.is_not(None))
            .order_by(distance)
            .limit(candidate_count)
        )
        rows = result.all()

        if not rows:
            return NaturalLanguageSearchResponse(query=query, interpreted_query=None, results=[])

        interpreted_query, explanations = await self._explain_candidates(query, rows)

        results = [
            SearchResultItem(
                manga=MangaSummary.model_validate(manga),
                similarity=max(0.0, 1.0 - distance),
                why_relevant=explanations.get(manga.mal_id),
            )
            for manga, distance in rows[:limit]
        ]

        logger.info("nl_search", query=query, candidates=len(rows), returned=len(results))
        return NaturalLanguageSearchResponse(query=query, interpreted_query=interpreted_query, results=results)

    async def _explain_candidates(self, query: str, rows) -> tuple[str | None, dict[int, str]]:
        candidate_lines = []
        for manga, _distance in rows:
            bits = [f'mal_id={manga.mal_id}', f'"{manga.title}"']
            if manga.genres:
                bits.append(f"genres: {', '.join(manga.genres)}")
            if manga.synopsis:
                bits.append(f"synopsis: {manga.synopsis.strip()[:_SYNOPSIS_CHARS]}")
            candidate_lines.append("- " + " | ".join(bits))

        user_prompt = (
            f"<query>{query}</query>\n\n"
            f"<candidates count=\"{len(candidate_lines)}\" note=\"pre-filtered by embedding similarity\">\n"
            + "\n".join(candidate_lines)
            + "\n</candidates>"
        )

        try:
            batch = await self._text_provider.generate_structured(
                system=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                output_model=SearchExplanationBatch,
                max_tokens=4096,
            )
        except Exception:
            logger.warning("nl_search_explanation_failed", query=query, exc_info=True)
            return None, {}

        return batch.interpreted_query, {item.mal_id: item.why_relevant for item in batch.explanations}
