"""
Keeps `Manga.embedding` populated so pgvector similarity search
(natural-language search, recommendations) has something to query against.

Embeddings are computed from a compact text representation — title, genres,
media type, and a trimmed synopsis — rather than the raw synopsis alone, since
genre/format signal materially improves nearest-neighbor quality for "vibe"-style
queries like "a slow-burn workplace romance".
"""
from __future__ import annotations

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.manga import Manga
from app.services.llm.base import EmbeddingProvider

logger = get_logger(__name__)

_BATCH_SIZE = 32
_SYNOPSIS_CHARS = 1200


def _embedding_text(manga: Manga) -> str:
    parts = [manga.title]
    if manga.media_type:
        parts.append(f"({manga.media_type})")
    if manga.genres:
        parts.append("Genres: " + ", ".join(manga.genres))
    if manga.synopsis:
        parts.append(manga.synopsis.strip()[:_SYNOPSIS_CHARS])
    return "\n".join(parts)


async def backfill_missing_embeddings(
    db: AsyncSession, embeddings: EmbeddingProvider, *, limit: int | None = None
) -> int:
    """Embed every manga that has none yet, or whose embedding used a different model."""
    query = select(Manga).where(
        or_(Manga.embedding.is_(None), Manga.embedding_model.is_(None), Manga.embedding_model != embeddings.model_name)
    )
    if limit is not None:
        query = query.limit(limit)

    result = await db.execute(query)
    pending = list(result.scalars().all())
    if not pending:
        return 0

    total = 0
    for start in range(0, len(pending), _BATCH_SIZE):
        batch = pending[start : start + _BATCH_SIZE]
        vectors = await embeddings.embed_documents([_embedding_text(m) for m in batch])
        for manga, vector in zip(batch, vectors, strict=True):
            manga.embedding = vector
            manga.embedding_model = embeddings.model_name
        await db.commit()
        total += len(batch)

    logger.info("manga_embeddings_backfilled", count=total, model=embeddings.model_name)
    return total
