"""
Builds the textual "reading history" context that's fed to the LLM as a system-
prompt block (implicitly cached by Gemini across repeated calls) across the
taste-profile, habit-analysis, and recommendation services.

Formatting choices here are deliberate for prompt-cache stability: deterministic
ordering, no timestamps/UUIDs in the rendered text, stable key ordering. Gemini's
implicit caching keys off exact-prefix matches, so re-running these analyses
without new activity should hit the cache rather than reprocess the prefix from
scratch — but only if the rendered text is byte-identical run to run.
"""
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.activity import ReadingActivity
from app.models.library import LibraryEntry, ReadingStatus
from app.models.manga import Manga
from app.models.review import Review

# Hard caps keep the prompt bounded for very large libraries — the LLM gets the
# most-informative slice (favorites, extremes, and recent activity) rather than
# everything, which would blow the context budget and dilute the signal.
_MAX_ENTRIES_PER_BUCKET = 60
_MAX_REVIEWS = 40
_MAX_ACTIVITIES = 150


async def build_library_snapshot(db: AsyncSession, user_id: uuid.UUID) -> dict:
    """Structured counts + bucketed entries — also stored as TasteProfile.source_stats."""
    result = await db.execute(
        select(LibraryEntry)
        .where(LibraryEntry.user_id == user_id)
        .options(selectinload(LibraryEntry.manga))
        .order_by(LibraryEntry.score.desc().nulls_last(), LibraryEntry.updated_at.desc())
    )
    entries = list(result.scalars().all())

    by_status: dict[str, list[LibraryEntry]] = {}
    for entry in entries:
        by_status.setdefault(entry.status.value, []).append(entry)

    return {
        "total_entries": len(entries),
        "counts_by_status": {status: len(rows) for status, rows in sorted(by_status.items())},
        "favorites": [e for e in entries if e.is_favorite],
        "top_rated": [e for e in entries if e.score is not None and e.score >= 8][:_MAX_ENTRIES_PER_BUCKET],
        "low_rated": [e for e in entries if e.score is not None and e.score <= 4][:_MAX_ENTRIES_PER_BUCKET],
        "currently_reading": by_status.get(ReadingStatus.READING.value, [])[:_MAX_ENTRIES_PER_BUCKET],
        "dropped": by_status.get(ReadingStatus.DROPPED.value, [])[:_MAX_ENTRIES_PER_BUCKET],
        "all_entries": entries,
    }


def _format_entry(entry: LibraryEntry) -> str:
    manga = entry.manga
    bits = [f'"{manga.title}"']
    if manga.genres:
        bits.append(f"genres: {', '.join(manga.genres)}")
    bits.append(f"status: {entry.status.value}")
    if entry.score is not None:
        bits.append(f"your score: {entry.score}/10")
    bits.append(f"progress: ch.{entry.progress_chapter}")
    if manga.mal_mean_score is not None:
        bits.append(f"MAL mean: {manga.mal_mean_score:.2f}")
    return " — ".join(bits)


def render_library_context(snapshot: dict) -> str:
    lines: list[str] = []

    lines.append(f"Library size: {snapshot['total_entries']} series.")
    lines.append("Counts by status: " + ", ".join(f"{k}={v}" for k, v in snapshot["counts_by_status"].items()))

    for label, key in (
        ("Favorites", "favorites"),
        ("Top rated (8-10/10)", "top_rated"),
        ("Low rated (1-4/10)", "low_rated"),
        ("Currently reading", "currently_reading"),
        ("Dropped", "dropped"),
    ):
        rows = snapshot[key]
        if not rows:
            continue
        lines.append(f"\n{label} ({len(rows)}):")
        lines.extend(f"  - {_format_entry(e)}" for e in rows)

    return "\n".join(lines)


async def render_reviews_context(db: AsyncSession, user_id: uuid.UUID) -> str:
    result = await db.execute(
        select(Review)
        .where(Review.user_id == user_id)
        .options(selectinload(Review.manga))
        .order_by(Review.created_at.desc())
        .limit(_MAX_REVIEWS)
    )
    reviews = list(result.scalars().all())
    if not reviews:
        return "No written reviews yet."

    lines = [f"Written reviews ({len(reviews)} most recent):"]
    for review in reviews:
        score = f"{review.score}/10" if review.score is not None else "no score"
        lines.append(f'  - "{review.manga.title}" ({score}): {review.body.strip()[:600]}')
    return "\n".join(lines)


async def render_activity_context(db: AsyncSession, user_id: uuid.UUID) -> str:
    result = await db.execute(
        select(ReadingActivity)
        .where(ReadingActivity.user_id == user_id)
        .options(selectinload(ReadingActivity.manga))
        .order_by(ReadingActivity.occurred_at.desc())
        .limit(_MAX_ACTIVITIES)
    )
    activities = list(result.scalars().all())
    if not activities:
        return "No recorded activity yet."

    lines = [f"Recent activity ({len(activities)} most recent events, newest first):"]
    for activity in activities:
        title = f'"{activity.manga.title}"' if activity.manga else "(general)"
        lines.append(f"  - [{activity.activity_type.value}] {title}: {activity.payload or {}}")
    return "\n".join(lines)


def summarize_source_stats(snapshot: dict) -> dict:
    """A compact, JSON-serializable footprint of the inputs that drove an analysis."""
    return {
        "total_entries": snapshot["total_entries"],
        "counts_by_status": snapshot["counts_by_status"],
        "favorites_count": len(snapshot["favorites"]),
        "top_rated_count": len(snapshot["top_rated"]),
        "low_rated_count": len(snapshot["low_rated"]),
    }
