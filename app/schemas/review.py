from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel
from app.schemas.manga import MangaSummary


class ReviewCreate(BaseModel):
    manga_id: uuid.UUID
    body: str = Field(min_length=1, max_length=20_000)
    score: int | None = Field(default=None, ge=1, le=10)


class ReviewUpdate(BaseModel):
    body: str | None = Field(default=None, min_length=1, max_length=20_000)
    score: int | None = Field(default=None, ge=1, le=10)


class ReviewAnalysis(BaseModel):
    """
    Structured LLM output describing a single review.

    Produced via `generate_structured(output_model=ReviewAnalysis)` — see
    services/ai/habit_analysis.py. Field descriptions double as the prompt
    guidance the LLM receives for each field.
    """

    sentiment: str = Field(description="Overall sentiment: positive, mixed, or negative")
    sentiment_score: float = Field(
        description="Sentiment strength from -1.0 (very negative) to 1.0 (very positive)",
        ge=-1.0,
        le=1.0,
    )
    themes: list[str] = Field(
        description="Notable themes, motifs, or topics the review highlights (e.g. 'found family', 'pacing issues')"
    )
    aspects_praised: list[str] = Field(
        description="Specific things the reviewer liked (e.g. 'art style', 'character growth')"
    )
    aspects_criticized: list[str] = Field(
        description="Specific things the reviewer disliked or felt were weak"
    )
    one_line_summary: str = Field(description="A single sentence capturing the review's takeaway")


class CommunityReviewDigest(BaseModel):
    """
    Structured LLM output summarizing what MyAnimeList's community says about a
    manga as a whole — distinct from `ReviewAnalysis`, which dissects a single
    review (and specifically the *user's own*).

    Produced by `CommunityReviewService` from reviews fetched via the Jikan API
    (services/jikan/client.py — MAL's official API has no reviews endpoint),
    persisted on `Manga.community_review_digest`, and folded into recommendation
    prompts so picks can be grounded in what readers actually report experiencing,
    not just synopsis/genre similarity.
    """

    consensus: str = Field(description="A one-to-two sentence summary of the overall community consensus on this manga")
    aspects_praised: list[str] = Field(
        description="Specific things community reviewers consistently praise (e.g. 'art direction', 'plot twists')"
    )
    aspects_criticized: list[str] = Field(
        description="Specific things community reviewers consistently criticize or warn newcomers about"
    )
    themes: list[str] = Field(
        description="Recurring themes, tones, or comparisons reviewers bring up (e.g. 'slow burn', 'compared to Vinland Saga')"
    )
    best_for: str = Field(
        description="A short phrase on the kind of reader this tends to land well with, per the reviews — "
        "e.g. 'fans of morally grey characters who don't mind a slow start'"
    )
    review_count_considered: int = Field(description="How many community reviews this digest was built from", ge=0)


class ReviewRead(ORMModel):
    id: uuid.UUID
    manga: MangaSummary
    body: str
    score: int | None
    source: str
    llm_analysis: ReviewAnalysis | None = None
    created_at: datetime
