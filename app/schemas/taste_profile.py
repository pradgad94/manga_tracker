from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.common import ORMModel


class GenrePreference(BaseModel):
    genre: str
    affinity: float = Field(description="Estimated affinity from 0.0 (dislikes) to 1.0 (loves)", ge=0.0, le=1.0)
    evidence: str = Field(description="Brief justification grounded in the user's actual library/reviews")


class TasteShift(BaseModel):
    description: str = Field(description="A specific way the user's taste has changed recently")
    direction: str = Field(description="e.g. 'emerging interest', 'fading interest', 'steady preference'")


class TasteProfileAnalysis(BaseModel):
    """
    Structured LLM output describing a user's reading taste at a point in time.

    Produced via `client.messages.parse(output_format=TasteProfileAnalysis)` from
    the user's library, ratings, reviews, and activity history — see
    services/ai/taste_profile.py. This is stored in TasteProfile.analysis (JSONB)
    and TasteProfile.summary, and embedded for recommendation matching.
    """

    summary: str = Field(
        description="A 3-6 sentence narrative description of this reader's taste, written in second person"
    )
    favorite_genres: list[GenrePreference] = Field(
        description="The genres this reader gravitates toward, ranked by inferred affinity"
    )
    favorite_themes: list[str] = Field(
        description="Recurring themes/motifs/tones the reader seems drawn to"
    )
    preferred_demographics: list[str] = Field(
        default_factory=list,
        description="Demographic targets the reader favors if discernible (e.g. shounen, seinen, josei)",
    )
    pacing_and_format_preferences: str = Field(
        description="Observations about preferred length, pacing, art style, or format"
    )
    rating_tendencies: str = Field(
        description="How this reader tends to rate things — generous, harsh, polarized, consistent, etc."
    )
    recent_shifts: list[TasteShift] = Field(
        default_factory=list,
        description="Notable changes in taste compared to the reader's earlier history, if any",
    )
    notable_outliers: list[str] = Field(
        default_factory=list,
        description="Manga that don't fit the reader's usual pattern but they rated highly (or vice versa)",
    )


class TasteProfileRead(ORMModel):
    id: uuid.UUID
    version: int
    is_current: bool
    summary: str
    analysis: TasteProfileAnalysis
    source_stats: dict
    created_at: datetime
