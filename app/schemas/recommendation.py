from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.manga import MangaSummary


class RecommendationItem(BaseModel):
    """
    One recommended manga plus the LLM's reasoning for *this specific user*.

    The `manga` field is filled in by the recommendation service after the LLM
    selects candidates by title/mal_id — the LLM never invents catalog metadata.
    """

    manga: MangaSummary
    reason: str = Field(description="Why this fits the user's taste profile, written directly to them")
    confidence: float = Field(description="How strongly this matches their taste, 0.0-1.0", ge=0.0, le=1.0)


class RecommendationCandidate(BaseModel):
    """Structured LLM output: a single candidate before catalog metadata is attached."""

    mal_id: int = Field(description="The MyAnimeList ID of the recommended manga")
    title: str = Field(description="The manga's title, for display while metadata is resolved")
    reason: str = Field(description="Why this fits the user's taste profile, written directly to them")
    confidence: float = Field(ge=0.0, le=1.0)


class RecommendationBatch(BaseModel):
    """Structured LLM output for a full recommendation request."""

    candidates: list[RecommendationCandidate]
    overall_rationale: str = Field(
        description="A short note on the overall strategy behind these picks (e.g. balancing comfort picks vs. stretch picks)"
    )


class RecommendationResponse(BaseModel):
    items: list[RecommendationItem]
    overall_rationale: str
    based_on_taste_profile_version: int | None = None
