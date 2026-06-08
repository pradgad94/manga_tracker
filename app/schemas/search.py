from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.manga import MangaSummary


class NaturalLanguageSearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=500,
        description="A free-form description of what the user wants to read, e.g. "
        "'a slow-burn romance with great art and no fanservice'",
    )
    limit: int = Field(default=10, ge=1, le=50)


class SearchResultItem(BaseModel):
    manga: MangaSummary
    similarity: float = Field(description="Cosine similarity between the query and manga embeddings, 0.0-1.0")
    why_relevant: str | None = Field(
        default=None, description="LLM-generated explanation of why this result matches the query"
    )


class NaturalLanguageSearchResponse(BaseModel):
    query: str
    interpreted_query: str | None = Field(
        default=None,
        description="How the LLM/embedding pipeline expanded or interpreted the raw query",
    )
    results: list[SearchResultItem]


class SearchExplanation(BaseModel):
    """One candidate's relevance explanation — matched back to results by `mal_id`."""

    mal_id: int = Field(description="The mal_id of the candidate this explanation is for")
    why_relevant: str = Field(description="A one-sentence explanation of why this matches the search, written to the searcher")


class SearchExplanationBatch(BaseModel):
    """Structured LLM output for explaining a batch of vector-search candidates."""

    interpreted_query: str = Field(
        description="A short restatement of what the searcher seems to be looking for, e.g. "
        "'character-driven fantasy with a slow-burn romance and minimal fanservice'"
    )
    explanations: list[SearchExplanation]
