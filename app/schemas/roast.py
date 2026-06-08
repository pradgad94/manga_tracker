from __future__ import annotations

from pydantic import BaseModel, Field


class MangaRoast(BaseModel):
    """
    Structured LLM output: a funny, affectionate roast of a manga the reader is
    tracking — personalized to *their* progress, score, status, and review, not a
    generic takedown of the title.

    Produced via `generate_structured(output_model=MangaRoast)` — see
    services/ai/roast.py and GET /ai/roast/{manga_id}. Deliberately not cached or
    persisted: regenerating it is half the fun, and Gemini's implicit prompt
    caching already keeps repeat calls over the same manga/library context cheap
    on the input side while still letting the output vary.
    """

    roast: str = Field(
        description="The main roast: a few punchy, funny paragraphs addressed directly to the reader, "
        "grounded in specifics about the manga and how *they* have responded to it so far"
    )
    signature_burn: str = Field(
        description="The single sharpest, most quotable line — the one the reader would screenshot and send to a friend"
    )
    backhanded_compliment: str = Field(
        description="One genuine strength of the manga (or of the reader's own taste/persistence), "
        "delivered as a compliment wrapped in a jab"
    )
    verdict: str = Field(description="A one-line, tongue-in-cheek closing verdict on whether they should keep going with it")
