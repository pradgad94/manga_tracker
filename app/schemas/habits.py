from __future__ import annotations

from pydantic import BaseModel, Field


class HabitInsight(BaseModel):
    observation: str = Field(description="A specific, evidence-grounded observation about how this person reads")
    supporting_evidence: str = Field(description="The data points that led to this observation")


class ReadingHabitAnalysis(BaseModel):
    """
    Structured LLM output summarizing *behavioral* reading patterns — distinct from
    TasteProfileAnalysis, which focuses on *content preferences*.

    Produced via `client.messages.parse(output_format=ReadingHabitAnalysis)` — see
    services/ai/habit_analysis.py.
    """

    reading_pace: str = Field(
        description="How quickly/consistently this person reads (e.g. binger, steady, sporadic) with rough cadence"
    )
    completion_behavior: str = Field(
        description="Tendency to finish vs. drop series, and under what circumstances they tend to drop something"
    )
    rating_patterns: str = Field(
        description="How they use the rating scale — generous, harsh, polarized, clustered, etc."
    )
    series_length_preference: str = Field(
        description="Whether they gravitate toward short one-shots, medium series, or long-running epics"
    )
    insights: list[HabitInsight] = Field(
        description="3-6 concrete, evidence-backed observations about this reader's habits"
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Gentle, optional suggestions the reader might find useful (e.g. 'you have 12 on-hold series — "
        "might be worth revisiting or dropping a few')",
    )
