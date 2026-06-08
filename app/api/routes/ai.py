from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import Cache, CurrentUser, DbSession
from app.core.cache import cache_key
from app.schemas.habits import ReadingHabitAnalysis
from app.schemas.recommendation import RecommendationResponse
from app.schemas.roast import MangaRoast
from app.schemas.search import NaturalLanguageSearchRequest, NaturalLanguageSearchResponse
from app.schemas.taste_profile import TasteProfileRead
from app.services.ai.habit_analysis import HabitAnalysisService, InsufficientDataError as HabitsInsufficientDataError
from app.services.ai.recommendations import NoTasteProfileError, RecommendationService
from app.services.ai.roast import MangaNotTrackedError, RoastService
from app.services.ai.search import SearchService
from app.services.ai.taste_profile import InsufficientDataError as ProfileInsufficientDataError, TasteProfileService
from app.services.factory import (
    get_habit_analysis_service,
    get_recommendation_service,
    get_roast_service,
    get_search_service,
    get_taste_profile_service,
)

router = APIRouter(prefix="/ai", tags=["ai"])


@router.get("/taste-profile", response_model=TasteProfileRead | None)
async def get_current_taste_profile(
    db: DbSession,
    current_user: CurrentUser,
    service: TasteProfileService = Depends(get_taste_profile_service),
) -> TasteProfileRead | None:
    profile = await service.get_current(db, current_user.id)
    return TasteProfileRead.model_validate(profile) if profile is not None else None


@router.get("/taste-profile/history", response_model=list[TasteProfileRead])
async def get_taste_profile_history(
    db: DbSession,
    current_user: CurrentUser,
    service: TasteProfileService = Depends(get_taste_profile_service),
) -> list[TasteProfileRead]:
    profiles = await service.list_versions(db, current_user.id)
    return [TasteProfileRead.model_validate(p) for p in profiles]


@router.post("/taste-profile", response_model=TasteProfileRead, status_code=status.HTTP_201_CREATED)
async def generate_taste_profile(
    db: DbSession,
    current_user: CurrentUser,
    service: TasteProfileService = Depends(get_taste_profile_service),
) -> TasteProfileRead:
    """Generates a new, versioned taste-profile snapshot from the user's current library/reviews/activity."""
    try:
        profile = await service.generate_new_version(db, current_user.id)
    except ProfileInsufficientDataError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return TasteProfileRead.model_validate(profile)


@router.get("/habits", response_model=ReadingHabitAnalysis)
async def get_reading_habits(
    db: DbSession,
    current_user: CurrentUser,
    cache: Cache,
    service: HabitAnalysisService = Depends(get_habit_analysis_service),
) -> ReadingHabitAnalysis:
    """On-demand LLM analysis of reading behavior (pace, completion rate, rating tendencies, ...)."""
    key = cache_key("habits", str(current_user.id))

    async def _compute() -> ReadingHabitAnalysis:
        try:
            return await service.analyze_reading_habits(db, current_user.id)
        except HabitsInsufficientDataError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return await cache.get_or_compute(key, ReadingHabitAnalysis, _compute)


@router.get("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    db: DbSession,
    current_user: CurrentUser,
    cache: Cache,
    service: RecommendationService = Depends(get_recommendation_service),
) -> RecommendationResponse:
    """
    AI picks from the (synced, embedded) catalog, reasoned against the user's
    current taste profile. Cached per (user, taste-profile version) so that the
    fairly expensive candidate-retrieval + LLM-selection pipeline only reruns
    when the user's taste actually changes (a fresh taste-profile version) or
    the cache entry expires — whichever comes first.
    """
    profile = await get_taste_profile_service().get_current(db, current_user.id)
    cache_version = str(profile.version) if profile is not None else "none"
    key = cache_key("recommendations", str(current_user.id), cache_version)

    async def _compute() -> RecommendationResponse:
        try:
            return await service.recommend(db, current_user.id)
        except NoTasteProfileError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return await cache.get_or_compute(key, RecommendationResponse, _compute)


@router.get("/roast/{manga_id}", response_model=MangaRoast)
async def roast_manga(
    manga_id: uuid.UUID,
    db: DbSession,
    current_user: CurrentUser,
    service: RoastService = Depends(get_roast_service),
) -> MangaRoast:
    """
    On-demand, funny/affectionate AI roast of a manga in the reader's library —
    grounded in catalog data plus their own progress, score, status, and review.

    Deliberately uncached: regenerating it for a different (or just plain better)
    joke is the point, not something to suppress with a stale cached response.
    """
    try:
        return await service.roast(db, current_user.id, manga_id)
    except MangaNotTrackedError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/search", response_model=NaturalLanguageSearchResponse)
async def natural_language_search(
    payload: NaturalLanguageSearchRequest,
    db: DbSession,
    current_user: CurrentUser,
    cache: Cache,
    service: SearchService = Depends(get_search_service),
) -> NaturalLanguageSearchResponse:
    """
    Free-form 'find me something like X but with Y' search over the synced catalog.
    Cached by normalized query text — repeat or near-simultaneous searches for the
    same phrase skip both the embedding call and the explanation LLM call.
    """
    key = cache_key("search", payload.query.strip().lower(), str(payload.limit))
    return await cache.get_or_compute(
        key,
        NaturalLanguageSearchResponse,
        lambda: service.search(db, payload.query, limit=payload.limit),
    )
