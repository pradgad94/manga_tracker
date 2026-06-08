"""Process-wide singletons for application services, used as FastAPI deps."""
from __future__ import annotations

from functools import lru_cache

import httpx

from app.core.config import get_settings
from app.services.ai.community_reviews import CommunityReviewService
from app.services.ai.habit_analysis import HabitAnalysisService
from app.services.ai.recommendations import RecommendationService
from app.services.ai.roast import RoastService
from app.services.ai.search import SearchService
from app.services.ai.taste_profile import TasteProfileService
from app.services.jikan.client import JikanClient
from app.services.llm.factory import get_embedding_provider, get_text_provider
from app.services.mal.client import MALClient
from app.services.mal.sync import MALSyncService


@lru_cache
def get_taste_profile_service() -> TasteProfileService:
    return TasteProfileService(get_text_provider(), get_embedding_provider())


@lru_cache
def get_habit_analysis_service() -> HabitAnalysisService:
    return HabitAnalysisService(get_text_provider())


@lru_cache
def get_recommendation_service() -> RecommendationService:
    return RecommendationService(get_text_provider(), get_taste_profile_service())


@lru_cache
def get_search_service() -> SearchService:
    return SearchService(get_text_provider(), get_embedding_provider())


@lru_cache
def get_roast_service() -> RoastService:
    return RoastService(get_text_provider())


@lru_cache
def get_jikan_client() -> JikanClient:
    return JikanClient(httpx.AsyncClient(timeout=30.0))


@lru_cache
def get_community_review_service() -> CommunityReviewService:
    return CommunityReviewService(get_jikan_client(), get_text_provider())


@lru_cache
def get_mal_client() -> MALClient:
    settings = get_settings()
    return MALClient(settings.mal_client_id, settings.mal_client_secret, httpx.AsyncClient(timeout=30.0))


@lru_cache
def get_mal_sync_service() -> MALSyncService:
    return MALSyncService(get_mal_client())
