from fastapi import APIRouter

from app.api.routes import ai, auth, library, manga, reviews, sync

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(manga.router)
api_router.include_router(library.router)
api_router.include_router(reviews.router)
api_router.include_router(sync.router)
api_router.include_router(ai.router)
