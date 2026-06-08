"""Import all ORM models so Base.metadata is fully populated for Alembic autogenerate."""
from app.db.base import Base
from app.models.activity import ReadingActivity, ReadingActivityType
from app.models.library import LibraryEntry, ReadingStatus
from app.models.mal_account import MALAccount
from app.models.manga import Manga
from app.models.review import Review
from app.models.taste_profile import TasteProfile
from app.models.user import User

__all__ = [
    "Base",
    "User",
    "Manga",
    "LibraryEntry",
    "ReadingStatus",
    "Review",
    "ReadingActivity",
    "ReadingActivityType",
    "TasteProfile",
    "MALAccount",
]
