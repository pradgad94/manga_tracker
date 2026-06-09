from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Boolean, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin

if TYPE_CHECKING:
    from app.models.activity import ReadingActivity
    from app.models.library import LibraryEntry
    from app.models.mal_account import MALAccount
    from app.models.review import Review
    from app.models.taste_profile import TasteProfile


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    library_entries: Mapped[list["LibraryEntry"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    reviews: Mapped[list["Review"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    activities: Mapped[list["ReadingActivity"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    taste_profiles: Mapped[list["TasteProfile"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    mal_account: Mapped["MALAccount | None"] = relationship(
        back_populates="user", cascade="all, delete-orphan", uselist=False
    )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<User id={self.id} email={self.email!r}>"
