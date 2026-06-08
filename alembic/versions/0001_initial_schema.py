"""initial schema: users, manga, library, reviews, activity, taste profiles, MAL accounts

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-06-08
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Must match Settings.embedding_dimensions at the time this migration is run.
# The default embedding provider (Gemini, model "gemini-embedding-001") is
# Matryoshka-trained and configured here at 1536 dimensions via output_dimensionality.
# Changing providers/models to a different dimensionality requires a follow-up
# migration that alters these columns.
EMBEDDING_DIM = 1536


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("display_name", sa.String(120), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("is_superuser", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )
    op.create_index("ix_users_email", "users", ["email"])

    op.create_table(
        "manga",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("mal_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("title_english", sa.String(500), nullable=True),
        sa.Column("title_japanese", sa.String(500), nullable=True),
        sa.Column("alternative_titles", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("synopsis", sa.Text(), nullable=True),
        sa.Column("background", sa.Text(), nullable=True),
        sa.Column("media_type", sa.String(40), nullable=True),
        sa.Column("status", sa.String(40), nullable=True),
        sa.Column("genres", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("authors", postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column("num_volumes", sa.Integer(), nullable=True),
        sa.Column("num_chapters", sa.Integer(), nullable=True),
        sa.Column("mal_mean_score", sa.Float(), nullable=True),
        sa.Column("mal_rank", sa.Integer(), nullable=True),
        sa.Column("mal_popularity", sa.Integer(), nullable=True),
        sa.Column("start_date", sa.Date(), nullable=True),
        sa.Column("end_date", sa.Date(), nullable=True),
        sa.Column("main_picture_url", sa.String(1000), nullable=True),
        sa.Column("mal_raw", postgresql.JSONB(), nullable=True),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column("embedding_model", sa.String(120), nullable=True),
        sa.UniqueConstraint("mal_id", name="uq_manga_mal_id"),
    )
    op.create_index("ix_manga_mal_id", "manga", ["mal_id"])
    op.create_index("ix_manga_title", "manga", ["title"])
    op.execute(
        "CREATE INDEX ix_manga_embedding_cosine ON manga "
        "USING hnsw (embedding vector_cosine_ops)"
    )

    op.create_table(
        "library_entries",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("manga_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("manga.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="plan_to_read"),
        sa.Column("score", sa.Integer(), nullable=True),
        sa.Column("progress_chapter", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("progress_volume", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("times_reread", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_favorite", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("started_at", sa.Date(), nullable=True),
        sa.Column("finished_at", sa.Date(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("mal_list_status_raw", postgresql.JSONB(), nullable=True),
        sa.Column("synced_with_mal_at", sa.Date(), nullable=True),
        sa.UniqueConstraint("user_id", "manga_id", name="uq_library_entry_user_manga"),
    )
    op.create_index("ix_library_entries_user_id", "library_entries", ["user_id"])
    op.create_index("ix_library_entries_manga_id", "library_entries", ["manga_id"])
    op.create_index("ix_library_entries_status", "library_entries", ["status"])

    op.create_table(
        "reviews",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("manga_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("manga.id", ondelete="CASCADE"), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("score", sa.Integer(), nullable=True),
        sa.Column("source", sa.String(20), nullable=False, server_default="local"),
        sa.Column("llm_analysis", postgresql.JSONB(), nullable=True),
        sa.Column("llm_analyzed_at", sa.String(40), nullable=True),
    )
    op.create_index("ix_reviews_user_id", "reviews", ["user_id"])
    op.create_index("ix_reviews_manga_id", "reviews", ["manga_id"])

    op.create_table(
        "reading_activities",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("manga_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("manga.id", ondelete="SET NULL"), nullable=True),
        sa.Column("activity_type", sa.String(30), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_reading_activities_user_id", "reading_activities", ["user_id"])
    op.create_index("ix_reading_activities_manga_id", "reading_activities", ["manga_id"])
    op.create_index("ix_reading_activities_activity_type", "reading_activities", ["activity_type"])
    op.create_index("ix_reading_activities_occurred_at", "reading_activities", ["occurred_at"])

    op.create_table(
        "taste_profiles",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("is_current", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("analysis", postgresql.JSONB(), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column("embedding_model", sa.String(120), nullable=True),
        sa.Column("source_stats", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.UniqueConstraint("user_id", "version", name="uq_taste_profile_user_version"),
    )
    op.create_index("ix_taste_profiles_user_id", "taste_profiles", ["user_id"])
    op.create_index("ix_taste_profiles_is_current", "taste_profiles", ["is_current"])

    op.create_table(
        "mal_accounts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("mal_username", sa.String(120), nullable=False),
        sa.Column("mal_user_id", sa.Integer(), nullable=True),
        sa.Column("access_token", sa.Text(), nullable=True),
        sa.Column("refresh_token", sa.Text(), nullable=True),
        sa.Column("token_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_synced_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_sync_status", sa.String(20), nullable=True),
        sa.Column("last_sync_error", sa.Text(), nullable=True),
        sa.UniqueConstraint("user_id", name="uq_mal_accounts_user_id"),
    )
    op.create_index("ix_mal_accounts_user_id", "mal_accounts", ["user_id"])


def downgrade() -> None:
    op.drop_table("mal_accounts")
    op.drop_table("taste_profiles")
    op.drop_table("reading_activities")
    op.drop_table("reviews")
    op.drop_table("library_entries")
    op.execute("DROP INDEX IF EXISTS ix_manga_embedding_cosine")
    op.drop_table("manga")
    op.drop_table("users")
    op.execute("DROP EXTENSION IF EXISTS vector")
