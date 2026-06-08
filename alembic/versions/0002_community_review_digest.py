"""add community review digest columns to manga

Revision ID: 0002_community_review_digest
Revises: 0001_initial_schema
Create Date: 2026-06-08
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0002_community_review_digest"
down_revision: Union[str, None] = "0001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("manga", sa.Column("community_review_digest", postgresql.JSONB(), nullable=True))
    op.add_column(
        "manga",
        sa.Column("community_review_digest_generated_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("manga", "community_review_digest_generated_at")
    op.drop_column("manga", "community_review_digest")
