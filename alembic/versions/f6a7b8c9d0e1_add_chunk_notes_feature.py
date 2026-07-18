"""add chunk notes feature

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-06-18 12:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f6a7b8c9d0e1"  # pragma: allowlist secret
down_revision: Union[str, Sequence[str], None] = "e5f6a7b8c9d0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("notes_status", sa.String(), nullable=True))
    op.add_column("documents", sa.Column("notes_error", sa.String(), nullable=True))
    op.add_column(
        "preferences",
        sa.Column(
            "generate_chunk_notes",
            sa.Boolean(),
            server_default="false",
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("preferences", "generate_chunk_notes")
    op.drop_column("documents", "notes_error")
    op.drop_column("documents", "notes_status")
