"""drop chunk notes columns (moved to on-demand generation)

Revision ID: a1b2c3d4e5f7
Revises: f6a7b8c9d0e1
Create Date: 2026-07-18 13:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "a1b2c3d4e5f7"  # pragma: allowlist secret
down_revision: Union[str, Sequence[str], None] = "f6a7b8c9d0e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("preferences", "generate_chunk_notes")
    op.drop_column("documents", "notes_error")
    op.drop_column("documents", "notes_status")


def downgrade() -> None:
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
