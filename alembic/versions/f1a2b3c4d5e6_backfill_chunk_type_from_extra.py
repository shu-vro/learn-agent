"""backfill chunk type from extra

Revision ID: f1a2b3c4d5e6
Revises: 7a6dc6a54c31
Create Date: 2026-06-01 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f1a2b3c4d5e6"  # pragma: allowlist secret
down_revision: Union[str, Sequence[str], None] = "7a6dc6a54c31"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        sa.text(
            """
            UPDATE chunks
            SET type = COALESCE(NULLIF(extra->>'type', ''), 'text_chunk')
            WHERE type IS NULL
            """
        )
    )


def downgrade() -> None:
    pass
