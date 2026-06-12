"""add chat model preferences

Revision ID: e5f6a7b8c9d0
Revises: 1a6198c7481e
Create Date: 2026-06-11 12:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e5f6a7b8c9d0"  # pragma: allowlist secret
down_revision: Union[str, Sequence[str], None] = "1a6198c7481e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "preferences",
        sa.Column(
            "default_llm_model",
            sa.String(),
            server_default="omlx:gemma-4-e4b-it-4bit",
            nullable=False,
        ),
    )
    op.add_column(
        "preferences",
        sa.Column("default_reasoning_effort", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("preferences", "default_reasoning_effort")
    op.drop_column("preferences", "default_llm_model")
