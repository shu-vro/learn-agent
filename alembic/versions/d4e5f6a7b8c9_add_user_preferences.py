"""add user preferences

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-05-30 12:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d4e5f6a7b8c9"  # pragma: allowlist secret
down_revision: Union[str, Sequence[str], None] = "c3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "preferences",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("theme", sa.String(), server_default="system", nullable=False),
        sa.Column(
            "use_vision_model", sa.Boolean(), server_default="true", nullable=False
        ),
        sa.Column(
            "use_image_descriptions",
            sa.Boolean(),
            server_default="true",
            nullable=False,
        ),
        sa.Column(
            "use_formula_transcription",
            sa.Boolean(),
            server_default="true",
            nullable=False,
        ),
        sa.Column(
            "equation_ocr_lib", sa.String(), server_default="local", nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id"),
    )


def downgrade() -> None:
    op.drop_table("preferences")
