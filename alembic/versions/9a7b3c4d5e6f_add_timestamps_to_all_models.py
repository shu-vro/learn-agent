"""add timestamps to all models

Revision ID: 9a7b3c4d5e6f
Revises: 6ccd788e9f3b
Create Date: 2026-05-28 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9a7b3c4d5e6f"  # pragma: allowlist secret
down_revision: Union[str, Sequence[str], None] = "6ccd788e9f3b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _add_timestamps(table_name: str) -> None:
    op.add_column(
        table_name,
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )
    op.add_column(
        table_name,
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
    )


def _drop_timestamps(table_name: str) -> None:
    op.drop_column(table_name, "updated_at")
    op.drop_column(table_name, "created_at")


def upgrade() -> None:
    """Upgrade schema."""
    _add_timestamps("users")
    _add_timestamps("projects")
    _add_timestamps("threads")
    _add_timestamps("chats")
    _add_timestamps("documents")
    _add_timestamps("chunks")
    _add_timestamps("projects_documents")


def downgrade() -> None:
    """Downgrade schema."""
    _drop_timestamps("projects_documents")
    _drop_timestamps("chunks")
    _drop_timestamps("documents")
    _drop_timestamps("chats")
    _drop_timestamps("threads")
    _drop_timestamps("projects")
    _drop_timestamps("users")
