"""add_chat_message_cache_token_and_usage_detail

Revision ID: 265e212698b7
Revises: 356842fcb852
Create Date: 2026-08-01 22:45:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "265e212698b7"  # pragma: allowlist secret
down_revision: Union[str, Sequence[str], None] = "356842fcb852"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "chat_messages",
        sa.Column("cache_token", sa.Integer(), nullable=True),
    )
    op.add_column(
        "chat_messages",
        sa.Column("usage_detail", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("chat_messages", "usage_detail")
    op.drop_column("chat_messages", "cache_token")
