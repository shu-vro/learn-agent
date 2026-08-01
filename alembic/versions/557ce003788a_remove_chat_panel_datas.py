"""remove chat panel datas

Revision ID: 557ce003788a
Revises: d4e5f6a7b8c9
Create Date: 2026-05-29 22:57:02.454793

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "557ce003788a"  # pragma: allowlist secret
down_revision: Union[str, Sequence[str], None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # LangGraph checkpointer tables are created/managed by the checkpointer at
    # runtime, not by Alembic. Drop defensively if a prior app run created them.
    op.execute("DROP TABLE IF EXISTS checkpoint_writes CASCADE")
    op.execute("DROP TABLE IF EXISTS checkpoint_migrations CASCADE")
    op.execute("DROP TABLE IF EXISTS checkpoint_blobs CASCADE")
    op.execute("DROP TABLE IF EXISTS checkpoints CASCADE")
    # chat_layout may not exist on databases built purely from this migration
    # chain, so drop it conditionally.
    op.execute("ALTER TABLE preferences DROP COLUMN IF EXISTS chat_layout")


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column(
        "preferences",
        sa.Column(
            "chat_layout",
            postgresql.JSON(astext_type=sa.Text()),
            autoincrement=False,
            nullable=False,
        ),
    )
