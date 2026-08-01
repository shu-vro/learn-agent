"""update chat schema for multiple chats

Revision ID: 342749791ebe
Revises: a1b2c3d4e5f7
Create Date: 2026-07-18 17:53:09.850747

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "342749791ebe"  # pragma: allowlist secret
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "chat_messages",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("chat_id", sa.String(), nullable=False),
        sa.Column("message", sa.String(), nullable=False),
        sa.Column("input_token", sa.Integer(), nullable=True),
        sa.Column("output_token", sa.Integer(), nullable=True),
        sa.Column("total_token", sa.Integer(), nullable=True),
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
        sa.ForeignKeyConstraint(["chat_id"], ["chats.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    # LangGraph checkpointer tables are managed at runtime — ignore them.
    op.drop_column("chats", "total_token")
    op.drop_column("chats", "message")
    op.drop_column("chats", "output_token")
    op.drop_column("chats", "input_token")


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column(
        "chats",
        sa.Column("input_token", sa.INTEGER(), autoincrement=False, nullable=True),
    )
    op.add_column(
        "chats",
        sa.Column("output_token", sa.INTEGER(), autoincrement=False, nullable=True),
    )
    op.add_column(
        "chats", sa.Column("message", sa.VARCHAR(), autoincrement=False, nullable=False)
    )
    op.add_column(
        "chats",
        sa.Column("total_token", sa.INTEGER(), autoincrement=False, nullable=True),
    )
    op.drop_table("chat_messages")
