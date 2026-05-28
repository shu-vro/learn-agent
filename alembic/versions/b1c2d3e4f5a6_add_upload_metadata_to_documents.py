"""add upload metadata to documents

Revision ID: b1c2d3e4f5a6
Revises: 018d361a9f1d
Create Date: 2026-05-28 01:15:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b1c2d3e4f5a6"  # pragma: allowlist secret
down_revision: Union[str, Sequence[str], None] = "018d361a9f1d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("documents", sa.Column("sha256", sa.String(), nullable=True))
    op.add_column("documents", sa.Column("mime_type", sa.String(), nullable=True))
    op.add_column(
        "documents",
        sa.Column(
            "file_size",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )
    op.alter_column("documents", "file_size", server_default=None)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("documents", "file_size")
    op.drop_column("documents", "mime_type")
    op.drop_column("documents", "sha256")
