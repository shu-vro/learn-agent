"""update chatmessage.image_url

Revision ID: 356842fcb852
Revises: e24906c2d073
Create Date: 2026-07-23 17:33:28.607271

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "356842fcb852"  # pragma: allowlist secret
down_revision: Union[str, Sequence[str], None] = "e24906c2d073"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "chat_messages",
        sa.Column("image_urls", postgresql.ARRAY(sa.String()), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("chat_messages", "image_urls")
