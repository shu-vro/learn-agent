"""update chat models v2

Revision ID: fdeb2736e02f
Revises: 342749791ebe
Create Date: 2026-07-18 18:05:14.327457

"""

from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "fdeb2736e02f"  # pragma: allowlist secret
down_revision: Union[str, Sequence[str], None] = "342749791ebe"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    No-op: autogenerate previously tried to drop LangGraph checkpointer tables,
    which are created/managed at runtime and must not be versioned.
    """
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
