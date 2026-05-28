"""add document ingestion state and link uniqueness

Revision ID: c3d4e5f6a7b8
Revises: b1c2d3e4f5a6
Create Date: 2026-05-28 01:50:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c3d4e5f6a7b8"  # pragma: allowlist secret
down_revision: Union[str, Sequence[str], None] = "b1c2d3e4f5a6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "documents",
        sa.Column(
            "ingestion_status",
            sa.String(),
            nullable=False,
            server_default="processing",
        ),
    )
    op.add_column("documents", sa.Column("ingestion_error", sa.String(), nullable=True))
    op.alter_column("documents", "ingestion_status", server_default=None)
    op.create_unique_constraint("uq_documents_sha256", "documents", ["sha256"])
    op.create_unique_constraint(
        "uq_project_document_link",
        "projects_documents",
        ["project_id", "document_id"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("uq_project_document_link", "projects_documents", type_="unique")
    op.drop_constraint("uq_documents_sha256", "documents", type_="unique")
    op.drop_column("documents", "ingestion_error")
    op.drop_column("documents", "ingestion_status")
