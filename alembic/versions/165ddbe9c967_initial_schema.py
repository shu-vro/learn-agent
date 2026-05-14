"""initial_schema

Revision ID: 165ddbe9c967
Revises:
Create Date: 2026-05-14 02:01:29.723816

Idempotent baseline: safe on empty DB, on legacy DBs created with the old
per-model ``create`` helper (which skipped association tables), and on DBs
that already match models.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "165ddbe9c967"  # pragma: allowlist secret
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    have = set(insp.get_table_names())

    if "chunks" not in have:
        op.create_table(
            "chunks",
            sa.Column("id", sa.String(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        have.add("chunks")
    if "documents" not in have:
        op.create_table(
            "documents",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("source", sa.String(), nullable=False),
            sa.Column("url", sa.String(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
        )
        have.add("documents")
    if "users" not in have:
        op.create_table(
            "users",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("email", sa.String(), nullable=False),
            sa.Column("password", sa.String(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("email"),
        )
        have.add("users")
    if "documents_chunks" not in have:
        op.create_table(
            "documents_chunks",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("document_id", sa.String(), nullable=True),
            sa.Column("chunks_id", sa.String(), nullable=True),
            sa.ForeignKeyConstraint(["chunks_id"], ["chunks.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(
                ["document_id"], ["documents.id"], ondelete="CASCADE"
            ),
            sa.PrimaryKeyConstraint("id"),
        )
        have.add("documents_chunks")
    if "projects" not in have:
        op.create_table(
            "projects",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("user_id", sa.String(), nullable=False),
            sa.Column("name", sa.String(), nullable=False),
            sa.Column("description", sa.String(), nullable=False),
            sa.Column("extra", sa.JSON(), nullable=True),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        have.add("projects")
    else:
        proj_cols = {c["name"] for c in insp.get_columns("projects")}
        if "description" not in proj_cols:
            op.add_column(
                "projects",
                sa.Column(
                    "description",
                    sa.String(),
                    nullable=False,
                    server_default="",
                ),
            )
            op.alter_column("projects", "description", server_default=None)

    if "projects_documents" not in have:
        op.create_table(
            "projects_documents",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("project_id", sa.String(), nullable=False),
            sa.Column("document_id", sa.String(), nullable=False),
            sa.ForeignKeyConstraint(["document_id"], ["documents.id"]),
            sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        have.add("projects_documents")
    if "threads" not in have:
        op.create_table(
            "threads",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("user_id", sa.String(), nullable=False),
            sa.Column("thread_name", sa.String(), nullable=False),
            sa.Column("project_id", sa.String(), nullable=False),
            sa.Column("extra", sa.JSON(), nullable=True),
            sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        have.add("threads")
    if "chats" not in have:
        op.create_table(
            "chats",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("thread_id", sa.String(), nullable=False),
            sa.Column("user_id", sa.String(), nullable=False),
            sa.Column("type", sa.String(), nullable=False),
            sa.Column("message", sa.String(), nullable=False),
            sa.Column("input_token", sa.Integer(), nullable=True),
            sa.Column("output_token", sa.Integer(), nullable=True),
            sa.Column("total_token", sa.Integer(), nullable=True),
            sa.Column("group_id", sa.String(), nullable=True),
            sa.Column("extra", sa.JSON(), nullable=True),
            sa.ForeignKeyConstraint(["thread_id"], ["threads.id"]),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        have.add("chats")
    if "chats_chunks" not in have:
        op.create_table(
            "chats_chunks",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("chats_id", sa.String(), nullable=True),
            sa.Column("chunks_id", sa.String(), nullable=True),
            sa.ForeignKeyConstraint(["chats_id"], ["chats.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["chunks_id"], ["chunks.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
        have.add("chats_chunks")


def downgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    tables = set(insp.get_table_names())
    for name in (
        "chats_chunks",
        "chats",
        "threads",
        "projects_documents",
        "projects",
        "documents_chunks",
        "users",
        "documents",
        "chunks",
    ):
        if name in tables:
            op.drop_table(name)
