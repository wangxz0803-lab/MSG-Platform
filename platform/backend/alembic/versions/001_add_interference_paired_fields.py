"""Add interference-aware paired channel fields to samples table.

Revision ID: 001
Revises:
Create Date: 2026-04-23
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("samples") as batch_op:
        batch_op.add_column(sa.Column("ul_sir_db", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("dl_sir_db", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("num_interfering_ues", sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column("link_pairing", sa.String(), nullable=True, server_default="single")
        )


def downgrade() -> None:
    with op.batch_alter_table("samples") as batch_op:
        batch_op.drop_column("link_pairing")
        batch_op.drop_column("num_interfering_ues")
        batch_op.drop_column("dl_sir_db")
        batch_op.drop_column("ul_sir_db")
