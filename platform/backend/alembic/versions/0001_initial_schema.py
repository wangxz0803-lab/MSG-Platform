"""Initial schema: jobs, runs, samples, model_artifacts.

Revision ID: 0001
Revises: None
Create Date: 2026-04-22
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("job_id", sa.String(), primary_key=True),
        sa.Column("type", sa.String(), index=True),
        sa.Column("status", sa.String(), index=True, server_default="queued"),
        sa.Column("display_name", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("params_json", sa.Text(), nullable=True),
        sa.Column("progress_pct", sa.Float(), server_default="0.0"),
        sa.Column("log_path", sa.String(), nullable=True),
        sa.Column("error_msg", sa.Text(), nullable=True),
        sa.Column("run_id", sa.String(), nullable=True, index=True),
    )

    op.create_table(
        "runs",
        sa.Column("run_id", sa.String(), primary_key=True),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("ckpt_path", sa.String(), nullable=True),
        sa.Column("ckpt_best", sa.String(), nullable=True),
        sa.Column("ckpt_last", sa.String(), nullable=True),
        sa.Column("metrics_json_path", sa.String(), nullable=True),
        sa.Column("config_path", sa.String(), nullable=True),
        sa.Column("metadata_path", sa.String(), nullable=True),
        sa.Column("git_sha", sa.String(), nullable=True),
        sa.Column("tags", sa.String(), nullable=True),
    )

    op.create_table(
        "samples",
        sa.Column("uuid", sa.String(), primary_key=True),
        sa.Column("sample_id", sa.String(), nullable=True, index=True),
        sa.Column("shard_id", sa.String(), nullable=True, index=True),
        sa.Column("source", sa.String(), nullable=True, index=True),
        sa.Column("link", sa.String(), nullable=True, index=True),
        sa.Column("snr_db", sa.Float(), nullable=True),
        sa.Column("sir_db", sa.Float(), nullable=True),
        sa.Column("sinr_db", sa.Float(), nullable=True),
        sa.Column("num_cells", sa.Integer(), nullable=True),
        sa.Column("ts", sa.DateTime(), nullable=True),
        sa.Column("status", sa.String(), nullable=True, index=True),
        sa.Column("job_id", sa.String(), nullable=True, index=True),
        sa.Column("run_id", sa.String(), nullable=True, index=True),
        sa.Column("path", sa.String(), nullable=True),
        sa.Column("split", sa.String(), nullable=True),
    )

    op.create_table(
        "model_artifacts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.String(),
            sa.ForeignKey("runs.run_id"),
            index=True,
        ),
        sa.Column("format", sa.String(), index=True),
        sa.Column("path", sa.String()),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("size_bytes", sa.Integer(), server_default="0"),
    )


def downgrade() -> None:
    op.drop_table("model_artifacts")
    op.drop_table("samples")
    op.drop_table("runs")
    op.drop_table("jobs")
