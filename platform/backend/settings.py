"""Backend settings -- pydantic-settings powered, env-overridable."""

from __future__ import annotations

import os as _os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_MSG_REPO_ROOT = Path(
    _os.environ.get("MSG_REPO_ROOT", str(Path(__file__).resolve().parent.parent.parent))
)


class BackendSettings(BaseSettings):
    """FastAPI backend configuration.

    All paths are stored as strings so environment overrides work cleanly.
    Override via env vars prefixed with ``MSG_BACKEND_`` or a ``.env`` file.
    """

    data_dir: str = Field(default=str(_MSG_REPO_ROOT / "bridge_out"))
    artifacts_dir: str = Field(default=str(_MSG_REPO_ROOT / "artifacts"))
    reports_dir: str = Field(default=str(_MSG_REPO_ROOT / "reports"))
    bridge_out_dir: str = Field(default=str(_MSG_REPO_ROOT / "bridge_out"))
    manifest_path: str = Field(default=str(_MSG_REPO_ROOT / "bridge_out" / "manifest.parquet"))
    db_url: str = Field(default=f"sqlite:///{_MSG_REPO_ROOT}/platform/backend/msg.db")
    api_port: int = Field(default=8000)
    worker_queue_dir: str = Field(
        default=str(_MSG_REPO_ROOT / "platform" / "worker" / "queue")
    )
    worker_logs_dir: str = Field(
        default=str(_MSG_REPO_ROOT / "platform" / "worker" / "logs")
    )
    worker_progress_dir: str = Field(
        default=str(_MSG_REPO_ROOT / "platform" / "worker" / "progress")
    )
    worker_cancel_dir: str = Field(
        default=str(_MSG_REPO_ROOT / "platform" / "worker" / "cancel")
    )
    configs_dir: str = Field(default=str(_MSG_REPO_ROOT / "configs"))
    schema_path: str = Field(default=str(_MSG_REPO_ROOT / "configs" / "_schema.json"))

    model_config = SettingsConfigDict(
        env_prefix="MSG_BACKEND_",
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def artifacts_path(self) -> Path:
        return Path(self.artifacts_dir)

    @property
    def reports_path(self) -> Path:
        return Path(self.reports_dir)

    @property
    def manifest_file(self) -> Path:
        return Path(self.manifest_path)

    @property
    def worker_queue_path(self) -> Path:
        return Path(self.worker_queue_dir)

    @property
    def worker_logs_path(self) -> Path:
        return Path(self.worker_logs_dir)

    @property
    def worker_progress_path(self) -> Path:
        return Path(self.worker_progress_dir)

    @property
    def worker_cancel_path(self) -> Path:
        return Path(self.worker_cancel_dir)

    @property
    def configs_path(self) -> Path:
        return Path(self.configs_dir)

    @property
    def schema_file(self) -> Path:
        return Path(self.schema_path)

    def ensure_dirs(self) -> None:
        """Create all runtime directories (idempotent)."""
        for p in (
            self.worker_queue_path,
            self.worker_logs_path,
            self.worker_progress_path,
            self.worker_cancel_path,
        ):
            p.mkdir(parents=True, exist_ok=True)


_settings: BackendSettings | None = None


def get_settings() -> BackendSettings:
    """Cached accessor -- tests can override via :func:`reset_settings`."""
    global _settings
    if _settings is None:
        _settings = BackendSettings()
    return _settings


def reset_settings(new: BackendSettings | None = None) -> None:
    """Swap in a fresh settings instance (or clear for env re-read)."""
    global _settings
    _settings = new
