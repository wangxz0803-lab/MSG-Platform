"""Worker settings -- environment-driven via pydantic-settings."""

from __future__ import annotations

import os as _os
import sys
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_MSG_REPO_ROOT = Path(
    _os.environ.get("MSG_REPO_ROOT", str(Path(__file__).resolve().parent.parent.parent))
)


class WorkerSettings(BaseSettings):
    """Worker configuration -- all paths expressed as strings for easy override."""

    redis_url: str = Field(default="redis://localhost:6379/0")
    queue_dir: str = Field(default=str(_MSG_REPO_ROOT / "platform" / "worker" / "queue"))
    log_dir: str = Field(default=str(_MSG_REPO_ROOT / "platform" / "worker" / "logs"))
    progress_dir: str = Field(
        default=str(_MSG_REPO_ROOT / "platform" / "worker" / "progress")
    )
    cancel_dir: str = Field(default=str(_MSG_REPO_ROOT / "platform" / "worker" / "cancel"))
    processed_dir: str = Field(
        default=str(_MSG_REPO_ROOT / "platform" / "worker" / "queue" / ".processed")
    )
    db_url: str = Field(default=f"sqlite:///{_MSG_REPO_ROOT}/platform/backend/msg.db")
    python_exe: str = Field(default=str(sys.executable))
    msg_repo_root: str = Field(default=str(_MSG_REPO_ROOT))
    worker_poll_interval_secs: float = Field(default=2.0)
    queue_poll_interval_secs: float = Field(default=1.0)
    task_timeout_hours: float = Field(default=12.0)
    max_retries: int = Field(default=2)
    cancel_grace_secs: float = Field(default=5.0)

    model_config = SettingsConfigDict(
        env_prefix="MSG_WORKER_",
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )

    @property
    def queue_path(self) -> Path:
        return Path(self.queue_dir)

    @property
    def log_path(self) -> Path:
        return Path(self.log_dir)

    @property
    def progress_path(self) -> Path:
        return Path(self.progress_dir)

    @property
    def cancel_path(self) -> Path:
        return Path(self.cancel_dir)

    @property
    def processed_path(self) -> Path:
        return Path(self.processed_dir)

    @property
    def repo_path(self) -> Path:
        return Path(self.msg_repo_root)

    def ensure_dirs(self) -> None:
        """Create all runtime directories (idempotent)."""
        for p in (
            self.queue_path,
            self.log_path,
            self.progress_path,
            self.cancel_path,
            self.processed_path,
        ):
            p.mkdir(parents=True, exist_ok=True)


_settings: WorkerSettings | None = None


def get_settings() -> WorkerSettings:
    """Cached accessor -- tests can reset via :func:`reset_settings`."""
    global _settings
    if _settings is None:
        _settings = WorkerSettings()
    return _settings


def reset_settings(new: WorkerSettings | None = None) -> None:
    """Replace the cached settings (or clear it for re-read from env)."""
    global _settings
    _settings = new
