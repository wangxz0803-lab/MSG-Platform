from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(os.environ.get("MSG_REPO_ROOT", str(Path(__file__).resolve().parents[3])))


class MSGSettings(BaseSettings):
    """Infrastructure configuration. ML hyperparameters stay in Hydra YAML."""

    model_config = SettingsConfigDict(
        env_prefix="MSG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    repo_root: Path = Field(default=_REPO_ROOT)

    # Database
    db_url: str = Field(default="sqlite:///msg.db")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0")

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # Logging
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="console")

    # Data paths
    data_dir: str = Field(default="data")
    artifacts_dir: str = Field(default="artifacts")
    reports_dir: str = Field(default="reports")
    bridge_out_dir: str = Field(default="bridge_out")

    # Worker
    worker_processes: int = Field(default=2)
    worker_threads: int = Field(default=4)
    worker_poll_interval: float = Field(default=2.0)
    worker_task_timeout_hours: float = Field(default=12.0)

    @property
    def data_path(self) -> Path:
        return self.repo_root / self.data_dir

    @property
    def artifacts_path(self) -> Path:
        return self.repo_root / self.artifacts_dir

    @property
    def reports_path(self) -> Path:
        return self.repo_root / self.reports_dir

    @property
    def bridge_out_path(self) -> Path:
        return self.repo_root / self.bridge_out_dir

    @property
    def configs_path(self) -> Path:
        return self.repo_root / "configs"

    def ensure_dirs(self) -> None:
        for p in (self.data_path, self.artifacts_path, self.reports_path, self.bridge_out_path):
            p.mkdir(parents=True, exist_ok=True)


_settings: MSGSettings | None = None


def get_settings() -> MSGSettings:
    global _settings
    if _settings is None:
        _settings = MSGSettings()
    return _settings


def reset_settings(new: MSGSettings | None = None) -> None:
    global _settings
    _settings = new


def load_hydra_config(overrides: list[str] | None = None) -> Any:
    """Load ML configuration via Hydra compose.

    Returns an OmegaConf DictConfig with model/train/eval/infer groups.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    settings = get_settings()
    config_dir = str(settings.configs_path.resolve())

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides or [])

    return cfg


def hydra_to_dict(cfg: Any) -> dict[str, Any]:
    """Convert Hydra DictConfig to a plain Python dict."""
    from omegaconf import OmegaConf

    return OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore[return-value]
