"""Training callbacks: checkpoint manager, early stopping, LR / grad-norm monitors."""

from __future__ import annotations

import contextlib
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from msg_embedding.core.logging import get_logger

_log = get_logger(__name__)

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

try:
    from omegaconf import DictConfig, OmegaConf  # type: ignore
except ImportError:  # pragma: no cover
    DictConfig = Any  # type: ignore[misc,assignment]
    OmegaConf = None  # type: ignore[assignment]


class CheckpointManager:
    """Save / load training checkpoints under ``artifacts/<run_id>/``."""

    CKPT_FILENAME_BY_TAG: dict[str, str] = {
        "last": "ckpt.pth",
        "best": "ckpt_best.pth",
    }

    def __init__(
        self,
        run_dir: str | Path,
        keep_last_n: int = 3,
        config: DictConfig | dict | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self._config = config
        self._config_written = (self.run_dir / "config.yaml").exists()
        self._metadata = dict(metadata or {})
        self._metadata.setdefault("created_at", _utc_now_iso())

    def save(self, state: dict[str, Any], tag: str = "last",
             is_best: bool = False) -> Path:
        filename = self._filename_for_tag(tag)
        target = self.run_dir / filename
        tmp_path = target.with_suffix(target.suffix + ".tmp")
        torch.save(state, tmp_path)
        os.replace(tmp_path, target)
        _log.info("checkpoint_saved", path=str(target), tag=tag)

        if is_best and tag != "best":
            best_path = self.run_dir / self.CKPT_FILENAME_BY_TAG["best"]
            shutil.copyfile(target, best_path)
            _log.info("checkpoint_promoted_to_best", path=str(best_path))

        self._maybe_write_config()
        self._write_metadata(tag=tag, is_best=is_best,
                             best_val_loss=state.get("best_val_loss"))
        self._prune_snapshots()
        return target

    def load_latest(self, map_location: str | torch.device | None = None
                    ) -> dict[str, Any] | None:
        return self._load_tag("last", map_location)

    def load_best(self, map_location: str | torch.device | None = None
                  ) -> dict[str, Any] | None:
        return self._load_tag("best", map_location)

    def load_tag(self, tag: str,
                 map_location: str | torch.device | None = None
                 ) -> dict[str, Any] | None:
        return self._load_tag(tag, map_location)

    def _filename_for_tag(self, tag: str) -> str:
        if tag in self.CKPT_FILENAME_BY_TAG:
            return self.CKPT_FILENAME_BY_TAG[tag]
        return f"ckpt_{tag}.pth"

    def _load_tag(self, tag: str,
                  map_location: str | torch.device | None
                  ) -> dict[str, Any] | None:
        path = self.run_dir / self._filename_for_tag(tag)
        if not path.exists():
            return None
        return torch.load(path, map_location=map_location, weights_only=False)

    def _maybe_write_config(self) -> None:
        if self._config_written or self._config is None:
            return
        target = self.run_dir / "config.yaml"
        try:
            if OmegaConf is not None and hasattr(self._config, "_content"):
                OmegaConf.save(config=self._config, f=str(target))
            elif yaml is not None:
                with open(target, "w", encoding="utf-8") as f:
                    yaml.safe_dump(dict(self._config), f, sort_keys=False)
            else:  # pragma: no cover
                target.write_text(repr(self._config), encoding="utf-8")
            self._config_written = True
        except Exception as exc:  # pragma: no cover
            _log.warning("config_snapshot_failed", error=str(exc))

    def _write_metadata(self, tag: str, is_best: bool,
                        best_val_loss: float | None) -> None:
        meta = dict(self._metadata)
        meta["last_tag"] = tag
        meta["last_saved_at"] = _utc_now_iso()
        if is_best:
            meta["best_tag"] = tag
            meta["best_saved_at"] = meta["last_saved_at"]
        if best_val_loss is not None:
            prev = meta.get("best_val_loss")
            try:
                if prev is None or (is_best and float(best_val_loss) < float(prev)):
                    meta["best_val_loss"] = float(best_val_loss)
            except (TypeError, ValueError):
                pass
        target = self.run_dir / "metadata.json"
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, target)
        self._metadata = meta

    def _prune_snapshots(self) -> None:
        if self.keep_last_n < 0:
            return
        snapshots = sorted(
            self.run_dir.glob("ckpt_epoch_*.pth"),
            key=lambda p: p.stat().st_mtime,
        )
        excess = len(snapshots) - self.keep_last_n
        for i in range(max(0, excess)):
            with contextlib.suppress(OSError):  # pragma: no cover
                snapshots[i].unlink()


@dataclass
class EarlyStopping:
    """Patience-based early stopping on a scalar validation metric."""

    patience: int = 10
    min_delta: float = 0.0
    mode: str = "min"

    best: float = field(default=float("inf"), init=False)
    bad_epochs: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode!r}")
        if self.patience < 0:
            raise ValueError(f"patience must be >= 0, got {self.patience}")
        if self.mode == "max":
            self.best = float("-inf")

    def __call__(self, val_metric: float) -> bool:
        improved = self._is_improvement(val_metric)
        if improved:
            self.best = float(val_metric)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs > self.patience

    def _is_improvement(self, val_metric: float) -> bool:
        val = float(val_metric)
        if self.mode == "min":
            return val < self.best - self.min_delta
        return val > self.best + self.min_delta

    def reset(self) -> None:
        self.best = float("inf") if self.mode == "min" else float("-inf")
        self.bad_epochs = 0


class LRMonitor:
    """Log optimizer learning rates to an ExperimentTracker."""

    def __init__(self, tracker: Any | None = None) -> None:
        self.tracker = tracker

    def step(self, optimizer: torch.optim.Optimizer, step: int) -> dict[str, float]:
        out: dict[str, float] = {}
        groups = optimizer.param_groups
        for i, g in enumerate(groups):
            lr = float(g.get("lr", 0.0))
            out[f"lr/group_{i}"] = lr
            if len(groups) == 1:
                out["lr"] = lr
        if self.tracker is not None:
            for name, value in out.items():
                self.tracker.log_scalar(name, value, step)
        return out


class GradNormMonitor:
    """Compute and optionally log the global parameter gradient norm."""

    def __init__(self, tracker: Any | None = None, norm_type: float = 2.0,
                 tag: str = "grad_norm") -> None:
        self.tracker = tracker
        self.norm_type = float(norm_type)
        self.tag = tag

    def step(self, parameters: torch.Tensor | list[torch.Tensor],
             step: int) -> float:
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        params = [p for p in parameters if p is not None and p.grad is not None]
        if not params:
            return 0.0
        device = params[0].grad.device
        total = torch.zeros((), device=device)
        for p in params:
            g = p.grad.detach()
            total = total + g.norm(self.norm_type) ** self.norm_type
        grad_norm = float(total.pow(1.0 / self.norm_type).item())
        if self.tracker is not None:
            self.tracker.log_scalar(self.tag, grad_norm, step)
        return grad_norm


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


__all__ = [
    "CheckpointManager",
    "EarlyStopping",
    "LRMonitor",
    "GradNormMonitor",
]
