"""Experiment tracking with TensorBoard + optional Weights & Biases double-write."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from msg_embedding.core.logging import get_logger

_log = get_logger(__name__)


class ExperimentTracker:
    """Dual-sink metric tracker (TensorBoard + optional Weights & Biases)."""

    def __init__(
        self,
        run_id: str,
        log_dir: str | Path,
        use_wandb: bool = False,
        wandb_project: str = "msg-embedding",
        wandb_config: dict[str, Any] | None = None,
        wandb_mode: str | None = None,
        rank: int = 0,
    ) -> None:
        self.run_id = run_id
        self.rank = int(rank)
        self.log_dir = Path(log_dir) / run_id
        self._tb_writer: Any | None = None
        self._wandb_run: Any | None = None
        self._wandb_module: Any | None = None

        if self.rank != 0:
            return

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._tb_writer = _try_open_tensorboard(self.log_dir)

        if use_wandb:
            self._wandb_run, self._wandb_module = _try_open_wandb(
                run_id=run_id,
                project=wandb_project,
                config=wandb_config or {},
                mode=wandb_mode,
                log_dir=self.log_dir,
            )

    def log_scalar(self, name: str, value: float, step: int) -> None:
        if self.rank != 0:
            return
        v = float(value)
        self._tb_call("add_scalar", name, v, step)
        self._wandb_log({name: v}, step)

    def log_histogram(self, name: str, tensor: torch.Tensor | np.ndarray,
                      step: int) -> None:
        if self.rank != 0:
            return
        arr = tensor if isinstance(tensor, np.ndarray) else tensor.detach().cpu().numpy()
        self._tb_call("add_histogram", name, arr, step)
        if self._wandb_run is not None and self._wandb_module is not None:
            try:
                hist = self._wandb_module.Histogram(arr)
                self._wandb_run.log({name: hist}, step=step)
            except Exception as exc:  # pragma: no cover
                _log.debug("wandb_histogram_failed", error=str(exc))

    def log_image(self, name: str, image: np.ndarray, step: int) -> None:
        if self.rank != 0:
            return
        arr = np.asarray(image)
        dataformats = "HWC" if arr.ndim == 3 else "HW"
        if self._tb_writer is not None:
            try:
                self._tb_writer.add_image(name, arr, step, dataformats=dataformats)
            except Exception as exc:  # pragma: no cover
                _log.debug("tb_image_failed", error=str(exc))
        if self._wandb_run is not None and self._wandb_module is not None:
            try:
                self._wandb_run.log(
                    {name: self._wandb_module.Image(arr)}, step=step
                )
            except Exception as exc:  # pragma: no cover
                _log.debug("wandb_image_failed", error=str(exc))

    def log_config(self, cfg: dict[str, Any]) -> None:
        if self.rank != 0:
            return
        if self._tb_writer is not None:
            try:
                flat = _flatten_dict(cfg)
                self._tb_writer.add_hparams(flat, {"hparam/placeholder": 0.0})
            except Exception as exc:  # pragma: no cover
                _log.debug("tb_hparams_failed", error=str(exc))
        if self._wandb_run is not None:
            try:
                self._wandb_run.config.update(cfg, allow_val_change=True)
            except Exception as exc:  # pragma: no cover
                _log.debug("wandb_config_update_failed", error=str(exc))

    def close(self) -> None:
        if self._tb_writer is not None:
            with contextlib.suppress(Exception):  # pragma: no cover
                self._tb_writer.flush()
                self._tb_writer.close()
            self._tb_writer = None
        if self._wandb_run is not None:
            with contextlib.suppress(Exception):  # pragma: no cover
                self._wandb_run.finish()
            self._wandb_run = None

    def __enter__(self) -> ExperimentTracker:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    def _tb_call(self, method: str, *args: Any, **kwargs: Any) -> None:
        if self._tb_writer is None:
            return
        try:
            getattr(self._tb_writer, method)(*args, **kwargs)
        except Exception as exc:  # pragma: no cover
            _log.debug("tb_call_failed", method=method, error=str(exc))

    def _wandb_log(self, payload: dict[str, Any], step: int) -> None:
        if self._wandb_run is None:
            return
        try:
            self._wandb_run.log(payload, step=step)
        except Exception as exc:  # pragma: no cover
            _log.debug("wandb_log_failed", error=str(exc))


def _try_open_tensorboard(log_dir: Path) -> Any | None:
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
    except ImportError:
        _log.info("tensorboard_not_available")
        return None
    try:
        return SummaryWriter(log_dir=str(log_dir))
    except Exception as exc:  # pragma: no cover
        _log.warning("summary_writer_init_failed", error=str(exc))
        return None


def _try_open_wandb(
    run_id: str,
    project: str,
    config: dict[str, Any],
    mode: str | None,
    log_dir: Path,
) -> tuple[Any | None, Any | None]:
    try:
        import wandb  # type: ignore
    except ImportError:
        _log.info("wandb_not_installed")
        return None, None
    try:
        run = wandb.init(
            project=project,
            name=run_id,
            config=config,
            dir=str(log_dir),
            mode=mode,
            reinit=True,
        )
        return run, wandb
    except Exception as exc:
        _log.warning("wandb_init_failed", error=str(exc))
        return None, wandb


def _flatten_dict(d: dict[str, Any], prefix: str = "",
                  sep: str = ".") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key, sep))
        elif isinstance(v, int | float | bool | str):
            out[key] = v
        else:
            out[key] = str(v)
    return out


__all__ = ["ExperimentTracker"]
