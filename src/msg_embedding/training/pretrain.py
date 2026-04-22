"""MAE pre-training loop with DDP + AMP + checkpoint manager + experiment tracking."""

from __future__ import annotations

import random
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from msg_embedding.core.logging import get_logger
from msg_embedding.features.extractor import FeatureExtractor
from msg_embedding.models.channel_mae import ChannelMAE

from .callbacks import CheckpointManager, EarlyStopping, GradNormMonitor, LRMonitor
from .distributed import DistEnv, distributed_context
from .experiment import ExperimentTracker
from .losses import contrastive_loss, mae_total_loss, reconstruction_loss

_log = get_logger(__name__)

try:
    from msg_embedding.data.dataset import ChannelDataset  # type: ignore
    _HAVE_DATASET = True
except ImportError:  # pragma: no cover
    ChannelDataset = None  # type: ignore[assignment]
    _HAVE_DATASET = False

try:
    import hydra  # type: ignore
    from omegaconf import DictConfig, OmegaConf  # type: ignore
except ImportError:  # pragma: no cover
    hydra = None  # type: ignore[assignment]
    DictConfig = Any  # type: ignore[misc,assignment]
    OmegaConf = None  # type: ignore[assignment]


def get_git_sha() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=2
        )
        return sha.decode("ascii").strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"


def seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_mask_ratio(epoch: int, total_epochs: int, mask_ratios: list[float],
                   use_multi_scale: bool, fallback: float) -> float:
    if not use_multi_scale:
        return fallback
    if epoch < total_epochs // 3:
        return mask_ratios[0]
    if epoch < 2 * total_epochs // 3:
        return mask_ratios[1 if len(mask_ratios) > 1 else 0]
    return mask_ratios[2 if len(mask_ratios) > 2 else -1]


def _make_grad_scaler(enabled: bool) -> torch.cuda.amp.GradScaler:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)  # type: ignore[attr-defined]
    except (AttributeError, TypeError):  # pragma: no cover
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_ctx(enabled: bool):
    try:
        return torch.amp.autocast("cuda", enabled=enabled)  # type: ignore[attr-defined]
    except (AttributeError, TypeError):  # pragma: no cover
        return torch.cuda.amp.autocast(enabled=enabled)


def loss_weight_schedule(epoch: int, total_epochs: int,
                         schedule: list[list[float]] | None = None
                         ) -> tuple[float, float]:
    if schedule is None:
        schedule = [[0.1, 1.5, 0.2], [0.3, 1.0, 0.5], [1.0, 0.8, 1.0]]
    frac = (epoch + 1) / max(1, total_epochs)
    for end_frac, rw, cw in schedule:
        if frac <= end_frac:
            return rw, cw
    last = schedule[-1] if schedule else [1.0, 0.8, 1.0]
    return last[1], last[2]


def _feat_to_device(feat: dict[str, torch.Tensor], device: torch.device
                    ) -> dict[str, torch.Tensor]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in feat.items()}


class _FakeChannelDataset(torch.utils.data.Dataset):
    """Synthetic dataset for smoke runs when ChannelDataset is unavailable."""

    def __init__(self, n_samples: int = 128, tx_ant: int = 64,
                 seed: int = 0) -> None:
        self.n_samples = int(n_samples)
        self.tx_ant = int(tx_ant)
        self._rng = np.random.default_rng(seed)
        self._cache = [self._make_sample(i) for i in range(self.n_samples)]

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self._cache[idx]

    def _make_sample(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(idx + 100)
        tx = self.tx_ant
        sample: dict[str, torch.Tensor] = {}
        for k in ("srs1", "srs2", "srs3", "srs4",
                  "pmi1", "pmi2", "pmi3", "pmi4",
                  "dft1", "dft2", "dft3", "dft4"):
            real = rng.standard_normal(tx, dtype=np.float32)
            imag = rng.standard_normal(tx, dtype=np.float32)
            sample[k] = torch.from_numpy(real + 1j * imag).to(torch.complex64)
        sample["pdp_crop"] = torch.from_numpy(
            rng.uniform(0.0, 1.0, size=64).astype(np.float32)
        )
        sample["rsrp_srs"] = torch.from_numpy(
            rng.uniform(-120.0, -70.0, size=tx).astype(np.float32)
        )
        sample["rsrp_cb"] = torch.from_numpy(
            rng.uniform(-120.0, -70.0, size=tx).astype(np.float32)
        )
        sample["cell_rsrp"] = torch.from_numpy(
            rng.uniform(-130.0, -80.0, size=16).astype(np.float32)
        )
        sample["cqi"] = torch.tensor(int(rng.integers(0, 16)), dtype=torch.int64)
        sample["srs_sinr"] = torch.tensor(
            float(rng.uniform(-10.0, 15.0)), dtype=torch.float32
        )
        sample["srs_cb_sinr"] = torch.tensor(
            float(rng.uniform(-10.0, 15.0)), dtype=torch.float32
        )
        for k in ("srs_w1", "srs_w2", "srs_w3", "srs_w4"):
            sample[k] = torch.tensor(0.25, dtype=torch.float32)
        sample["_gt"] = {k: v.clone() for k, v in sample.items() if not k.startswith("_")}
        return sample


def _collate_feat(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = [k for k in batch[0] if k != "_gt"]
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    if "_gt" in batch[0]:
        out["_gt"] = {
            k: torch.stack([b["_gt"][k] for b in batch], dim=0)
            for k in batch[0]["_gt"]
        }
    return out


def build_dataset(cfg: Any) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    if _HAVE_DATASET and ChannelDataset is not None:
        try:  # pragma: no cover
            train = ChannelDataset(cfg=cfg, split="train")
            val = ChannelDataset(cfg=cfg, split="val")
            return train, val
        except Exception as exc:
            _log.warning("channel_dataset_fallback", error=str(exc))

    n_train = int(getattr(cfg.train, "_fake_train_n", 128))
    n_val = int(getattr(cfg.train, "_fake_val_n", 32))
    tx = int(cfg.data.tx_ant_num_max)
    train = _FakeChannelDataset(n_samples=n_train, tx_ant=tx, seed=0)
    val = _FakeChannelDataset(n_samples=n_val, tx_ant=tx, seed=99)
    return train, val


class PretrainRunner:
    """Encapsulates one end-to-end MAE pre-training run."""

    def __init__(self, cfg: Any, env: DistEnv,
                 run_id: str | None = None) -> None:
        self.cfg = cfg
        self.env = env
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.device = env.device

        seed_everything(getattr(cfg.project, "seed", 42))

        repo_root = Path(__file__).resolve().parents[3]
        self.artifact_dir = repo_root / "artifacts" / self.run_id
        self.log_dir = repo_root / "artifacts" / "tb"

        self.ckpt_mgr = CheckpointManager(
            run_dir=self.artifact_dir,
            keep_last_n=3,
            config=cfg if env.is_main else None,
            metadata={
                "run_id": self.run_id,
                "job_id": str(getattr(getattr(cfg, "project", None), "job_id", "")),
                "git_sha": get_git_sha(),
                "world_size": env.world_size,
            },
        )

        self.tracker: ExperimentTracker | None = None
        if env.is_main:
            self.tracker = ExperimentTracker(
                run_id=self.run_id,
                log_dir=self.log_dir,
                use_wandb=bool(getattr(cfg.train, "use_wandb", False)),
                wandb_project=str(getattr(cfg.train, "wandb_project", "msg-embedding")),
                wandb_config=OmegaConf.to_container(cfg, resolve=True)
                if OmegaConf is not None else None,
                rank=env.rank,
            )

        self.lr_monitor = LRMonitor(tracker=self.tracker)
        self.grad_monitor = GradNormMonitor(tracker=self.tracker)
        self.early_stop = EarlyStopping(
            patience=int(cfg.train.early_stop_patience),
            min_delta=0.0,
            mode="min",
        )

        model_cfg = {}
        if hasattr(cfg, "model"):
            model_cfg = OmegaConf.to_container(cfg.model, resolve=True) if OmegaConf is not None else {}
        if hasattr(cfg, "train"):
            model_cfg["use_snr_condition"] = bool(getattr(cfg.train, "use_snr_condition", True))
        if hasattr(cfg, "data"):
            model_cfg["tx_ant_num_max"] = int(getattr(cfg.data, "tx_ant_num_max", 64))
            model_cfg["cell_rsrp_dim"] = int(getattr(cfg.data, "cell_rsrp_dim", 16))
        self.mae = ChannelMAE(model_cfg).to(self.device)
        self.feat_ext = FeatureExtractor().to(self.device)

        self.mae_ddp: torch.nn.Module = self.mae
        if env.is_distributed:
            self.mae_ddp = DDP(
                self.mae,
                device_ids=[env.local_rank] if torch.cuda.is_available() else None,
                output_device=env.local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=False,
            )

        self.optimizer = torch.optim.AdamW(
            self.mae.parameters(),
            lr=float(cfg.train.learning_rate),
            weight_decay=1e-4,
        )
        warmup_epochs = int(cfg.train.warmup_epochs)
        total_epochs = int(cfg.train.epochs)
        self.warmup = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda e: (e + 1) / max(1, warmup_epochs)
            if e < warmup_epochs else 1.0,
        )
        self.cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, total_epochs - warmup_epochs),
            eta_min=float(cfg.train.min_learning_rate),
        )
        self.use_amp = torch.cuda.is_available() and bool(
            getattr(cfg.train, "use_amp", True)
        )
        self.scaler = _make_grad_scaler(enabled=self.use_amp)

        self.recon_weights = dict(cfg.train.recon_weights) if hasattr(
            cfg.train, "recon_weights"
        ) else None

        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        if bool(getattr(cfg.train, "resume", False)):
            self._resume_from_ckpt()

    def _resume_from_ckpt(self) -> None:
        state = self.ckpt_mgr.load_latest(map_location=self.device)
        if state is None:
            _log.info("no_checkpoint_found", artifact_dir=str(self.artifact_dir))
            return
        self.mae.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if state.get("warmup") is not None:
            self.warmup.load_state_dict(state["warmup"])
        if state.get("cosine") is not None:
            self.cosine.load_state_dict(state["cosine"])
        if state.get("scaler") is not None and self.use_amp:
            self.scaler.load_state_dict(state["scaler"])
        self.epoch = int(state.get("epoch", 0))
        self.global_step = int(state.get("step", 0))
        self.best_val_loss = float(state.get("best_val_loss", float("inf")))
        rng = state.get("rng")
        if rng is not None:
            try:
                torch.set_rng_state(rng["torch"])
                if rng.get("cuda") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(rng["cuda"])
                if rng.get("numpy") is not None:
                    np.random.set_state(rng["numpy"])
                if rng.get("python") is not None:
                    random.setstate(rng["python"])
            except Exception as exc:  # pragma: no cover
                _log.warning("rng_restore_failed", error=str(exc))
        _log.info("checkpoint_resumed", epoch=self.epoch, step=self.global_step,
                  best_val=round(self.best_val_loss, 4))

    def build_loaders(self) -> tuple[DataLoader, DataLoader]:
        train_ds, val_ds = build_dataset(self.cfg)
        batch_size = int(self.cfg.train.batch_size)

        train_sampler: DistributedSampler | None = None
        val_sampler: DistributedSampler | None = None
        if self.env.is_distributed:
            train_sampler = DistributedSampler(
                train_ds, num_replicas=self.env.world_size,
                rank=self.env.rank, shuffle=True,
            )
            val_sampler = DistributedSampler(
                val_ds, num_replicas=self.env.world_size,
                rank=self.env.rank, shuffle=False,
            )

        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            drop_last=True,
            num_workers=int(getattr(self.cfg.train, "num_workers", 0)),
            collate_fn=_collate_feat,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size,
            sampler=val_sampler,
            shuffle=False,
            drop_last=False,
            num_workers=int(getattr(self.cfg.train, "num_workers", 0)),
            collate_fn=_collate_feat,
            pin_memory=torch.cuda.is_available(),
        )
        return train_loader, val_loader

    def _forward_loss(self, feat_noisy: dict[str, torch.Tensor],
                      feat_clean: dict[str, torch.Tensor] | None,
                      mask_ratio: float,
                      recon_w: float, cont_w: float
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        snr = feat_noisy.get("srs_sinr", torch.zeros(
            next(iter(feat_noisy.values())).shape[0], device=self.device
        ))
        tokens_noisy, norm_stats = self.feat_ext(feat_noisy)
        token_mask = norm_stats.get("token_mask")

        recon_tokens = self.mae_ddp(tokens_noisy, snr, mask_ratio=mask_ratio)
        loss_recon = reconstruction_loss(
            recon_tokens=recon_tokens,
            feat=feat_noisy,
            mae=self.mae,
            feature_extractor=self.feat_ext,
            norm_stats=norm_stats,
            weights=self.recon_weights,
            token_mask=token_mask,
            feat_gt=feat_clean,
        )

        if feat_clean is not None:
            tokens_clean, _ = self.feat_ext(feat_clean)
        else:
            tokens_clean = tokens_noisy
        z1 = self.mae.get_latent(tokens_clean, snr)
        z2 = self.mae.get_latent(tokens_noisy, snr)
        loss_cont = contrastive_loss(z1, z2, regularization=True, reg_weight=0.01)

        loss_total = mae_total_loss(
            loss_recon, loss_cont,
            weights={"recon": recon_w, "contrastive": cont_w},
        )
        return loss_total, loss_recon, loss_cont

    def train_one_epoch(self, loader: DataLoader) -> float:
        self.mae_ddp.train()
        total_epochs = int(self.cfg.train.epochs)
        mask_ratios = list(self.cfg.train.mask_ratios)
        mask_ratio = get_mask_ratio(
            self.epoch, total_epochs, mask_ratios,
            bool(self.cfg.train.use_multi_scale_mask),
            fallback=mask_ratios[0] if mask_ratios else 0.3,
        )
        schedule = list(getattr(self.cfg.train, "loss_weight_schedule", None) or [])
        recon_w, cont_w = loss_weight_schedule(self.epoch, total_epochs, schedule or None)
        accum_steps = max(1, int(self.cfg.train.gradient_accumulation_steps))

        running = 0.0
        n_batches = 0
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader):
            feat_clean = batch.pop("_gt", None)
            feat_noisy = _feat_to_device(batch, self.device)
            if feat_clean is not None:
                feat_clean = _feat_to_device(feat_clean, self.device)

            with _autocast_ctx(self.use_amp):
                loss, loss_recon, loss_cont = self._forward_loss(
                    feat_noisy, feat_clean, mask_ratio, recon_w, cont_w,
                )
                loss_scaled = loss / accum_steps

            self.scaler.scale(loss_scaled).backward()

            if (step + 1) % accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = self.grad_monitor.step(
                    list(self.mae.parameters()), self.global_step
                )
                torch.nn.utils.clip_grad_norm_(self.mae.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

                if self.tracker is not None:
                    self.tracker.log_scalar("train/loss", float(loss.item()),
                                            self.global_step)
                    self.tracker.log_scalar("train/loss_recon",
                                            float(loss_recon.item()),
                                            self.global_step)
                    self.tracker.log_scalar("train/loss_cont",
                                            float(loss_cont.item()),
                                            self.global_step)
                    self.tracker.log_scalar("train/mask_ratio", mask_ratio,
                                            self.global_step)
                    self.tracker.log_scalar("train/grad_norm", grad_norm,
                                            self.global_step)
                self.lr_monitor.step(self.optimizer, self.global_step)
                self.global_step += 1

            running += float(loss.item())
            n_batches += 1

        return running / max(1, n_batches)

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict[str, float]:
        self.mae_ddp.eval()
        mask_ratio = float(self.cfg.model.mask_ratio)
        schedule = list(getattr(self.cfg.train, "loss_weight_schedule", None) or [])
        recon_w, cont_w = loss_weight_schedule(self.epoch, int(self.cfg.train.epochs), schedule or None)
        total = total_recon = total_cont = 0.0
        n = 0
        for batch in loader:
            feat_clean = batch.pop("_gt", None)
            feat_noisy = _feat_to_device(batch, self.device)
            if feat_clean is not None:
                feat_clean = _feat_to_device(feat_clean, self.device)
            with _autocast_ctx(self.use_amp):
                loss, loss_recon, loss_cont = self._forward_loss(
                    feat_noisy, feat_clean, mask_ratio, recon_w, cont_w,
                )
            total += float(loss.item())
            total_recon += float(loss_recon.item())
            total_cont += float(loss_cont.item())
            n += 1
        n = max(1, n)
        return {
            "val/loss": total / n,
            "val/loss_recon": total_recon / n,
            "val/loss_cont": total_cont / n,
        }

    def fit(self) -> None:
        train_loader, val_loader = self.build_loaders()
        total_epochs = int(self.cfg.train.epochs)

        if self.tracker is not None and OmegaConf is not None:
            self.tracker.log_config({"run_id": self.run_id})

        if self.env.is_main:
            print(f"[run_id] {self.run_id}", file=sys.stdout, flush=True)  # noqa: T201

        start_epoch = self.epoch
        for epoch in range(start_epoch, total_epochs):
            self.epoch = epoch
            if isinstance(getattr(train_loader, "sampler", None),
                          DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            train_loss = self.train_one_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            if epoch < int(self.cfg.train.warmup_epochs):
                self.warmup.step()
            else:
                self.cosine.step()

            if self.tracker is not None:
                for name, value in val_metrics.items():
                    self.tracker.log_scalar(name, value, self.global_step)
                self.tracker.log_scalar("train/epoch_loss", train_loss,
                                        self.global_step)

            val_loss = val_metrics["val/loss"]
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            if self.env.is_main:
                self._save_ckpt(epoch, tag="last", is_best=is_best)
                _log.info(
                    "epoch_complete",
                    epoch=epoch + 1,
                    total=total_epochs,
                    train_loss=round(train_loss, 4),
                    val_loss=round(val_metrics["val/loss"], 4),
                    recon=round(val_metrics["val/loss_recon"], 4),
                    cont=round(val_metrics["val/loss_cont"], 4),
                    lr=self.optimizer.param_groups[0]["lr"],
                )
                pct = (epoch + 1 - start_epoch) / max(1, total_epochs - start_epoch) * 100
                print(  # noqa: T201
                    f"[progress] pct={pct:.1f} step=epoch_{epoch + 1}/{total_epochs}",
                    file=sys.stdout, flush=True,
                )

            if self.early_stop(val_loss):
                _log.info("early_stop", epoch=epoch + 1,
                          best_val=round(self.best_val_loss, 4))
                break

        if self.tracker is not None:
            self.tracker.close()

    def _save_ckpt(self, epoch: int, tag: str, is_best: bool) -> None:
        state = {
            "model": self.mae.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "warmup": self.warmup.state_dict(),
            "cosine": self.cosine.state_dict(),
            "scaler": self.scaler.state_dict() if self.use_amp else None,
            "epoch": epoch + 1,
            "step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }
        self.ckpt_mgr.save(state, tag=tag, is_best=is_best)


def run(cfg: Any) -> None:
    """Programmatic entry point (bypasses the @hydra.main decorator)."""
    with distributed_context() as env:
        runner = PretrainRunner(cfg=cfg, env=env)
        runner.fit()


if hydra is not None:  # pragma: no cover

    @hydra.main(config_path="../../../configs", config_name="config",
                version_base=None)
    def main(cfg: DictConfig) -> None:
        run(cfg)

else:  # pragma: no cover

    def main(cfg: Any = None) -> None:
        raise RuntimeError(
            "hydra is required to invoke training.pretrain.main; install hydra-core"
        )


__all__ = [
    "PretrainRunner",
    "build_dataset",
    "get_git_sha",
    "get_mask_ratio",
    "loss_weight_schedule",
    "main",
    "run",
    "seed_everything",
]
