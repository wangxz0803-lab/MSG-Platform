"""Online fine-tuning loop: freeze encoder, train only the LatentAdapter."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, DistributedSampler

from msg_embedding.core.logging import get_logger
from msg_embedding.features.extractor import FeatureExtractor
from msg_embedding.models.channel_mae import ChannelMAE
from msg_embedding.models.ema import SimpleEMA

from .callbacks import CheckpointManager, GradNormMonitor, LRMonitor
from .distributed import DistEnv, distributed_context
from .experiment import ExperimentTracker
from .losses import contrastive_loss
from .pretrain import (
    _autocast_ctx,
    _collate_feat,
    _feat_to_device,
    _make_grad_scaler,
    build_dataset,
    get_git_sha,
    seed_everything,
)

_log = get_logger(__name__)

try:
    import hydra  # type: ignore
    from omegaconf import DictConfig, OmegaConf  # type: ignore
except ImportError:  # pragma: no cover
    hydra = None  # type: ignore[assignment]
    DictConfig = Any  # type: ignore[misc,assignment]
    OmegaConf = None  # type: ignore[assignment]


def setup_trainable(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Freeze everything except latent_proj, proj_shortcut, latent_adapter."""
    for p in model.parameters():
        p.requires_grad = False

    trainable: list[torch.nn.Parameter] = []
    for module_name in ("latent_proj", "proj_shortcut", "latent_adapter"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for p in module.parameters():
            p.requires_grad = True
            trainable.append(p)
    return trainable


def split_param_groups(model: torch.nn.Module,
                       base_wd: float,
                       adapter_wd: float) -> list[dict[str, Any]]:
    """Return two-group optimizer spec: adapter gets adapter_wd, rest gets base_wd."""
    adapter_params: list[torch.nn.Parameter] = []
    other_params: list[torch.nn.Parameter] = []
    for name, module in model.named_modules():
        if name == "latent_adapter":
            for p in module.parameters():
                if p.requires_grad:
                    adapter_params.append(p)
    seen = {id(p) for p in adapter_params}
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            other_params.append(p)
    groups: list[dict[str, Any]] = []
    if other_params:
        groups.append({"params": other_params, "weight_decay": base_wd})
    if adapter_params:
        groups.append({"params": adapter_params, "weight_decay": adapter_wd})
    return groups


class FinetuneRunner:
    """End-to-end online fine-tuning driver."""

    def __init__(self, cfg: Any, env: DistEnv,
                 pretrained_path: str | Path | None = None,
                 run_id: str | None = None) -> None:
        self.cfg = cfg
        self.env = env
        self.device = env.device
        self.run_id = run_id or "ft-" + time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        seed_everything(getattr(cfg.project, "seed", 42))

        self.mae = ChannelMAE(None).to(self.device)
        self.feat_ext = FeatureExtractor().to(self.device)
        for p in self.feat_ext.parameters():
            p.requires_grad = False
        self.feat_ext.eval()

        if pretrained_path is not None and Path(pretrained_path).exists():
            state = torch.load(str(pretrained_path), map_location=self.device,
                               weights_only=False)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self.mae.load_state_dict(state, strict=False)
            _log.info("pretrained_weights_loaded", path=str(pretrained_path))
        else:
            _log.info("no_pretrained_weights")

        self.trainable_params = setup_trainable(self.mae)
        total_trainable = sum(p.numel() for p in self.trainable_params)
        total_params = sum(p.numel() for p in self.mae.parameters())
        _log.info("trainable_params",
                  trainable=total_trainable, total=total_params,
                  pct=round(100.0 * total_trainable / max(1, total_params), 2))

        online_cfg = getattr(cfg.train, "online", cfg.train)
        base_wd = float(getattr(online_cfg, "weight_decay", 1e-5))
        adapter_wd = float(getattr(online_cfg, "adapter_weight_decay", base_wd))
        groups = split_param_groups(self.mae, base_wd=base_wd, adapter_wd=adapter_wd)
        self.optimizer = torch.optim.AdamW(
            groups or [{"params": self.trainable_params, "weight_decay": base_wd}],
            lr=float(getattr(online_cfg, "learning_rate", 5e-5)),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, int(getattr(online_cfg, "epochs", 15))),
            eta_min=1e-6,
        )
        self.use_amp = torch.cuda.is_available() and bool(
            getattr(cfg.train, "use_amp", True)
        )
        self.scaler = _make_grad_scaler(enabled=self.use_amp)
        self.ema = SimpleEMA(self.mae, decay=0.999)

        repo_root = Path(__file__).resolve().parents[3]
        self.artifact_dir = repo_root / "artifacts" / self.run_id
        self.ckpt_mgr = CheckpointManager(
            run_dir=self.artifact_dir, keep_last_n=2,
            config=cfg if env.is_main else None,
            metadata={"run_id": self.run_id, "git_sha": get_git_sha(),
                      "stage": "finetune"},
        )
        self.tracker: ExperimentTracker | None = None
        if env.is_main:
            self.tracker = ExperimentTracker(
                run_id=self.run_id,
                log_dir=repo_root / "artifacts" / "tb",
                use_wandb=bool(getattr(cfg.train, "use_wandb", False)),
                wandb_project=str(getattr(cfg.train, "wandb_project", "msg-embedding")),
                wandb_config=OmegaConf.to_container(cfg, resolve=True)
                if OmegaConf is not None else None,
                rank=env.rank,
            )
        self.lr_monitor = LRMonitor(self.tracker)
        self.grad_monitor = GradNormMonitor(self.tracker)

        self.global_step = 0
        self.best_loss = float("inf")

    def build_loader(self) -> DataLoader:
        train_ds, _ = build_dataset(self.cfg)
        sampler: DistributedSampler | None = None
        if self.env.is_distributed:
            sampler = DistributedSampler(
                train_ds, num_replicas=self.env.world_size,
                rank=self.env.rank, shuffle=True,
            )
        return DataLoader(
            train_ds,
            batch_size=int(self.cfg.train.batch_size),
            sampler=sampler,
            shuffle=(sampler is None),
            drop_last=True,
            num_workers=int(getattr(self.cfg.train, "num_workers", 0)),
            collate_fn=_collate_feat,
            pin_memory=torch.cuda.is_available(),
        )

    def _step(self, feat: dict[str, torch.Tensor]) -> tuple[torch.Tensor, float]:
        snr = feat.get("srs_sinr", torch.zeros(
            next(iter(feat.values())).shape[0], device=self.device
        ))
        with torch.no_grad():
            tokens, _ = self.feat_ext(feat)

        online_cfg = getattr(self.cfg.train, "online", self.cfg.train)
        use_adapter = bool(getattr(online_cfg, "use_adapter", True))
        use_reg = bool(getattr(online_cfg, "use_regularization", True))

        enc1 = self.mae.encode(tokens, snr, perturb_type="gauss")
        z1 = self.mae.pool_and_project(enc1, use_adapter=use_adapter)
        enc2 = self.mae.encode(tokens, snr, perturb_type="shift")
        z2 = self.mae.pool_and_project(enc2, use_adapter=use_adapter)

        loss = contrastive_loss(
            z1, z2, temperature=0.07,
            regularization=use_reg, reg_weight=0.01,
        )
        return loss, float(loss.item())

    def fit(self) -> None:
        loader = self.build_loader()
        epochs = int(getattr(self.cfg.train.online, "epochs",
                             self.cfg.train.epochs))

        for epoch in range(epochs):
            self.mae.train()
            if isinstance(getattr(loader, "sampler", None), DistributedSampler):
                loader.sampler.set_epoch(epoch)

            running = 0.0
            n = 0
            for batch in loader:
                batch.pop("_gt", None)
                feat = _feat_to_device(batch, self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with _autocast_ctx(self.use_amp):
                    loss, loss_val = self._step(feat)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = self.grad_monitor.step(
                    self.trainable_params, self.global_step
                )
                torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.ema.update(self.mae)
                if self.tracker is not None:
                    self.tracker.log_scalar("finetune/loss", loss_val,
                                            self.global_step)
                    self.tracker.log_scalar("finetune/grad_norm", grad_norm,
                                            self.global_step)
                self.lr_monitor.step(self.optimizer, self.global_step)
                self.global_step += 1
                running += loss_val
                n += 1

            self.scheduler.step()
            avg = running / max(1, n)
            if avg < self.best_loss:
                self.best_loss = avg
            if self.env.is_main:
                _log.info("finetune_epoch_complete",
                          epoch=epoch + 1, total=epochs,
                          loss=round(avg, 4),
                          lr=self.optimizer.param_groups[0]["lr"])
                self._save_ckpt(epoch, best=(avg == self.best_loss))

        self.ema.apply_shadow(self.mae)
        if self.env.is_main:
            self._save_ckpt(epochs - 1, best=False, tag="final")
        if self.tracker is not None:
            self.tracker.close()

    def _save_ckpt(self, epoch: int, best: bool, tag: str = "last") -> None:
        state = {
            "model": self.mae.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.use_amp else None,
            "epoch": epoch + 1,
            "step": self.global_step,
            "best_val_loss": self.best_loss,
        }
        self.ckpt_mgr.save(state, tag=tag, is_best=best)


def run(cfg: Any, pretrained_path: str | Path | None = None) -> None:
    """Programmatic entry point for fine-tuning."""
    with distributed_context() as env:
        FinetuneRunner(cfg=cfg, env=env, pretrained_path=pretrained_path).fit()


if hydra is not None:  # pragma: no cover

    @hydra.main(config_path="../../../configs", config_name="config",
                version_base=None)
    def main(cfg: DictConfig) -> None:
        pretrained = getattr(cfg.train, "pretrained_path", None)
        run(cfg, pretrained_path=pretrained)

else:  # pragma: no cover

    def main(cfg: Any = None) -> None:
        raise RuntimeError("hydra is required to invoke training.finetune.main")


__all__ = [
    "FinetuneRunner",
    "main",
    "run",
    "setup_trainable",
    "split_param_groups",
]
