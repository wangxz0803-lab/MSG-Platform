"""Inference-time wrapper around ChannelMAE's encoder + projection head."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from msg_embedding.core.logging import get_logger
from msg_embedding.features.extractor import FeatureExtractor
from msg_embedding.models.channel_mae import ChannelMAE

_log = get_logger(__name__)


class EncoderWrapper(nn.Module):
    """Deterministic encoder + projection head with mask-aware pooling."""

    def __init__(self, model: nn.Module, use_adapter: bool = False) -> None:
        super().__init__()
        self.model = model
        self.use_adapter = bool(use_adapter)

    def forward(self, tokens: torch.Tensor,
                token_mask: torch.Tensor) -> torch.Tensor:
        model = self.model
        x = model.input_proj(tokens) + model.pos_emb

        pad_mask = token_mask.clone()
        all_pad = pad_mask.all(dim=1, keepdim=True)
        unmask_first = torch.zeros_like(pad_mask)
        unmask_first[:, 0:1] = all_pad
        pad_mask = pad_mask & (~unmask_first)

        enc = model.encoder(x, src_key_padding_mask=pad_mask)

        mlp_out = model.mlp(enc)
        norm_energy = torch.norm(mlp_out, dim=-1, keepdim=True)
        mask_logit = pad_mask.unsqueeze(-1).to(norm_energy.dtype) * (-1e4)
        norm_energy = norm_energy + mask_logit
        weight = torch.softmax(norm_energy, dim=1)
        pooled = torch.sum(mlp_out * weight, dim=1)

        proj = model.latent_proj(pooled) + model.proj_shortcut(pooled)
        if self.use_adapter:
            proj = model.latent_adapter(proj)
        return F.normalize(proj, dim=-1)


def load_state_dict_flexible(ckpt_path: str | Path,
                             map_location: str | torch.device | None = None
                             ) -> dict[str, torch.Tensor]:
    """Load a state dict, handling CheckpointManager wrappers."""
    payload = torch.load(str(ckpt_path), map_location=map_location,
                         weights_only=False)
    if isinstance(payload, dict) and "model" in payload and isinstance(
        payload["model"], dict
    ):
        return payload["model"]
    if isinstance(payload, dict):
        return payload
    raise TypeError(
        f"expected dict state dict at {ckpt_path}, got {type(payload).__name__}"
    )


def load_channel_mae_class() -> type:
    return ChannelMAE


def load_feature_extractor_class() -> type:
    return FeatureExtractor


def build_encoder_wrapper(ckpt_path: str | Path,
                          device: str | torch.device = "cpu",
                          use_adapter: bool = False,
                          strict: bool = False) -> EncoderWrapper:
    """Instantiate ChannelMAE, load the ckpt, and wrap it for inference."""
    model = ChannelMAE(None)
    sd = load_state_dict_flexible(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing or unexpected:
        _log.info("load_state_dict_info",
                  missing=len(missing), unexpected=len(unexpected))
    model.to(device)
    model.eval()
    wrapper = EncoderWrapper(model, use_adapter=use_adapter)
    wrapper.eval()
    wrapper.to(device)
    return wrapper


__all__ = [
    "EncoderWrapper",
    "build_encoder_wrapper",
    "load_channel_mae_class",
    "load_feature_extractor_class",
    "load_state_dict_flexible",
]
