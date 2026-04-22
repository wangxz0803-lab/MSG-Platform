"""Loss functions for MAE pre-training and contrastive online fine-tuning."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from msg_embedding.features.extractor import FeatureExtractor
    from msg_embedding.models.channel_mae import ChannelMAE

_TOKEN_SLOT: dict[str, int] = {
    "pdp_crop": 0,
    "srs1": 1, "srs2": 2, "srs3": 3, "srs4": 4,
    "pmi1": 5, "pmi2": 6, "pmi3": 7, "pmi4": 8,
    "dft1": 9, "dft2": 10, "dft3": 11, "dft4": 12,
    "rsrp_srs": 13,
    "rsrp_cb": 14,
    "cell_rsrp": 15,
}

_COMPLEX_FEATURES: tuple[str, ...] = (
    "srs1", "srs2", "srs3", "srs4",
    "pmi1", "pmi2", "pmi3", "pmi4",
    "dft1", "dft2", "dft3", "dft4",
)

_DEFAULT_WEIGHTS: dict[str, float] = {
    "pdp_crop": 1.0,
    "srs1": 2.0, "srs2": 2.0, "srs3": 2.0, "srs4": 2.0,
    "pmi1": 1.5, "pmi2": 1.5, "pmi3": 1.5, "pmi4": 1.5,
    "dft1": 1.0, "dft2": 1.0, "dft3": 1.0, "dft4": 1.0,
    "rsrp_srs": 1.0, "rsrp_cb": 1.0, "cell_rsrp": 0.5,
}

_NORM_EPS = 1e-8


def _slot_is_active(token_mask: torch.Tensor | None, slot_idx: int) -> torch.Tensor | None:
    if token_mask is None:
        return None
    active = (~token_mask[:, slot_idx]).float()
    return active


def _masked_mse(recon: torch.Tensor, target: torch.Tensor,
                active: torch.Tensor | None) -> torch.Tensor:
    if active is None:
        return F.mse_loss(recon, target)

    diff_sq = (recon - target) ** 2
    while active.dim() < diff_sq.dim():
        active = active.unsqueeze(-1)

    numerator = (diff_sq * active).sum()
    denom = active.expand_as(diff_sq).sum().clamp_min(1.0)
    return numerator / denom


def reconstruction_loss(
    recon_tokens: torch.Tensor,
    feat: dict[str, torch.Tensor],
    mae: ChannelMAE,
    feature_extractor: FeatureExtractor,
    norm_stats: dict[str, Any],
    weights: dict[str, float] | None = None,
    token_mask: torch.Tensor | None = None,
    feat_gt: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Weighted MAE reconstruction loss in the normalized [-1, 1] domain."""
    if weights is None:
        weights = _DEFAULT_WEIGHTS
    if token_mask is None:
        token_mask = norm_stats.get("token_mask") if norm_stats is not None else None

    target = feat_gt if feat_gt is not None else feat
    recon_feat = mae.reconstruct(recon_tokens)

    normalizer = feature_extractor.normalizer
    device = recon_tokens.device
    loss = torch.zeros((), device=device, dtype=recon_tokens.dtype)

    # PDP
    if "pdp_crop" in recon_feat and "pdp_crop" in target:
        t_pdp = torch.clamp(target["pdp_crop"], 0.0, 1.0) * 2.0 - 1.0
        active = _slot_is_active(token_mask, _TOKEN_SLOT["pdp_crop"])
        loss = loss + weights.get("pdp_crop", 1.0) * _masked_mse(
            recon_feat["pdp_crop"], t_pdp, active
        )

    # Complex features
    for k in _COMPLEX_FEATURES:
        if k not in recon_feat or k not in target:
            continue
        r = recon_feat[k]
        t = target[k]
        if norm_stats is not None and k in norm_stats:
            rms = norm_stats[k]["rms"].unsqueeze(-1)
            t_norm = t / (rms + _NORM_EPS)
        else:
            t_norm = t

        active = _slot_is_active(token_mask, _TOKEN_SLOT[k])
        w = weights.get(k, 1.0)
        loss = loss + w * _masked_mse(r.real, t_norm.real, active)
        loss = loss + w * _masked_mse(r.imag, t_norm.imag, active)

    # RSRP
    rsrp_gates = norm_stats.get("rsrp_gates", {}) if norm_stats is not None else {}
    for k in ("rsrp_srs", "rsrp_cb"):
        if k not in recon_feat or k not in target:
            continue
        r = recon_feat[k]
        t_norm = normalizer.normalize(target[k], k)
        if k in rsrp_gates:
            gate = rsrp_gates[k].view(-1, 1)
            t_norm = t_norm * gate
        active = _slot_is_active(token_mask, _TOKEN_SLOT[k])
        loss = loss + weights.get(k, 1.0) * _masked_mse(r, t_norm, active)

    # Cell RSRP
    if "cell_rsrp" in recon_feat and "cell_rsrp" in target:
        r = recon_feat["cell_rsrp"]
        t_norm = normalizer.normalize(target["cell_rsrp"], "cell_rsrp")
        active = _slot_is_active(token_mask, _TOKEN_SLOT["cell_rsrp"])
        loss = loss + weights.get("cell_rsrp", 0.5) * _masked_mse(r, t_norm, active)

    return loss


def contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.07,
    regularization: bool = False,
    reg_weight: float = 0.01,
) -> torch.Tensor:
    """Symmetric InfoNCE with optional latent-space regularization."""
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    sim_12 = torch.mm(z1, z2.T) / temperature
    sim_21 = torch.mm(z2, z1.T) / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)

    loss_cont = (F.cross_entropy(sim_12, labels) + F.cross_entropy(sim_21, labels)) / 2

    if not regularization:
        return loss_cont

    batch_sim = torch.mm(z1, z1.T)
    eye = torch.eye(batch_sim.shape[0], device=z1.device)
    loss_ortho = torch.mean(torch.abs(batch_sim * (1 - eye)))
    loss_smooth = 1.0 - F.cosine_similarity(z1, z2, dim=-1).mean()
    mean_vec = z1.mean(dim=0, keepdim=True)
    loss_uniform = torch.norm(mean_vec, p=2) ** 2

    return loss_cont + reg_weight * (loss_ortho + loss_smooth + loss_uniform)


def mae_total_loss(
    recon_loss: torch.Tensor,
    cont_loss: torch.Tensor,
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Convex combination of reconstruction and contrastive losses."""
    if weights is None:
        weights = {"recon": 1.0, "contrastive": 0.5}
    w_recon = float(weights.get("recon", 1.0))
    w_cont = float(weights.get("contrastive", 0.5))
    return w_recon * recon_loss + w_cont * cont_loss


__all__ = [
    "reconstruction_loss",
    "contrastive_loss",
    "mae_total_loss",
]
