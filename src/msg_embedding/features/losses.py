from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from msg_embedding.features.normalizer import ProtocolNormalizer

if TYPE_CHECKING:
    from msg_embedding.features.extractor import FeatureExtractor
    from msg_embedding.models.channel_mae import ChannelMAE

_NORM_EPS: float = 1e-8

DEFAULT_RECON_WEIGHTS: dict[str, float] = {
    "pdp_crop": 1.0,
    "srs1": 2.0,
    "srs2": 2.0,
    "srs3": 2.0,
    "srs4": 2.0,
    "pmi1": 1.5,
    "pmi2": 1.5,
    "pmi3": 1.5,
    "pmi4": 1.5,
    "dft1": 1.0,
    "dft2": 1.0,
    "dft3": 1.0,
    "dft4": 1.0,
    "rsrp_srs": 1.0,
    "rsrp_cb": 1.0,
    "cell_rsrp": 0.5,
}

_COMPLEX_KEYS = [
    "srs1",
    "srs2",
    "srs3",
    "srs4",
    "pmi1",
    "pmi2",
    "pmi3",
    "pmi4",
    "dft1",
    "dft2",
    "dft3",
    "dft4",
]


def calculate_reconstruction_loss(
    recon_tokens: torch.Tensor,
    feat: dict[str, torch.Tensor],
    mae: ChannelMAE,
    ext: FeatureExtractor,
    norm_stats: dict[str, Any] | None = None,
    use_ground_truth: bool = False,
    feat_gt: dict[str, torch.Tensor] | None = None,
    recon_weights: dict[str, float] | None = None,
    use_feature_weighting: bool = True,
) -> torch.Tensor:
    target = feat_gt if use_ground_truth and feat_gt is not None else feat
    recon_feat = mae.reconstruct(recon_tokens)

    loss = torch.tensor(0.0, device=recon_tokens.device)
    normalizer: ProtocolNormalizer = ext.normalizer  # type: ignore[attr-defined]
    _weights = recon_weights or DEFAULT_RECON_WEIGHTS
    weights = _weights if use_feature_weighting else {k: 1.0 for k in _weights}

    if "pdp_crop" in recon_feat and "pdp_crop" in target:
        target_pdp_norm = torch.clamp(target["pdp_crop"], 0.0, 1.0) * 2.0 - 1.0
        loss = loss + weights.get("pdp_crop", 1.0) * F.mse_loss(
            recon_feat["pdp_crop"], target_pdp_norm
        )

    for k in _COMPLEX_KEYS:
        if k not in recon_feat or k not in target:
            continue
        r = recon_feat[k]
        t = target[k]

        if norm_stats and k in norm_stats:
            rms = norm_stats[k]["rms"].unsqueeze(-1)
            t_norm = t / (rms + _NORM_EPS)
        else:
            t_norm = t

        w = weights.get(k, 1.0)
        loss = loss + w * (F.mse_loss(r.real, t_norm.real) + F.mse_loss(r.imag, t_norm.imag))

    for k in ["rsrp_srs", "rsrp_cb"]:
        if k not in recon_feat or k not in target:
            continue
        r = recon_feat[k]
        t_norm = normalizer.normalize(target[k], k)

        if norm_stats and "rsrp_gates" in norm_stats and k in norm_stats["rsrp_gates"]:
            gate = norm_stats["rsrp_gates"][k].view(-1, 1)
            t_norm = t_norm * gate

        w = weights.get(k, 1.0)
        loss = loss + w * F.mse_loss(r, t_norm)

    if "cell_rsrp" in recon_feat and "cell_rsrp" in target:
        r = recon_feat["cell_rsrp"]
        t_norm = normalizer.normalize(target["cell_rsrp"], "cell_rsrp")
        loss = loss + weights.get("cell_rsrp", 0.5) * F.mse_loss(r, t_norm)

    return loss


def contrastive_loss_fn(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.07,
    use_regularization: bool = False,
    reg_weight: float = 0.01,
) -> torch.Tensor:
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    sim_12 = torch.mm(z1, z2.T) / temperature
    sim_21 = torch.mm(z2, z1.T) / temperature

    labels = torch.arange(z1.shape[0], device=z1.device)
    loss_cont = (F.cross_entropy(sim_12, labels) + F.cross_entropy(sim_21, labels)) / 2

    if not use_regularization:
        return loss_cont

    batch_sim = torch.mm(z1, z1.T)
    eye = torch.eye(batch_sim.shape[0], device=z1.device)
    loss_ortho = torch.mean(torch.abs(batch_sim * (1 - eye)))

    loss_smooth = 1.0 - F.cosine_similarity(z1, z2, dim=-1).mean()

    mean_vec = z1.mean(dim=0, keepdim=True)
    loss_uniform = torch.norm(mean_vec, p=2) ** 2

    return loss_cont + reg_weight * (loss_ortho + loss_smooth + loss_uniform)
