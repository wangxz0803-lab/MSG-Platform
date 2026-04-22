"""Channel-prediction / compression-feedback evaluation metrics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch

from msg_embedding.core.logging import get_logger

_log = get_logger(__name__)

_COMPLEX_FEATURES: tuple[str, ...] = (
    "srs1", "srs2", "srs3", "srs4",
    "pmi1", "pmi2", "pmi3", "pmi4",
    "dft1", "dft2", "dft3", "dft4",
)
_REAL_FEATURES: tuple[str, ...] = ("rsrp_srs", "rsrp_cb", "cell_rsrp", "pdp_crop")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_device(obj: Any, device: str | torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


def _safe_nmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    """Return ``||pred - target||^2 / ||target||^2`` as a float (NaN-safe)."""
    if pred.is_complex() or target.is_complex():
        num = torch.sum(torch.abs(pred - target) ** 2).item()
        den = torch.sum(torch.abs(target) ** 2).item()
    else:
        num = torch.sum((pred.float() - target.float()) ** 2).item()
        den = torch.sum(target.float() ** 2).item()
    return float(num / max(den, eps))


def _linear_to_db(x: float, eps: float = 1e-12) -> float:
    return float(10.0 * np.log10(max(x, eps)))


# ---------------------------------------------------------------------------
# Reconstruction NMSE
# ---------------------------------------------------------------------------

def nmse_reconstruction(
    model: Any,
    feature_extractor: Any,
    samples: Iterable[dict[str, Any]],
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Per-feature reconstruction NMSE in dB for a batch of samples."""
    model = model.to(device).eval()
    feature_extractor = feature_extractor.to(device).eval()

    sum_sq_err: dict[str, float] = {}
    sum_sq_target: dict[str, float] = {}
    n_total = 0

    with torch.no_grad():
        for chunk in samples:
            feat = _to_device(chunk.get("feat"), device)
            if feat is None:
                raise ValueError("each sample dict must contain a 'feat' field")
            feat_gt = _to_device(chunk.get("feat_gt", feat), device)

            any_t = next(iter(feat.values()))
            b = int(any_t.shape[0])
            n_total += b

            tokens, norm_stats = feature_extractor(feat)
            snr = feat.get("srs_sinr")
            recon_tokens = model(tokens, snr)
            recon_feat = model.reconstruct(recon_tokens)

            for k in _COMPLEX_FEATURES:
                if k not in recon_feat or k not in feat_gt:
                    continue
                r = recon_feat[k]
                t = feat_gt[k]
                rms_entry = norm_stats.get(k) if isinstance(norm_stats, dict) else None
                if rms_entry is not None and "rms" in rms_entry:
                    rms = rms_entry["rms"].unsqueeze(-1)
                    t_norm = t / (rms + 1e-8)
                else:
                    t_norm = t
                min_d = min(r.shape[-1], t_norm.shape[-1])
                r_c = r[..., :min_d]
                t_c = t_norm[..., :min_d]
                err = torch.sum(torch.abs(r_c - t_c) ** 2).item()
                tgt = torch.sum(torch.abs(t_c) ** 2).item()
                sum_sq_err[k] = sum_sq_err.get(k, 0.0) + err
                sum_sq_target[k] = sum_sq_target.get(k, 0.0) + tgt

            for k in _REAL_FEATURES:
                if k not in recon_feat or k not in feat_gt:
                    continue
                r = recon_feat[k].float()
                t = feat_gt[k].float()
                if t.shape != r.shape:
                    min_d = min(t.shape[-1], r.shape[-1])
                    r = r[..., :min_d]
                    t = t[..., :min_d]
                err = torch.sum((r - t) ** 2).item()
                tgt = torch.sum(t ** 2).item()
                sum_sq_err[k] = sum_sq_err.get(k, 0.0) + err
                sum_sq_target[k] = sum_sq_target.get(k, 0.0) + tgt

    per_feature: dict[str, float] = {}
    for k, num in sum_sq_err.items():
        den = sum_sq_target.get(k, 0.0)
        per_feature[k] = _linear_to_db(num / max(den, 1e-12))

    if per_feature:
        total_num = sum(sum_sq_err.values())
        total_den = sum(sum_sq_target.values())
        nmse_overall_db = _linear_to_db(total_num / max(total_den, 1e-12))
    else:
        nmse_overall_db = float("nan")

    return {
        "nmse_dB": nmse_overall_db,
        "nmse_per_feature": per_feature,
        "n_samples_total": n_total,
    }


# ---------------------------------------------------------------------------
# Cosine distribution / pair-label separation
# ---------------------------------------------------------------------------

def cosine_distribution(
    embeddings: np.ndarray,
    pair_labels: Sequence[str] | np.ndarray,
) -> dict[str, float]:
    """Cosine-similarity summary statistics, grouped by pair label."""
    emb = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(list(pair_labels))
    if emb.ndim != 2:
        raise ValueError(f"embeddings must be 2-D [2P, D], got {emb.shape}")
    if emb.shape[0] % 2 != 0:
        raise ValueError(f"embeddings must have even N (pairs); got {emb.shape[0]}")
    p = emb.shape[0] // 2
    if labels.shape[0] != p:
        raise ValueError(f"pair_labels length {labels.shape[0]} != {p}")

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    unit = emb / norms
    anchor = unit[0::2]
    partner = unit[1::2]
    cos = np.sum(anchor * partner, axis=1)

    out: dict[str, float] = {}
    for label in ("clean_noisy", "same_ue", "diff_ue"):
        mask = labels == label
        if mask.any():
            out[f"{label}_mean"] = float(cos[mask].mean())
            out[f"{label}_std"] = float(cos[mask].std(ddof=0))
            out[f"{label}_n"] = int(mask.sum())
        else:
            out[f"{label}_mean"] = float("nan")
            out[f"{label}_std"] = float("nan")
            out[f"{label}_n"] = 0

    clean_noisy_mean = out["clean_noisy_mean"]
    diff_ue_mean = out["diff_ue_mean"]
    if np.isnan(clean_noisy_mean) or np.isnan(diff_ue_mean):
        out["separation_margin"] = 0.0
    else:
        out["separation_margin"] = float(clean_noisy_mean - diff_ue_mean)
    return out


# ---------------------------------------------------------------------------
# Compression feedback equivalence
# ---------------------------------------------------------------------------

def _svd_rank1_baseline(feat: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return a normalised ``[B, 2 * TX_ANT]`` float baseline payload."""
    pmi1 = feat.get("pmi1")
    if pmi1 is None:
        raise ValueError("svd_rank1 baseline requires 'pmi1' in feat dict")
    if not pmi1.is_complex():
        raise ValueError("pmi1 must be complex-valued for SVD baseline")
    real = pmi1.real.float()
    imag = pmi1.imag.float()
    return torch.cat([real, imag], dim=-1)


def _pmi_type1_baseline(feat: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return a quantised ``[B, 2 * TX_ANT]`` float Type-I PMI baseline."""
    pmi1 = feat.get("pmi1")
    if pmi1 is None:
        raise ValueError("pmi_type1 baseline requires 'pmi1' in feat dict")
    phases = torch.angle(pmi1)
    q_phases = torch.round(phases / (np.pi / 2.0)) * (np.pi / 2.0)
    q = torch.abs(pmi1) * torch.exp(1j * q_phases)
    real = q.real.float()
    imag = q.imag.float()
    return torch.cat([real, imag], dim=-1)


def _sinr_proxy_from_payload(payload: torch.Tensor, reference: torch.Tensor) -> float:
    """Return a SINR proxy (dB) based on cosine alignment with the reference."""
    flat_pay = payload.reshape(payload.shape[0], -1).float()
    flat_ref = reference.reshape(reference.shape[0], -1).float()
    d = min(flat_pay.shape[-1], flat_ref.shape[-1])
    fp = flat_pay[:, :d]
    fr = flat_ref[:, :d]
    fp = fp / (torch.linalg.norm(fp, dim=-1, keepdim=True) + 1e-12)
    fr = fr / (torch.linalg.norm(fr, dim=-1, keepdim=True) + 1e-12)
    rho = (fp * fr).sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    rho2 = (rho ** 2).mean().item()
    sinr_lin = rho2 / max(1.0 - rho2, 1e-12)
    return _linear_to_db(sinr_lin)


def compression_feedback_equiv(
    model: Any,
    fe: Any,
    samples: Iterable[dict[str, Any]],
    baseline: str = "pmi_type1",
) -> dict[str, float]:
    """Compare 16-D latent-feedback SINR vs a classical CSI baseline."""
    if baseline not in {"pmi_type1", "svd_rank1"}:
        raise ValueError(f"unknown baseline {baseline!r}")
    model = model.eval()
    fe = fe.eval()

    lat_sinr_accum = 0.0
    base_sinr_accum = 0.0
    n_batches = 0
    latent_dim_seen = 0
    baseline_dim_seen = 0

    with torch.no_grad():
        for chunk in samples:
            feat = chunk.get("feat")
            if feat is None:
                raise ValueError("each sample dict must contain 'feat'")
            feat_gt = chunk.get("feat_gt", feat)

            ref = _svd_rank1_baseline(feat_gt)

            tokens, _stats = fe(feat)
            snr = feat.get("srs_sinr")
            z = model.get_latent(tokens, snr, is_infer=True)
            latent_dim_seen = int(z.shape[-1])

            if baseline == "pmi_type1":
                base_payload = _pmi_type1_baseline(feat)
            else:
                base_payload = _svd_rank1_baseline(feat)
            baseline_dim_seen = int(base_payload.shape[-1])

            lat_sinr_accum += _sinr_proxy_from_payload(z, ref)
            base_sinr_accum += _sinr_proxy_from_payload(base_payload, ref)
            n_batches += 1

    if n_batches == 0:
        return {
            "sinr_diff_dB": 0.0,
            "compression_ratio": 0.0,
            "latent_sinr_dB": float("nan"),
            "baseline_sinr_dB": float("nan"),
        }

    lat_db = lat_sinr_accum / n_batches
    base_db = base_sinr_accum / n_batches
    ratio = (
        baseline_dim_seen / latent_dim_seen if latent_dim_seen > 0 else float("nan")
    )
    return {
        "sinr_diff_dB": float(lat_db - base_db),
        "compression_ratio": float(ratio),
        "latent_sinr_dB": float(lat_db),
        "baseline_sinr_dB": float(base_db),
    }


__all__ = [
    "nmse_reconstruction",
    "cosine_distribution",
    "compression_feedback_equiv",
]
