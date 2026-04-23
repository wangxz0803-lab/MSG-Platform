"""Channel Explorer endpoints -- serves bridged channel data from bridge_out/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import APIRouter, HTTPException, Query

from ..settings import get_settings

router = APIRouter(prefix="/api/channels", tags=["channels"])


def _bridge_out_dir() -> Path:
    return get_settings().data_path


def _sorted_pt_files() -> list[Path]:
    """Return all .pt files in bridge_out and its subdirectories."""
    d = _bridge_out_dir()
    if not d.is_dir():
        return []
    return sorted(d.rglob("sample_*.pt"))


def _load_sample(path: Path) -> dict[str, Any]:
    """Load a single .pt sample from disk."""
    return torch.load(path, map_location="cpu", weights_only=False)


def _to_list(v: Any) -> Any:
    """Recursively convert torch tensors / numpy arrays to plain Python lists."""
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy().tolist()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _to_list(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_list(item) for item in v]
    if isinstance(v, np.floating | np.integer):
        return v.item()
    return v


def _magnitude_heatmap(ch: dict[str, Any]) -> list[list[float]]:
    """Compute magnitude heatmap [T][RB] averaged across BS x UE dimensions."""
    real = np.asarray(ch["real"], dtype=np.float32)
    imag = np.asarray(ch["imag"], dtype=np.float32)
    mag = np.sqrt(real**2 + imag**2)
    mag_avg = mag.mean(axis=(2, 3))
    return mag_avg.tolist()


def _error_heatmap(
    ideal: dict[str, Any], est: dict[str, Any]
) -> list[list[float]]:
    """Compute |ideal - est| heatmap [T][RB] averaged across BS x UE."""
    r_i = np.asarray(ideal["real"], dtype=np.float32)
    i_i = np.asarray(ideal["imag"], dtype=np.float32)
    r_e = np.asarray(est["real"], dtype=np.float32)
    i_e = np.asarray(est["imag"], dtype=np.float32)
    err = np.sqrt((r_i - r_e) ** 2 + (i_i - i_e) ** 2)
    err_avg = err.mean(axis=(2, 3))
    return err_avg.tolist()


_COMPLEX_PREFIXES = ("srs", "pmi", "dft")


def _is_complex_feature(name: str) -> bool:
    """Check if a feature key belongs to a complex-valued group."""
    lower = name.lower()
    return any(lower.startswith(p) for p in _COMPLEX_PREFIXES)


def _extract_features(feat_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a feat_clean / feat_noisy dict into a JSON-friendly list."""
    results: list[dict[str, Any]] = []
    for key, val in feat_dict.items():
        if isinstance(val, torch.Tensor):
            raw = val.detach().cpu()
            is_cplx = raw.is_complex()
            arr = raw.numpy()
        else:
            arr = np.asarray(val)
            is_cplx = np.iscomplexobj(arr)

        entry: dict[str, Any] = {
            "name": key,
            "shape": list(arr.shape),
        }
        if is_cplx or _is_complex_feature(key):
            carr = arr.astype(np.complex64) if is_cplx else arr.astype(np.float32)
            if is_cplx:
                mag = np.abs(carr).astype(np.float32)
                phase = np.angle(carr).astype(np.float32)
                entry["magnitude"] = mag.tolist()
                entry["phase"] = phase.tolist()
            else:
                entry["values"] = carr.tolist()
        else:
            entry["values"] = np.asarray(arr, dtype=np.float32).tolist()
        results.append(entry)
    return results


@router.get("")
def list_channels(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> dict[str, Any]:
    """List available bridged channel samples with pagination."""
    files = _sorted_pt_files()
    total = len(files)
    page = files[offset : offset + limit]

    items: list[dict[str, Any]] = []
    bridge_out = _bridge_out_dir()
    for idx_offset, f in enumerate(page):
        global_idx = offset + idx_offset
        try:
            sample = _load_sample(f)
            meta_raw = sample.get("meta", {})
            meta_summary = {
                k: _to_list(v)
                for k, v in meta_raw.items()
                if k
                in (
                    "snr_dB",
                    "sinr_dB",
                    "sir_dB",
                    "ul_sir_dB",
                    "dl_sir_dB",
                    "source",
                    "link",
                    "link_pairing",
                    "channel_est_mode",
                    "serving_cell_id",
                    "num_interfering_ues",
                )
            }
        except Exception:
            meta_summary = {}

        if "source" not in meta_summary and f.parent != bridge_out:
            meta_summary["source"] = f.parent.name

        items.append(
            {
                "index": global_idx,
                "filename": f.name,
                "source": meta_summary.get("source", f.parent.name),
                "meta": meta_summary,
            }
        )

    return {"total": total, "offset": offset, "limit": limit, "items": items}


@router.get("/{index}")
def get_channel(index: int) -> dict[str, Any]:
    """Return full visualization payload for a single sample."""
    files = _sorted_pt_files()
    if index < 0 or index >= len(files):
        raise HTTPException(
            status_code=404,
            detail=f"Sample index {index} out of range (0..{len(files) - 1}).",
        )

    try:
        sample = _load_sample(files[index])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load sample: {exc}",
        ) from exc

    ch_ideal = sample.get("channel_ideal", {})
    ch_est = sample.get("channel_est", {})
    shape = ch_ideal.get("shape", ch_est.get("shape", []))

    channel_ideal_hm = _magnitude_heatmap(ch_ideal) if ch_ideal else []
    channel_est_hm = _magnitude_heatmap(ch_est) if ch_est else []
    channel_error_hm = _error_heatmap(ch_ideal, ch_est) if ch_ideal and ch_est else []

    feat_clean = sample.get("feat_clean", {})
    feat_noisy = sample.get("feat_noisy", {})
    feat_src = feat_clean if feat_clean else feat_noisy
    feature_list = _extract_features(feat_src)
    features = {f["name"]: f for f in feature_list}

    meta = _to_list(sample.get("meta", {}))

    return {
        "index": index,
        "filename": files[index].name,
        "shape": _to_list(shape),
        "channel_ideal": channel_ideal_hm,
        "channel_est": channel_est_hm,
        "channel_error": channel_error_hm,
        "features": features,
        "meta": meta,
    }
