"""End-to-end evaluation orchestrator for a trained ChannelMAE checkpoint."""

from __future__ import annotations

import json
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from msg_embedding.core.logging import get_logger
from msg_embedding.features.extractor import FeatureExtractor
from msg_embedding.models.channel_mae import ChannelMAE

from ..data.dataset import ChannelDataset
from . import channel_charting as cc
from . import prediction as pred

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataclass result
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Evaluation result — the JSON representation is the platform contract."""

    run_id: str
    ckpt_path: str
    git_sha: str
    timestamp: str
    ct: float
    tw: float
    knn_consistency: float
    nmse_dB: float
    cosine_sep_margin: float
    kendall_tau: float
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_git_sha(repo_root: Path | None = None) -> str:
    """Return the git SHA of ``repo_root`` (defaults to repo root of this file)."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return sha.decode("ascii").strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unknown"


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(v):
        return None
    return v


def _feat_dict_from_record(record: dict[str, Any]) -> dict[str, torch.Tensor] | None:
    """Best-effort synthesis of a FeatureExtractor feat dict from a dataset record."""
    if "feat" in record and isinstance(record["feat"], dict):
        return record["feat"]
    return None


def _extract_ue_xyz(record: dict[str, Any]) -> np.ndarray | None:
    """Pull ``ue_position`` out of ``record['meta']`` if available."""
    meta = record.get("meta") or {}
    pos = meta.get("ue_position") or meta.get("ue_xyz")
    if pos is None:
        return None
    arr = np.asarray(pos, dtype=np.float64).reshape(-1)
    if arr.shape[0] < 2:
        return None
    return arr


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def _collect_embeddings(
    model: Any,
    fe: Any,
    dataset: ChannelDataset,
    device: str | torch.device,
    limit: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Iterate ``dataset`` and return ``(embeddings[N, D], metadata_df)``."""
    model.eval()
    fe.eval()
    n = len(dataset) if limit is None else min(limit, len(dataset))

    rows: list[dict[str, Any]] = []
    latents: list[np.ndarray] = []

    with torch.no_grad():
        for i in range(n):
            record = dataset[i]
            feat = _feat_dict_from_record(record)
            if feat is None:
                rows.append(_row_from_record(record, None))
                continue
            feat_dev = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in feat.items()
            }
            tokens, _stats = fe(feat_dev)
            snr = feat_dev.get("srs_sinr")
            z = model.get_latent(tokens, snr, is_infer=True)
            latents.append(z.detach().cpu().numpy().reshape(-1))
            rows.append(_row_from_record(record, z.detach().cpu().numpy().reshape(-1)))

    df = pd.DataFrame(rows)
    if not latents:
        return np.zeros((0, 0), dtype=np.float32), df
    return np.stack(latents, axis=0).astype(np.float32), df


def _row_from_record(record: dict[str, Any], z: np.ndarray | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "uuid": record.get("uuid"),
        "sinr_dB": _maybe_float(record.get("sinr_dB")),
        "snr_dB": _maybe_float(record.get("snr_dB")),
        "link": record.get("link"),
        "source": record.get("source"),
    }
    ue = _extract_ue_xyz(record)
    if ue is not None:
        row["ue_x"] = float(ue[0])
        row["ue_y"] = float(ue[1])
        row["ue_z"] = float(ue[2]) if ue.shape[0] > 2 else 0.0
    if z is not None:
        for i, v in enumerate(z, start=1):
            row[f"z_{i}"] = float(v)
    return row


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _load_state_dict(ckpt_path: Path, device: str | torch.device) -> dict[str, Any]:
    """Load a checkpoint, handling both ``{'model': sd, ...}`` and raw state dicts."""
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        return state["model"]
    if isinstance(state, dict):
        return state
    raise TypeError(
        f"expected dict state from {ckpt_path}, got {type(state).__name__}"
    )


def run_eval(
    ckpt_path: Path | str,
    dataset: ChannelDataset,
    output_dir: Path | str,
    eval_cfg: dict[str, Any] | None = None,
    device: str | torch.device = "cpu",
) -> EvalResult:
    """Run the full downstream evaluation and persist artefacts."""
    cfg = dict(eval_cfg or {})
    run_id = str(cfg.get("run_id") or _make_run_id())
    output_dir = Path(output_dir)
    out_run = output_dir / run_id
    out_run.mkdir(parents=True, exist_ok=True)

    model = ChannelMAE(None).to(device)
    fe = FeatureExtractor().to(device)
    sd = _load_state_dict(Path(ckpt_path), device=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        _log.info(
            "load_state_dict_partial",
            missing=len(missing),
            unexpected=len(unexpected),
        )

    limit = cfg.get("limit")
    embeddings, meta_df = _collect_embeddings(
        model, fe, dataset, device=device, limit=limit
    )

    if "z_1" in meta_df.columns:
        has_z = meta_df["z_1"].notna()
        valid_df = meta_df.loc[has_z].reset_index(drop=True)
    else:
        valid_df = meta_df.iloc[0:0].reset_index(drop=True)
    n_valid = int(len(valid_df))
    if embeddings.ndim != 2 or embeddings.shape[0] != n_valid:
        n_valid = min(n_valid, int(embeddings.shape[0]) if embeddings.size else 0)
        valid_df = valid_df.iloc[:n_valid].reset_index(drop=True)
        if embeddings.size:
            embeddings = embeddings[:n_valid]

    # -------- Channel charting metrics ---------------------------------
    k_neigh = int(cfg.get("k_neighbors", 10))
    knn_k = int(cfg.get("knn_k", 5))

    attr_cols: list[np.ndarray] = []
    if "sinr_dB" in valid_df:
        attr_cols.append(valid_df["sinr_dB"].fillna(0.0).to_numpy(dtype=np.float64)[:, None])
    if "link" in valid_df:
        attr_cols.append(
            pd.get_dummies(valid_df["link"].fillna("UL")).to_numpy(dtype=np.float64)
        )
    if "source" in valid_df:
        attr_cols.append(
            pd.get_dummies(valid_df["source"].fillna("unknown")).to_numpy(dtype=np.float64)
        )
    attrs = (
        np.concatenate(attr_cols, axis=1) if attr_cols else np.zeros((n_valid, 1))
    )

    if {"ue_x", "ue_y"}.issubset(valid_df.columns):
        coords = valid_df[["ue_x", "ue_y"]].fillna(0.0).to_numpy(dtype=np.float64)
    elif "sinr_dB" in valid_df.columns and n_valid > 0:
        coords = valid_df["sinr_dB"].fillna(0.0).to_numpy(dtype=np.float64)[:, None]
    else:
        coords = np.zeros((n_valid, 2))

    can_knn = n_valid > max(k_neigh, knn_k) + 1
    tw = cc.trustworthiness(coords, embeddings, k=min(k_neigh, n_valid - 1)) if can_knn else float("nan")
    ct = cc.continuity(coords, embeddings, k=min(k_neigh, n_valid - 1)) if can_knn else float("nan")
    knn_c = (
        cc.knn_consistency(embeddings, attrs, k=min(knn_k, n_valid - 1))
        if can_knn and attrs.size
        else float("nan")
    )
    tau = (
        cc.kendall_tau_global(embeddings, coords) if n_valid >= 3 else float("nan")
    )

    # -------- Prediction / compression metrics -------------------------
    nmse_db = float("nan")
    cos_margin = 0.0
    if cfg.get("compute_nmse", False):
        nmse_samples = cfg.get("nmse_samples", [])
        nmse_result = pred.nmse_reconstruction(
            model=model, feature_extractor=fe, samples=nmse_samples, device=device
        )
        nmse_db = float(nmse_result["nmse_dB"])

    if cfg.get("cosine_pairs") is not None:
        pair_emb = np.asarray(cfg["cosine_pairs"]["embeddings"])
        pair_labels = cfg["cosine_pairs"]["labels"]
        cos_stats = pred.cosine_distribution(pair_emb, pair_labels)
        cos_margin = float(cos_stats.get("separation_margin", 0.0))

    # -------- Persist --------------------------------------------------
    git_sha = get_git_sha()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    result = EvalResult(
        run_id=run_id,
        ckpt_path=str(ckpt_path),
        git_sha=git_sha,
        timestamp=timestamp,
        ct=float(ct),
        tw=float(tw),
        knn_consistency=float(knn_c),
        nmse_dB=float(nmse_db),
        cosine_sep_margin=float(cos_margin),
        kendall_tau=float(tau),
        meta={
            "ckpt": str(ckpt_path),
            "git_sha": git_sha,
            "dataset_size": len(dataset),
            "embeddings_used": n_valid,
            "device": str(device),
            "eval_config": cfg,
        },
    )

    _write_metrics_json(result, out_run / "metrics.json")
    _write_embeddings_parquet(valid_df, out_run / "embeddings.parquet")
    return result


def _make_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _write_metrics_json(result: EvalResult, path: Path) -> None:
    data = result.to_dict()
    payload = _sanitize_for_json(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_embeddings_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(df) == 0:
        pd.DataFrame(columns=["uuid"]).to_parquet(path, index=False)
        return
    df.to_parquet(path, index=False)


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.floating | np.integer):
        obj = obj.item()
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if obj is None or isinstance(obj, str | int | bool):
        return obj
    try:
        return json.loads(json.dumps(obj, default=str))
    except (TypeError, ValueError):
        return str(obj)


# ---------------------------------------------------------------------------
# Readiness check
# ---------------------------------------------------------------------------

def is_ready() -> bool:
    """Return True if ChannelMAE / FeatureExtractor are importable."""
    try:
        from msg_embedding.features.extractor import FeatureExtractor as _FE  # noqa: F401
        from msg_embedding.models.channel_mae import ChannelMAE as _MAE  # noqa: F401
        return True
    except Exception:
        return False


__all__ = [
    "EvalResult",
    "get_git_sha",
    "is_ready",
    "run_eval",
]
