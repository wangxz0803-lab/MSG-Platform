"""Post-QuaDRiGa pipeline: convert real .mat -> .pt, then bridge.

Waits for all 20 QuaDRiGa .mat shards (1000 samples), converts to
ChannelSample .pt, then runs bridge to produce combined channel+features.
Sources: QuaDRiGa (real MATLAB) + Sionna RT.

Usage:
    python scripts/post_quadriga_pipeline.py
"""

from __future__ import annotations

import gc
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from msg_embedding.data.bridge import _build_feat_dict
from msg_embedding.data.contract import ChannelSample
from msg_embedding.features.extractor import FeatureExtractor
from msg_embedding.models.channel_mae import ChannelMAE

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

QUADRIGA_MAT_DIR = _PROJECT_ROOT / "artifacts" / "quadriga_real_mat"
COLLECT_DIR = _PROJECT_ROOT / "artifacts" / "collect_quadriga_real"
BRIDGE_OUT = _PROJECT_ROOT / "bridge_out"
NUM_SHARDS = 20
TOTAL_SAMPLES = 1000


def step1_wait_for_mats():
    """Wait until all 20 .mat shards exist."""
    print("[Step 1] Waiting for QuaDRiGa .mat files...")
    while True:
        existing = sorted(QUADRIGA_MAT_DIR.glob("QDRG_real_shard*.mat"))
        if len(existing) >= NUM_SHARDS:
            print(f"  All {NUM_SHARDS} shards found!")
            return existing
        print(f"  {len(existing)}/{NUM_SHARDS} shards ready, waiting 5min...")
        time.sleep(300)


def step2_convert_mat_to_pt(mat_files):
    """Convert .mat files to ChannelSample .pt files."""
    print(f"\n[Step 2] Converting {len(mat_files)} .mat files to ChannelSample .pt...")
    COLLECT_DIR.mkdir(parents=True, exist_ok=True)

    existing_pts = sorted(COLLECT_DIR.glob("sample_*.pt"))
    if len(existing_pts) >= TOTAL_SAMPLES:
        print(f"  Already have {len(existing_pts)} .pt files, skipping conversion")
        return len(existing_pts)

    import scipy.io as sio

    sample_idx = len(existing_pts)
    ues_per_shard = 50
    start_from_shard = sample_idx // ues_per_shard

    for mat_path in sorted(mat_files):
        shard_num = int(mat_path.stem.split("shard")[-1])
        if shard_num < start_from_shard:
            continue

        print(f"  Loading {mat_path.name}...")
        try:
            d = sio.loadmat(str(mat_path))
        except Exception:
            import h5py

            d = {}
            with h5py.File(str(mat_path), "r") as f:
                for key in [
                    "Hf_serving_est",
                    "Hf_serving_ideal",
                    "rsrp_per_cell",
                    "snr_dB",
                    "sir_dB",
                    "sinr_dB",
                ]:
                    if key in f:
                        d[key] = np.array(f[key]).T
                meta_grp = f["meta"]
                d["meta"] = {}
                for k in meta_grp:
                    d["meta"][k] = np.array(meta_grp[k]).T

        Hf_est = d["Hf_serving_est"]
        Hf_ideal = d["Hf_serving_ideal"]
        mat_snr = np.asarray(d["snr_dB"]).flatten()
        mat_sir = np.asarray(d["sir_dB"]).flatten()
        mat_sinr = np.asarray(d["sinr_dB"]).flatten()
        mat_rsrp = np.asarray(d["rsrp_per_cell"])

        if isinstance(d.get("meta"), np.ndarray) and d["meta"].dtype.names:
            meta = d["meta"][0, 0]
        elif isinstance(d.get("meta"), dict):
            meta = d["meta"]
        else:
            meta = d.get("meta", {})

        no_ue = Hf_est.shape[0]

        def _meta_get(m, key, default=None):
            if isinstance(m, dict):
                return m.get(key, default)
            try:
                return m[key]
            except (KeyError, ValueError, IndexError):
                return default

        try:
            ue_pos = _meta_get(meta, "ue_positions", np.zeros((3, no_ue)))
            k_cells = int(_meta_get(meta, "num_cells", 7))
            noise_dBm_val = float(_meta_get(meta, "noise_dBm", -89.1))
            ptx_dBm_val = float(_meta_get(meta, "Ptx_BS_dBm", _meta_get(meta, "Ptx_dBm", 46.0)))
        except (KeyError, TypeError):
            ue_pos = np.zeros((3, no_ue))
            k_cells = 7
            noise_dBm_val = -89.1
            ptx_dBm_val = 46.0

        n_re = 273 * 12
        ptx_per_re_dBm = ptx_dBm_val - 10.0 * np.log10(n_re)

        for i_ue in range(no_ue):
            if sample_idx >= TOTAL_SAMPLES:
                break

            h_ideal = np.transpose(Hf_ideal[i_ue], (3, 2, 0, 1))
            h_est = np.transpose(Hf_est[i_ue], (3, 2, 0, 1))

            snr_dB = float(np.clip(mat_snr[i_ue], -50, 50))
            sir_dB = float(np.clip(mat_sir[i_ue], -50, 50))
            sinr_dB = float(np.clip(mat_sinr[i_ue], -50, 50))

            raw_gain = np.mean(np.abs(h_ideal) ** 2)
            if raw_gain > 0:
                scale = np.sqrt(raw_gain)
                h_ideal = h_ideal / scale
                h_est = h_est / scale

            try:
                ue_p = ue_pos[:, i_ue].flatten()
            except (IndexError, TypeError):
                ue_p = np.array([0.0, 0.0, 1.5])

            rsrp_row = mat_rsrp[i_ue] if i_ue < mat_rsrp.shape[0] else np.zeros(k_cells)
            ssb_rsrp_list = [
                float(ptx_per_re_dBm + 10 * np.log10(max(float(g), 1e-30)))
                for g in rsrp_row
            ]

            sample = ChannelSample(
                h_serving_true=h_ideal.astype(np.complex64),
                h_serving_est=h_est.astype(np.complex64),
                h_interferers=None,
                interference_signal=None,
                noise_power_dBm=float(noise_dBm_val),
                snr_dB=snr_dB,
                sir_dB=sir_dB,
                sinr_dB=sinr_dB,
                ssb_rsrp_dBm=ssb_rsrp_list,
                link="UL",
                channel_est_mode="ls_linear",
                serving_cell_id=0,
                ue_position=ue_p,
                source="quadriga_real",
                sample_id=str(uuid.uuid4()),
                created_at=datetime.now(),
                meta={
                    "num_cells": k_cells,
                    "isd_m": 500,
                    "scenario": "3GPP_38.901_UMa_NLOS",
                    "carrier_freq_hz": 3.5e9,
                    "shard": shard_num,
                    "ue_idx": i_ue,
                },
            )

            out_path = COLLECT_DIR / f"sample_{sample_idx:06d}.pt"
            torch.save(sample.to_dict(), str(out_path))
            sample_idx += 1

        del Hf_est, Hf_ideal, d
        gc.collect()
        print(f"    Shard {shard_num}: converted, total {sample_idx} samples")

    summary = {
        "status": "completed",
        "saved": sample_idx,
        "requested": TOTAL_SAMPLES,
        "errors": 0,
        "finished_at": datetime.now().isoformat(),
        "source": "quadriga_real",
    }
    (COLLECT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    run_meta = {
        "config": {
            "scenario": "3GPP_38.901_UMa_NLOS",
            "num_cells": 7,
            "num_ues": 1000,
            "isd_m": 500,
            "carrier_freq_hz": 3.5e9,
            "channel_est_mode": "ls_linear",
            "link": "UL",
            "bs_antennas": 64,
            "ue_antennas": 2,
            "num_rb": 273,
            "bandwidth_mhz": 100,
            "scs_khz": 30,
            "seed": 42,
            "engine": "QuaDRiGa 2.8.1 (real MATLAB)",
            "estimation": "per-port LS with real multi-cell interference",
        }
    }
    (COLLECT_DIR / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

    print(f"  Conversion complete: {sample_idx} samples -> {COLLECT_DIR}")
    return sample_idx


def step3_recompute_stats():
    """Recompute dataset stats with real QuaDRiGa data."""
    print("\n[Step 3] Recomputing dataset statistics...")

    collect_dirs = {
        "sionna_rt": _PROJECT_ROOT / "artifacts" / "collect_sionna_rt",
        "quadriga_real": COLLECT_DIR,
    }

    all_stats = {}
    for name, d in collect_dirs.items():
        pts = sorted(d.glob("sample_*.pt"))[:1000]
        snrs, sinrs, powers, links, est_modes = [], [], [], [], []
        ue_positions = []

        for i, p in enumerate(pts):
            data = torch.load(str(p), map_location="cpu", weights_only=False)
            snrs.append(float(data.get("snr_dB", 0)))
            sinrs.append(float(data.get("sinr_dB", 0)) if data.get("sinr_dB") is not None else 0)
            links.append(data.get("link", "DL"))
            est_modes.append(data.get("channel_est_mode", "ideal"))

            h = data.get("h_serving_true") or data.get("h_serving_est")
            if h is not None:
                if isinstance(h, list | tuple):
                    h_arr = (np.array(h[0]) + 1j * np.array(h[1])).astype(np.complex64)
                elif isinstance(h, np.ndarray):
                    h_arr = h
                else:
                    h_arr = np.zeros((1,))
                powers.append(float(np.mean(np.abs(h_arr) ** 2)))
            else:
                powers.append(1.0)

            pos = data.get("ue_position")
            if pos is not None:
                pos_arr = np.array(pos).flatten()
                if len(pos_arr) >= 2:
                    ue_positions.append(pos_arr[:3])

            del data
            if (i + 1) % 1000 == 0:
                print(f"  {name}: {i + 1}/{len(pts)}")

        snrs_a = np.array(snrs)
        sinrs_a = np.array(sinrs)
        powers_a = np.array(powers)

        stats = {
            "count": len(pts),
            "snr_dB": {
                "mean": float(np.mean(snrs_a)),
                "std": float(np.std(snrs_a)),
                "min": float(np.min(snrs_a)),
                "max": float(np.max(snrs_a)),
                "p25": float(np.percentile(snrs_a, 25)),
                "p50": float(np.percentile(snrs_a, 50)),
                "p75": float(np.percentile(snrs_a, 75)),
            },
            "sinr_dB": {
                "mean": float(np.mean(sinrs_a)),
                "std": float(np.std(sinrs_a)),
                "min": float(np.min(sinrs_a)),
                "max": float(np.max(sinrs_a)),
                "p25": float(np.percentile(sinrs_a, 25)),
                "p50": float(np.percentile(sinrs_a, 50)),
                "p75": float(np.percentile(sinrs_a, 75)),
            },
            "channel_power": {
                "mean": float(np.mean(powers_a)),
                "std": float(np.std(powers_a)),
                "min": float(np.min(powers_a)),
                "max": float(np.max(powers_a)),
            },
            "links": dict(zip(*np.unique(links, return_counts=True), strict=False)),
            "channel_est_modes": dict(zip(*np.unique(est_modes, return_counts=True), strict=False)),
        }
        if ue_positions:
            pos = np.array(ue_positions)
            stats["ue_position"] = {
                "x_range": [float(pos[:, 0].min()), float(pos[:, 0].max())],
                "y_range": [float(pos[:, 1].min()), float(pos[:, 1].max())],
            }

        run_meta_path = d / "run_meta.json"
        if run_meta_path.exists():
            stats["config"] = json.loads(run_meta_path.read_text()).get("config", {})
        stats["source"] = name

        all_stats[name] = stats
        print(
            f"  {name}: {stats['count']} samples, "
            f"SNR={stats['snr_dB']['mean']:.1f}+/-{stats['snr_dB']['std']:.1f} dB"
        )

    stats_path = BRIDGE_OUT / "dataset_stats.json"
    stats_path.write_text(json.dumps(all_stats, indent=2, default=str), encoding="utf-8")
    print(f"  Stats saved -> {stats_path}")
    return all_stats


def step4_bridge():
    """Bridge all sources."""
    print("\n[Step 4] Running bridge pipeline...")

    sources = {
        "sionna_rt": _PROJECT_ROOT / "artifacts" / "collect_sionna_rt",
        "quadriga_real": COLLECT_DIR,
    }

    for name, collect_dir in sources.items():
        out_dir = BRIDGE_OUT / name
        out_dir.mkdir(parents=True, exist_ok=True)

        existing = sorted(out_dir.glob("shard_*.pt"))
        if len(existing) >= 50:
            print(f"  {name}: already bridged ({len(existing)} shards), skipping")
            continue

        pts = sorted(collect_dir.glob("sample_*.pt"))[:1000]
        batch_size = 100
        total_saved = 0
        shard_idx = len(existing)
        resume_from = shard_idx

        if resume_from > 0:
            total_saved = resume_from * batch_size
            print(f"  {name}: resuming from shard {resume_from}")

        for start in range(0, len(pts), batch_size):
            current_shard = start // batch_size
            if current_shard < resume_from:
                continue

            chunk_paths = pts[start : start + batch_size]
            feats = []
            for p in chunk_paths:
                try:
                    d = torch.load(str(p), map_location="cpu", weights_only=False)
                    s = ChannelSample.from_dict(d)
                    fd, _ = _build_feat_dict(s, use_legacy_pmi=False)
                    feats.append(fd)
                    del s, d
                except Exception:
                    pass

            if feats:
                stacked = {}
                for key in feats[0]:
                    stacked[key] = torch.cat([f[key] for f in feats], dim=0)
                shard_path = out_dir / f"shard_{shard_idx:04d}.pt"
                torch.save(stacked, shard_path)
                total_saved += len(feats)
                del stacked, feats
                shard_idx += 1

            pct = min(100, (start + len(chunk_paths)) / len(pts) * 100)
            if int(pct) % 20 == 0 or start + len(chunk_paths) >= len(pts):
                print(f"  Bridge [{name}] {pct:.0f}% ({total_saved} saved)")

        print(f"  {name}: {total_saved} samples bridged")


def step5_train():
    """Train MAE model."""
    print("\n[Step 5] Training ChannelMAE (30 epochs)...")

    shard_files = sorted(BRIDGE_OUT.rglob("shard_*.pt"))
    if not shard_files:
        print("  No shards found!")
        return {}

    all_feats: dict[str, list] = {}
    for sf in shard_files:
        feat = torch.load(str(sf), map_location="cpu", weights_only=False)
        for k, v in feat.items():
            all_feats.setdefault(k, []).append(v)

    merged = {k: torch.cat(vs, dim=0) for k, vs in all_feats.items()}
    n_total = merged[list(merged.keys())[0]].shape[0]
    print(f"  Training on {n_total} samples")

    perm = torch.randperm(n_total)
    n_train = int(n_total * 0.9)
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    train_feat = {k: v[train_idx] for k, v in merged.items()}
    val_feat = {k: v[val_idx] for k, v in merged.items()}

    feat_ext = FeatureExtractor()
    mae = ChannelMAE(None)
    mae.train()

    optimizer = torch.optim.AdamW(mae.parameters(), lr=1e-4, weight_decay=0.01)
    epochs = 30
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    batch_size = 64

    train_log: dict[str, list] = {"epochs": [], "train_loss": [], "val_loss": [], "learning_rate": []}

    for epoch in range(epochs):
        mae.train()
        epoch_loss, n_batches = 0.0, 0
        perm_e = torch.randperm(n_train)

        for i in range(0, n_train, batch_size):
            idx = perm_e[i : i + batch_size]
            batch = {k: v[idx] for k, v in train_feat.items()}
            with torch.no_grad():
                tokens, _ = feat_ext(batch)
            snr = batch.get("srs_sinr", torch.zeros(tokens.shape[0]))
            mask_ratio = [0.15, 0.3, 0.5][epoch % 3]

            recon = mae(tokens, snr, mask_ratio=mask_ratio)
            loss = torch.nn.functional.mse_loss(recon, tokens)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = epoch_loss / max(n_batches, 1)

        mae.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val_idx), batch_size):
                idx_v = torch.arange(i, min(i + batch_size, len(val_idx)))
                batch_v = {k: v[idx_v] for k, v in val_feat.items()}
                tok_v, _ = feat_ext(batch_v)
                snr_v = batch_v.get("srs_sinr", torch.zeros(tok_v.shape[0]))
                recon_v = mae(tok_v, snr_v, mask_ratio=0.3)
                val_loss += torch.nn.functional.mse_loss(recon_v, tok_v).item()
                n_val += 1

        avg_val = val_loss / max(n_val, 1)
        train_log["epochs"].append(epoch + 1)
        train_log["train_loss"].append(round(avg_train, 6))
        train_log["val_loss"].append(round(avg_val, 6))
        train_log["learning_rate"].append(round(scheduler.get_last_lr()[0], 8))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: train={avg_train:.4f} val={avg_val:.4f}")

    ckpt_dir = _PROJECT_ROOT / "artifacts" / "train_5k"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": mae.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epochs, "train_log": train_log},
        ckpt_dir / "best.pth",
    )

    train_log["final_train_loss"] = train_log["train_loss"][-1]
    train_log["final_val_loss"] = train_log["val_loss"][-1]
    train_log["best_epoch"] = int(np.argmin(train_log["val_loss"])) + 1

    log_path = BRIDGE_OUT / "train_log.json"
    log_path.write_text(json.dumps(train_log, indent=2))

    metrics = {
        **train_log,
        "total_samples": n_total,
        "embedding_dim": 16,
        "sources": ["sionna_rt", "quadriga_real"],
        "meta": {
            "run_id": "train_1k",
            "ckpt": str(ckpt_dir / "best.pth"),
            "timestamp": datetime.now().isoformat(),
        },
    }
    (ckpt_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"  Training complete: val_loss={avg_val:.4f}")
    return train_log


def step6_inference():
    """Run inference to get embeddings."""
    print("\n[Step 6] Running inference...")

    ckpt_path = _PROJECT_ROOT / "artifacts" / "train_5k" / "best.pth"
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    feat_ext = FeatureExtractor()
    mae = ChannelMAE(None)
    mae.load_state_dict(ckpt["model"])
    mae.eval()

    shard_files = sorted(BRIDGE_OUT.rglob("shard_*.pt"))
    all_z = []
    for sf in shard_files:
        feat = torch.load(str(sf), map_location="cpu", weights_only=False)
        with torch.no_grad():
            tokens, _ = feat_ext(feat)
            snr = feat.get("srs_sinr", torch.zeros(tokens.shape[0]))
            z = mae.get_latent(tokens, snr)
            all_z.append(z.numpy())

    embeddings = np.concatenate(all_z, axis=0)
    emb_path = BRIDGE_OUT / "embeddings.npy"
    np.save(str(emb_path), embeddings)
    print(f"  Embeddings: {embeddings.shape}")
    return embeddings


def step7_generate_ppt():
    """Generate PPT report."""
    print("\n[Step 7] Generating PPT report...")
    from generate_report_ppt import generate_ppt

    stats_path = BRIDGE_OUT / "dataset_stats.json"
    train_log_path = BRIDGE_OUT / "train_log.json"
    figures_dir = _PROJECT_ROOT / "reports" / "figures"
    output_path = _PROJECT_ROOT / "reports" / "MSG_Embedding_Report.pptx"

    generate_ppt(stats_path, train_log_path, output_path, figures_dir)


def step8_update_platform():
    """Update manifest and restart platform."""
    print("\n[Step 8] Updating platform data...")
    import pandas as pd

    collect_dirs = {
        "sionna_rt": _PROJECT_ROOT / "artifacts" / "collect_sionna_rt",
        "quadriga_real": COLLECT_DIR,
    }

    rows = []
    for source_name, collect_dir in collect_dirs.items():
        pts = sorted(collect_dir.glob("sample_*.pt"))[:1000]
        for i, p in enumerate(pts):
            d = torch.load(str(p), map_location="cpu", weights_only=False)
            rows.append({
                "uuid": str(uuid.uuid4()),
                "sample_id": f"{source_name}_{i:05d}",
                "shard_id": f"shard_{i // 100:04d}",
                "source": source_name,
                "link": d.get("link", "DL"),
                "snr_dB": float(d.get("snr_dB", 0)),
                "sir_dB": float(d.get("sir_dB", 0)) if d.get("sir_dB") is not None else None,
                "sinr_dB": float(d.get("sinr_dB", 0)) if d.get("sinr_dB") is not None else None,
                "num_cells": 7,
                "created_at": datetime.now().isoformat(),
                "status": "ready",
                "job_id": f"collect_5k_{source_name}",
                "path": str(p),
                "split": "train" if i < 4500 else "val",
            })
            del d
            if (i + 1) % 2000 == 0:
                print(f"  {source_name}: {i + 1}/{len(pts)}")

    df = pd.DataFrame(rows)
    out = _PROJECT_ROOT / "bridge_out" / "manifest.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(out), index=False)
    print(f"  Manifest: {len(df)} rows -> {out}")


if __name__ == "__main__":
    print("=" * 60)
    print(f"Post-QuaDRiGa Pipeline  ({datetime.now():%Y-%m-%d %H:%M})")
    print("=" * 60)

    mat_files = step1_wait_for_mats()
    step2_convert_mat_to_pt(mat_files)

    step3_recompute_stats()
    step4_bridge()
    step8_update_platform()

    print("\n" + "=" * 60)
    print("ALL DONE! QuaDRiGa + Sionna RT data collected & bridged.")
    print("=" * 60)
