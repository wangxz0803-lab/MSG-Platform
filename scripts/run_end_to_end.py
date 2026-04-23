"""End-to-end pipeline: wait for collection -> bridge -> train -> infer -> analyze -> PPT.

Usage:
    python scripts/run_end_to_end.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from msg_embedding.data.bridge import _build_feat_dict
from msg_embedding.data.contract import ChannelSample
from msg_embedding.features.extractor import FeatureExtractor
from msg_embedding.models.channel_mae import ChannelMAE

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _wait_for_collections(dirs: dict[str, Path], timeout: int = 7200) -> bool:
    print("Waiting for data collections to finish...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        all_done = True
        for name, d in dirs.items():
            summary = d / "summary.json"
            if not summary.exists():
                all_done = False
                progress = d / "progress.json"
                if progress.exists():
                    p = json.loads(progress.read_text())
                    print(
                        f"  {name}: {p.get('saved', '?')}/{p.get('total', '?')} "
                        f"({p.get('pct', 0):.1f}%)"
                    )
            else:
                s = json.loads(summary.read_text())
                print(f"  {name}: DONE ({s['saved']} samples)")
        if all_done:
            print("All collections complete!")
            return True
        time.sleep(30)
    print("TIMEOUT waiting for collections")
    return False


def _iter_sample_files(collect_dir: Path, max_n: int = 5000) -> list[Path]:
    """Yield .pt file paths without loading them."""
    return sorted(collect_dir.glob("sample_*.pt"))[:max_n]


def _load_one(pt_path: Path) -> ChannelSample:
    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    return ChannelSample.from_dict(d)


def _compute_stats_streaming(collect_dir: Path, max_n: int = 5000) -> dict:
    """Compute statistics by streaming samples one at a time (no OOM)."""
    pts = _iter_sample_files(collect_dir, max_n)
    snrs: list[float] = []
    sinrs: list[float] = []
    powers: list[float] = []
    links: list[str] = []
    est_modes: list[str] = []
    ue_positions: list[Any] = []
    count = 0

    for p in pts:
        try:
            s = _load_one(p)
            snrs.append(float(s.snr_dB))
            sinrs.append(float(s.sinr_dB))
            powers.append(float(np.mean(np.abs(s.h_serving_true) ** 2)))
            links.append(s.link)
            est_modes.append(s.channel_est_mode)
            if s.ue_position is not None:
                ue_positions.append(s.ue_position.copy())
            count += 1
            del s
        except Exception as e:
            print(f"  Stats error {p.name}: {e}")

        if count % 500 == 0:
            print(f"  Stats: {count}/{len(pts)} scanned")

    snrs_a, sinrs_a, powers_a = np.array(snrs), np.array(sinrs), np.array(powers)
    stats: dict[str, Any] = {
        "count": count,
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
    return stats


def _bridge_streaming(
    collect_dir: Path, out_dir: Path, source: str, max_n: int = 5000
) -> int:
    """Bridge samples in streaming batches with resume support."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pts = _iter_sample_files(collect_dir, max_n)
    batch_size = 100
    total_saved = 0
    shard_idx = 0

    existing_shards = sorted(out_dir.glob("shard_*.pt"))
    resume_from = len(existing_shards)
    if resume_from > 0:
        total_saved = resume_from * batch_size
        shard_idx = resume_from
        print(
            f"  Bridge [{source}] resuming: {resume_from} shards exist, "
            f"skipping first {total_saved} samples"
        )

    for start in range(0, len(pts), batch_size):
        current_shard = start // batch_size
        if current_shard < resume_from:
            continue

        chunk_paths = pts[start : start + batch_size]
        feats = []
        for p in chunk_paths:
            try:
                s = _load_one(p)
                fd, _ = _build_feat_dict(s, use_legacy_pmi=False)
                feats.append(fd)
                del s
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
            print(f"  Bridge [{source}] {pct:.0f}% ({total_saved} saved)")

    return total_saved


def _train_on_bridge(bridge_dir: Path, epochs: int = 30) -> dict:
    """Train MAE on bridged features, return training log."""
    shard_files = sorted(bridge_dir.rglob("shard_*.pt"))
    if not shard_files:
        print("  No bridge shards found, skipping training")
        return {}

    all_feats: dict[str, list[torch.Tensor]] = {}
    for sf in shard_files:
        feat = torch.load(sf, map_location="cpu", weights_only=False)
        for k, v in feat.items():
            all_feats.setdefault(k, []).append(v)

    merged: dict[str, torch.Tensor] = {}
    for k, vs in all_feats.items():
        merged[k] = torch.cat(vs, dim=0)

    n_total = merged[list(merged.keys())[0]].shape[0]
    print(f"  Training on {n_total} samples, {epochs} epochs")

    perm = torch.randperm(n_total)
    n_train = int(n_total * 0.9)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_feat = {k: v[train_idx] for k, v in merged.items()}
    val_feat = {k: v[val_idx] for k, v in merged.items()}

    feat_ext = FeatureExtractor()
    with torch.no_grad():
        sample_tokens, sample_ns = feat_ext({k: v[:2] for k, v in train_feat.items()})

    mae = ChannelMAE(None)
    mae.train()

    optimizer = torch.optim.AdamW(mae.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    batch_size = 64
    train_log: dict[str, list] = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }

    for epoch in range(epochs):
        mae.train()
        epoch_loss = 0.0
        n_batches = 0

        perm_e = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm_e[i : i + batch_size]
            batch = {k: v[idx] for k, v in train_feat.items()}

            with torch.no_grad():
                tokens, ns = feat_ext(batch)

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
        avg_train_loss = epoch_loss / max(n_batches, 1)

        mae.eval()
        val_loss = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for i in range(0, len(val_idx), batch_size):
                idx_v = torch.arange(i, min(i + batch_size, len(val_idx)))
                batch_v = {k: v[idx_v] for k, v in val_feat.items()}
                tok_v, ns_v = feat_ext(batch_v)
                snr_v = batch_v.get("srs_sinr", torch.zeros(tok_v.shape[0]))
                recon_v = mae(tok_v, snr_v, mask_ratio=0.3)
                val_loss += torch.nn.functional.mse_loss(recon_v, tok_v).item()
                n_val_batches += 1

        avg_val_loss = val_loss / max(n_val_batches, 1)

        train_log["epochs"].append(epoch + 1)
        train_log["train_loss"].append(round(avg_train_loss, 6))
        train_log["val_loss"].append(round(avg_val_loss, 6))
        train_log["learning_rate"].append(round(scheduler.get_last_lr()[0], 8))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch + 1}/{epochs}: train={avg_train_loss:.4f} val={avg_val_loss:.4f} "
                f"lr={scheduler.get_last_lr()[0]:.6f}"
            )

    ckpt_dir = _PROJECT_ROOT / "artifacts" / "train_5k"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": mae.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epochs,
            "train_log": train_log,
        },
        ckpt_dir / "best.pth",
    )
    print(f"  Checkpoint saved: {ckpt_dir / 'best.pth'}")

    train_log["final_train_loss"] = train_log["train_loss"][-1]
    train_log["final_val_loss"] = train_log["val_loss"][-1]
    best_epoch = int(np.argmin(train_log["val_loss"])) + 1
    train_log["best_epoch"] = best_epoch

    return train_log


def _run_inference(bridge_dir: Path, ckpt_path: Path) -> np.ndarray:
    """Run inference to get embeddings."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    feat_ext = FeatureExtractor()

    shard_files = sorted(bridge_dir.rglob("shard_*.pt"))
    if not shard_files:
        return np.array([])

    mae = ChannelMAE(None)
    mae.load_state_dict(ckpt["model"])
    mae.eval()

    all_z = []
    for sf in shard_files:
        feat = torch.load(sf, map_location="cpu", weights_only=False)
        with torch.no_grad():
            tokens, _ = feat_ext(feat)
            snr = feat.get("srs_sinr", torch.zeros(tokens.shape[0]))
            z = mae.get_latent(tokens, snr)
            all_z.append(z.numpy())

    embeddings = np.concatenate(all_z, axis=0)
    print(f"  Inference: {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")
    return embeddings


if __name__ == "__main__":
    print("=" * 60)
    print(f"MSG-Embedding End-to-End Pipeline  ({datetime.now():%Y-%m-%d %H:%M})")
    print("=" * 60)

    collect_dirs = {
        "internal_sim": _PROJECT_ROOT / "artifacts" / "collect_internal_sim_fast",
        "sionna_rt": _PROJECT_ROOT / "artifacts" / "collect_sionna_rt",
        "quadriga_real": _PROJECT_ROOT / "artifacts" / "collect_quadriga_real",
    }

    bridge_out = _PROJECT_ROOT / "artifacts" / "bridge_out_5k"
    bridge_out.mkdir(parents=True, exist_ok=True)

    print("\n[Step 1] Checking data collections...")
    _wait_for_collections(collect_dirs)

    print("\n[Step 2] Computing statistics (streaming)...")
    stats_path = bridge_out / "dataset_stats.json"
    all_stats: dict[str, Any] = {}
    active_sources: list[str] = []

    if stats_path.exists():
        all_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        active_sources = list(all_stats.keys())
        print(f"  Stats already computed for {active_sources}, reusing")
    else:
        for name, d in collect_dirs.items():
            if not (d / "summary.json").exists():
                print(f"  {name}: SKIP (not collected)")
                continue
            active_sources.append(name)
            run_meta = (
                json.loads((d / "run_meta.json").read_text())
                if (d / "run_meta.json").exists()
                else {}
            )
            stats = _compute_stats_streaming(d, 5000)
            stats["config"] = run_meta.get("config", {})
            stats["source"] = name
            all_stats[name] = stats
            print(
                f"  {name}: {stats['count']} samples, "
                f"SNR={stats['snr_dB']['mean']:.1f}+/-{stats['snr_dB']['std']:.1f} dB"
            )

        stats_path.write_text(json.dumps(all_stats, indent=2, default=str), encoding="utf-8")
        print(f"  Stats -> {stats_path}")

    print("\n[Step 3] Running bridge pipeline (streaming)...")
    for name in active_sources:
        src_dir = bridge_out / name
        n = _bridge_streaming(collect_dirs[name], src_dir, name, 5000)
        print(f"  {name}: {n} samples bridged -> {src_dir}")

    print("\n[Step 4] Training MAE model (30 epochs)...")
    train_log = _train_on_bridge(bridge_out, epochs=30)

    if train_log:
        train_log_path = bridge_out / "train_log.json"
        train_log_path.write_text(json.dumps(train_log, indent=2), encoding="utf-8")
        print(f"  Train log -> {train_log_path}")

    ckpt_path = _PROJECT_ROOT / "artifacts" / "train_5k" / "best.pth"
    if ckpt_path.exists():
        print("\n[Step 5] Running inference...")
        embeddings = _run_inference(bridge_out, ckpt_path)
        if embeddings.size > 0:
            emb_path = bridge_out / "embeddings.npy"
            np.save(str(emb_path), embeddings)
            print(f"  Embeddings -> {emb_path} ({embeddings.shape})")

            print("\n[Step 6] Embedding analysis...")
            print(f"  Shape: {embeddings.shape}")
            print(f"  Mean norm: {np.mean(np.linalg.norm(embeddings, axis=1)):.4f}")
            print(f"  Std norm: {np.std(np.linalg.norm(embeddings, axis=1)):.4f}")

            offset = 0
            for name in active_sources:
                n = all_stats[name]["count"]
                src_emb = embeddings[offset : offset + n]
                if src_emb.shape[0] > 0:
                    norms = np.linalg.norm(src_emb, axis=1)
                    print(
                        f"  {name}: norm={np.mean(norms):.4f}+/-{np.std(norms):.4f}, "
                        f"range=[{norms.min():.4f}, {norms.max():.4f}]"
                    )
                offset += n

    print("\n[Step 7] Generating PPT report...")
    try:
        from generate_report_ppt import generate_ppt

        figures_dir = _PROJECT_ROOT / "reports" / "figures"
        ppt_path = _PROJECT_ROOT / "reports" / "MSG_Embedding_Report.pptx"
        generate_ppt(stats_path, bridge_out / "train_log.json", ppt_path, figures_dir)
        print(f"  PPT -> {ppt_path}")
    except Exception as e:
        print(f"  PPT generation error: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
