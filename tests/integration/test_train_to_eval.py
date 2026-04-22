"""Integration: trained MAE -> eval metrics -> report generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from msg_embedding.eval import channel_charting as cc


def test_eval_metrics_on_trained_model(trained_mae) -> None:
    """Run channel charting + prediction metrics on a trained model."""
    mae, feat_ext, ckpt_info = trained_mae
    stacked = ckpt_info["stacked_feats"]

    with torch.no_grad():
        tokens, _ = feat_ext(stacked)
        snr = stacked.get("srs_sinr", torch.zeros(tokens.shape[0]))
        latent = mae.get_latent(tokens, snr)
        recon = mae(tokens, snr, mask_ratio=0.0)

    z = latent.numpy()
    assert z.shape[0] == 16
    assert z.shape[1] == 16

    positions = np.random.default_rng(0).uniform(-200, 200, size=(z.shape[0], 2)).astype(np.float32)

    tw = cc.trustworthiness(z, positions, k=3)
    ct = cc.continuity(z, positions, k=3)
    assert 0.0 <= tw <= 1.0
    assert 0.0 <= ct <= 1.0

    knn = cc.knn_consistency(z, positions, k=3)
    assert 0.0 <= knn <= 1.0

    tok_np, rec_np = tokens.numpy(), recon.numpy()
    nmse = float(np.mean(np.abs(tok_np - rec_np) ** 2) / (np.mean(np.abs(tok_np) ** 2) + 1e-12))
    assert np.isfinite(nmse)


def test_report_generation_with_real_metrics(trained_mae, tmp_path: Path) -> None:
    """Generate an HTML report from a trained model's metrics."""
    import json

    mae, feat_ext, ckpt_info = trained_mae

    artifacts_dir = tmp_path / "artifacts"
    reports_dir = tmp_path / "reports"
    run_id = "integ_test_run"
    run_artifacts = artifacts_dir / run_id
    run_reports = reports_dir / run_id
    run_artifacts.mkdir(parents=True)
    run_reports.mkdir(parents=True)

    (run_artifacts / "metadata.json").write_text(
        json.dumps({"run_id": run_id, "git_sha": "test123", "timestamp": "2026-04-22T00:00:00Z"}),
        encoding="utf-8",
    )

    import yaml

    (run_artifacts / "config.yaml").write_text(
        yaml.safe_dump({"train": {"batch_size": 16, "epochs": 2}, "model": {"embed_dim": 64}}),
        encoding="utf-8",
    )

    (run_reports / "metrics.json").write_text(
        json.dumps(
            {
                "ct": 0.75,
                "tw": 0.80,
                "knn_consistency": 0.65,
                "nmse_db": -15.2,
                "cos_clean_noisy_mean": 0.90,
                "cos_cross_ue_mean": 0.15,
                "meta": {"run_id": run_id, "ckpt": "test.pth", "git_sha": "test123"},
            }
        ),
        encoding="utf-8",
    )

    from msg_embedding.report import ReportGenerator

    gen = ReportGenerator(run_id, artifacts_dir=artifacts_dir, reports_dir=reports_dir)
    out_path = gen.build()

    assert out_path.exists()
    html = out_path.read_text(encoding="utf-8")
    assert run_id in html
    assert "0.80" in html or "0.8" in html
    assert "Metrics" in html
