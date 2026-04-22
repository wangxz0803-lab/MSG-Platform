"""Tests for eval runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("pyarrow")

from msg_embedding.eval.runner import (  # noqa: E402
    EvalResult,
    get_git_sha,
    is_ready,
    run_eval,
)
from msg_embedding.models.channel_mae import ChannelMAE  # noqa: E402

# ---------------------------------------------------------------------------
# Fake dataset
# ---------------------------------------------------------------------------

class _FakeDataset:
    def __init__(self, feats: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
        assert len(feats) == len(records)
        self._feats = feats
        self._records = records

    def __len__(self) -> int:
        return len(self._feats)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = dict(self._records[idx])
        rec["feat"] = self._feats[idx]
        return rec


def _make_feat_dict(n: int = 1, tx: int = 64) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(abs(hash(("feat", n, tx))) % (2**32))
    feat: dict[str, torch.Tensor] = {}
    for k in (
        "srs1", "srs2", "srs3", "srs4",
        "pmi1", "pmi2", "pmi3", "pmi4",
        "dft1", "dft2", "dft3", "dft4",
    ):
        arr = (
            rng.standard_normal((n, tx), dtype=np.float32)
            + 1j * rng.standard_normal((n, tx), dtype=np.float32)
        ).astype(np.complex64)
        feat[k] = torch.from_numpy(arr)
    feat["pdp_crop"] = torch.from_numpy(
        rng.uniform(0.0, 1.0, size=(n, 64)).astype(np.float32)
    )
    feat["rsrp_srs"] = torch.from_numpy(
        rng.uniform(-120.0, -70.0, size=(n, tx)).astype(np.float32)
    )
    feat["rsrp_cb"] = torch.from_numpy(
        rng.uniform(-120.0, -70.0, size=(n, tx)).astype(np.float32)
    )
    feat["cell_rsrp"] = torch.full((n, 16), -110.0, dtype=torch.float32)
    feat["cqi"] = torch.randint(0, 16, (n,), dtype=torch.int64)
    feat["srs_sinr"] = torch.empty(n, dtype=torch.float32).uniform_(-10.0, 15.0)
    feat["srs_cb_sinr"] = torch.empty(n, dtype=torch.float32).uniform_(-10.0, 15.0)
    for k in ("srs_w1", "srs_w2", "srs_w3", "srs_w4"):
        feat[k] = torch.full((n,), 0.25, dtype=torch.float32)
    return feat


def _make_record(i: int, *, ue_xy: tuple[float, float] | None = None) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "uuid": f"fake-{i:03d}",
        "sinr_dB": float(5.0 + i * 0.1),
        "snr_dB": float(10.0 + i * 0.2),
        "link": "UL",
        "source": "quadriga_multi",
        "meta": {},
    }
    if ue_xy is not None:
        rec["meta"]["ue_position"] = [ue_xy[0], ue_xy[1], 1.5]
    return rec


def _write_fake_ckpt(tmp_path: Path) -> Path:
    mae = ChannelMAE(None)
    ckpt_path = tmp_path / "fake_ckpt.pth"
    torch.save({"model": mae.state_dict(), "epoch": 0}, ckpt_path)
    return ckpt_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_is_ready() -> None:
    assert is_ready() is True


def test_get_git_sha_returns_str() -> None:
    sha = get_git_sha()
    assert isinstance(sha, str)
    assert len(sha) > 0


def test_eval_result_flat_metrics_contract() -> None:
    res = EvalResult(
        run_id="r1",
        ckpt_path="/tmp/x.pt",
        git_sha="abc123",
        timestamp="2026-04-17T00:00:00Z",
        ct=0.9,
        tw=0.85,
        knn_consistency=0.8,
        nmse_dB=-20.0,
        cosine_sep_margin=0.5,
        kendall_tau=0.7,
        meta={"dataset_size": 10},
    )
    d = res.to_dict()
    for key in ("ct", "tw", "knn_consistency", "nmse_dB",
                "cosine_sep_margin", "kendall_tau"):
        assert key in d
        assert isinstance(d[key], float)
    assert d["meta"]["dataset_size"] == 10


def test_run_eval_produces_artifacts(tmp_path: Path) -> None:
    n = 24
    feats = [_make_feat_dict(n=1, tx=64) for _ in range(n)]
    records = [
        _make_record(i, ue_xy=(float(i), float(i * 2))) for i in range(n)
    ]
    dataset = _FakeDataset(feats=feats, records=records)

    ckpt = _write_fake_ckpt(tmp_path)
    out = tmp_path / "reports"

    result = run_eval(
        ckpt_path=ckpt,
        dataset=dataset,  # type: ignore[arg-type]
        output_dir=out,
        eval_cfg={"k_neighbors": 5, "knn_k": 3, "run_id": "smoke-001"},
        device="cpu",
    )

    assert isinstance(result, EvalResult)
    assert result.run_id == "smoke-001"
    run_dir = out / "smoke-001"
    metrics_file = run_dir / "metrics.json"
    parquet_file = run_dir / "embeddings.parquet"
    assert metrics_file.exists()
    assert parquet_file.exists()

    payload = json.loads(metrics_file.read_text(encoding="utf-8"))
    for key in ("ct", "tw", "knn_consistency", "nmse_dB",
                "cosine_sep_margin", "kendall_tau"):
        assert key in payload
    assert "meta" in payload
    assert payload["meta"]["embeddings_used"] == n


def test_run_eval_handles_empty_feat(tmp_path: Path) -> None:
    n = 10
    records = [_make_record(i) for i in range(n)]

    class _NoFeat:
        def __len__(self) -> int:
            return len(records)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            return records[idx]

    ckpt = _write_fake_ckpt(tmp_path)
    out = tmp_path / "reports"
    result = run_eval(
        ckpt_path=ckpt,
        dataset=_NoFeat(),  # type: ignore[arg-type]
        output_dir=out,
        eval_cfg={"run_id": "empty-feat"},
        device="cpu",
    )
    metrics_path = out / "empty-feat" / "metrics.json"
    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert payload["meta"]["embeddings_used"] == 0
    assert payload["tw"] is None
    assert payload["ct"] is None
    assert isinstance(result, EvalResult)


def test_run_eval_respects_limit(tmp_path: Path) -> None:
    n = 32
    feats = [_make_feat_dict(n=1, tx=64) for _ in range(n)]
    records = [_make_record(i) for i in range(n)]
    dataset = _FakeDataset(feats=feats, records=records)

    ckpt = _write_fake_ckpt(tmp_path)
    out = tmp_path / "reports"
    result = run_eval(
        ckpt_path=ckpt,
        dataset=dataset,  # type: ignore[arg-type]
        output_dir=out,
        eval_cfg={"limit": 6, "run_id": "limited", "k_neighbors": 3, "knn_k": 2},
        device="cpu",
    )
    assert result.meta["embeddings_used"] == 6
