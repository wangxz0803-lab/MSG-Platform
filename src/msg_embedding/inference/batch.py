"""Batch inference runner for MSG-Embedding."""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from msg_embedding.core.logging import get_logger

from .wrapper import (
    EncoderWrapper,
    build_encoder_wrapper,
    load_feature_extractor_class,
)

_log = get_logger(__name__)


class BatchInferenceRunner:
    """Stateful batch inference driver."""

    def __init__(
        self,
        ckpt_path: str | Path,
        device: str | torch.device = "cuda",
        half: bool = False,
        use_adapter: bool = False,
    ) -> None:
        self.ckpt_path = Path(ckpt_path)
        resolved_device = _resolve_device(device)
        self.device: torch.device = resolved_device
        self.half = bool(half and resolved_device.type == "cuda")

        self.wrapper: EncoderWrapper = build_encoder_wrapper(
            ckpt_path=self.ckpt_path,
            device=self.device,
            use_adapter=use_adapter,
        )
        feat_cls = load_feature_extractor_class()
        self.feature_extractor = feat_cls().to(self.device)
        self.feature_extractor.eval()
        _log.info("batch_inference_ready", ckpt=str(self.ckpt_path),
                  device=str(self.device), half=self.half)

    @torch.no_grad()
    def embed_feat_dict(self, feat: dict[str, torch.Tensor]) -> np.ndarray:
        feat_dev = {
            k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in feat.items()
        }
        tokens, norm_stats = self.feature_extractor(feat_dev)
        token_mask = norm_stats["token_mask"]
        with self._autocast_ctx():
            z = self.wrapper(tokens, token_mask)
        return z.detach().float().cpu().numpy()

    @torch.no_grad()
    def embed_tokens(self, tokens: torch.Tensor,
                     token_mask: torch.Tensor) -> np.ndarray:
        tokens = tokens.to(self.device)
        token_mask = token_mask.to(self.device)
        with self._autocast_ctx():
            z = self.wrapper(tokens, token_mask)
        return z.detach().float().cpu().numpy()

    def infer_dataset(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 0,
    ) -> Iterator[tuple[list[str], np.ndarray]]:
        loader = _make_record_loader(dataset, batch_size=batch_size,
                                     num_workers=num_workers)
        for batch_records in loader:
            sample_ids: list[str] = []
            feat_list: list[dict[str, torch.Tensor]] = []
            for rec in batch_records:
                fd = _feat_from_record(rec)
                if fd is None:
                    _log.debug("skip_record",
                               sample_id=rec.get("sample_id") if isinstance(rec, dict) else None)
                    continue
                feat_list.append(fd)
                sid = (rec.get("sample_id") or rec.get("uuid") or "")\
                    if isinstance(rec, dict) else str(rec)
                sample_ids.append(str(sid))

            if not feat_list:
                continue
            stacked = _stack_feat_dicts(feat_list)
            z_np = self.embed_feat_dict(stacked)
            yield sample_ids, z_np

    def infer_directory(
        self,
        bridge_out_dir: str | Path,
        output_parquet_path: str | Path,
        batch_size: int = 64,
        num_workers: int = 0,
        half: bool | None = None,
        glob: str = "*.pt",
    ) -> Path:
        _ = num_workers
        src_dir = Path(bridge_out_dir)
        if not src_dir.exists():
            raise FileNotFoundError(f"bridge_out_dir not found: {src_dir}")
        out_path = Path(output_parquet_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if half is not None:
            self.half = bool(half and self.device.type == "cuda")

        paths = sorted(src_dir.glob(glob))
        if not paths:
            raise FileNotFoundError(
                f"no .pt files match {glob!r} under {src_dir}"
            )
        _log.info("infer_directory_start", n_files=len(paths),
                  src_dir=str(src_dir))

        writer = _ParquetStreamer(out_path)
        try:
            for pt_path in paths:
                payload = torch.load(pt_path, map_location="cpu",
                                     weights_only=False)
                for feat, ids in _iter_feat_chunks(payload, pt_path,
                                                   chunk=batch_size):
                    z_np = self.embed_feat_dict(feat)
                    writer.write_chunk(ids, z_np, source_file=pt_path.name)
        finally:
            writer.close()

        _log.info("infer_directory_done", total_rows=writer.total_rows,
                  path=str(out_path))
        return out_path

    def _autocast_ctx(self) -> Any:
        if self.half and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()


def _resolve_device(device: str | torch.device) -> torch.device:
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type == "cuda" and not torch.cuda.is_available():
        _log.warning("cuda_fallback_to_cpu")
        return torch.device("cpu")
    return dev


def _make_record_loader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=lambda batch: list(batch),
    )


def _feat_from_record(rec: Any) -> dict[str, torch.Tensor] | None:
    if isinstance(rec, dict):
        if "feat" in rec and isinstance(rec["feat"], dict):
            return rec["feat"]
        if _looks_like_feat_dict(rec):
            return rec
    return None


_FEAT_PROBE_KEYS = ("pdp_crop", "srs1", "pmi1", "dft1", "rsrp_srs", "cell_rsrp")


def _looks_like_feat_dict(rec: dict[str, Any]) -> bool:
    return any(
        k in rec and torch.is_tensor(rec[k]) for k in _FEAT_PROBE_KEYS
    )


def _stack_feat_dicts(
    feats: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if not feats:
        return {}
    keys = list(feats[0].keys())
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        tensors = [f[k] for f in feats if k in f]
        if len(tensors) != len(feats):
            continue
        normed: list[torch.Tensor] = []
        for t in tensors:
            if t.dim() == 0:
                normed.append(t.view(1))
            else:
                normed.append(t.unsqueeze(0) if t.shape[0] != 1 else t)
        try:
            out[k] = torch.cat(normed, dim=0)
        except RuntimeError:
            continue
    return out


def _iter_feat_chunks(
    payload: Any,
    source_path: Path,
    chunk: int,
) -> Iterable[tuple[dict[str, torch.Tensor], list[str]]]:
    stem = source_path.stem
    if isinstance(payload, dict) and _looks_like_feat_dict(payload):
        feat, batch_size = _ensure_batched(payload)
        ids = [f"{stem}_{i}" for i in range(batch_size)]
        for start in range(0, batch_size, chunk):
            stop = min(start + chunk, batch_size)
            sub = {k: v[start:stop] for k, v in feat.items()}
            yield sub, ids[start:stop]
        return

    if isinstance(payload, list):
        buf: list[dict[str, torch.Tensor]] = []
        buf_ids: list[str] = []
        for i, item in enumerate(payload):
            fd = _feat_from_record(item)
            if fd is None:
                continue
            buf.append(fd)
            buf_ids.append(f"{stem}_{i}")
            if len(buf) >= chunk:
                yield _stack_feat_dicts(buf), buf_ids
                buf = []
                buf_ids = []
        if buf:
            yield _stack_feat_dicts(buf), buf_ids
        return

    raise TypeError(
        f"unsupported .pt payload at {source_path}: {type(payload).__name__}"
    )


def _ensure_batched(
    feat: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], int]:
    probe_key = next((k for k in _FEAT_PROBE_KEYS if k in feat), None)
    if probe_key is None:
        raise ValueError("feat dict has no recognised field for batch probing")
    probe = feat[probe_key]
    if probe.dim() == 0:
        batch_size = 1
        return {k: v.view(1) if v.dim() == 0 else v.unsqueeze(0) for k, v in feat.items()}, batch_size
    batch_size = int(probe.shape[0])
    return feat, batch_size


class _ParquetStreamer:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.total_rows = 0
        self._writer: Any = None
        self._schema: Any = None
        self._arrow: Any = None
        self._fallback_buffer: list[dict[str, Any]] = []
        self._use_arrow = _try_import_pyarrow() is not None

    def write_chunk(self, sample_ids: list[str], embeddings: np.ndarray,
                    source_file: str) -> None:
        n = embeddings.shape[0]
        if n == 0:
            return
        d = embeddings.shape[1]
        if len(sample_ids) != n:
            sample_ids = list(sample_ids)[:n] + [""] * max(0, n - len(sample_ids))

        if self._use_arrow:
            self._write_chunk_arrow(sample_ids, embeddings, d, source_file)
        else:
            for i in range(n):
                row: dict[str, Any] = {
                    "sample_id": sample_ids[i],
                    "source_file": source_file,
                }
                for j in range(d):
                    row[f"embedding_{j}"] = float(embeddings[i, j])
                self._fallback_buffer.append(row)
        self.total_rows += n

    def close(self) -> None:
        if self._use_arrow and self._writer is not None:
            self._writer.close()
            self._writer = None
            return
        if not self._use_arrow:
            import pandas as pd
            df = pd.DataFrame(self._fallback_buffer)
            if df.empty:
                df = pd.DataFrame(columns=["sample_id"])
            try:
                df.to_parquet(self.path, index=False)
            except Exception:
                df.to_parquet(self.path, index=False, engine="fastparquet")
            self._fallback_buffer = []

    def _write_chunk_arrow(self, sample_ids: list[str], embeddings: np.ndarray,
                           d: int, source_file: str) -> None:
        pa = self._arrow or _try_import_pyarrow()
        self._arrow = pa
        import pyarrow.parquet as pq  # type: ignore

        arrays = {
            "sample_id": pa.array(sample_ids, type=pa.string()),
            "source_file": pa.array([source_file] * len(sample_ids),
                                    type=pa.string()),
        }
        for j in range(d):
            arrays[f"embedding_{j}"] = pa.array(
                embeddings[:, j].astype(np.float32), type=pa.float32()
            )
        table = pa.table(arrays)

        if self._writer is None:
            self._schema = table.schema
            self._writer = pq.ParquetWriter(str(self.path), self._schema)
        self._writer.write_table(table)


def _try_import_pyarrow() -> Any:
    try:
        import pyarrow  # type: ignore
        import pyarrow.parquet  # type: ignore  # noqa: F401
        return pyarrow
    except Exception:
        return None


__all__ = [
    "BatchInferenceRunner",
]
