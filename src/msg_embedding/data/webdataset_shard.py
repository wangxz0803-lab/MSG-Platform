"""Simple tar-based shard writer / reader for :class:`ChannelSample` payloads.

This is a *simplified* alternative to the `webdataset` package: we pack each
sample as a single ``.pkl`` entry inside a tar archive. Sharding is automatic
based on ``shard_size`` — large jobs produce ``shard-000000.tar``,
``shard-000001.tar``, ... so shuffling and streaming remain cheap.

The dependency footprint is ``stdlib`` only (``tarfile`` + ``pickle``) so this
is safe to use in offline / constrained environments where the real
webdataset package is not installed.
"""

from __future__ import annotations

import io
import pickle
import tarfile
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from .contract import ChannelSample


def _shard_name(output_dir: Path, prefix: str, idx: int) -> Path:
    return output_dir / f"{prefix}-{idx:06d}.tar"


def pack_shard(
    samples: Iterable[ChannelSample],
    output_path: Path | str,
    shard_size: int = 1000,
    prefix: str = "shard",
) -> list[Path]:
    """Serialise ``samples`` into one or more tar archives.

    Parameters
    ----------
    samples :
        Iterable of :class:`ChannelSample` instances.
    output_path :
        Target directory for the shard files. Created if missing. A single
        file path is also accepted — its parent dir is used and the filename
        stem becomes the ``prefix`` (the suffix is ignored).
    shard_size :
        Maximum samples per shard.
    prefix :
        Filename prefix. Ignored if ``output_path`` is a single file.

    Returns
    -------
    Paths of the written shards in creation order.
    """
    if shard_size < 1:
        raise ValueError(f"shard_size must be >= 1, got {shard_size}")

    output_path = Path(output_path)
    if output_path.suffix:
        output_dir = output_path.parent
        prefix = output_path.stem
    else:
        output_dir = output_path
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    shard_idx = 0
    tar: tarfile.TarFile | None = None
    in_shard = 0

    def _open_new() -> tarfile.TarFile:
        nonlocal shard_idx
        path = _shard_name(output_dir, prefix, shard_idx)
        t = tarfile.open(path, "w")
        written.append(path)
        shard_idx += 1
        return t

    try:
        for i, sample in enumerate(samples):
            if tar is None or in_shard >= shard_size:
                if tar is not None:
                    tar.close()
                tar = _open_new()
                in_shard = 0

            payload = pickle.dumps(sample.to_dict(), protocol=pickle.HIGHEST_PROTOCOL)
            # Use sample.sample_id (uuid) so names stay stable across re-packs
            # but prefix with ``i`` to guarantee lexicographic ordering inside
            # the tar even if the same sample_id shows up twice in tests.
            member_name = f"{i:08d}-{sample.sample_id}.pkl"
            info = tarfile.TarInfo(name=member_name)
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
            in_shard += 1
    finally:
        if tar is not None:
            tar.close()

    return written


def stream_shard(shard_paths: Sequence[Path | str]) -> Iterator[ChannelSample]:
    """Yield :class:`ChannelSample` records from one or more shard tars.

    Samples are emitted in the tar's native (lexicographic) member order so
    callers can rely on determinism when packing with :func:`pack_shard`.
    """
    for raw_path in shard_paths:
        path = Path(raw_path)
        with tarfile.open(path, "r") as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            members.sort(key=lambda m: m.name)
            for member in members:
                extracted = tar.extractfile(member)
                if extracted is None:  # pragma: no cover - dir entries
                    continue
                data = extracted.read()
                payload = pickle.loads(data)
                if isinstance(payload, ChannelSample):
                    yield payload
                elif isinstance(payload, dict):
                    yield ChannelSample.from_dict(payload)
                else:  # pragma: no cover - defensive
                    raise TypeError(
                        f"{path}::{member.name}: expected dict/ChannelSample, "
                        f"got {type(payload).__name__}"
                    )


__all__ = ["pack_shard", "stream_shard"]
