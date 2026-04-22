"""Process-pool wrapper for batched sample processing with a live manifest.

The pipeline converts raw source files into validated ``ChannelSample`` records
in parallel. Each worker is expected to return ``(uuid, fields_to_patch)`` for
a successful run, or raise to signal failure. This module:

* Submits work through :class:`concurrent.futures.ProcessPoolExecutor`.
* Updates the :class:`~msg_embedding.data.manifest.Manifest` in the driver
  process as futures resolve (workers never touch the manifest directly, to
  avoid pickling / locking issues).
* Persists failed samples both on the manifest (``status='failed'`` +
  ``error_msg``) and as a parquet sidecar ``failures.parquet`` next to the
  manifest, so they can be re-queued later.
* Handles ``KeyboardInterrupt`` gracefully: cancels pending futures, saves the
  manifest, and re-raises so the CLI can exit with an informative code.
"""

from __future__ import annotations

import os
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is a soft dep

    def tqdm(iterable=None, **_kwargs):  # type: ignore[no-redef]
        return iterable if iterable is not None else iter(())


from .manifest import Manifest

FAILURE_SCHEMA = pa.schema(
    [
        ("uuid", pa.string()),
        ("item_repr", pa.string()),
        ("error_msg", pa.string()),
        ("traceback", pa.string()),
        ("failed_at", pa.timestamp("us")),
    ]
)


def _default_workers() -> int:
    """Return a sensible default worker count for parallel processing."""
    cpu = os.cpu_count() or 1
    return max(1, min(cpu - 1, 8))


def _append_failures(failures_path: Path, rows: list[dict[str, Any]]) -> None:
    """Append ``rows`` to ``failures_path``, creating the file if needed."""
    if not rows:
        return
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    # Schema is tz-naive; convert tz-aware inputs to naive UTC first.
    ts = pd.to_datetime(df["failed_at"], utc=True)
    df["failed_at"] = ts.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[us]")
    new_table = pa.Table.from_pandas(df, schema=FAILURE_SCHEMA, preserve_index=False)
    if failures_path.exists():
        existing = pq.read_table(failures_path).cast(FAILURE_SCHEMA)
        combined = pa.concat_tables([existing, new_table])
    else:
        combined = new_table
    tmp = failures_path.with_suffix(failures_path.suffix + ".tmp")
    pq.write_table(combined, tmp)
    tmp.replace(failures_path)


def parallel_process(
    items: Sequence[Any],
    worker_fn: Callable[[Any], tuple[str, dict[str, Any]]],
    manifest: Manifest,
    num_workers: int | None = None,
    chunk_size: int = 32,
    failures_path: Path | str | None = None,
    desc: str = "processing",
) -> tuple[int, int]:
    """Run ``worker_fn`` over ``items`` in parallel, updating ``manifest``.

    Parameters
    ----------
    items :
        Sequence of pickleable work items. Each is passed verbatim to
        ``worker_fn``.
    worker_fn :
        Callable executed in a subprocess. Must return ``(uuid, fields)`` where
        ``fields`` is a dict of manifest columns to patch. Any raised exception
        is captured and the item is marked failed.
    manifest :
        The manifest to update as results arrive. Not modified if ``items`` is
        empty.
    num_workers :
        Subprocess count. Defaults to ``min(cpu_count - 1, 8)``.
    chunk_size :
        How many futures to keep in-flight at once. Larger values improve
        throughput at the cost of memory.
    failures_path :
        Where to append failure rows. Defaults to ``failures.parquet`` next to
        the manifest.
    desc :
        Progress bar label.

    Returns
    -------
    ``(succeeded, failed)`` counts.
    """
    items = list(items)
    if not items:
        return 0, 0

    if num_workers is None:
        num_workers = _default_workers()
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    failures_path = (
        Path(failures_path)
        if failures_path is not None
        else manifest.path.parent / "failures.parquet"
    )

    succeeded = 0
    failed = 0
    failure_rows: list[dict[str, Any]] = []

    pending: set = set()
    item_by_future: dict = {}
    next_idx = 0
    progress = tqdm(total=len(items), desc=desc)

    interrupted = False

    def _flush_failures() -> None:
        nonlocal failure_rows
        if failure_rows:
            _append_failures(failures_path, failure_rows)
            failure_rows = []

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            # Prime the pool with up to chunk_size * num_workers futures.
            initial = min(len(items), chunk_size * num_workers)
            for _ in range(initial):
                fut = pool.submit(worker_fn, items[next_idx])
                item_by_future[fut] = items[next_idx]
                pending.add(fut)
                next_idx += 1

            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    item = item_by_future.pop(fut)
                    try:
                        uuid, fields = fut.result()
                    except KeyboardInterrupt:
                        interrupted = True
                        break
                    except BaseException as exc:  # noqa: BLE001 - capture everything
                        failed += 1
                        tb = traceback.format_exc()
                        err = f"{type(exc).__name__}: {exc}"
                        uuid = getattr(item, "uuid", None)
                        if isinstance(item, dict):
                            uuid = item.get("uuid", uuid)
                        failure_rows.append(
                            {
                                "uuid": str(uuid) if uuid is not None else "",
                                "item_repr": repr(item)[:500],
                                "error_msg": err,
                                "traceback": tb,
                                "failed_at": datetime.now(timezone.utc).replace(tzinfo=None),
                            }
                        )
                        if uuid is not None:
                            try:
                                manifest.update(
                                    str(uuid),
                                    status="failed",
                                    error_msg=err,
                                )
                            except KeyError:
                                # Item was not pre-registered — just leave it
                                # in the failures sidecar.
                                pass
                    else:
                        succeeded += 1
                        patch: dict[str, Any] = dict(fields)
                        patch.setdefault("status", "succeeded")
                        try:
                            manifest.update(str(uuid), **patch)
                        except KeyError:
                            # uuid missing → append a new row then patch it.
                            patch["uuid"] = str(uuid)
                            manifest.append([patch])
                    progress.update(1)

                    if next_idx < len(items) and not interrupted:
                        nxt = items[next_idx]
                        f2 = pool.submit(worker_fn, nxt)
                        item_by_future[f2] = nxt
                        pending.add(f2)
                        next_idx += 1

                if interrupted:
                    for f in pending:
                        f.cancel()
                    pending.clear()
                    break

    except KeyboardInterrupt:
        interrupted = True
    finally:
        progress.close()
        _flush_failures()
        manifest.save()

    if interrupted:
        raise KeyboardInterrupt("parallel_process interrupted; manifest saved")

    return succeeded, failed


__all__ = ["parallel_process", "FAILURE_SCHEMA"]
