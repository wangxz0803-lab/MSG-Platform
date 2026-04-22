"""CLI entry point for Phase 4 HTML report generation.

Examples
--------
Single-run report::

    python scripts/run_report.py \\
        report.run_id=<latest> \\
        report.artifacts_dir=artifacts/ \\
        report.reports_dir=reports/

Multi-run comparison (Hydra list override)::

    python scripts/run_report.py \\
        +report.compare=[run1,run2,run3] \\
        report.out=reports/compare.html

Notes
-----
The script is intentionally Hydra-driven so it slots cleanly into the
rest of the ``scripts/`` entry points, but it does **not** require a
dedicated ``configs/report/default.yaml`` — all knobs are provided via
overrides with sensible defaults.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
except ImportError as exc:  # pragma: no cover - Hydra is a first-party dep
    print(f"hydra / omegaconf missing: {exc}", file=sys.stderr)
    raise

from msg_embedding.core.logging import get_logger
from msg_embedding.report import MultiRunComparator, ReportGenerator

_log = get_logger(__name__)


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    report_block = getattr(cfg, "report", None)
    if report_block is None:
        return default
    return getattr(report_block, key, default)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Top-level CLI dispatcher."""
    artifacts_dir = Path(str(_get(cfg, "artifacts_dir", "artifacts")))
    reports_dir = Path(str(_get(cfg, "reports_dir", "reports")))

    compare_arg = _get(cfg, "compare", None)
    if compare_arg:
        run_ids = list(OmegaConf.to_container(compare_arg, resolve=True))  # type: ignore[arg-type]
        run_ids = [str(r) for r in run_ids]
        out = _get(cfg, "out", None)
        out_path = Path(str(out)) if out else None
        comparator = MultiRunComparator(artifacts_dir=artifacts_dir, reports_dir=reports_dir)
        written = comparator.compare(run_ids, out_path=out_path)
        _log.info("multi_run_report_written", path=str(written))
        return

    run_id = _get(cfg, "run_id", None)
    if not run_id:
        raise RuntimeError(
            "report.run_id is required — pass via Hydra override, e.g. "
            "'report.run_id=<latest>'"
        )

    gen = ReportGenerator(
        run_id=str(run_id),
        artifacts_dir=artifacts_dir,
        reports_dir=reports_dir,
    )
    out_path = gen.build()
    _log.info("single_run_report_written", path=str(out_path))


if __name__ == "__main__":
    main()
