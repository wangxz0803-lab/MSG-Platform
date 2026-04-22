"""Multi-run comparison report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jinja2

from msg_embedding.core.logging import get_logger

from ..viz.training_curves import read_tb_scalars
from .generator import _METRIC_ROWS, run_id_hash

_log = get_logger(__name__)

_DEFAULT_TEMPLATE_DIR: Path = Path(__file__).resolve().parent / "templates"
_COMPARE_TEMPLATE: str = "compare.html.j2"


class MultiRunComparator:
    """Build a side-by-side comparison HTML file for multiple runs."""

    def __init__(
        self,
        artifacts_dir: str | Path,
        reports_dir: str | Path,
        template_dir: str | Path | None = None,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.reports_dir = Path(reports_dir)
        self.template_dir = Path(template_dir) if template_dir else _DEFAULT_TEMPLATE_DIR
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_dir)),
            autoescape=jinja2.select_autoescape(["html", "xml"]),
        )

    def compare(self, run_ids: list[str], out_path: str | Path | None = None) -> Path:
        if not run_ids:
            raise ValueError("compare requires at least one run id")

        metrics_per_run: dict[str, dict[str, Any]] = {
            rid: self._load_metrics(rid) for rid in run_ids
        }

        rows: list[tuple[str, list[str]]] = []
        for key, label, fmt in _METRIC_ROWS:
            cells: list[str] = []
            for rid in run_ids:
                val = metrics_per_run[rid].get(key, None)
                if val is None:
                    cells.append("N/A")
                else:
                    try:
                        cells.append(fmt.format(float(val)))
                    except Exception:
                        cells.append(str(val))
            rows.append((label, cells))

        training_html = self._build_overlaid_curves(run_ids)

        template = self._env.get_template(_COMPARE_TEMPLATE)
        html = template.render(
            run_ids=run_ids,
            metric_rows=rows,
            training_curves_html=training_html,
        )

        if out_path is None:
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            out_path = self.reports_dir / f"compare_{run_id_hash(run_ids)}.html"
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
        _log.info("compare_report_written", path=str(out_path))
        return out_path

    def _load_metrics(self, run_id: str) -> dict[str, Any]:
        path = self.reports_dir / run_id / "metrics.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _build_overlaid_curves(self, run_ids: list[str]) -> str | None:
        try:
            import plotly.graph_objects as go  # type: ignore[import-untyped]
        except ImportError:
            _log.warning("plotly_missing", detail="skipping overlaid curves")
            return None

        scalars_per_run: dict[str, dict[str, tuple[list[int], list[float]]]] = {}
        for rid in run_ids:
            tb_dir = self.artifacts_dir / rid / "tb"
            if tb_dir.exists():
                scalars_per_run[rid] = read_tb_scalars(tb_dir)

        tag_choice: str | None = None
        for scal in scalars_per_run.values():
            for tag in ("loss/train", "loss"):
                if tag in scal:
                    tag_choice = tag
                    break
            if tag_choice is not None:
                break
        if tag_choice is None:
            for scal in scalars_per_run.values():
                if scal:
                    tag_choice = next(iter(scal.keys()))
                    break
        if tag_choice is None:
            return None

        fig = go.Figure()
        for rid in run_ids:
            scal = scalars_per_run.get(rid, {})
            if tag_choice not in scal:
                continue
            steps, values = scal[tag_choice]
            fig.add_trace(go.Scatter(x=steps, y=values, mode="lines", name=rid))
        fig.update_layout(
            title=f"Training curve overlay: {tag_choice}",
            xaxis_title="step",
            yaxis_title=tag_choice,
            height=420,
            margin=dict(l=40, r=20, t=50, b=40),
        )
        return fig.to_html(include_plotlyjs="cdn", full_html=False)
