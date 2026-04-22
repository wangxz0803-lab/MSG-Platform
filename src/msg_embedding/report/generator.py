"""Render a self-contained HTML report for a single training / eval run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from msg_embedding.core.logging import get_logger

from ..viz.dataset_stats import plot_manifest_histograms
from ..viz.latent import plot_latent_scatter_plotly, umap_2d
from ..viz.training_curves import training_curves_div

try:
    import jinja2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "jinja2 is required for the Phase 4 report generator"
    ) from exc

_log = get_logger(__name__)

_DEFAULT_TEMPLATE_DIR: Path = Path(__file__).resolve().parent / "templates"
_DEFAULT_TEMPLATE_NAME: str = "report.html.j2"

_METRIC_ROWS: tuple[tuple[str, str, str], ...] = (
    ("ct", "Continuity (CT)", "{:.4f}"),
    ("tw", "Trustworthiness (TW)", "{:.4f}"),
    ("knn_consistency", "kNN consistency", "{:.4f}"),
    ("nmse_db", "Reconstruction NMSE (dB)", "{:.2f}"),
    ("cos_clean_noisy_mean", "Cosine clean<->noisy (mean)", "{:.4f}"),
    ("cos_cross_ue_mean", "Cosine cross-UE (mean)", "{:.4f}"),
)


@dataclass
class ReportContext:
    """Structured payload fed into the Jinja template."""

    run_id: str
    timestamp: str
    git_sha: str
    ckpt_path: str
    config_table: list[tuple[str, str]]
    metrics_rows: list[tuple[str, str]]
    training_curves_html: str | None
    latent_html: str | None
    dataset_summary: dict[str, Any] | None
    notes: list[str]


class ReportGenerator:
    """Top-level entry point for producing ``reports/<run_id>/report.html``."""

    def __init__(
        self,
        run_id: str,
        artifacts_dir: str | Path,
        reports_dir: str | Path,
        template_dir: str | Path | None = None,
        deterministic: bool = True,
    ) -> None:
        self.run_id = str(run_id)
        self.artifacts_dir = Path(artifacts_dir)
        self.reports_dir = Path(reports_dir)
        self.template_dir = Path(template_dir) if template_dir else _DEFAULT_TEMPLATE_DIR
        self.deterministic = deterministic

        self._run_artifacts: Path = self.artifacts_dir / self.run_id
        self._run_reports: Path = self.reports_dir / self.run_id
        self._run_reports.mkdir(parents=True, exist_ok=True)

        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_dir)),
            autoescape=jinja2.select_autoescape(["html", "xml"]),
            keep_trailing_newline=True,
        )

    def build(self) -> Path:
        """Render the report and return the path to the written HTML file."""
        context = self._build_context()
        template = self._env.get_template(_DEFAULT_TEMPLATE_NAME)
        html = template.render(**context.__dict__)
        out_path = self._run_reports / "report.html"
        out_path.write_text(html, encoding="utf-8")
        _log.info("report_written", path=str(out_path))
        return out_path

    def _build_context(self) -> ReportContext:
        notes: list[str] = []

        metadata = self._load_metadata(notes)
        config_table = self._load_config_table(notes)
        metrics_rows = self._load_metrics_rows(notes)
        training_html = self._build_training_curves_html(notes)
        latent_html = self._build_latent_html(notes)
        dataset_summary = self._load_dataset_summary(notes)

        timestamp = (
            metadata.get("timestamp")
            or (
                "deterministic"
                if self.deterministic
                else datetime.now(timezone.utc).isoformat(timespec="seconds")
            )
        )
        git_sha = metadata.get("git_sha", "unknown")
        ckpt_path = metadata.get("ckpt") or metadata.get("ckpt_path") or "N/A"

        return ReportContext(
            run_id=self.run_id,
            timestamp=str(timestamp),
            git_sha=str(git_sha),
            ckpt_path=str(ckpt_path),
            config_table=config_table,
            metrics_rows=metrics_rows,
            training_curves_html=training_html,
            latent_html=latent_html,
            dataset_summary=dataset_summary,
            notes=notes,
        )

    def _load_metadata(self, notes: list[str]) -> dict[str, Any]:
        path = self._run_artifacts / "metadata.json"
        if not path.exists():
            notes.append(f"metadata.json missing under {self._run_artifacts}")
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            notes.append(f"failed to parse metadata.json: {exc}")
            return {}

    def _load_config_table(self, notes: list[str]) -> list[tuple[str, str]]:
        path = self._run_artifacts / "config.yaml"
        if not path.exists():
            notes.append(f"config.yaml missing under {self._run_artifacts}")
            return []
        try:
            import yaml  # type: ignore[import-untyped]

            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            notes.append(f"failed to parse config.yaml: {exc}")
            return []
        if not isinstance(data, dict):
            return []

        rows: list[tuple[str, str]] = []

        def _walk(prefix: str, node: Any) -> None:
            if isinstance(node, dict):
                for k, v in node.items():
                    _walk(f"{prefix}.{k}" if prefix else str(k), v)
            elif isinstance(node, list):
                rows.append((prefix, ", ".join(str(x) for x in node)))
            else:
                rows.append((prefix, str(node)))

        _walk("", data)
        return sorted(rows, key=lambda pair: pair[0])

    def _load_metrics_rows(self, notes: list[str]) -> list[tuple[str, str]]:
        path = self.reports_dir / self.run_id / "metrics.json"
        metrics: dict[str, Any] = {}
        if path.exists():
            try:
                metrics = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                notes.append(f"failed to parse metrics.json: {exc}")
        else:
            notes.append(f"metrics.json missing at {path}")

        rows: list[tuple[str, str]] = []
        for key, label, fmt in _METRIC_ROWS:
            value = metrics.get(key, None)
            if value is None or (isinstance(value, float) and not _finite(value)):
                rows.append((label, "N/A"))
            else:
                try:
                    rows.append((label, fmt.format(float(value))))
                except Exception:
                    rows.append((label, str(value)))
        extras = {
            k: v
            for k, v in metrics.items()
            if k not in {r[0] for r in _METRIC_ROWS}
            and k != "meta"
            and isinstance(v, int | float)
        }
        for k in sorted(extras):
            rows.append((k, f"{float(extras[k]):.4f}"))
        return rows

    def _build_training_curves_html(self, notes: list[str]) -> str | None:
        tb_dir = self._run_artifacts / "tb"
        if not tb_dir.exists():
            notes.append(f"tb/ directory missing under {self._run_artifacts}")
            return None
        html = training_curves_div(tb_dir)
        if html is None:
            notes.append("training curves unavailable (plotly / tensorboard missing)")
        return html

    def _build_latent_html(self, notes: list[str]) -> str | None:
        parquet_path = self._run_artifacts / "embeddings.parquet"
        if not parquet_path.exists():
            notes.append(f"embeddings.parquet missing under {self._run_artifacts}")
            return None
        try:
            import pandas as pd

            df = pd.read_parquet(parquet_path)
        except Exception as exc:
            notes.append(f"failed to load embeddings.parquet: {exc}")
            return None

        z_cols = sorted([c for c in df.columns if c.startswith("z_")])
        if not z_cols:
            notes.append("embeddings.parquet has no z_* columns")
            return None
        embeddings = df[z_cols].to_numpy(dtype="float32")
        if embeddings.shape[0] < 3:
            notes.append("too few embedding rows for UMAP")
            return None

        coords = umap_2d(embeddings)
        color_by = None
        color_label = None
        for candidate in ("sinr_dB", "sinr_db", "link", "source"):
            if candidate in df.columns:
                color_by = df[candidate].to_numpy()
                color_label = candidate
                break

        fig = plot_latent_scatter_plotly(
            coords,
            color_by=color_by,
            color_label=color_label,
            title=f"Latent UMAP ({self.run_id})",
        )
        if hasattr(fig, "to_html"):
            return fig.to_html(include_plotlyjs="cdn", full_html=False)  # type: ignore[no-any-return]
        notes.append("plotly unavailable — latent section falls back to static image")
        png_path = self._run_reports / "latent_umap.png"
        try:
            fig.savefig(png_path, dpi=130)  # type: ignore[attr-defined]
        except Exception:
            return None
        return f"<img src='{png_path.name}' alt='latent UMAP'/>"

    def _load_dataset_summary(self, notes: list[str]) -> dict[str, Any] | None:
        manifest_path = self._run_artifacts / "manifest.parquet"
        if not manifest_path.exists():
            manifest_path = self.artifacts_dir / "manifest.parquet"
        if not manifest_path.exists():
            return None
        try:
            import pandas as pd

            df = pd.read_parquet(manifest_path)
        except Exception as exc:
            notes.append(f"failed to load manifest.parquet: {exc}")
            return None
        try:
            from ..viz.dataset_stats import manifest_stats

            summary = manifest_stats(df)
        except Exception as exc:
            notes.append(f"manifest_stats failed: {exc}")
            return None
        try:
            plot_manifest_histograms(df, self._run_reports / "manifest_hist.png")
            summary["histogram_png"] = "manifest_hist.png"
        except Exception as exc:
            notes.append(f"plot_manifest_histograms failed: {exc}")
        return summary


def _finite(x: float) -> bool:
    import math

    return math.isfinite(x)


def run_id_hash(run_ids: list[str]) -> str:
    """Short stable hash used for compare-report filenames."""
    h = hashlib.sha1()
    for r in sorted(run_ids):
        h.update(r.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:10]
