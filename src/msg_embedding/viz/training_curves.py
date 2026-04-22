"""TensorBoard scalar extraction + Plotly rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from msg_embedding.core.logging import get_logger  # noqa: E402

_log = get_logger(__name__)

_PRIMARY_TAGS: tuple[str, ...] = ("loss", "lr", "grad_norm", "mask_ratio")


def read_tb_scalars(tb_logdir: str | Path) -> dict[str, tuple[list[int], list[float]]]:
    """Extract all scalar tags from a TensorBoard event directory."""
    logdir = Path(tb_logdir)
    if not logdir.exists():
        _log.warning("tb_logdir_missing", path=str(logdir))
        return {}

    try:
        from tensorboard.backend.event_processing import (
            event_accumulator,  # type: ignore[import-untyped]
        )
    except ImportError:
        _log.warning("tensorboard_missing")
        return {}

    size_guidance = {event_accumulator.SCALARS: 0}
    out: dict[str, tuple[list[int], list[float]]] = {}

    candidate_dirs: list[Path] = []
    has_events_here = any(p.name.startswith("events.out.tfevents") for p in logdir.iterdir()) \
        if logdir.is_dir() else False
    if has_events_here:
        candidate_dirs.append(logdir)
    if logdir.is_dir():
        for sub in sorted(logdir.iterdir()):
            if sub.is_dir():
                candidate_dirs.append(sub)

    if not candidate_dirs:
        candidate_dirs = [logdir]

    for run_dir in candidate_dirs:
        try:
            ea = event_accumulator.EventAccumulator(str(run_dir), size_guidance=size_guidance)
            ea.Reload()
        except Exception:  # pragma: no cover
            continue
        for tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            steps = [int(e.step) for e in events]
            values = [float(e.value) for e in events]
            if tag in out:
                out[tag] = (out[tag][0] + steps, out[tag][1] + values)
            else:
                out[tag] = (steps, values)

    return out


def _canonical_subplot_order(tags: list[str]) -> list[str]:
    """Return ``tags`` ordered with loss/lr/grad_norm/mask_ratio first."""
    ordered: list[str] = []
    for stem in _PRIMARY_TAGS:
        for tag in tags:
            matches = (
                tag == stem
                or tag.startswith(stem + "/")
                or tag.startswith(stem + "_")
            )
            if matches and tag not in ordered:
                ordered.append(tag)
    for tag in tags:
        if tag not in ordered:
            ordered.append(tag)
    return ordered


def plot_training_curves(tb_logdir: str | Path, out_html: str | Path) -> Path:
    """Render TB scalars as a multi-subplot Plotly HTML file."""
    scalars = read_tb_scalars(tb_logdir)
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    if not scalars:
        _log.warning("no_scalars", logdir=str(tb_logdir))
        out_html.write_text(
            "<html><body><p>No training curves available.</p></body></html>",
            encoding="utf-8",
        )
        return out_html

    tags = _canonical_subplot_order(list(scalars.keys()))

    try:
        import plotly.graph_objects as go  # type: ignore[import-untyped]
        from plotly.subplots import make_subplots  # type: ignore[import-untyped]
    except ImportError:
        _log.warning("plotly_fallback", reason="matplotlib PNG")
        png_path = out_html.with_suffix(".png")
        n = len(tags)
        cols = 2
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(10, 3 * rows), squeeze=False)
        for i, tag in enumerate(tags):
            r, c = divmod(i, cols)
            steps, values = scalars[tag]
            axes[r][c].plot(steps, values)
            axes[r][c].set_title(tag)
            axes[r][c].set_xlabel("step")
        for i in range(len(tags), rows * cols):
            r, c = divmod(i, cols)
            axes[r][c].axis("off")
        fig.tight_layout()
        fig.savefig(png_path, dpi=130)
        plt.close(fig)
        out_html.write_text(
            f"<html><body><img src='{png_path.name}' alt='training curves'/></body></html>",
            encoding="utf-8",
        )
        return out_html

    n = len(tags)
    cols = 2
    rows = (n + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=tags)
    for i, tag in enumerate(tags):
        r = i // cols + 1
        c = i % cols + 1
        steps, values = scalars[tag]
        fig.add_trace(
            go.Scatter(x=steps, y=values, mode="lines", name=tag, showlegend=False),
            row=r,
            col=c,
        )
        fig.update_xaxes(title_text="step", row=r, col=c)
    fig.update_layout(
        height=260 * rows + 80,
        title_text="Training curves",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig.write_html(str(out_html), include_plotlyjs="cdn", full_html=True)
    _log.info("training_curves_saved", path=str(out_html))
    return out_html


def training_curves_div(tb_logdir: str | Path) -> str | None:
    """Return an inline Plotly ``<div>`` for embedding into the report."""
    scalars = read_tb_scalars(tb_logdir)
    if not scalars:
        return None
    try:
        import plotly.graph_objects as go  # type: ignore[import-untyped]
        from plotly.subplots import make_subplots  # type: ignore[import-untyped]
    except ImportError:
        return None

    tags = _canonical_subplot_order(list(scalars.keys()))
    n = len(tags)
    cols = 2
    rows = (n + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=tags)
    for i, tag in enumerate(tags):
        r = i // cols + 1
        c = i % cols + 1
        steps, values = scalars[tag]
        fig.add_trace(
            go.Scatter(x=steps, y=values, mode="lines", name=tag, showlegend=False),
            row=r,
            col=c,
        )
        fig.update_xaxes(title_text="step", row=r, col=c)
    fig.update_layout(height=260 * rows + 80, margin=dict(l=40, r=20, t=40, b=40))
    return fig.to_html(include_plotlyjs="cdn", full_html=False)


def _unused_guard() -> Any:  # pragma: no cover
    return plt
