from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import plotly.graph_objects as go

if TYPE_CHECKING:
    from intervention.base import InterventionSweepResult


def plot_intervention_sweep(
    result: "InterventionSweepResult",
    title: str = "Intervention Sweep",
    yaxis_title: str = "Metric",
) -> go.Figure:
    """Plot metric value as a function of intervention scale, grouped by mode.

    All interventions share the same hook_layers, so the x-axis is scale
    (interpretation depends on mode). A horizontal dashed line marks the
    baseline (no intervention) if provided.
    """
    fig = go.Figure()

    by_mode: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for intervention, metric in zip(result.interventions, result.metric_values):
        by_mode[intervention.mode].append((intervention.scale, metric))

    for mode, points in by_mode.items():
        points.sort(key=lambda p: p[0])
        xs, ys = zip(*points)

        fig.add_trace(go.Scatter(
            x=list(xs),
            y=list(ys),
            mode="lines+markers",
            name=mode,
        ))

    if result.baseline_value is not None:
        fig.add_hline(
            y=result.baseline_value,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"baseline ({result.baseline_value:.2f})",
            annotation_position="top left",
        )

    if len(result.hook_layers) == 1:
        layer_str = f"layer {result.hook_layers[0]}"
    elif len(result.hook_layers) <= 6:
        layer_str = f"layers {result.hook_layers}"
    else:
        layer_str = f"{len(result.hook_layers)} layers"

    fig.update_layout(
        title=f"{title} ({layer_str})",
        xaxis_title="Scale",
        yaxis_title=yaxis_title,
    )

    return fig
