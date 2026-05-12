"""
Cross-run visualizations for `RunResults` loaded from disk.

These are about *post-hoc* analysis after `probelab.experiments.geometry_of_truth.pipeline` has run
each model and written its outputs. For live in-memory plots over a single
sweep / ablation, see `probelab.train.viz` instead.

Each plot function takes `dict[model_id, RunResults]` (as returned by
`probelab.experiments.geometry_of_truth.results.load_runs`) and returns a Plotly Figure.
"""

from __future__ import annotations

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from probelab.experiments.geometry_of_truth.results import RunResults
from probelab.viz import _stack_unit


def _short(model_id: str) -> str:
    return model_id.split("/")[-1]


def plot_predictive_vs_causal(
    runs: dict[str, RunResults],
    title: str = "Probe predictive power vs. causal effect, per layer",
) -> go.Figure:
    """One subplot per model overlaying:
      - dev accuracy and OOD accuracy on the left y-axis,
      - per-layer causal metric (e.g. log-odds gap) on the right y-axis.

    Vertical guides mark the best probe layer (by dev acc) and the best
    causal layer (by intervention metric); the gap between them is the
    "predictive vs causal" question made visible.
    """
    n = len(runs)

    fig = make_subplots(
        rows=n, cols=1,
        subplot_titles=[_short(r.model_id) for r in runs.values()],
        specs=[[{"secondary_y": True}] for _ in range(n)],
        vertical_spacing=0.12,
    )

    for row, r in enumerate(runs.values(), 1):
        layers = sorted(r.dev_accs)
        dev    = [r.dev_accs[l]                 for l in layers]
        ood    = [r.ood_accs.get(l, None)       for l in layers]
        cau    = [r.causal_per_layer[l]         for l in layers]

        first = (row == 1)

        fig.add_trace(go.Scatter(
            x=layers, y=dev, name="dev acc",
            mode="lines+markers", legendgroup="dev", showlegend=first,
        ), row=row, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=layers, y=ood, name="OOD acc",
            mode="lines+markers", line=dict(dash="dash"),
            legendgroup="ood", showlegend=first,
        ), row=row, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=layers, y=cau, name="causal metric",
            mode="lines+markers", line=dict(color="firebrick"),
            legendgroup="cau", showlegend=first,
        ), row=row, col=1, secondary_y=True)

        fig.add_vline(x=r.best_probe_layer,  line_dash="dot",
                      line_color="steelblue", row=row, col=1)
        fig.add_vline(x=r.best_causal_layer, line_dash="dot",
                      line_color="firebrick", row=row, col=1)

        fig.add_hline(y=r.causal_baseline, line_dash="dash",
                      line_color="gray", row=row, col=1, secondary_y=True)

    fig.update_yaxes(title_text="accuracy", range=[0.4, 1.0], secondary_y=False)
    fig.update_yaxes(title_text="causal metric", secondary_y=True)
    fig.update_xaxes(title_text="layer")
    fig.update_layout(height=350 * n, title=title)

    return fig


def plot_probe_direction_stability(
    runs: dict[str, RunResults],
    *,
    signed: bool = True,
    title: str = "Probe direction cosine similarity across layers",
) -> go.Figure:
    """Per-model heatmap of cos(probe[i], probe[j]) for every pair of layers.

    A bright square in the middle indicates a stable probe direction across
    a contiguous block of layers — what M&T call a "geometric truth direction".
    A noisy off-diagonal means the direction rotates and isn't really one axis.

    Pass `signed=False` to take |cos| — useful because DIM probe sign is
    arbitrary, so a sign flip across layers is "same axis", not "opposite".
    """
    n = len(runs)

    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=[_short(r.model_id) for r in runs.values()],
        horizontal_spacing=0.12,
    )

    for col, r in enumerate(runs.values(), 1):
        layers, stack = _stack_unit(r.load_probes())
        sim = stack @ stack.T

        if not signed:
            sim = sim.abs()

        fig.add_trace(go.Heatmap(
            z=sim.numpy(), x=layers, y=layers,
            zmin=(0 if not signed else -1), zmax=1,
            colorscale="RdBu", reversescale=True,
            colorbar=dict(title=("|cos|" if not signed else "cos sim"),
                          x=col / n - 0.02),
            showscale=(col == n),
        ), row=1, col=col)

        fig.update_xaxes(title_text="layer", row=1, col=col)
        fig.update_yaxes(title_text="layer", row=1, col=col)

    fig.update_layout(height=500, title=title)

    return fig


def plot_probe_direction_vs_best_causal(
    runs: dict[str, RunResults],
    *,
    signed: bool = False,
    title: str = "Probe direction cosine to best causal layer, by layer",
) -> go.Figure:
    """1D slice of the stability heatmap: cos(probe[i], probe[best_causal_layer])
    per model on shared axes.

    A flat line at y=1 means every layer's probe is colinear with the layer
    that does the most causal work — i.e. the model has *one* truth direction
    and uses it everywhere. A high plateau around the anchor that decays at
    the extremes is the typical shape; the plateau width is the
    truth-direction band.

    Default `signed=False` (|cos|) collapses arbitrary DIM sign flips into
    the band; pass `signed=True` if the sign itself matters to you.
    """
    fig = go.Figure()

    for r in runs.values():
        layers, stack = _stack_unit(r.load_probes())
        anchor_idx    = layers.index(r.best_causal_layer)
        sims          = stack @ stack[anchor_idx]

        if not signed:
            sims = sims.abs()

        fig.add_trace(go.Scatter(
            x=layers, y=sims.numpy(), mode="lines+markers",
            name=f"{_short(r.model_id)} (anchor={r.best_causal_layer})",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="layer",
        yaxis_title=("|cos|" if not signed else "cos") + " similarity to best causal layer",
        yaxis=dict(range=[-1.05, 1.05] if signed else [0, 1.05]),
    )

    return fig


def plot_best_layers(
    runs: dict[str, RunResults],
    title: str = "Where in the model does the probe direction live? (depth-normalized)",
) -> go.Figure:
    """Side-by-side bars of best-probe-layer / depth and best-causal-layer / depth
    per model. Depth-normalized so models of different sizes are comparable."""
    names = [_short(r.model_id) for r in runs.values()]

    probe_frac  = [r.best_probe_layer  / r.n_layers for r in runs.values()]
    causal_frac = [r.best_causal_layer / r.n_layers for r in runs.values()]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=probe_frac,  name="best probe / depth"))
    fig.add_trace(go.Bar(x=names, y=causal_frac, name="best causal / depth"))

    fig.update_layout(
        barmode="group",
        yaxis_title="fraction of total depth",
        yaxis=dict(range=[0, 1]),
        title=title,
    )

    return fig


def plot_overfitting(
    runs: dict[str, RunResults],
    title: str = "Probe overfitting check: train acc − dev acc by layer",
) -> go.Figure:
    """Per-layer (train − dev) gap, one line per model. Big positive spikes
    flag layers where the probe is fitting noise rather than signal."""
    fig = go.Figure()

    for r in runs.values():
        layers = sorted(set(r.train_accs) & set(r.dev_accs))
        gap    = [r.train_accs[l] - r.dev_accs[l] for l in layers]

        fig.add_trace(go.Scatter(
            x=layers, y=gap, mode="lines+markers", name=_short(r.model_id),
        ))

    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title=title,
        xaxis_title="layer",
        yaxis_title="train − dev",
    )

    return fig
