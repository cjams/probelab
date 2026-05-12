"""
Visualizations for refusal-pipeline results.

Two axes of comparison are interesting here:
  - Across configs *within* a model: did changing the token selector change
    where the causal direction lives, or how strong it is?
  - Across models *for a fixed config*: does Arditi's last-token extraction
    reproduce on every architecture, or does the best config differ by model?

Each plot returns a Plotly Figure.
"""

from __future__ import annotations

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from probelab.experiments.refusal.results import RunResults
from probelab.viz import _stack_unit


def _short(model_id: str) -> str:
    return model_id.split("/")[-1]


def _layer_pairs(d: dict[int, float]) -> tuple[list[int], list[float]]:
    layers = sorted(d)
    return layers, [d[l] for l in layers]


# ---------------------------------------------------------------------------
# Within-model: configs side by side
# ---------------------------------------------------------------------------

def plot_refusal_by_config(
    run: RunResults,
    *,
    title: str | None = None,
) -> go.Figure:
    """Per-layer post-ablation refusal rate, one line per config, on shared axes.

    Lower is better (less refusal after ablating the probe direction at every
    layer during generation). The horizontal dashed line is the no-intervention
    baseline. The gap between the line and the baseline at the best layer is
    the causal "lift" the direction provides.
    """
    fig = go.Figure()
    baseline_values: list[float] = []

    for c in run.configs.values():
        layers, vals = _layer_pairs(c.causal_per_layer)
        fig.add_trace(go.Scatter(
            x=layers, y=vals, mode="lines+markers", name=c.name,
        ))
        baseline_values.append(c.causal_baseline)

    if baseline_values:
        baseline = sum(baseline_values) / len(baseline_values)
        fig.add_hline(
            y=baseline, line_dash="dash", line_color="gray",
            annotation_text=f"baseline ≈ {baseline:.2%}",
            annotation_position="top right",
        )

    fig.update_layout(
        title=title or f"Refusal rate after ablation by layer — {_short(run.model_id)}",
        xaxis_title="layer (direction source)",
        yaxis_title="refusal rate (lower is better)",
        yaxis=dict(range=[0, 1]),
    )

    return fig


def plot_probe_accuracy_by_config(
    run: RunResults,
    *,
    title: str | None = None,
) -> go.Figure:
    """Per-layer probe dev accuracy, one line per config — the *predictive*
    side of the comparison. Useful when paired with `plot_refusal_by_config`:
    a config can have higher dev acc but worse causal effect, which is the
    Arditi "predictive ≠ causal" point made flexible across hyperparameters."""
    fig = go.Figure()

    for c in run.configs.values():
        layers, vals = _layer_pairs(c.dev_accs)
        fig.add_trace(go.Scatter(
            x=layers, y=vals, mode="lines+markers", name=c.name,
        ))

    fig.add_hline(y=0.5, line_dash="dot", line_color="gray",
                  annotation_text="chance", annotation_position="top right")

    fig.update_layout(
        title=title or f"Probe dev accuracy by layer — {_short(run.model_id)}",
        xaxis_title="layer",
        yaxis_title="accuracy",
        yaxis=dict(range=[0.4, 1.0]),
    )

    return fig


def plot_predictive_vs_causal_by_config(
    run: RunResults,
    *,
    title: str | None = None,
) -> go.Figure:
    """One subplot per config, dual y-axes: dev acc (left), refusal rate (right).
    Vertical guides at the best probe layer (blue) and best causal layer (red).
    Lets you see, per config, whether the layer that linearly separates harmful
    vs benign is the same layer the model uses to *generate* refusals."""
    n = len(run.configs)

    fig = make_subplots(
        rows=n, cols=1,
        subplot_titles=list(run.configs.keys()),
        specs=[[{"secondary_y": True}] for _ in range(n)],
        vertical_spacing=0.15,
    )

    for row, c in enumerate(run.configs.values(), 1):
        layers, dev = _layer_pairs(c.dev_accs)
        _,      ref = _layer_pairs(c.causal_per_layer)
        first = (row == 1)

        fig.add_trace(go.Scatter(
            x=layers, y=dev, mode="lines+markers", name="dev acc",
            legendgroup="dev", showlegend=first,
        ), row=row, col=1, secondary_y=False)

        fig.add_trace(go.Scatter(
            x=layers, y=ref, mode="lines+markers",
            line=dict(color="firebrick"), name="refusal after ablation",
            legendgroup="ref", showlegend=first,
        ), row=row, col=1, secondary_y=True)

        fig.add_vline(x=c.best_probe_layer,  line_dash="dot",
                      line_color="steelblue", row=row, col=1)
        fig.add_vline(x=c.best_causal_layer, line_dash="dot",
                      line_color="firebrick", row=row, col=1)
        fig.add_hline(y=c.causal_baseline, line_dash="dash",
                      line_color="gray", row=row, col=1, secondary_y=True)

    fig.update_yaxes(title_text="dev acc", range=[0.4, 1.0], secondary_y=False)
    fig.update_yaxes(title_text="refusal rate", range=[0, 1], secondary_y=True)
    fig.update_xaxes(title_text="layer")
    fig.update_layout(
        height=350 * n,
        title=title or f"Predictive vs. causal, per config — {_short(run.model_id)}",
    )

    return fig


def plot_config_probe_similarity(
    run: RunResults,
    *,
    layer: int | None = None,
    signed: bool = False,
    title: str | None = None,
) -> go.Figure:
    """Pairwise cosine similarity between configs' probe directions at one layer.

    If `layer` is None, uses the best causal layer of the first config. A
    high-similarity matrix (~1.0 off-diagonal) means all configs are picking
    up the same axis; lower similarity means different positions are
    extracting different directions."""
    config_names = list(run.configs.keys())
    n = len(config_names)

    if layer is None:
        layer = next(iter(run.configs.values())).best_causal_layer

    # Build a (n_configs, d_model) stack of unit-normed directions at `layer`.
    probes_at_layer = {}

    for name, c in run.configs.items():
        all_probes = c.load_probes()

        if layer not in all_probes:
            raise KeyError(
                f"Config {name!r} has no probe at layer {layer}; "
                f"available: {sorted(all_probes)[:5]}..."
            )

        probes_at_layer[name] = all_probes[layer]

    # _stack_unit wants int-keyed; reuse by mapping config names to indices.
    indexed = {i: probes_at_layer[name] for i, name in enumerate(config_names)}
    _, stack = _stack_unit(indexed)
    sim = stack @ stack.T

    if not signed:
        sim = sim.abs()

    fig = go.Figure(go.Heatmap(
        z=sim.numpy(), x=config_names, y=config_names,
        zmin=(0 if not signed else -1), zmax=1,
        colorscale="RdBu", reversescale=True,
        colorbar=dict(title=("|cos|" if not signed else "cos sim")),
    ))

    fig.update_layout(
        title=title or (
            f"Config probe direction similarity at layer {layer} — "
            f"{_short(run.model_id)}"
        ),
        height=400,
    )

    return fig


# ---------------------------------------------------------------------------
# Cross-model: fixed config across models
# ---------------------------------------------------------------------------

def plot_refusal_across_models(
    runs: dict[str, RunResults],
    config_name: str,
    *,
    depth_normalize: bool = False,
    title: str | None = None,
) -> go.Figure:
    """Per-layer post-ablation refusal rate for a single config, one line per
    model. Set `depth_normalize=True` to plot layer-fraction-of-depth on the
    x-axis so models with different depths are comparable."""
    fig = go.Figure()
    baseline_values: list[float] = []

    for run in runs.values():
        if config_name not in run.configs:
            continue

        c = run.configs[config_name]
        layers, vals = _layer_pairs(c.causal_per_layer)
        baseline_values.append(c.causal_baseline)

        if depth_normalize:
            xs = [l / c.n_layers for l in layers]
        else:
            xs = layers

        fig.add_trace(go.Scatter(
            x=xs, y=vals, mode="lines+markers", name=_short(run.model_id),
        ))

    if baseline_values:
        baseline = sum(baseline_values) / len(baseline_values)
        fig.add_hline(
            y=baseline, line_dash="dash", line_color="gray",
            annotation_text=f"baseline ≈ {baseline:.2%}",
            annotation_position="top right",
        )

    fig.update_layout(
        title=title or (
            f"Refusal rate after ablation across models, config={config_name}"
            + (" (depth-normalized)" if depth_normalize else "")
        ),
        xaxis_title=("layer / depth" if depth_normalize else "layer"),
        yaxis_title="refusal rate (lower is better)",
        yaxis=dict(range=[0, 1]),
    )

    return fig


def plot_best_config_per_model(
    runs: dict[str, RunResults],
    *,
    title: str = "Best config per model (lowest refusal after ablation)",
) -> go.Figure:
    """For each model, bar of post-ablation refusal at the best (config, layer)
    pair. Annotates which config won so cross-architecture differences are
    visible at a glance."""
    names: list[str] = []
    bars:  list[float] = []
    annot: list[str] = []

    for run in runs.values():
        # Find the (config, layer) with the lowest causal metric.
        best_name = None
        best_layer = None
        best_value = float("inf")

        for cname, c in run.configs.items():
            layer = c.best_causal_layer
            value = c.causal_per_layer[layer]

            if value < best_value:
                best_value = value
                best_name  = cname
                best_layer = layer

        names.append(_short(run.model_id))
        bars.append(best_value)
        annot.append(f"{best_name} @ layer {best_layer}")

    fig = go.Figure(go.Bar(
        x=names, y=bars, text=annot, textposition="outside",
    ))

    fig.update_layout(
        title=title,
        yaxis_title="refusal rate (after best (config, layer) ablation)",
        yaxis=dict(range=[0, 1]),
    )

    return fig
