"""
Generic single-model probe-sweep visualizations.

These take primitive inputs (`dict[int, torch.Tensor]` of probe directions,
`dict[int, float]` of accuracies) and return single-figure Plotly plots —
nothing about a specific experiment is assumed. Multi-model orchestration
and experiment-specific framing live in `probelab.<experiment>.viz`.

For live results from `sweep_layers` / `validate_by_ablation`, see
`probelab.train.viz`.
"""

from __future__ import annotations

import torch
import plotly.graph_objects as go


def _stack_unit(probes: dict[int, torch.Tensor]) -> tuple[list[int], torch.Tensor]:
    """Return (sorted layer ids, (n_layers, d_model) unit-normed tensor)."""
    layers = sorted(probes)
    stack = torch.stack([probes[l] for l in layers]).float()
    stack = stack / stack.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    return layers, stack


def plot_probe_pairwise_cosine(
    probes: dict[int, torch.Tensor],
    *,
    signed: bool = True,
    title: str = "Probe direction cosine similarity across layers",
) -> go.Figure:
    """Heatmap of cos(probe[i], probe[j]) for every pair of layers in `probes`.

    Each direction is unit-normed before the dot product so probe magnitude
    doesn't influence the comparison. With `signed=False` the absolute value
    is taken — useful for DIM probes, whose sign is arbitrary (depends on the
    incidental ordering of class means), so a sign flip across layers reads
    as "same axis" rather than "opposite direction".
    """
    layers, stack = _stack_unit(probes)
    sim = stack @ stack.T

    if not signed:
        sim = sim.abs()

    fig = go.Figure(go.Heatmap(
        z=sim.numpy(), x=layers, y=layers,
        zmin=(0 if not signed else -1), zmax=1,
        colorscale="RdBu", reversescale=True,
        colorbar=dict(title=("|cos|" if not signed else "cos sim")),
    ))

    fig.update_xaxes(title_text="layer")
    fig.update_yaxes(title_text="layer")
    fig.update_layout(title=title, height=500)

    return fig


def plot_probe_cosine_to_anchor(
    probes: dict[int, torch.Tensor],
    anchor_layer: int,
    *,
    signed: bool = False,
    title: str | None = None,
) -> go.Figure:
    """Per-layer cos(probe[i], probe[anchor_layer]) as a 1D line.

    A flat line at y=1 means every layer's probe is colinear with the anchor —
    "one direction throughout". A high plateau around the anchor that decays
    at the extremes is the typical shape and tells you how broad the
    truth-direction band is.

    Args:
        probes:       layer -> probe direction tensor (any d_model).
        anchor_layer: which layer to compare every other layer against. Must
                      be a key in `probes`.
        signed:       if False (default), plot |cos| so DIM sign flips don't
                      produce a misleading dip to -1.
    """
    if anchor_layer not in probes:
        raise KeyError(f"anchor_layer {anchor_layer} not present in probes "
                       f"(have {sorted(probes)[:5]}...).")

    layers, stack = _stack_unit(probes)
    anchor_idx    = layers.index(anchor_layer)
    sims          = (stack @ stack[anchor_idx])

    if not signed:
        sims = sims.abs()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=layers, y=sims.numpy(), mode="lines+markers",
    ))

    fig.add_vline(
        x=anchor_layer, line_dash="dot", line_color="firebrick",
        annotation_text=f"anchor (layer {anchor_layer})",
        annotation_position="top right",
    )

    fig.update_layout(
        title=title or f"Probe direction cosine to layer {anchor_layer}",
        xaxis_title="layer",
        yaxis_title=("|cos|" if not signed else "cos") + " similarity to anchor",
        yaxis=dict(range=[-1.05, 1.05] if signed else [0, 1.05]),
    )

    return fig


def plot_layer_acc_gap(
    train: dict[int, float],
    dev: dict[int, float],
    *,
    name: str | None = None,
    title: str = "Probe overfitting check: train − dev accuracy by layer",
) -> go.Figure:
    """Per-layer (train − dev) gap. Big positive spikes at specific layers
    flag where the probe is fitting noise rather than signal."""
    layers = sorted(set(train) & set(dev))
    gap    = [train[l] - dev[l] for l in layers]

    fig = go.Figure(go.Scatter(
        x=layers, y=gap, mode="lines+markers", name=name,
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title=title,
        xaxis_title="layer",
        yaxis_title="train − dev",
    )

    return fig
