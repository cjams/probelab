import plotly.graph_objects as go

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from train.sweep import LayerSweepResult, MultiModelSweepResult


def plot_layer_sweep(
    result: "LayerSweepResult",
    test_accs: dict[int, float] | None = None,
    title: str = "Probe Accuracy by Layer",
) -> go.Figure:
    """
    Plot train, dev, and optionally test accuracy for each swept layer.

    Args:
        result:    LayerSweepResult from sweep_layers().
        test_accs: Optional dict mapping layer index -> test accuracy, returned
                   by LayerSweepResult.evaluate(). Pass this only after the sweep
                   is complete and the best layer has been chosen.
        title:     Plot title.

    Returns:
        A Plotly Figure.
    """
    layers = sorted(result.probes.keys())

    train_accs = [result.train_accs[l] for l in layers]
    dev_accs = [result.dev_accs[l] for l in layers]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=layers,
        y=train_accs,
        mode="lines+markers",
        name="Train",
    ))

    fig.add_trace(go.Scatter(
        x=layers,
        y=dev_accs,
        mode="lines+markers",
        name="Dev",
    ))

    if test_accs is not None:
        fig.add_trace(go.Scatter(
            x=layers,
            y=[test_accs[l] for l in layers],
            mode="lines+markers",
            name="Test",
        ))

    fig.add_vline(
        x=result.best_layer,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"best (layer {result.best_layer})",
        annotation_position="top right",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0, 1]),
    )

    return fig


def plot_multi_model_sweep(
    result: "MultiModelSweepResult",
    test_accs: dict[str, dict[int, float]] | None = None,
    title: str = "Probe Accuracy by Layer",
) -> go.Figure:
    """
    Overlay dev (and optionally test) accuracy curves for each model.

    Each model gets its own color. Train curves are shown as faint dashed
    lines; dev curves are solid. A vertical marker indicates the best layer
    chosen per model. Test curves, when provided, are shown as dotted lines.

    Args:
        result:    MultiModelSweepResult from multi_model_sweep().
        test_accs: Optional dict mapping model_id -> (layer -> test accuracy),
                   returned by MultiModelSweepResult.evaluate().
        title:     Plot title.

    Returns:
        A Plotly Figure.
    """
    fig = go.Figure()

    for model_id, sweep in result.results.items():
        layers = sorted(sweep.probes.keys())

        train_accs = [sweep.train_accs[l] for l in layers]
        dev_accs = [sweep.dev_accs[l] for l in layers]

        fig.add_trace(go.Scatter(
            x=layers,
            y=train_accs,
            mode="lines",
            name=f"{model_id} train",
            line=dict(dash="dash"),
            opacity=0.4,
            legendgroup=model_id,
        ))

        fig.add_trace(go.Scatter(
            x=layers,
            y=dev_accs,
            mode="lines+markers",
            name=f"{model_id} dev",
            legendgroup=model_id,
        ))

        if test_accs is not None and model_id in test_accs:
            fig.add_trace(go.Scatter(
                x=layers,
                y=[test_accs[model_id][l] for l in layers],
                mode="lines+markers",
                name=f"{model_id} test",
                line=dict(dash="dot"),
                legendgroup=model_id,
            ))

        fig.add_vline(
            x=sweep.best_layer,
            line_dash="dash",
            line_color="gray",
            opacity=0.4,
            annotation_text=f"best (layer {sweep.best_layer})",
            annotation_position="top right",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Accuracy",
        yaxis=dict(range=[0, 1]),
    )

    return fig
