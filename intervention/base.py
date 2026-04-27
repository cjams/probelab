from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

import torch

if TYPE_CHECKING:
    from dataset.base import ProbeDataset
    from evaluate.generate import ModelResponses
    from train.token import TokenSelector
    import plotly.graph_objects as go


Mode = Literal["add", "subtract", "ablate"]

LayerSpec = int | list[int] | Literal["all"]


@dataclass
class Intervention:
    """A single intervention to apply at a specific layer.

    Modes (scale is always a non-negative magnitude/fraction):
      add       hidden += scale * direction                   (steer toward direction)
      subtract  hidden -= scale * direction                   (steer away from direction)
      ablate    hidden -= scale * (hidden · direction) * direction   (fractional projection ablation;
                                                              scale=1 is full ablation)

    component: which hook point(s) to apply the intervention at. Uses the
    same vocabulary as ActivationSpec.component. A single string hooks one
    point per layer; a list hooks every listed component at every hook layer
    in the same forward pass (so e.g. component=["resid_pre", "resid_mid",
    "resid_post", "mlp_out", "attn_out"] simultaneously ablates the direction
    from all five residual-stream hook points at every layer).

    Only "resid_post" is supported by the HuggingFace backend; the
    TransformerLens backend supports every value ("resid_pre", "resid_mid",
    "resid_post", "mlp_out", "attn_out") and any combination thereof.
    """
    direction: torch.Tensor
    scale: float
    mode: Mode = "add"
    apply_on: Literal["prefill", "all"] = "all"
    component: str | list[str] = "resid_post"

    def components(self) -> list[str]:
        """Normalize self.component to a list of component strings."""
        return [self.component] if isinstance(self.component, str) else list(self.component)


def apply_intervention(
    hidden: torch.Tensor,       # (batch, seq_len, d_model)
    mask: torch.Tensor,         # (batch, seq_len) bool
    direction: torch.Tensor,    # (d_model,)
    intervention: Intervention,
) -> None:
    """Apply intervention in-place at positions where mask is True.

    Shared between HF and TL backends so the semantics of (mode, scale,
    direction) match exactly across them.
    """
    direction = direction.to(hidden.dtype)

    # Boolean indexing flattens the masked positions: (n_selected, d_model).
    selected = hidden[mask]

    if intervention.mode == "add":
        # direction broadcasts from (d_model,) to (n_selected, d_model).
        hidden[mask] = selected + intervention.scale * direction

    elif intervention.mode == "subtract":
        hidden[mask] = selected - intervention.scale * direction

    elif intervention.mode == "ablate":
        # (selected @ direction) is (n_selected,); unsqueeze gives (n_selected, 1)
        # so the outer product with direction is (n_selected, d_model).
        # scale=1 removes the full projection; scale<1 removes a fraction.
        proj = (selected @ direction).unsqueeze(-1) * direction
        hidden[mask] = selected - intervention.scale * proj

    else:
        raise ValueError(f"Unknown intervention mode: {intervention.mode!r}")


class InterventionBackend(ABC):
    @abstractmethod
    def collect_responses(
        self,
        dataset: "ProbeDataset",
        hook_layers: int | list[int],
        token_selector: "TokenSelector",
        intervention: Intervention | None,
        batch_size: int = 8,
        prompt_fn: Callable | None = None,
        command_fn: Callable | None = None,
        **generate_kwargs,
    ) -> "ModelResponses":
        """Generate responses with the intervention applied at every layer in
        hook_layers within a single forward pass. The same (direction, scale,
        mode) is installed as a hook at each listed layer."""
        ...

    @abstractmethod
    def num_transformer_layers(self) -> int:
        """Number of transformer blocks in the underlying model.

        Used to resolve LayerSpec="all" into the concrete list [1..N].
        Indexing matches the hidden_states convention where 0 is the
        embedding output, so transformer layers run from 1 to N.
        """
        ...


@dataclass
class InterventionSweepResult:
    hook_layers: list[int]
    interventions: list[Intervention]
    # One metric per intervention — all interventions share the same hook_layers.
    metric_values: list[float]
    baseline_value: float | None = None

    def plot(self, **kwargs) -> "go.Figure":
        from intervention.viz import plot_intervention_sweep
        return plot_intervention_sweep(self, **kwargs)


def make_scale_sweep(
    direction: torch.Tensor,
    scales: list[float],
    mode: Mode = "add",
    apply_on: Literal["prefill", "all"] = "all",
    component: str | list[str] = "resid_post",
) -> list[Intervention]:
    """Build a list of interventions that share direction/mode/component and vary by scale."""
    return [
        Intervention(
            direction=direction,
            scale=s,
            mode=mode,
            apply_on=apply_on,
            component=component,
        )
        for s in scales
    ]


def intervention_sweep(
    backend: InterventionBackend,
    dataset: "ProbeDataset",
    hook_layers: LayerSpec,
    token_selector: "TokenSelector",
    interventions: list[Intervention],
    metric_fn: Callable[["ModelResponses"], float],
    include_baseline: bool = True,
    **collect_kwargs,
) -> InterventionSweepResult:
    """Run each intervention (with hooks at all hook_layers simultaneously) and
    collect metric values.

    Args:
        hook_layers: Single layer index, list of layers, or "all" to hook every
                     transformer layer (1..N). For each intervention, the same
                     (direction, scale, mode) is applied at every listed layer
                     within one forward pass.

    Use make_scale_sweep to build a simple scale sweep, or construct the list
    directly to mix modes/directions.
    """
    if hook_layers == "all":
        layers = list(range(1, backend.num_transformer_layers() + 1))
    elif isinstance(hook_layers, int):
        layers = [hook_layers]
    else:
        layers = list(hook_layers)

    baseline_value = None
    n = len(interventions)

    if include_baseline:
        baseline_responses = backend.collect_responses(
            dataset=dataset,
            hook_layers=layers,
            token_selector=token_selector,
            intervention=None,
            **collect_kwargs,
        )
        baseline_value = metric_fn(baseline_responses)
        print(f"  baseline: {baseline_value:.3f}", flush=True)

    metric_values: list[float] = []

    for i, intervention in enumerate(interventions, 1):
        responses = backend.collect_responses(
            dataset=dataset,
            hook_layers=layers,
            token_selector=token_selector,
            intervention=intervention,
            **collect_kwargs,
        )
        value = metric_fn(responses)
        metric_values.append(value)
        print(f"  [{i}/{n}] scale={intervention.scale:g} mode={intervention.mode}: {value:.3f}", flush=True)

    return InterventionSweepResult(
        hook_layers=layers,
        interventions=interventions,
        metric_values=metric_values,
        baseline_value=baseline_value,
    )
