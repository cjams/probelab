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
    """
    direction: torch.Tensor
    scale: float
    mode: Mode = "add"
    apply_on: Literal["prefill", "all"] = "all"


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
) -> list[Intervention]:
    """Build a list of interventions that share direction/mode and vary by scale."""
    return [
        Intervention(direction=direction, scale=s, mode=mode, apply_on=apply_on)
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
    from tqdm.auto import tqdm

    if hook_layers == "all":
        layers = list(range(1, backend.num_transformer_layers() + 1))
    elif isinstance(hook_layers, int):
        layers = [hook_layers]
    else:
        layers = list(hook_layers)

    baseline_value = None

    if include_baseline:
        baseline_responses = backend.collect_responses(
            dataset=dataset,
            hook_layers=layers,
            token_selector=token_selector,
            intervention=None,
            **collect_kwargs,
        )
        baseline_value = metric_fn(baseline_responses)

    metric_values: list[float] = []

    for intervention in tqdm(interventions, desc="intervention sweep"):
        responses = backend.collect_responses(
            dataset=dataset,
            hook_layers=layers,
            token_selector=token_selector,
            intervention=intervention,
            **collect_kwargs,
        )
        metric_values.append(metric_fn(responses))

    return InterventionSweepResult(
        hook_layers=layers,
        interventions=interventions,
        metric_values=metric_values,
        baseline_value=baseline_value,
    )
