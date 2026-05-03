from __future__ import annotations

import gc
import torch

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal

from probelab.dataset.base import ProbeDataset, Example
from probelab.train.activation import ActivationCollector, ActivationDataset
from probelab.train.probe import Probe, ProbeTrainer
from probelab.train.token import TokenSelector, TokenReducer, LayerSelection

if TYPE_CHECKING:
    from probelab.model import ModelHandle
    from probelab.intervention.base import InterventionBackend, LayerSpec
    from probelab.evaluate.generate import ModelResponses
    import plotly.graph_objects as go


@dataclass
class LayerSweepResult:
    """
    Output of sweep_layers(). Holds all trained probes and their dev accuracies.

    Call evaluate() with held-out test activations to get final test accuracy
    after the sweep is complete.
    """

    probes: dict[int, Probe]
    train_accs: dict[int, float]
    dev_accs: dict[int, float]
    best_layer: int
    best_probe: Probe

    # stored so evaluate() uses the same pipeline as the sweep
    selector: TokenSelector
    reducer: TokenReducer

    def evaluate(self, test_dataset: ActivationDataset) -> dict[int, float]:
        """
        Evaluate all layer probes on held-out test activations.

        This should only be called once, after the sweep is complete and the
        best configuration has been chosen using dev_accs.
        """
        layers = list(self.probes.keys())
        acts_3d, labels, mask = self.selector.select(test_dataset, layers)

        reduced = self.reducer.reduce(acts_3d, labels, mask)
        test_acts, test_labels = reduced[0], reduced[1]

        return {
            layer: (probe.predict(test_acts[layer]) == test_labels).float().mean().item()
            for layer, probe in self.probes.items()
        }

    def plot(self, test_accs: "dict[int, float] | None" = None, title: str = "Probe Accuracy by Layer"):
        from probelab.train.viz import plot_layer_sweep
        return plot_layer_sweep(self, test_accs=test_accs, title=title)

    def validate_by_ablation(
        self,
        dataset: ProbeDataset,
        backend: "InterventionBackend",
        metric_fn: Callable[["ModelResponses"], float],
        token_selector: TokenSelector | None = None,
        hook_layers: "LayerSpec" = "all_transformer",
        component: str | list[str] = "resid_post",
        scale: float = 1.0,
        mode: Literal["add", "subtract", "ablate"] = "ablate",
        objective: Literal["min", "max"] = "min",
        include_baseline: bool = True,
        batch_size: int = 8,
        max_new_tokens: int = 256,
        prompt_fn: Callable | None = None,
        command_fn: Callable | None = None,
        target_tokens: "dict[str, int | list[int]] | None" = None,
    ) -> "LayerAblationResult":
        """
        Re-rank the trained probes by the behavioral effect of ablating each
        layer's direction across the model.

        For each probe in self.probes, runs a generation pass over `dataset`
        with the probe's direction ablated (mode="ablate") at every layer in
        `hook_layers` and every token selected by `token_selector`, then scores
        the responses with `metric_fn`, then picks the probe that minimises
        (or maximises) the metric per `objective`.

        Args:
            dataset:          Held-out behavioural validation set. Generation
                              runs once per probe, so keep this small.
            backend:          InterventionBackend (HF or TL) for generation.
            metric_fn:        ModelResponses -> float. e.g. positive_rate from
                              a SemanticJudge, but works for any scalar metric.
            token_selector:   Where the intervention applies in the prompt.
                              Defaults to AllTokenSelector() (all real tokens).
            hook_layers:      Which transformer layers to ablate at.
                              "all_transformer" ablates at every layer (1..N).
            component:        Hook point per layer. Pass a list to TL backend
                              to ablate at multiple residual-stream points
                              simultaneously. Default is resid_post
            scale:            Magnitude. For mode="ablate" this is the
                              projection fraction (1.0 = full removal); for
                              "add" / "subtract" it scales the direction.
            mode:             How the direction is applied. "ablate" (default,
                              the Arditi pattern) zeros the projection;
                              "add" / "subtract" steer along the direction.
            objective:        "min" picks the layer that minimises metric_fn
                              (typical for "ablation removes the direction's
                              effect"). "max" picks the maximiser.
            include_baseline: If True, also runs once with no intervention to
                              get the metric without ablation.
            batch_size:       Generation batch size.
            max_new_tokens:   Per-example generation budget.
            prompt_fn:        Optional formatter applied to each ProbeDataset example.
            command_fn:       Optional formatter producing the command text
                              passed to ModelResponses.commands (useful for downstream
                              judging tasks)
            target_tokens:    Forwarded to the backend; when set, the metric_fn
                              receives ModelResponses.target_logits populated
                              with first-generated-position logits per label.
                              See evaluate.generate.ModelResponses.

        Returns:
            LayerAblationResult with per-layer metrics and the best direction.
        """
        from probelab.train.token import AllTokenSelector
        from probelab.intervention.base import Intervention

        selector = token_selector if token_selector is not None else AllTokenSelector()

        # Resolve hook_layers up front. The backend's collect_responses
        # signature is int | list[int]; "all_transformer" is a sweep-level
        # convenience that expands to range(1, N+1) against the live model's
        # transformer-block count. The embedding (index 0) is not hookable
        # in the HF backend and is a passthrough in TL, which is why this
        # path doesn't accept the looser "all" literal that the
        # activation-collection vocabulary uses.
        if hook_layers == "all_transformer":
            resolved_hook_layers: list[int] = list(range(1, backend.num_transformer_layers() + 1))
        elif isinstance(hook_layers, int):
            resolved_hook_layers = [hook_layers]
        else:
            resolved_hook_layers = list(hook_layers)

        baseline_metric: float | None = None

        if include_baseline:
            baseline_responses = backend.collect_responses(
                dataset=dataset,
                hook_layers=resolved_hook_layers,
                token_selector=selector,
                intervention=None,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                prompt_fn=prompt_fn,
                command_fn=command_fn,
                target_tokens=target_tokens,
            )
            baseline_metric = metric_fn(baseline_responses)
            print(f"  baseline: {baseline_metric:.3f}", flush=True)

        per_layer_metric: dict[int, float] = {}
        layer_keys = sorted(self.probes.keys())
        n_layers = len(layer_keys)

        for i, layer in enumerate(layer_keys, 1):
            direction = self.probes[layer].direction

            intervention = Intervention(
                direction=direction,
                scale=scale,
                mode=mode,
                component=component,
            )

            responses = backend.collect_responses(
                dataset=dataset,
                hook_layers=resolved_hook_layers,
                token_selector=selector,
                intervention=intervention,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                prompt_fn=prompt_fn,
                command_fn=command_fn,
                target_tokens=target_tokens,
            )

            value = metric_fn(responses)
            per_layer_metric[layer] = value
            print(f"  [{i}/{n_layers}] layer {layer}: {value:.3f}", flush=True)

        best_layer = (
            min(per_layer_metric, key=per_layer_metric.__getitem__)
            if objective == "min"
            else max(per_layer_metric, key=per_layer_metric.__getitem__)
        )

        return LayerAblationResult(
            per_layer_metric=per_layer_metric,
            baseline_metric=baseline_metric,
            best_layer=best_layer,
            best_direction=self.probes[best_layer].direction,
            objective=objective,
        )


@dataclass
class LayerAblationResult:
    """
    Output of LayerSweepResult.validate_by_ablation(). Holds per-layer
    behavioural metric values and identifies the layer whose direction
    produced the largest ablation delta.
    """
    per_layer_metric: dict[int, float]
    baseline_metric: float | None
    best_layer: int
    best_direction: torch.Tensor   # (d_model,)
    objective: Literal["min", "max"]

    def best_delta(self) -> float:
        """Signed change at best_layer vs baseline; positive = improvement."""
        if self.baseline_metric is None:
            return float("nan")

        delta = self.per_layer_metric[self.best_layer] - self.baseline_metric
        return -delta if self.objective == "min" else delta

    def plot(self, title: str = "Ablation Effect by Layer") -> "go.Figure":
        from probelab.train.viz import plot_layer_ablation
        return plot_layer_ablation(self, title=title)


def _obj_summary(obj) -> str:
    """One-line summary of an object: ClassName(attr=val, ...)."""
    name = type(obj).__name__
    attrs = {k: v for k, v in vars(obj).items() if not k.startswith("_")}

    if not attrs:
        return name

    parts = ", ".join(f"{k}={v!r}" for k, v in attrs.items())
    return f"{name}({parts})"


def sweep_layers(
    train_dataset: ActivationDataset,
    dev_dataset: ActivationDataset,
    selector: TokenSelector,
    reducer: TokenReducer,
    trainer: ProbeTrainer,
    layers: LayerSelection = "all",
) -> LayerSweepResult:
    """
    Train a probe per layer and pick the best by dev accuracy.
    """
    train_acts_3d, train_labels, train_mask = selector.select(train_dataset, layers)
    dev_acts_3d, dev_labels, dev_mask = selector.select(dev_dataset, layers)

    n_layers = len(train_acts_3d)
    n_train = next(iter(train_acts_3d.values())).shape[0]
    n_dev = next(iter(dev_acts_3d.values())).shape[0]

    print(f"  layers={n_layers}  train={n_train}  dev={n_dev}")

    train_reduced = reducer.reduce(train_acts_3d, train_labels, train_mask)
    train_acts, train_labels_r = train_reduced[0], train_reduced[1]

    dev_reduced = reducer.reduce(dev_acts_3d, dev_labels, dev_mask)
    dev_acts, dev_labels_r = dev_reduced[0], dev_reduced[1]

    probes = trainer.fit_layers(train_acts, train_labels_r)

    train_accs = {}
    dev_accs = {}

    for layer, probe in probes.items():
        train_preds = probe.predict(train_acts[layer])
        train_accs[layer] = (train_preds == train_labels_r).float().mean().item()

        dev_preds = probe.predict(dev_acts[layer])
        dev_accs[layer] = (dev_preds == dev_labels_r).float().mean().item()

    best_layer = max(dev_accs, key=dev_accs.__getitem__)

    return LayerSweepResult(
        probes=probes,
        train_accs=train_accs,
        dev_accs=dev_accs,
        best_layer=best_layer,
        best_probe=probes[best_layer],
        selector=selector,
        reducer=reducer,
    )


@dataclass
class MultiModelSweepResult:
    """
    Output of multi_model_sweep(). One LayerSweepResult per model.

    Stores the model-loading factories so that downstream operations
    (test-set evaluation, ablation validation) can re-instantiate each model
    serially, releasing GPU memory between iterations.
    """

    results: dict[str, LayerSweepResult]
    handle_factory:    Callable[[str], "ModelHandle"]                                   = field(repr=False)
    collector_factory: Callable[["ModelHandle"], ActivationCollector]                   = field(repr=False)
    prompt_fn_factory: Callable[["ModelHandle"], Callable[[Example], str]] | None      = field(default=None, repr=False)

    def evaluate(
        self,
        test_probe_ds: ProbeDataset,
        batch_size: int = 16,
    ) -> dict[str, dict[int, float]]:
        """
        Evaluate every model's layer probes on held-out test data.

        Re-instantiates each handle/collector serially to avoid holding
        multiple models in memory simultaneously.
        """
        test_accs = {}
        n_models = len(self.results)

        for i, (model_id, result) in enumerate(self.results.items(), 1):
            prefix = f"[{i}/{n_models}] {model_id}"

            print(f"\n{prefix}  loading model...", flush=True)
            handle = self.handle_factory(model_id)
            collector = self.collector_factory(handle)
            prompt_fn = self.prompt_fn_factory(handle) if self.prompt_fn_factory is not None else None

            print(f"{prefix}  collecting test activations...", flush=True)
            test_acts = collector.collect(test_probe_ds, batch_size, prompt_fn)
            test_accs[model_id] = result.evaluate(test_acts)

            print(f"{prefix}  freeing GPU memory...", flush=True)
            
            del prompt_fn, collector, handle
            gc.collect()
            torch.cuda.empty_cache()
            
            print(f"{prefix}  done", flush=True)

        return test_accs

    def validate_by_ablation_per_model(
        self,
        dataset: ProbeDataset,
        backend_factory: Callable[["ModelHandle"], "InterventionBackend"],
        metric_fn: Callable[["ModelResponses"], float],
        token_selector_factory: Callable[["ModelHandle"], TokenSelector] | None = None,
        prompt_fn_factory: Callable[["ModelHandle"], Callable[[Example], str]] | None = None,
        command_fn_factory: Callable[["ModelHandle"], Callable[[Example], str]] | None = None,
        target_tokens_factory: Callable[["ModelHandle"], "dict[str, int | list[int]]"] | None = None,
        hook_layers: "LayerSpec" = "all_transformer",
        component: str | list[str] = "resid_post",
        scale: float = 1.0,
        mode: Literal["add", "subtract", "ablate"] = "ablate",
        objective: Literal["min", "max"] = "min",
        include_baseline: bool = True,
        batch_size: int = 8,
        max_new_tokens: int = 256,
    ) -> dict[str, LayerAblationResult]:
        """
        Run validate_by_ablation for each model serially.

        Loads each model via handle_factory, builds an InterventionBackend
        via backend_factory, runs the per-model validation, then frees GPU
        memory before loading the next.

        Override prompt_fn_factory or token_selector_factory if the ablation
        run needs different formatting/positions than the activation sweep
        used (e.g. AllTokenSelector for steering vs PostInstructionTokenSelector
        for extraction). When omitted, prompt_fn_factory falls back to the one
        stored on this result; token_selector_factory falls back to
        AllTokenSelector().

        target_tokens_factory is a per-model factory because token IDs differ
        across tokenizers — keys (e.g. "true", "false") stay shared across
        models so a single metric_fn can read ModelResponses.target_logits
        without knowing which model produced them.
        """
        from probelab.train.token import AllTokenSelector

        prompt_fn_factory = prompt_fn_factory or self.prompt_fn_factory
        token_selector_factory = token_selector_factory or (lambda _h: AllTokenSelector())

        ablation_results: dict[str, LayerAblationResult] = {}
        n_models = len(self.results)

        for i, (model_id, sweep_result) in enumerate(self.results.items(), 1):
            prefix = f"[{i}/{n_models}] {model_id}"

            print(f"\n{prefix}  loading model...", flush=True)
            handle = self.handle_factory(model_id)
            print(f"{prefix}  model loaded", flush=True)

            backend = backend_factory(handle)
            token_selector = token_selector_factory(handle)
            prompt_fn = prompt_fn_factory(handle) if prompt_fn_factory is not None else None
            command_fn = command_fn_factory(handle) if command_fn_factory is not None else None
            target_tokens = target_tokens_factory(handle) if target_tokens_factory is not None else None

            print(f"{prefix}  ablating across {len(sweep_result.probes)} layer directions...", flush=True)
            ablation_results[model_id] = sweep_result.validate_by_ablation(
                dataset=dataset,
                backend=backend,
                metric_fn=metric_fn,
                token_selector=token_selector,
                hook_layers=hook_layers,
                component=component,
                scale=scale,
                mode=mode,
                objective=objective,
                include_baseline=include_baseline,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                prompt_fn=prompt_fn,
                command_fn=command_fn,
                target_tokens=target_tokens,
            )

            print(f"{prefix}  freeing GPU memory...", flush=True)
            # prompt_fn / command_fn / selector may capture handle.tokenizer,
            # so they must be deleted before handle for the model to free.
            del prompt_fn, command_fn, token_selector, backend, handle
            gc.collect()
            torch.cuda.empty_cache()
            print(f"{prefix}  done", flush=True)

        return ablation_results

    def plot(
        self,
        test_accs: "dict[str, dict[int, float]] | None" = None,
        title: str = "Probe Accuracy by Layer",
    ):
        from probelab.train.viz import plot_multi_model_sweep
        return plot_multi_model_sweep(self, test_accs=test_accs, title=title)


def multi_model_sweep(
    model_ids: list[str],
    handle_factory: Callable[[str], "ModelHandle"],
    collector_factory: Callable[["ModelHandle"], ActivationCollector],
    selector_factory: Callable[["ModelHandle"], TokenSelector],
    prompt_fn_factory: Callable[["ModelHandle"], Callable[[Example], str]] | None,
    reducer: TokenReducer,
    trainer: ProbeTrainer,
    train_probe_ds: ProbeDataset,
    dev_probe_ds: ProbeDataset,
    layers: LayerSelection = "all_transformer",
    batch_size: int = 16,
) -> MultiModelSweepResult:
    """
    Run sweep_layers() for each model serially, releasing GPU memory between models.

    Args:
        model_ids:         Models to sweep, in order.
        handle_factory:    Called with a model_id to load the model and return
                           a ModelHandle. The single source of truth for "how
                           do I load model X" — reused by evaluate() and
                           validate_by_ablation_per_model().
        collector_factory: Called with the live handle to produce an
                           ActivationCollector.
        selector_factory:  Called with the live handle to produce a
                           TokenSelector. Receives the handle so it can access
                           the tokenizer (e.g. for PostInstructionTokenSelector).
        prompt_fn_factory: Optional, called with the live handle to produce a
                           per-model prompt formatter. Use this when the prompt
                           depends on the model's tokenizer / chat template.
        reducer:           Shared across all models.
        trainer:           Shared across all models.
        train_probe_ds:    Training split (ProbeDataset).
        dev_probe_ds:      Dev split used to rank layers.
        layers:            Which layers to sweep per model.
        batch_size:        Examples per forward pass.

    Returns:
        MultiModelSweepResult keyed by model_id.
    """
    print(f"Sweeping {len(model_ids)} model(s):")

    for mid in model_ids:
        print(f"  {mid}")

    print(f"  layers     : {layers}")
    print(f"  batch_size : {batch_size}")
    print(f"  trainer    : {_obj_summary(trainer)}")
    print(f"  reducer    : {_obj_summary(reducer)}")
    print()

    results = {}

    n_models = len(model_ids)

    for i, model_id in enumerate(model_ids, 1):
        prefix = f"[{i}/{n_models}] {model_id}"

        print(f"\n{prefix}  loading model...", flush=True)
        handle = handle_factory(model_id)
        print(f"{prefix}  model loaded", flush=True)

        collector = collector_factory(handle)
        selector = selector_factory(handle)

        print(f"{prefix}  selector: {_obj_summary(selector)}")
        prompt_fn = prompt_fn_factory(handle) if prompt_fn_factory is not None else None

        print(f"{prefix}  collecting train activations...", flush=True)
        train_acts = collector.collect(train_probe_ds, batch_size, prompt_fn)
        print(f"{prefix}  collecting dev activations...", flush=True)
        dev_acts = collector.collect(dev_probe_ds, batch_size, prompt_fn)
        print(f"{prefix}  collection done", flush=True)

        results[model_id] = sweep_layers(
            train_acts, dev_acts, selector, reducer, trainer, layers
        )

        print(f"{prefix}  freeing GPU memory...", flush=True)
        # prompt_fn/selector may close over handle.tokenizer, so they must be
        # deleted before handle for the model weights to actually free.
        del prompt_fn, selector, train_acts, dev_acts, collector, handle
        gc.collect()
        torch.cuda.empty_cache()
        print(f"{prefix}  done", flush=True)

    return MultiModelSweepResult(
        results=results,
        handle_factory=handle_factory,
        collector_factory=collector_factory,
        prompt_fn_factory=prompt_fn_factory,
    )
