import torch

from dataclasses import dataclass, field
from typing import Callable

from tqdm.auto import tqdm

from probelab.dataset.base import ProbeDataset, Example
from train.activation import ActivationCollector, ActivationDataset
from train.probe import Probe, ProbeTrainer
from train.token import TokenSelector, TokenReducer, LayerSelection


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

        Args:
            test_dataset: ActivationDataset for the held-out test split.

        Returns:
            dict mapping layer index -> test accuracy.
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
        from train.viz import plot_layer_sweep
        return plot_layer_sweep(self, test_accs=test_accs, title=title)


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

    The selector and reducer define how token positions are collapsed into
    per-example vectors before probes are fit. The same pipeline is applied
    to both train and dev splits.

    Args:
        train_dataset: ActivationDataset for the training split.
        dev_dataset:   ActivationDataset for the dev (validation) split.
                       Used to rank layers and select the best probe.
                       Do not use the test split here.
        selector:      TokenSelector that identifies which positions to use.
        reducer:       TokenReducer that collapses selected positions into
                       (n, d_model) tensors ready for probe training.
        trainer:       ProbeTrainer that fits one probe per layer.
        layers:        Which layers to sweep. "all" uses every layer in
                       train_dataset. Can also be an int or list[int].

    Returns:
        LayerSweepResult with probes and accuracies for every swept layer,
        the best layer by dev accuracy, and the selector/reducer stored for
        use in evaluate().
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

    for layer, probe in tqdm(probes.items(), desc="Evaluating layers", unit="layer"):
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

    Call evaluate() to collect test activations for each model and compute
    held-out accuracy. The collector_factory stored here is reused so you
    don't need to pass it again.
    """

    results: dict[str, LayerSweepResult]
    collector_factory: Callable[[str], ActivationCollector] = field(repr=False)
    prompt_fn_factory: Callable[[ActivationCollector], Callable[[Example], str]] | None = field(default=None, repr=False)

    def evaluate(
        self,
        test_probe_ds: ProbeDataset,
        batch_size: int = 16,
    ) -> dict[str, dict[int, float]]:
        """
        Evaluate every model's layer probes on held-out test data.

        Re-instantiates each collector serially to avoid holding multiple
        models in memory simultaneously. Uses the same prompt_fn_factory
        that was passed to multi_model_sweep().

        Args:
            test_probe_ds: ProbeDataset for the held-out test split.
            batch_size:    Examples per forward pass.

        Returns:
            dict mapping model_id -> (dict mapping layer index -> test accuracy).
        """
        test_accs = {}

        for model_id, result in tqdm(self.results.items(), desc="Evaluating models", unit="model"):
            collector = self.collector_factory(model_id)
            prompt_fn = self.prompt_fn_factory(collector) if self.prompt_fn_factory is not None else None
            test_acts = collector.collect(test_probe_ds, batch_size, prompt_fn)
            test_accs[model_id] = result.evaluate(test_acts)

            del collector
            torch.cuda.empty_cache()

        return test_accs

    def plot(
        self,
        test_accs: "dict[str, dict[int, float]] | None" = None,
        title: str = "Probe Accuracy by Layer",
    ):
        from train.viz import plot_multi_model_sweep
        return plot_multi_model_sweep(self, test_accs=test_accs, title=title)


def multi_model_sweep(
    model_ids: list[str],
    collector_factory: Callable[[str], ActivationCollector],
    selector_factory: Callable[[ActivationCollector], TokenSelector],
    prompt_fn_factory: Callable[[ActivationCollector], Callable[[Example], str]] | None,
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
        model_ids:          Models to sweep, in order.
        collector_factory:  Called with a model_id to produce an ActivationCollector.
                            The collector loads the model; it is deleted after each
                            model's sweep to free GPU memory.
        selector_factory:   Called with the live collector to produce a TokenSelector.
                            Receives the collector so it can access the tokenizer or
                            other model-specific attributes (e.g. for
                            PostInstructionTokenSelector).
        reducer:            Shared across all models.
        trainer:            Shared across all models.
        train_probe_ds:     Training split (ProbeDataset).
        dev_probe_ds:       Dev split used to rank layers. Do not pass test here.
        layers:             Which layers to sweep per model.
        batch_size:         Examples per forward pass.
        prompt_fn_factory:  Optional factory called with the live collector to produce
                            a per-model prompt formatter. Use this when the prompt
                            depends on the model's tokenizer or chat template (e.g.
                            applying ChatFormatter with the model's own tokenizer).

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

    for model_id in tqdm(model_ids, desc="Sweeping models", unit="model"):
        print(f"\n[{model_id}]  loading model...", flush=True)
        collector = collector_factory(model_id)
        print(f"[{model_id}]  model loaded", flush=True)

        selector = selector_factory(collector)

        print(f"[{model_id}]  selector: {_obj_summary(selector)}")
        prompt_fn = prompt_fn_factory(collector) if prompt_fn_factory is not None else None

        print(f"[{model_id}]  collecting train activations...", flush=True)
        train_acts = collector.collect(train_probe_ds, batch_size, prompt_fn)
        print(f"[{model_id}]  collecting dev activations...", flush=True)
        dev_acts = collector.collect(dev_probe_ds, batch_size, prompt_fn)
        print(f"[{model_id}]  collection done", flush=True)

        results[model_id] = sweep_layers(
            train_acts, dev_acts, selector, reducer, trainer, layers
        )

        print(f"[{model_id}]  freeing GPU memory...", flush=True)
        del collector
        torch.cuda.empty_cache()
        print(f"[{model_id}]  done", flush=True)

    return MultiModelSweepResult(
        results=results,
        collector_factory=collector_factory,
        prompt_fn_factory=prompt_fn_factory,
    )
