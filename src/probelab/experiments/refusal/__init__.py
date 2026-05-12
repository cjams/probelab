"""
Refusal-direction probing pipeline (after Arditi et al. 2024).

Components:
  pipeline.py  — single-model end-to-end script. Invoke as
                 `python -m probelab.experiments.refusal.pipeline <model_id> --out <dir>`.
                 Loads the model once, collects activations once, then iterates
                 over a small `CONFIGS` list of (selector, reducer) hyperparameter
                 combinations, training probes and validating each config
                 causally via Claude refusal-rate ablation.
  results.py   — typed loader (`RunResults`, `ConfigResults`, `load_run`,
                 `load_runs`) for the per-model output directories.
  viz.py       — within-model and cross-model visualizations comparing configs.

The dataset loaders (HarmBench, AdvBench, Alpaca, etc.) live in
`probelab.dataset.loaders`.
"""

from probelab.experiments.refusal.results import (
    ConfigResults,
    RunResults,
    load_run,
    load_runs,
)
from probelab.experiments.refusal.viz import (
    plot_refusal_by_config,
    plot_probe_accuracy_by_config,
    plot_predictive_vs_causal_by_config,
    plot_config_probe_similarity,
    plot_refusal_across_models,
    plot_best_config_per_model,
)

__all__ = [
    "ConfigResults",
    "RunResults",
    "load_run",
    "load_runs",
    "plot_refusal_by_config",
    "plot_probe_accuracy_by_config",
    "plot_predictive_vs_causal_by_config",
    "plot_config_probe_similarity",
    "plot_refusal_across_models",
    "plot_best_config_per_model",
]
