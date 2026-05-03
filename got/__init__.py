"""
Geometry-of-Truth pipeline (after Marks & Tegmark, 2023).

Components:
  pipeline.py — single-model end-to-end script. Invoke as
                `python -m probelab.got.pipeline <model_id> --out <dir>`.
                Designed to run once per model in a fresh subprocess so the
                OS reclaims VRAM between runs.
  results.py  — typed loader (`RunResults`, `load_run`, `load_runs`) for
                the per-model output directories the script writes.
  viz.py      — cross-run Plotly visualizations over `RunResults`.

The dataset loader for the M&T CSVs lives in
`probelab.dataset.loaders.geometry_of_truth` (alongside the other dataset
loaders), since dataset ingestion is a generic concern, not GoT-specific.
"""

from probelab.got.results import RunResults, load_run, load_runs
from probelab.got.viz import (
    plot_predictive_vs_causal,
    plot_probe_direction_stability,
    plot_probe_direction_vs_best_causal,
    plot_best_layers,
    plot_overfitting,
)

__all__ = [
    "RunResults",
    "load_run",
    "load_runs",
    "plot_predictive_vs_causal",
    "plot_probe_direction_stability",
    "plot_probe_direction_vs_best_causal",
    "plot_best_layers",
    "plot_overfitting",
]
