"""
Loaders for per-model refusal-pipeline results.

A run directory contains:
    meta.json                model_id, dtype, splits, configs
    <config_name>/
        probes.pt            dict {layer: direction tensor}
        accs.json            {"train": {...}, "dev": {...}, "best_layer": int}
        causal.json          {"baseline", "per_layer", "best_layer",
                              "best_delta", "objective"}

Multiple `(selector, reducer)` configurations are evaluated against the same
collected activations per model; this loader collapses each into a typed
`ConfigResults`, with all configs grouped under a single `RunResults`.
"""

from __future__ import annotations

import json

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class ConfigResults:
    """Per-(model, selector-config) results."""
    name:               str

    train_accs:         dict[int, float]
    dev_accs:           dict[int, float]
    best_probe_layer:   int

    causal_baseline:    float
    causal_per_layer:   dict[int, float]
    best_causal_layer:  int
    best_causal_delta:  float
    causal_objective:   str  # always "min" for refusal-rate ablation

    probes_path: Path | None = None

    @property
    def n_layers(self) -> int:
        return len(self.dev_accs)

    def load_probes(self) -> dict[int, torch.Tensor]:
        if self.probes_path is None:
            raise ValueError("probes_path is not set on this ConfigResults.")

        return torch.load(self.probes_path)


@dataclass
class RunResults:
    """All-configs results for one model."""
    model_id: str
    configs:  dict[str, ConfigResults]  # keyed by config name
    meta:     dict[str, Any] = field(default_factory=dict)

    @property
    def n_layers(self) -> int:
        # All configs share the model, so layer count is the same.
        return next(iter(self.configs.values())).n_layers

    def by_layer(self, attr: str) -> dict[str, dict[int, float]]:
        """Quick accessor: {config_name: {layer: value}} for any per-layer
        attribute (e.g. "dev_accs", "causal_per_layer")."""
        return {name: getattr(c, attr) for name, c in self.configs.items()}


def _intkeys(d: dict[str, float]) -> dict[int, float]:
    return {int(k): v for k, v in d.items()}


def _load_config(cfg_dir: Path, name: str) -> ConfigResults:
    accs   = json.loads((cfg_dir / "accs.json").read_text())
    causal = json.loads((cfg_dir / "causal.json").read_text())

    return ConfigResults(
        name              = name,
        train_accs        = _intkeys(accs["train"]),
        dev_accs          = _intkeys(accs["dev"]),
        best_probe_layer  = accs["best_layer"],
        causal_baseline   = causal["baseline"],
        causal_per_layer  = _intkeys(causal["per_layer"]),
        best_causal_layer = causal["best_layer"],
        best_causal_delta = causal["best_delta"],
        causal_objective  = causal["objective"],
        probes_path       = cfg_dir / "probes.pt",
    )


def load_run(run_dir: Path | str) -> RunResults:
    """Load a single model's run directory into a RunResults.

    Discovers configs by listing subdirectories that contain a causal.json
    (so a partial run that crashed mid-config still loads cleanly with
    whatever configs completed)."""
    run_dir = Path(run_dir)

    meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").exists() else {}

    configs: dict[str, ConfigResults] = {}

    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue

        if not (child / "causal.json").exists():
            continue

        configs[child.name] = _load_config(child, child.name)

    if not configs:
        raise FileNotFoundError(
            f"No completed configs found under {run_dir}. Expected at least "
            f"one subdirectory containing causal.json."
        )

    return RunResults(
        model_id = meta.get("model_id", run_dir.name),
        configs  = configs,
        meta     = meta,
    )


def load_runs(
    results_dir: Path | str,
    model_ids: list[str],
) -> dict[str, RunResults]:
    """Load multiple models' run directories, keyed by model_id."""
    results_dir = Path(results_dir)
    out: dict[str, RunResults] = {}

    for mid in model_ids:
        slug = mid.replace("/", "__")
        out[mid] = load_run(results_dir / slug)

    return out
