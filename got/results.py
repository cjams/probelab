"""
Loaders for per-model run results written by `probelab.got.pipeline`.

A run directory contains:
    probes.pt              dict {layer: direction tensor}
    accs.json              {"train": {...}, "dev": {...}, "best_layer": int}
    ood_sp_en_trans.json   {layer: acc} (or any single-dataset OOD eval)
    causal.json            {"baseline", "per_layer", "best_layer",
                            "best_delta", "objective"}
    meta.json              model_id, seed, shots, splits, target_tokens

`load_run`/`load_runs` collapse those files into a typed `RunResults` so
plotting and analysis don't have to re-parse JSON every time.
"""

from __future__ import annotations

import json

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class RunResults:
    model_id: str

    # Probe-training side.
    train_accs:       dict[int, float]
    dev_accs:         dict[int, float]
    best_probe_layer: int

    # Out-of-distribution accuracies, per layer (e.g. sp_en_trans for cities-trained probes).
    ood_accs: dict[int, float]

    # Causal validation side.
    causal_baseline:    float
    causal_per_layer:   dict[int, float]
    best_causal_layer:  int
    best_causal_delta:  float
    causal_objective:   str  # "min" or "max"

    # Free-form metadata (model id, shots, splits, target_tokens, ...).
    meta: dict[str, Any] = field(default_factory=dict)

    # Lazy-loaded probe directions.
    probes_path: Path | None = None

    @property
    def n_layers(self) -> int:
        return len(self.dev_accs)

    def load_probes(self) -> dict[int, torch.Tensor]:
        """Load probe directions from disk. Tensors are CPU."""
        if self.probes_path is None:
            raise ValueError("probes_path is not set on this RunResults.")

        return torch.load(self.probes_path)


def _intkeys(d: dict[str, float]) -> dict[int, float]:
    return {int(k): v for k, v in d.items()}


def load_run(run_dir: Path | str) -> RunResults:
    """Load a single model's run directory into a RunResults."""
    run_dir = Path(run_dir)

    accs   = json.loads((run_dir / "accs.json").read_text())
    ood    = json.loads((run_dir / "ood_sp_en_trans.json").read_text())
    causal = json.loads((run_dir / "causal.json").read_text())
    meta   = json.loads((run_dir / "meta.json").read_text())

    return RunResults(
        model_id          = meta["model_id"],
        train_accs        = _intkeys(accs["train"]),
        dev_accs          = _intkeys(accs["dev"]),
        best_probe_layer  = accs["best_layer"],
        ood_accs          = _intkeys(ood),
        causal_baseline   = causal["baseline"],
        causal_per_layer  = _intkeys(causal["per_layer"]),
        best_causal_layer = causal["best_layer"],
        best_causal_delta = causal["best_delta"],
        causal_objective  = causal["objective"],
        meta              = meta,
        probes_path       = run_dir / "probes.pt",
    )


def load_runs(
    results_dir: Path | str,
    model_ids: list[str],
) -> dict[str, RunResults]:
    """Load multiple models, keyed by model_id. Resolves each model_id to
    `<results_dir>/<slug>` where slug = model_id.replace('/', '__')."""
    results_dir = Path(results_dir)
    out: dict[str, RunResults] = {}

    for mid in model_ids:
        slug = mid.replace("/", "__")
        out[mid] = load_run(results_dir / slug)

    return out
