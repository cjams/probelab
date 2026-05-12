"""
Single-model refusal-direction pipeline (after Arditi et al. 2024).

For one model, collects residual activations once over the train/dev splits,
then iterates over a small set of (selector, reducer) configurations and for
each config:
  - trains DIM probes per layer over the *same* collected activations
  - validates causally by ablating each probe direction and measuring the
    refusal rate of the resulting generations on a held-out harmful set
    (Claude judge)
  - persists per-config results so the notebook can compare across configs.

The point of varying configs over a shared activation pass is to demonstrate
how much the *post-hoc* hyperparameters (where you read in the prompt) shape
the resulting probe — at near-zero additional cost, since the model forward
pass is the expensive part.

Output layout under <out>/:
  meta.json                    model_id, dtype, splits, configs
  <config_name>/
    probes.pt                  dict {layer: direction tensor (cpu)}
    accs.json                  {"train": {...}, "dev": {...}, "best_layer": int}
    causal.json                {"baseline": float, "per_layer": {...},
                                "best_layer": int, "best_delta": float,
                                "objective": "min"}

Usage (preferred — works from any cwd):
    python -m probelab.experiments.refusal.pipeline <model_id> --out <output_dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Cosmetic warnings that fire once per generation batch and bury the actual
# log lines. The behaviour they describe is exactly what we want
# (max_new_tokens overrides max_length, do_sample=False ignores temp/top_p).
warnings.filterwarnings("ignore", message=r".*max_length.*max_new_tokens.*")
warnings.filterwarnings("ignore", message=r".*generation flags are not valid.*")
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

import torch

from probelab.dataset.loaders import (
    HarmBenchLoader, AdvBenchLoader, AlpacaLoader,
    MaliciousInstructLoader, TDC2023Loader,
)
from probelab.evaluate.claude import ClaudeRefusalJudge
from probelab.evaluate.generate import ModelResponses
from probelab.intervention.huggingface import HFInterventionBackend
from probelab.model import load_hf, HFModelHandle
from probelab.prompt import ChatFormatter
from probelab.train.activation import ActivationSpec
from probelab.train.huggingface import HFActivationCollector
from probelab.train.probe import DifferenceOfMeansTrainer
from probelab.train.sweep import sweep_layers
from probelab.train.token import (
    AllTokenSelector, LastNTokenSelector, MeanReducer,
    PostInstructionTokenSelector, TokenReducer, TokenSelector,
)


SEED = 42


# ---------------------------------------------------------------------------
# Config — the set of (selector, reducer) hyperparameter combinations to try.
#
# Each config shares the activation collection (one forward pass per split,
# all transformer layers, resid_post). The selector decides which positions
# of the collected activations the probe trains on; the reducer collapses
# multi-position selections into a single vector per example.
#
# Adding a config: append to CONFIGS. The selector factory takes a tokenizer
# because PostInstructionTokenSelector needs the tokenizer's chat template
# to locate the user-content boundary; fixed-position selectors ignore it.
# ---------------------------------------------------------------------------

@dataclass
class ProbeConfig:
    name: str
    selector_factory: Callable[[Any], TokenSelector]
    reducer_factory:  Callable[[],   TokenReducer]
    description:      str = ""


CONFIGS: list[ProbeConfig] = [
    ProbeConfig(
        name="arditi_last_token",
        selector_factory=lambda _tok: LastNTokenSelector(n=1),
        reducer_factory=MeanReducer,
        description=(
            "Arditi's canonical extraction: the single last token of the "
            "chat-formatted prompt (the assistant-header opener)."
        ),
    ),
    ProbeConfig(
        name="last_5_mean",
        selector_factory=lambda _tok: LastNTokenSelector(n=5),
        reducer_factory=MeanReducer,
        description=(
            "Mean over the last 5 tokens. Captures the assistant-turn "
            "scaffolding plus the final tokens of user content."
        ),
    ),
    ProbeConfig(
        name="post_instruction_mean",
        selector_factory=lambda tok: PostInstructionTokenSelector(tok),
        reducer_factory=MeanReducer,
        description=(
            "Mean over every token after the user-content boundary "
            "(closing user-turn tokens + assistant-turn opener)."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Splits — mirrors the existing probe.ipynb composition: HarmBench + AdvBench
# + TDC2023 + MaliciousInstruct as harmful, Alpaca as benign, balanced.
# Held-out harmful-only set used for causal validation generation.
# ---------------------------------------------------------------------------

def build_splits():
    print("[refusal] loading datasets...", flush=True)

    harmbench = HarmBenchLoader(subset="standard").load()
    advbench  = AdvBenchLoader().load()
    tdc       = TDC2023Loader().load()
    malicious = MaliciousInstructLoader().load()
    alpaca    = AlpacaLoader().load()

    ds = harmbench.join(advbench).join(tdc).join(malicious).join(alpaca)
    ds = ds.balance(seed=SEED)

    trn_ds, rest = ds.split(left_ratio=0.8, seed=SEED)
    dev_ds, tst_ds = rest.split(left_ratio=0.5, seed=SEED)

    # Harmful-only held-out validation set for causal ablation (mirrors
    # Arditi: ablate the refusal direction during generation on harmful
    # prompts, judge whether refusal rate drops).
    harmful_val_ds = dev_ds.positives()

    return trn_ds, dev_ds, tst_ds, harmful_val_ds


def maybe_empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_config_against_handle(
    config:           ProbeConfig,
    handle:           HFModelHandle,
    formatter:        ChatFormatter,
    train_acts,
    dev_acts,
    harmful_val_ds,
    judge:            ClaudeRefusalJudge,
    out_dir:          Path,
    *,
    intervention_batch_size: int,
    max_new_tokens:          int,
) -> None:
    """Train + causally validate one config against already-collected activations."""

    cfg_dir = out_dir / config.name
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # Per-config resume: if a previous invocation completed this config,
    # don't redo the (expensive) generation pass over harmful_val_ds.
    if (cfg_dir / "causal.json").exists():
        print(f"\n[refusal][{config.name}] already complete at {cfg_dir}, skipping",
              flush=True)
        return

    print(f"\n[refusal][{config.name}] {config.description}", flush=True)
    print(f"[refusal][{config.name}] training probes...", flush=True)

    selector = config.selector_factory(handle.tokenizer)
    reducer  = config.reducer_factory()

    sweep = sweep_layers(
        train_dataset=train_acts,
        dev_dataset=dev_acts,
        selector=selector,
        reducer=reducer,
        trainer=DifferenceOfMeansTrainer(),
        layers="all_transformer",
    )

    print(f"[refusal][{config.name}] best by probe accuracy: layer "
          f"{sweep.best_layer} (dev={sweep.dev_accs[sweep.best_layer]:.2%})",
          flush=True)

    # Causal validation: ablate each layer's direction during generation,
    # judge refusal rate, pick the layer that minimises it.
    backend = HFInterventionBackend(handle)

    def metric_fn(r: ModelResponses) -> float:
        return r.judge(judge).positive_rate

    print(f"[refusal][{config.name}] causal validation by ablation...", flush=True)
    causal = sweep.validate_by_ablation(
        dataset=harmful_val_ds,
        backend=backend,
        metric_fn=metric_fn,
        token_selector=AllTokenSelector(),
        prompt_fn=formatter.format,
        command_fn=formatter.user_content,
        hook_layers="all_transformer",
        scale=1.0,
        mode="ablate",
        objective="min",
        include_baseline=True,
        batch_size=intervention_batch_size,
        max_new_tokens=max_new_tokens,
    )

    # Persist.
    torch.save(
        {layer: probe.direction.cpu() for layer, probe in sweep.probes.items()},
        cfg_dir / "probes.pt",
    )

    (cfg_dir / "accs.json").write_text(json.dumps({
        "train":      sweep.train_accs,
        "dev":        sweep.dev_accs,
        "best_layer": sweep.best_layer,
    }, indent=2))

    (cfg_dir / "causal.json").write_text(json.dumps({
        "baseline":   causal.baseline_metric,
        "per_layer":  causal.per_layer_metric,
        "best_layer": causal.best_layer,
        "best_delta": causal.best_delta(),
        "objective":  causal.objective,
    }, indent=2))

    print(f"[refusal][{config.name}] best by causal effect: layer "
          f"{causal.best_layer} (refusal: {causal.baseline_metric:.2%} -> "
          f"{causal.per_layer_metric[causal.best_layer]:.2%})", flush=True)


def main(
    model_id:                str,
    out_dir:                 Path,
    *,
    dtype:                   str = "auto",
    activation_batch_size:   int = 16,
    intervention_batch_size: int = 4,
    max_new_tokens:          int = 256,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[refusal] model_id = {model_id}", flush=True)
    print(f"[refusal] out_dir  = {out_dir}", flush=True)
    print(f"[refusal] dtype    = {dtype}", flush=True)
    print(f"[refusal] configs  = {[c.name for c in CONFIGS]}", flush=True)

    trn_ds, dev_ds, tst_ds, harmful_val_ds = build_splits()
    print(f"[refusal] trn={len(trn_ds)}  dev={len(dev_ds)}  tst={len(tst_ds)}  "
          f"harmful_val={len(harmful_val_ds)}", flush=True)

    print(f"[refusal] loading model...", flush=True)
    handle = load_hf(model_id, dtype=dtype)

    formatter = ChatFormatter(handle.tokenizer, instructionify=True, system_prompt=None)

    spec = ActivationSpec(targets="all_transformer", component="resid_post")
    collector = HFActivationCollector(handle, spec)

    print(f"[refusal] collecting train activations...", flush=True)
    train_acts = collector.collect(trn_ds, batch_size=activation_batch_size, prompt_fn=formatter.format)

    print(f"[refusal] collecting dev activations...", flush=True)
    dev_acts = collector.collect(dev_ds, batch_size=activation_batch_size, prompt_fn=formatter.format)

    judge = ClaudeRefusalJudge()  # uses ANTHROPIC_API_KEY env var

    for config in CONFIGS:
        run_config_against_handle(
            config=config,
            handle=handle,
            formatter=formatter,
            train_acts=train_acts,
            dev_acts=dev_acts,
            harmful_val_ds=harmful_val_ds,
            judge=judge,
            out_dir=out_dir,
            intervention_batch_size=intervention_batch_size,
            max_new_tokens=max_new_tokens,
        )

        # Tidy up between configs in case anything was holding GPU memory.
        maybe_empty_cache()

    (out_dir / "meta.json").write_text(json.dumps({
        "model_id": model_id,
        "dtype":    dtype,
        "seed":     SEED,
        "splits":   {
            "trn":           len(trn_ds),
            "dev":           len(dev_ds),
            "tst":           len(tst_ds),
            "harmful_val":   len(harmful_val_ds),
        },
        "configs":  [
            {"name": c.name, "description": c.description}
            for c in CONFIGS
        ],
    }, indent=2))

    print(f"\n[refusal] done.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", help="HF model id, e.g. meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--dtype", default="auto",
                        help='Model dtype. "auto" reads from checkpoint config '
                             '(needed for FP8 quantised checkpoints). Override '
                             'with e.g. "bfloat16" to force.')
    parser.add_argument("--activation-batch-size",   type=int, default=16)
    parser.add_argument("--intervention-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens",          type=int, default=256)
    args = parser.parse_args()

    main(
        model_id                = args.model_id,
        out_dir                 = Path(args.out),
        dtype                   = args.dtype,
        activation_batch_size   = args.activation_batch_size,
        intervention_batch_size = args.intervention_batch_size,
        max_new_tokens          = args.max_new_tokens,
    )
