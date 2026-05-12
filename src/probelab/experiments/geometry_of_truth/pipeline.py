"""
Single-model Geometry-of-Truth pipeline.

Runs end-to-end for one model and writes results to disk:
  - probes.pt            dict {layer: direction tensor}
  - accs.json            {"train": {layer: acc}, "dev": {layer: acc},
                          "best_layer": int}
  - ood_sp_en_trans.json {layer: acc}
  - causal.json          {"baseline": float, "per_layer": {layer: float},
                          "best_layer": int, "best_delta": float,
                          "objective": "max"}
  - meta.json            model_id, seed, shots, split sizes, target token ids

Designed to be invoked once per model in its own subprocess so the OS
reclaims all VRAM between runs (notebook references / accelerate hooks /
caching allocator can otherwise pin weights even after del + empty_cache).

Usage (preferred — works from any cwd):
    python -m probelab.experiments.geometry_of_truth.pipeline <model_id> --out <output_dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings

from pathlib import Path

# Silence two HF generation warnings that fire on every batch and bury the
# actually-useful log lines — both are cosmetic given our generation kwargs:
#   - "Both max_new_tokens and max_length seem to have been set." We pass
#     max_new_tokens=1; max_length comes from the model's stock generation
#     config and is overridden, exactly as the warning text itself confirms.
#   - "The following generation flags are not valid and may be ignored:
#     ['temperature', 'top_p']." We pass do_sample=False, so sampling flags
#     are unused.
warnings.filterwarnings("ignore", message=r".*max_length.*max_new_tokens.*")
warnings.filterwarnings("ignore", message=r".*generation flags are not valid.*")
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

import torch

from probelab.dataset.loaders import GeometryOfTruthLoader
from probelab.evaluate.generate import ModelResponses
from probelab.intervention.huggingface import HFInterventionBackend
from probelab.model import load_hf
from probelab.prompt import FewShotFormatter
from probelab.train.activation import ActivationSpec
from probelab.train.huggingface import HFActivationCollector
from probelab.train.probe import DifferenceOfMeansTrainer
from probelab.train.sweep import sweep_layers
from probelab.train.token import LastNTokenSelector, MeanReducer


SEED = 42

TRUE_FORMS  = ["TRUE", " TRUE", "True", " True", "true", " true"]
FALSE_FORMS = ["FALSE", " FALSE", "False", " False", "false", " false"]


def first_token_ids(tokenizer, surface_forms: list[str]) -> list[int]:
    """First token id of each surface form, deduped."""
    ids = []

    for s in surface_forms:
        # Multimodal models live behind a processor; fall back to .tokenizer.
        tok = getattr(tokenizer, "tokenizer", tokenizer)
        encoded = tok.encode(s, add_special_tokens=False)

        if encoded:
            ids.append(encoded[0])

    return sorted(set(ids))


def truth_logit_diff(r: ModelResponses) -> float:
    return r.logit_diff("true", "false").mean().item()


def build_splits():
    """Set aside one positive and one negative shot from cities (deterministic)
    and split the rest into train/dev/test. Identical across model invocations
    so probes from different models are trained on the same examples."""
    cities = GeometryOfTruthLoader("cities").load().balance(seed=SEED)

    pos_shots = cities.positives()[:1]
    neg_shots = cities.negatives()[:1]
    shots = pos_shots.join(neg_shots)
    shot_ids = {ex.id for ex in shots}

    rest = cities.filter(lambda ex: ex.id not in shot_ids)
    trn_ds, dev_ds = rest.split(left_ratio=0.8, seed=SEED)
    dev_ds, tst_ds = dev_ds.split(left_ratio=0.5, seed=SEED)

    return shots, trn_ds, dev_ds, tst_ds


def maybe_empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main(model_id: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[got_per_model] model_id = {model_id}", flush=True)
    print(f"[got_per_model] out_dir  = {out_dir}", flush=True)

    shots, trn_ds, dev_ds, tst_ds = build_splits()
    false_val_ds = dev_ds.negatives()
    sp_en = GeometryOfTruthLoader("sp_en_trans").load()

    print(f"[got_per_model] trn={len(trn_ds)}  dev={len(dev_ds)}  "
          f"tst={len(tst_ds)}  false_val={len(false_val_ds)}  "
          f"sp_en={len(sp_en)}", flush=True)

    formatter = FewShotFormatter(
        shots=shots,
        pos_completion="TRUE",
        neg_completion="FALSE",
        shot_template="{text} This statement is: {label}",
        query_template="{text} This statement is:",
        separator="\n",
    )

    print(f"[got_per_model] loading model...", flush=True)
    handle = load_hf(model_id)

    target_tokens = {
        "true":  first_token_ids(handle.tokenizer, TRUE_FORMS),
        "false": first_token_ids(handle.tokenizer, FALSE_FORMS),
    }
    print(f"[got_per_model] target_tokens = {target_tokens}", flush=True)

    spec = ActivationSpec(targets="all_transformer", component="resid_post")
    collector = HFActivationCollector(handle, spec)

    print(f"[got_per_model] collecting train activations...", flush=True)
    train_acts = collector.collect(trn_ds, batch_size=16, prompt_fn=formatter.format)

    print(f"[got_per_model] collecting dev activations...", flush=True)
    dev_acts = collector.collect(dev_ds, batch_size=16, prompt_fn=formatter.format)

    print(f"[got_per_model] training probes...", flush=True)
    sweep = sweep_layers(
        train_dataset=train_acts,
        dev_dataset=dev_acts,
        selector=LastNTokenSelector(n=1),
        reducer=MeanReducer(),
        trainer=DifferenceOfMeansTrainer(),
        layers="all_transformer",
    )

    print(f"[got_per_model] OOD eval on sp_en_trans...", flush=True)
    sp_en_acts = collector.collect(sp_en, batch_size=16, prompt_fn=formatter.format)
    ood_accs = sweep.evaluate(sp_en_acts)

    # Free activation tensors before the generation pass.
    del train_acts, dev_acts, sp_en_acts
    maybe_empty_cache()

    print(f"[got_per_model] causal validation (steering on truth direction)...", flush=True)
    backend = HFInterventionBackend(handle)

    causal = sweep.validate_by_ablation(
        dataset=false_val_ds,
        backend=backend,
        metric_fn=truth_logit_diff,
        prompt_fn=formatter.format,
        hook_layers="all_transformer",
        scale=2.0,
        mode="add",
        objective="max",
        include_baseline=True,
        batch_size=4,
        max_new_tokens=1,
        target_tokens=target_tokens,
    )

    print(f"[got_per_model] saving results to {out_dir}...", flush=True)

    torch.save(
        {layer: probe.direction.cpu() for layer, probe in sweep.probes.items()},
        out_dir / "probes.pt",
    )

    (out_dir / "accs.json").write_text(json.dumps({
        "train":      sweep.train_accs,
        "dev":        sweep.dev_accs,
        "best_layer": sweep.best_layer,
    }, indent=2))

    (out_dir / "ood_sp_en_trans.json").write_text(json.dumps(ood_accs, indent=2))

    (out_dir / "causal.json").write_text(json.dumps({
        "baseline":   causal.baseline_metric,
        "per_layer":  causal.per_layer_metric,
        "best_layer": causal.best_layer,
        "best_delta": causal.best_delta(),
        "objective":  causal.objective,
    }, indent=2))

    (out_dir / "meta.json").write_text(json.dumps({
        "model_id":      model_id,
        "seed":          SEED,
        "shots":         [{"text": ex.text, "label": ex.label} for ex in shots],
        "splits":        {
            "trn": len(trn_ds), "dev": len(dev_ds), "tst": len(tst_ds),
            "false_val": len(false_val_ds), "sp_en": len(sp_en),
        },
        "target_tokens": target_tokens,
    }, indent=2))

    print(f"[got_per_model] done.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", help="HF model id, e.g. meta-llama/Llama-2-13b-hf")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    main(args.model_id, Path(args.out))
