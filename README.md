# probelab

probelab is a python library designed to enable end-to-end experiments
for finding linear probes in open-weight transformer residual streams, with
support for both huggingface- and transformer_lens-based models.

## Probe Flow
Finding linear probes consists of several high-level steps:

0. Figure out the concept of interest, e.g., "refusal" a la [Arditi et al.](https://arxiv.org/abs/2406.11717), "truth" a la [Marks & Tegmark](https://arxiv.org/abs/2310.06824), or "code vulnerability" a la [Yu et al.](https://arxiv.org/html/2507.09508v1). Picking a concept is up to you and your taste.

1. Construct a dataset of token sequences designed to elicit the concept of interest
and its antithesis from the model's activations. This repo has some refusal and
truth related datasets as used from the papers referenced above. Datasets for
concepts other than those would need to be provided by you. The library provides [several helper classes](dataset/loaders) for importing a raw dataset into a normalized [ProbeDataset](dataset/base.py) class. The ProbeDataset is consumed by downstream probe training, evaluation, and evaluation classes.

2. Load the model you want to probe. probelab supports both huggingface (via [load_hf](model.py)) and transformer_lens (via [load_tl](model.py)) backends behind a common `ModelHandle`.

3. Experimentally decide *where* in the model to read activations from, and *which tokens* to read at. The "where" is an [ActivationSpec](train/activation.py) - a pair of (target layers, residual component). `targets="all_transformer"` collects every transformer layer; `component="resid_post"` reads the residual stream at the end of the layer. The "which" is a [TokenSelector](train/token.py) - `LastNTokenSelector(n=1)` reads the last N tokens; `PostInstructionTokenSelector` reads all tokens after user commands of a chat-formatted prompt; `AllTokenSelector` reads the whole sequence. A `TokenReducer` then collapses the selected tokens into a single vector per example.

   You can use a [ChatFormatter](prompt.py) to apply the model's chat template, optionally wrapping raw `activity` examples into instruction form (with `instructionify=True`). Then run an [HFActivationCollector](train/huggingface.py) (or the transformer_lens equivalent) over your train/dev splits to produce an [ActivationDataset](train/activation.py). Other formatters are available such as
   for few-shot prompts (used in work like Mixture of Corrections and Geometry of Truth).

4. Train one probe per layer with [sweep_layers](train/sweep.py). Pass a `ProbeTrainer` e.g., [DifferenceOfMeansTrainer](train/probe.py), or a logistic regression trainer if you want a learned classifier, the selector and reducer from step 3, and the train/dev `ActivationDataset`s. The result is a [LayerSweepResult](train/sweep.py) holding every trained probe keyed by layer, plus train and dev accuracies. The "best" layer here is best by probe accuracy on the dev set - which is necessary but not sufficient for a probe direction that *causally* matters like refusal.

5. To measure causality, you can re-rank layers by causal effect via [validate_by_ablation](train/sweep.py). For each layer's probe direction, this runs a generation pass over a held-out behavioural set with that direction ablated at every transformer layer, scores the generations with a metric of your choice (e.g., refusal rate via [ClaudeRefusalJudge](evaluate/claude.py)), and reports per-layer effect against a non-intervened baseline. You can pick the optimal layer by `best_delta()`, which gives the layer resulting in the largest delta of the chosen metric.

6. Sweep interventions more broadly with [intervention_sweep](intervention/base.py). Once you have a probe direction you trust, you can scan over scales and modes (`add`, `subtract`, `ablate`) and over which layers to hook, using either an [HFInterventionBackend](intervention/huggingface.py) or a [TLInterventionBackend](intervention/transformer_lens.py).

Visualization utilities are also provided for viewing results of different hyperparameter
sweeps for probe training and interventions.