# probelab

probelab is a python library designed to enable end-to-end experiments
for finding linear probes in open-weight transformer residual streams, with
support for both huggingface- and transformer_lens-based models.

## Install
```pip install probelab-py```

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

5. To measure causality, you can re-rank layers by causal effect via [validate_by_ablation](train/sweep.py). For each layer's probe direction, this runs a generation pass over a held-out behavioural set with that direction ablated at every transformer layer, scores the generations with a metric of your choice, and reports per-layer effect against a non-intervened baseline. You can pick the optimal layer by `best_delta()`, which gives the layer resulting in the largest delta of the chosen metric.

   The metric is just `metric_fn: ModelResponses -> float`, so anything you can compute from the (command, response) pairs works. For behaviours that need semantic understanding of the response, probelab ships an LLM-judge architecture split along two axes: the *property* being judged and the *backend* doing the judging. [SemanticJudge](evaluate/base.py) is the top-level ABC: `judge(command, response) -> 1 | 0 | None` (positive class / negative class / unclassifiable), with a default `judge_batch` that returns a [SemanticScore](evaluate/base.py) carrying `positive_rate`, `negative_rate`, and per-example breakdowns. [APIJudge](evaluate/api.py) is an abstract subclass for third-party-API providers; [ClaudeJudge](evaluate/api.py) is the concrete Anthropic implementation, parameterized over (`system_prompt`, `tool_name`, `tool_schema`, `output_field`) so the same provider class works for any binary property. [LocalJudge](evaluate/local.py) is the corresponding abstract base for self-hosted backends - HF, transformer_lens, vLLM, etc.

   [ClaudeRefusalJudge](evaluate/claude.py) is the worked refusal example: a thin `ClaudeJudge` subclass that bakes in a refusal-specific system prompt and a tool-use schema that forces a binary output (the system prompt also frames the judge as a safety-evaluation tool, which sidesteps the judge itself refusing on harmful inputs). To target a different concept - truthfulness a la Marks & Tegmark, code-vulnerability a la Yu et al., sycophancy, hallucination, jailbreak success, etc. - write a sibling subclass with the right prompt + tool schema and pass `metric_fn=lambda r: r.judge(my_judge).positive_rate` to `validate_by_ablation`. To swap providers, write the analogous subclass against an `OpenAIJudge` or a `LocalJudge` implementation. The probe direction `validate_by_ablation` surfaces is then the one that most causally moves the target behaviour.

6. Sweep interventions more broadly with [intervention_sweep](intervention/base.py). Once you have a probe direction you trust, you can scan over scales and modes (`add`, `subtract`, `ablate`) and over which layers to hook, using either an [HFInterventionBackend](intervention/huggingface.py) or a [TLInterventionBackend](intervention/transformer_lens.py).

Visualization utilities are also provided for viewing results of different hyperparameter
sweeps for probe training and interventions.
