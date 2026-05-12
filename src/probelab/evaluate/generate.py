from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable

import torch

from probelab.dataset.base import Example, ProbeDataset
from probelab.model import HFModelHandle, TLModelHandle, underlying_tokenizer

from .base import SemanticJudge, SemanticScore


def tl_generate_left_padded(
    model,                                 # HookedTransformer
    tokens: torch.Tensor,                  # (batch, prompt_len) left-padded input ids
    attention_mask: torch.Tensor,          # (batch, prompt_len) 1 for real tokens
    max_new_tokens: int,
    eos_token_id: int | None,
    pad_token_id: int,
    fwd_hooks: list | None = None,
) -> torch.Tensor:
    """Greedy generate with proper left-padding attention masking.

    HookedTransformer.generate does not feed an explicit attention_mask to
    forward during the prefill step — it calls forward(residual, ...) with
    start_at_layer=0, which skips input_to_embed and therefore skips the
    automatic attention_mask inference. This causes attention to attend to
    pad tokens for left-padded batches, silently polluting activations.

    This helper works around it by running prefill as a plain forward pass
    with both attention_mask and past_kv_cache, then advancing token-by-token
    using the cached attention_mask (append_attention_mask keeps it extended).

    fwd_hooks is a list of (hook_name, hook_fn) pairs installed for the full
    generation (prefill + autoregressive) via model.hooks.
    """
    from transformer_lens import HookedTransformerKeyValueCache

    batch_size = tokens.shape[0]
    device = model.cfg.device

    tokens = tokens.to(device)
    attention_mask = attention_mask.to(device)

    past_kv_cache = HookedTransformerKeyValueCache.init_cache(
        model.cfg, device, batch_size
    )

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    generated: list[torch.Tensor] = []

    hooks_ctx = model.hooks(fwd_hooks=fwd_hooks) if fwd_hooks else nullcontext()

    with hooks_ctx, torch.no_grad():
        # Prefill: full prompt with attention_mask. This populates past_kv_cache
        # and registers the mask inside it so later steps stay left-padding-aware.
        logits = model.forward(
            tokens,
            attention_mask=attention_mask,
            past_kv_cache=past_kv_cache,
            return_type="logits",
        )
        
        next_token = logits[:, -1, :].argmax(dim=-1)
        generated.append(next_token)

        if eos_token_id is not None:
            finished.logical_or_(next_token == eos_token_id)

        for _ in range(max_new_tokens - 1):
            if eos_token_id is not None and finished.all():
                break

            # A single-token step extends the cache; append_attention_mask
            # inside input_to_embed tacks a "real" position onto the cached
            # mask, preserving left-padding semantics.
            next_input = next_token.unsqueeze(-1)
            new_mask = torch.ones_like(next_input, dtype=attention_mask.dtype)

            logits = model.forward(
                next_input,
                attention_mask=new_mask,
                past_kv_cache=past_kv_cache,
                return_type="logits",
            )

            next_token = logits[:, -1, :].argmax(dim=-1)

            # For sequences already stopped, pad with pad_token so shapes stay
            # uniform and the decode step strips them cleanly.
            if eos_token_id is not None:
                next_token = torch.where(
                    finished,
                    torch.full_like(next_token, pad_token_id),
                    next_token,
                )
                
                finished.logical_or_(next_token == eos_token_id)

            generated.append(next_token)

    new_tokens = torch.stack(generated, dim=1)
    return torch.cat([tokens, new_tokens], dim=1)


@dataclass
class ModelResponses:
    """Generated responses paired with the commands that produced them."""
    # Raw command text from each Example (ex.text), used as judge input.
    commands: list[str]

    # Decoded model-generated text (new tokens only, prompt stripped).
    responses: list[str]

    # Per-example logits at the first generated position, keyed by the user
    # label provided via `target_tokens` to the collector (e.g. "true",
    # "false"). Each tensor is 1D, on CPU, shape (n_examples,). When the
    # user passes multiple token IDs per key (e.g. {"True", " True",
    # "true", " true"}), the values are aggregated via logsumexp so the
    # tensor reads as a single per-example log-score for that class.
    #
    # Use this for logit-difference metrics in the style of Marks & Tegmark,
    # where the causal validation signal is a logit gap between two classes
    # at the position immediately after the prompt rather than a property
    # of the decoded text. The partition function Z cancels in the difference,
    # so (target_logits["true"] - target_logits["false"]) is exactly the
    # log-odds the model assigns to true-vs-false at that position.
    target_logits: dict[str, torch.Tensor] | None = None

    def __len__(self) -> int:
        return len(self.commands)

    def judge(
        self,
        judge: SemanticJudge,
        *,
        samples_per_call: int = 10,
        max_concurrency: int = 8,
    ) -> SemanticScore:
        """Convenience method to score these responses with a SemanticJudge.

        Forwards `samples_per_call` and `max_concurrency` to `judge_batch`;
        defaults give k=10 samples per API call across 8 concurrent threads
        (~80 samples in flight at any moment). Tune for your provider's rate
        limits.
        """
        return judge.judge_batch(
            commands=self.commands,
            responses=self.responses,
            samples_per_call=samples_per_call,
            max_concurrency=max_concurrency,
        )

    def logit_diff(self, pos_key: str, neg_key: str) -> torch.Tensor:
        """Per-example (target_logits[pos_key] - target_logits[neg_key]).

        Requires target_tokens to have been passed to the collector with
        both keys populated. Returns a 1D CPU tensor of shape (n_examples,).
        """
        if self.target_logits is None:
            raise ValueError(
                "target_logits is None — pass target_tokens to the collector "
                "to populate it."
            )

        for k in (pos_key, neg_key):
            if k not in self.target_logits:
                raise KeyError(
                    f"target_logits has no key {k!r}; available keys: "
                    f"{list(self.target_logits.keys())}"
                )

        return self.target_logits[pos_key] - self.target_logits[neg_key]


# Type alias for the target_tokens parameter accepted by collectors and
# intervention backends. A single int captures one surface form; a list of
# ints captures a class made up of several tokenizations of the same concept
# (e.g. {"True", " True", "true", " true"}) and is aggregated via logsumexp
# at the collector so callers see a single per-example log-score.
TargetTokens = dict[str, "int | list[int]"]


def _aggregate_target_logits(
    first_scores: torch.Tensor,                          # (batch, vocab)
    target_tokens: TargetTokens,
) -> dict[str, torch.Tensor]:
    """Reduce raw first-position logits down to one value per target class.

    A single token id passes through as the bare logit. A list of ids is
    combined via logsumexp on the float-cast logits (bf16 logsumexp can lose
    precision when summands span many orders of magnitude).
    """
    out: dict[str, torch.Tensor] = {}

    for key, ids in target_tokens.items():
        if isinstance(ids, int):
            out[key] = first_scores[:, ids].float().cpu()
        else:
            ids_list = list(ids)
            selected = first_scores[:, ids_list].float()    # (batch, n_ids)
            out[key] = torch.logsumexp(selected, dim=-1).cpu()

    return out


class ResponseCollector(ABC):
    """
    Abstract response collector — runs a backend-specific generation loop over
    a ProbeDataset and returns a backend-independent ModelResponses.

    # Consistency with activation collection
    When using a response collector together with an ActivationCollector (e.g.
    collecting activations and then measuring refusal rate), three things
    must match exactly or the responses will not correspond to the activation
    inputs:

    1. model — pass the same ModelHandle to both. A different checkpoint or
       quantisation level changes both the activations and the generated text.

    2. prompt_fn — pass the identical callable to both collect() calls. The
       prompt_fn encodes instructionify transforms, the chat template, system
       prompt, and few-shot shots. Any mismatch silently changes the input
       distribution.

    3. add_generation_prompt — for generation this must be True so the model
       receives the assistant-turn opener and knows to produce a response.
       Activation collection may have been run with it False (e.g. collecting
       over a labelled completion). Use a separate ChatFormatter instance
       constructed with add_generation_prompt=True for this collector.

    The handle's tokenizer is expected to be configured for left-padding so
    batched generation terminates cleanly. load_hf / load_tl set
    this by default.
    """

    @abstractmethod
    def collect(
        self,
        dataset: ProbeDataset,
        batch_size: int = 8,
        prompt_fn: Callable[[Example], str] | None = None,
        command_fn: Callable[[Example], str] | None = None,
        max_new_tokens: int = 256,
        target_tokens: TargetTokens | None = None,
    ) -> ModelResponses:
        """Generate responses for every example in `dataset`.

        Args:
            dataset:        Source ProbeDataset. Must be the same split used
                            for activation collection — order and contents
                            determine which example_ids map to which responses.
            batch_size:     Examples per generation batch.
            prompt_fn:      Formats each Example into a model prompt string.
                            Must be constructed with add_generation_prompt=True
                            and identical to the prompt_fn passed to the
                            matching ActivationCollector.
            command_fn:     Extracts the user-facing instruction text to store
                            as ModelResponses.commands — what the judge sees.
                            Defaults to ex.text.
            max_new_tokens: Maximum tokens to generate per example.
            target_tokens:  Optional map from user label -> token id (or list
                            of token ids). When set, ModelResponses.target_logits
                            is populated with first-generated-position logits
                            for each label, aggregated via logsumexp when a
                            list is provided. Useful for logit-difference
                            metrics that don't depend on decoded text.
        """
        ...


class HFResponseCollector(ResponseCollector):
    """Generates text responses from a HuggingFace causal LM over a ProbeDataset.

    Uses left-padding so batched generation terminates cleanly: all prompts end
    at the same position and new tokens are sliced off uniformly.

    Args:
        handle: Loaded model + tokenizer + model_id.
    """

    def __init__(self, handle: HFModelHandle) -> None:
        self.model = handle.model
        self.tokenizer = handle.tokenizer
        self.model_id = handle.model_id

    def collect(
        self,
        dataset: ProbeDataset,
        batch_size: int = 8,
        prompt_fn: Callable[[Example], str] | None = None,
        command_fn: Callable[[Example], str] | None = None,
        max_new_tokens: int = 256,
        target_tokens: TargetTokens | None = None,
    ) -> ModelResponses:
        examples = list(dataset)
        prompts = [
            prompt_fn(ex) if prompt_fn is not None else ex.text
            for ex in examples
        ]
        commands = [
            command_fn(ex) if command_fn is not None else ex.text
            for ex in examples
        ]

        all_responses: list[str] = []
        target_logits_chunks: dict[str, list[torch.Tensor]] | None = (
            {k: [] for k in target_tokens} if target_tokens is not None else None
        )

        # Tokenizer-only attributes (pad_token_id, batch_decode) live on the
        # underlying tokenizer, which is `self.tokenizer.tokenizer` for
        # multimodal AutoProcessors and `self.tokenizer` for plain ones.
        tok = underlying_tokenizer(self.tokenizer)

        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start : batch_start + batch_size]

            # Pass text by keyword: multimodal processors (e.g. Gemma 4)
            # bind the first positional arg to `images`.
            encoded = self.tokenizer(
                text=batch_prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.model.device)

            prompt_length = encoded["input_ids"].shape[1]

            with torch.no_grad():
                if target_tokens is not None:
                    output = self.model.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tok.pad_token_id,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                    output_ids = output.sequences

                    # output.scores[0]: (batch, vocab) logits at the first
                    # newly-generated position.
                    batch_target = _aggregate_target_logits(output.scores[0], target_tokens)

                    for k, t in batch_target.items():
                        target_logits_chunks[k].append(t)
                else:
                    output_ids = self.model.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tok.pad_token_id,
                    )

            # Slice off the prompt tokens; only decode what the model generated.
            new_tokens = output_ids[:, prompt_length:]
            decoded = tok.batch_decode(new_tokens, skip_special_tokens=True)

            all_responses.extend(decoded)

        target_logits = (
            {k: torch.cat(v) for k, v in target_logits_chunks.items()}
            if target_logits_chunks is not None else None
        )

        return ModelResponses(
            commands=commands,
            responses=all_responses,
            target_logits=target_logits,
        )


class TLResponseCollector(ResponseCollector):
    """Generates text responses from a TransformerLens HookedTransformer.

    Uses tl_generate_left_padded to match HF's left-padding semantics during
    attention — TL's built-in generate would otherwise skip the prefill mask
    and attend to pad tokens.

    Args:
        handle: Loaded HookedTransformer + tokenizer + model_id.
    """

    def __init__(self, handle: TLModelHandle) -> None:
        self.model = handle.model
        self.tokenizer = handle.tokenizer
        self.model_id = handle.model_id

    def collect(
        self,
        dataset: ProbeDataset,
        batch_size: int = 8,
        prompt_fn: Callable[[Example], str] | None = None,
        command_fn: Callable[[Example], str] | None = None,
        max_new_tokens: int = 256,
        target_tokens: TargetTokens | None = None,
    ) -> ModelResponses:
        if target_tokens is not None:
            raise NotImplementedError(
                "target_tokens is not yet supported in TLResponseCollector. "
                "Use HFResponseCollector for logit-based metrics, or extend "
                "tl_generate_left_padded to surface first-generated-position "
                "logits."
            )

        examples = list(dataset)
        prompts = [
            prompt_fn(ex) if prompt_fn is not None else ex.text
            for ex in examples
        ]
        commands = [
            command_fn(ex) if command_fn is not None else ex.text
            for ex in examples
        ]

        all_responses: list[str] = []
        tok = underlying_tokenizer(self.tokenizer)

        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start : batch_start + batch_size]

            encoded = self.tokenizer(
                text=batch_prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            prompt_length = encoded["input_ids"].shape[1]

            output_ids = tl_generate_left_padded(
                model=self.model,
                tokens=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=max_new_tokens,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.pad_token_id,
            )

            new_tokens = output_ids[:, prompt_length:]
            decoded = tok.batch_decode(new_tokens, skip_special_tokens=True)

            all_responses.extend(decoded)

        return ModelResponses(
            commands=commands,
            responses=all_responses,
        )
