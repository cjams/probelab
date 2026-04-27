from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Callable

import torch
from tqdm.auto import tqdm

from probelab.dataset.base import Example, ProbeDataset
from model import HFModelHandle, TLModelHandle

from .base import RefusalJudge, RefusalScore


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

    def __len__(self) -> int:
        return len(self.commands)

    def judge(self, judge: RefusalJudge) -> RefusalScore:
        """Convenience method to score these responses with a RefusalJudge."""
        return judge.judge_batch(
            commands=self.commands,
            responses=self.responses,
        )


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
        n_batches = (len(prompts) + batch_size - 1) // batch_size

        for batch_start in tqdm(range(0, len(prompts), batch_size), total=n_batches, desc="generating", unit="batch"):
            batch_prompts = prompts[batch_start : batch_start + batch_size]

            encoded = self.tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.model.device)

            prompt_length = encoded["input_ids"].shape[1]

            with torch.no_grad():
                output_ids = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Slice off the prompt tokens; only decode what the model generated.
            new_tokens = output_ids[:, prompt_length:]
            decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            all_responses.extend(decoded)

        return ModelResponses(
            commands=commands,
            responses=all_responses,
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
        n_batches = (len(prompts) + batch_size - 1) // batch_size

        for batch_start in tqdm(range(0, len(prompts), batch_size), total=n_batches, desc="generating", unit="batch"):
            batch_prompts = prompts[batch_start : batch_start + batch_size]

            encoded = self.tokenizer(
                batch_prompts,
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
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            new_tokens = output_ids[:, prompt_length:]
            decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            all_responses.extend(decoded)

        return ModelResponses(
            commands=commands,
            responses=all_responses,
        )
