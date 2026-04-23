from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from tqdm.auto import tqdm

from probelab.dataset.base import Example, ProbeDataset
from model import HFModelBundle

from .base import RefusalJudge, RefusalScore


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


class HFResponseCollector:
    """Generates text responses from a HuggingFace causal LM over a ProbeDataset.

    Uses left-padding so batched generation terminates cleanly: all prompts end
    at the same position and new tokens are sliced off uniformly.

    # Consistency with HFActivationCollector
    This class is intentionally separate from HFActivationCollector. When using
    both together (e.g. collecting activations and then measuring refusal rate),
    three things must match exactly or the responses will not correspond to the
    activation inputs:

    1. model — pass the same HFModelBundle to both. A different checkpoint or
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

    The bundle's tokenizer is expected to be configured for left-padding so
    batched generation terminates cleanly. load_hf_bundle() sets this by
    default.

    Args:
        bundle: Loaded model + tokenizer + model_id.
    """

    def __init__(self, bundle: HFModelBundle) -> None:
        self.model = bundle.model
        self.tokenizer = bundle.tokenizer
        self.model_id = bundle.model_id

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
            batch_size:     Examples per generation batch. Lower than activation
                            collection since KV-cache grows with new tokens.
            prompt_fn:      Formats each Example into a model prompt string.
                            Must be constructed with add_generation_prompt=True
                            (e.g. ChatFormatter(tok, add_generation_prompt=True))
                            and must otherwise be identical to the prompt_fn
                            passed to HFActivationCollector.collect(). When
                            None, ex.text is used directly (no chat template).
            command_fn:     Extracts the user-facing instruction text to store
                            as ModelResponses.commands — what the judge sees.
                            Defaults to ex.text, which is wrong when the
                            dataset's raw text is a statement that gets
                            instructionified. Pass formatter.user_content to
                            get the actual instruction.
            max_new_tokens: Maximum tokens to generate per example.

        Returns:
            ModelResponses with one entry per dataset example.
        """
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
