from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from model import HFModelHandle
from intervention.base import Intervention, InterventionBackend, apply_intervention

if TYPE_CHECKING:
    from dataset.base import ProbeDataset
    from evaluate.generate import ModelResponses
    from train.token import TokenSelector


def _get_layers_module(model):
    """Return the nn.ModuleList of transformer blocks.

    Supports LLaMA/Mistral/Qwen/Gemma (model.model.layers), GPT-2
    (model.transformer.h), and GPT-NeoX (model.gpt_neox.layers).
    """
    for model_attr in ("model", "transformer", "gpt_neox"):
        inner = getattr(model, model_attr, None)
        if inner is None:
            continue

        for layers_attr in ("layers", "h", "blocks"):
            layers = getattr(inner, layers_attr, None)
            if layers is not None:
                return layers

    raise ValueError(
        f"Cannot auto-detect transformer blocks for {type(model).__name__}. "
        f"Supported: LLaMA/Mistral/Qwen/Gemma (model.model.layers), "
        f"GPT-2 (model.transformer.h), GPT-NeoX (model.gpt_neox.layers)."
    )


def _get_layer_module(model, layer_idx: int):
    """Return the transformer block whose output corresponds to hidden_states[layer_idx].

    Convention matches HFActivationCollector: hidden_states[0] is the embedding
    output, hidden_states[i] is the output of the (i-1)-th transformer block.
    """
    if layer_idx == 0:
        raise ValueError("Cannot hook at the embedding layer (layer_idx=0).")

    layers = _get_layers_module(model)
    block_idx = layer_idx - 1

    if block_idx >= len(layers):
        raise ValueError(f"Layer {layer_idx} out of range (model has {len(layers)} transformer blocks).")

    return layers[block_idx]


class HFInterventionBackend(InterventionBackend):
    """Intervention backend for HuggingFace causal LMs using PyTorch forward hooks.

    Args:
        handle: Loaded model and tokenizer.
    """

    def __init__(self, handle: HFModelHandle) -> None:
        self.model = handle.model
        self.tokenizer = handle.tokenizer

    def num_transformer_layers(self) -> int:
        return len(_get_layers_module(self.model))

    def collect_responses(
        self,
        dataset: "ProbeDataset",
        hook_layers: int | list[int],
        token_selector: "TokenSelector",
        intervention: Intervention | None,
        batch_size: int = 8,
        prompt_fn: Callable | None = None,
        command_fn: Callable | None = None,
        max_new_tokens: int = 256,
        target_tokens: "dict[str, int | list[int]] | None" = None,
        **generate_kwargs,
    ) -> "ModelResponses":
        import torch as _torch
        from evaluate.generate import ModelResponses, _aggregate_target_logits

        examples = list(dataset)
        prompts = [
            prompt_fn(ex) if prompt_fn is not None else ex.text
            for ex in examples
        ]
        commands = [
            command_fn(ex) if command_fn is not None else ex.text
            for ex in examples
        ]

        if intervention is not None and set(intervention.components()) != {"resid_post"}:
            raise ValueError(
                f"HFInterventionBackend only supports component='resid_post', "
                f"got {intervention.component!r}. Use TLInterventionBackend "
                f"for other residual-stream positions or component outputs."
            )

        layers = [hook_layers] if isinstance(hook_layers, int) else list(hook_layers)
        layer_modules = (
            [_get_layer_module(self.model, l) for l in layers]
            if intervention is not None else []
        )

        all_responses: list[str] = []
        target_logits_chunks: dict[str, list[_torch.Tensor]] | None = (
            {k: [] for k in target_tokens} if target_tokens is not None else None
        )

        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start:batch_start + batch_size]

            # encoded["input_ids"]:      (batch, seq_len) long
            # encoded["attention_mask"]: (batch, seq_len) long, 1 for real tokens
            encoded = self.tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.model.device)

            prompt_length = encoded["input_ids"].shape[1]
            hook_handles: list = []

            if intervention is not None:
                # One hook per layer — all share the same (direction, scale, mode),
                # each with its own independent prefill_done flag.
                for lm in layer_modules:
                    hook_handles.append(self._register_hook(
                        layer_module=lm,
                        token_selector=token_selector,
                        intervention=intervention,
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                    ))

            try:
                with torch.no_grad():
                    if target_tokens is not None:
                        # Capture first-generated-position logits so the
                        # caller can compute logit-difference metrics. The
                        # intervention is still applied during prefill via
                        # the registered hooks.
                        output = self.model.generate(
                            **encoded,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                            output_scores=True,
                            return_dict_in_generate=True,
                            **generate_kwargs,
                        )
                        output_ids = output.sequences

                        batch_target = _aggregate_target_logits(output.scores[0], target_tokens)

                        for k, t in batch_target.items():
                            target_logits_chunks[k].append(t)
                    else:
                        # output_ids: (batch, prompt_length + n_new_tokens)
                        output_ids = self.model.generate(
                            **encoded,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                            **generate_kwargs,
                        )
            finally:
                for h in hook_handles:
                    h.remove()

            # Slice off the prompt tokens — only decode the newly generated ones.
            new_tokens = output_ids[:, prompt_length:]
            decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            all_responses.extend(decoded)

        target_logits = (
            {k: _torch.cat(v) for k, v in target_logits_chunks.items()}
            if target_logits_chunks is not None else None
        )

        return ModelResponses(
            commands=commands,
            responses=all_responses,
            target_logits=target_logits,
        )

    def _register_hook(
        self,
        layer_module,
        token_selector: "TokenSelector",
        intervention: Intervention,
        input_ids: torch.Tensor,        # (batch, seq_len) — prefill only
        attention_mask: torch.Tensor,   # (batch, seq_len) — prefill only
    ):
        # Single-element list so the flag is mutable from within the closure.
        prefill_done = [False]
        direction = intervention.direction.to(self.model.device)  # (d_model,)

        def hook(module, input, output):
            # HF transformer blocks return either a tuple (hidden, ...) or a bare tensor.
            # hidden shape: (batch, seq_len_this_pass, d_model)
            #   - prefill:        seq_len_this_pass == prompt seq_len
            #   - autoregressive: seq_len_this_pass == 1 (with KV cache)
            hidden = output[0] if isinstance(output, tuple) else output

            if not prefill_done[0]:
                # pos_mask: (batch, seq_len) bool — which prompt positions to steer.
                pos_mask = token_selector.positions(input_ids, attention_mask).to(hidden.device)
                apply_intervention(hidden, pos_mask, direction, intervention)
                prefill_done[0] = True

            elif intervention.apply_on == "all":
                # Steer every position of the current autoregressive step
                # (typically just the one newly generated token).
                all_mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
                apply_intervention(hidden, all_mask, direction, intervention)

            if isinstance(output, tuple):
                return (hidden,) + output[1:]

            return hidden

        return layer_module.register_forward_hook(hook)
