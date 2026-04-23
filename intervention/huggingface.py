from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
from tqdm.auto import tqdm

from model import HFModelBundle
from intervention.base import Intervention, InterventionBackend

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


def _apply_intervention(
    hidden: torch.Tensor,       # (batch, seq_len, d_model)
    mask: torch.Tensor,         # (batch, seq_len) bool
    direction: torch.Tensor,    # (d_model,)
    intervention: Intervention,
) -> None:
    """Apply intervention in-place at positions where mask is True."""
    direction = direction.to(hidden.dtype)

    # Boolean indexing flattens the masked positions: (n_selected, d_model).
    selected = hidden[mask]

    if intervention.mode == "add":
        # direction broadcasts from (d_model,) to (n_selected, d_model).
        hidden[mask] = selected + intervention.scale * direction

    elif intervention.mode == "subtract":
        hidden[mask] = selected - intervention.scale * direction

    elif intervention.mode == "ablate":
        # (selected @ direction) is (n_selected,); unsqueeze gives (n_selected, 1)
        # so the outer product with direction is (n_selected, d_model).
        # scale=1 removes the full projection; scale<1 removes a fraction.
        proj = (selected @ direction).unsqueeze(-1) * direction
        hidden[mask] = selected - intervention.scale * proj

    else:
        raise ValueError(f"Unknown intervention mode: {intervention.mode!r}")


class HFInterventionBackend(InterventionBackend):
    """Intervention backend for HuggingFace causal LMs using PyTorch forward hooks.

    Args:
        bundle: Loaded model and tokenizer.
    """

    def __init__(self, bundle: HFModelBundle) -> None:
        self.model = bundle.model
        self.tokenizer = bundle.tokenizer

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
        **generate_kwargs,
    ) -> "ModelResponses":
        from evaluate.generate import ModelResponses

        examples = list(dataset)
        prompts = [
            prompt_fn(ex) if prompt_fn is not None else ex.text
            for ex in examples
        ]
        commands = [
            command_fn(ex) if command_fn is not None else ex.text
            for ex in examples
        ]

        layers = [hook_layers] if isinstance(hook_layers, int) else list(hook_layers)
        layer_modules = (
            [_get_layer_module(self.model, l) for l in layers]
            if intervention is not None else []
        )

        all_responses: list[str] = []
        n_batches = (len(prompts) + batch_size - 1) // batch_size

        for batch_start in tqdm(range(0, len(prompts), batch_size), total=n_batches, desc="generating", unit="batch"):
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

        return ModelResponses(
            commands=commands,
            responses=all_responses,
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
                _apply_intervention(hidden, pos_mask, direction, intervention)
                prefill_done[0] = True

            elif intervention.apply_on == "all":
                # Steer every position of the current autoregressive step
                # (typically just the one newly generated token).
                all_mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)
                _apply_intervention(hidden, all_mask, direction, intervention)

            if isinstance(output, tuple):
                return (hidden,) + output[1:]

            return hidden

        return layer_module.register_forward_hook(hook)
