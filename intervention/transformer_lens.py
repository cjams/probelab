from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from model import TLModelHandle
from intervention.base import Intervention, InterventionBackend, apply_intervention
from evaluate.generate import tl_generate_left_padded
from train.transformer_lens import resolve_hook_name

if TYPE_CHECKING:
    from dataset.base import ProbeDataset
    from evaluate.generate import ModelResponses
    from train.token import TokenSelector


class TLInterventionBackend(InterventionBackend):
    """Intervention backend for TransformerLens HookedTransformer.

    Applies the intervention at the hook point chosen by intervention.component
    (resid_pre / resid_mid / resid_post / mlp_out / attn_out). The same
    Intervention semantics (mode, scale, direction, apply_on) as the HF backend
    are used — only the attach point differs.

    Args:
        handle: Loaded HookedTransformer + tokenizer.
    """

    def __init__(self, handle: TLModelHandle) -> None:
        self.model = handle.model
        self.tokenizer = handle.tokenizer

    def num_transformer_layers(self) -> int:
        return self.model.cfg.n_layers

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

        all_responses: list[str] = []

        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start:batch_start + batch_size]

            encoded = self.tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            prompt_length = encoded["input_ids"].shape[1]

            hooks = (
                self._build_hooks(
                    layers=layers,
                    token_selector=token_selector,
                    intervention=intervention,
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                )
                if intervention is not None else None
            )

            output_ids = tl_generate_left_padded(
                model=self.model,
                tokens=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                fwd_hooks=hooks,
            )

            new_tokens = output_ids[:, prompt_length:]
            decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            all_responses.extend(decoded)

        return ModelResponses(
            commands=commands,
            responses=all_responses,
        )

    def _build_hooks(
        self,
        layers: list[int],
        token_selector: "TokenSelector",
        intervention: Intervention,
        input_ids: torch.Tensor,        # (batch, seq_len) — prefill
        attention_mask: torch.Tensor,   # (batch, seq_len) — prefill
    ) -> list[tuple[str, Callable]]:
        """Build (hook_name, hook_fn) pairs over the (layer x component) cross.

        When intervention.component is a list, every listed component is hooked
        at every layer in the same forward pass. Each hook owns its own
        prefill_done flag so they fire independently across the prefill ->
        autoregressive transition.
        """
        direction = intervention.direction.to(self.model.cfg.device)

        hooks: list[tuple[str, Callable]] = []

        for layer in layers:
            for component in intervention.components():
                hook_name = resolve_hook_name(component, layer)
                prefill_done = [False]

                def hook(value: torch.Tensor, hook, _pd=prefill_done):
                    # value shape: (batch, seq_len_this_pass, d_model)
                    #   - prefill:        seq_len_this_pass == prompt seq_len
                    #   - autoregressive: seq_len_this_pass == 1 (KV cache)
                    if not _pd[0]:
                        pos_mask = token_selector.positions(input_ids, attention_mask).to(value.device)
                        apply_intervention(value, pos_mask, direction, intervention)
                        _pd[0] = True

                    elif intervention.apply_on == "all":
                        all_mask = torch.ones(value.shape[:2], dtype=torch.bool, device=value.device)
                        apply_intervention(value, all_mask, direction, intervention)

                    return value

                hooks.append((hook_name, hook))

        return hooks
