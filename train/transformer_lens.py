import torch

from typing import Callable
from tqdm.auto import tqdm

from probelab.dataset.base import Example, ProbeDataset
from model import TLModelHandle
from train.activation import ActivationCollector, ActivationDataset, ActivationSpec


# Per-component hook-name resolver.
#
# Layer-index convention mirrors HFActivationCollector's hidden_states:
#   idx 0         -> embedding output (pre-block-0 state)
#   idx i in 1..N -> state "around" transformer block i-1
#
# See ActivationSpec.component for the meaning of each component string.
_BLOCK_HOOK_SUFFIX: dict[str, str] = {
    "resid_post": "hook_resid_post",
    "resid_pre": "hook_resid_pre",
    "resid_mid": "hook_resid_mid",
    "mlp_out": "hook_mlp_out",
    "attn_out": "hook_attn_out",
}

# Components for which idx 0 (the pre-block-0 state) is meaningful and
# resolves to hook_embed. Any other component + idx 0 is an error.
_IDX0_AS_EMBED: frozenset[str] = frozenset({"resid_post", "resid_pre"})


def resolve_hook_name(component: str, layer_idx: int) -> str:
    """
    Resolve a (component, hidden-state index) pair to a TransformerLens hook
    name. Shared by the activation collector and the intervention backend so
    both pin to identical hook points.
    """
    if component not in _BLOCK_HOOK_SUFFIX:
        raise ValueError(
            f"Unknown component {component!r}. Supported: "
            f"{sorted(_BLOCK_HOOK_SUFFIX)}."
        )

    if layer_idx == 0:
        if component not in _IDX0_AS_EMBED:
            raise ValueError(
                f"Component {component!r} is not defined at layer index 0 "
                f"(the embedding state). Valid at idx 0: "
                f"{sorted(_IDX0_AS_EMBED)}."
            )
        return "hook_embed"

    return f"blocks.{layer_idx - 1}.{_BLOCK_HOOK_SUFFIX[component]}"


class TLActivationCollector(ActivationCollector):
    """
    Collects activations from a TransformerLens HookedTransformer.

    Supports every residual-stream position exposed by TransformerLens:
    resid_pre/mid/post at each block, plus the additive mlp_out and attn_out
    contributions. Layer-index convention matches HFActivationCollector so
    ActivationSpec and downstream TokenSelectors are portable across backends
    for the "resid_post" component.

    The handle's tokenizer must be configured for left-padding — load_tl
    sets this by default.

    Args:
        handle: Loaded HookedTransformer + tokenizer + model_id.
        spec:   Declares which layers and component to capture.
    """

    def __init__(
        self,
        handle: TLModelHandle,
        spec: ActivationSpec,
    ):
        if spec.component not in _BLOCK_HOOK_SUFFIX:
            raise ValueError(
                f"TLActivationCollector does not support component="
                f"{spec.component!r}. Supported: "
                f"{sorted(_BLOCK_HOOK_SUFFIX)}."
            )

        self.model = handle.model
        self.tokenizer = handle.tokenizer
        self.model_id = handle.model_id
        self.spec = spec

    def _n_states(self) -> int:
        # Mirror the HF convention: embedding + one state per transformer block.
        return self.model.cfg.n_layers + 1

    def collect(
        self,
        dataset: ProbeDataset,
        batch_size: int = 16,
        prompt_fn: Callable[[Example], str] | None = None,
    ) -> ActivationDataset:
        texts = [
            prompt_fn(ex) if prompt_fn is not None else ex.text
            for ex in dataset
        ]

        labels = torch.tensor([ex.label for ex in dataset], dtype=torch.bool)
        example_ids = [ex.id for ex in dataset]

        resolved_targets = self.spec.resolve_targets(self._n_states())
        hook_names = {
            idx: resolve_hook_name(self.spec.component, idx)
            for idx in resolved_targets
        }
        wanted_names = set(hook_names.values())

        layer_buffers: dict[int, list[torch.Tensor]] = {idx: [] for idx in resolved_targets}
        all_input_ids: list[torch.Tensor] = []
        all_attention_masks: list[torch.Tensor] = []

        n_batches = (len(texts) + batch_size - 1) // batch_size

        for batch_start in tqdm(range(0, len(texts), batch_size), total=n_batches, desc="collecting", unit="batch"):
            batch_texts = texts[batch_start : batch_start + batch_size]

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.model.cfg.device)

            with torch.no_grad():
                _, cache = self.model.run_with_cache(
                    encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    names_filter=lambda name: name in wanted_names,
                    return_type=None,
                )

            for idx, name in hook_names.items():
                layer_buffers[idx].append(cache[name].detach().cpu())

            all_input_ids.append(encoded["input_ids"].cpu())
            all_attention_masks.append(encoded["attention_mask"].cpu())

        input_ids = _pad_and_stack(all_input_ids, pad_value=self.tokenizer.pad_token_id)
        attention_mask = _pad_and_stack(all_attention_masks, pad_value=0)

        activations: dict[int, torch.Tensor] = {
            idx: _pad_and_stack(chunks, pad_value=0.0)
            for idx, chunks in layer_buffers.items()
        }

        return ActivationDataset(
            activations=activations,
            labels=labels,
            input_ids=input_ids,
            attention_mask=attention_mask,
            example_ids=example_ids,
            concept=dataset.concept,
            model_id=self.model_id,
            spec=self.spec,
        )

    def d_model(self) -> int:
        return self.model.cfg.d_model

    def d_vocab(self) -> int:
        return self.model.cfg.d_vocab


def _pad_and_stack(tensors: list[torch.Tensor], pad_value) -> torch.Tensor:
    """
    Stack a list of tensors along dim 0, left-padding dim 1 to the global
    maximum sequence length so all tensors are the same shape.

    Works for both 2D (n, seq) and 3D (n, seq, d) tensors.
    """
    max_len = max(t.shape[1] for t in tensors)

    padded = []
    for t in tensors:
        pad_len = max_len - t.shape[1]

        if pad_len > 0:
            if t.dim() == 2:
                pad = torch.full((t.shape[0], pad_len), pad_value, dtype=t.dtype)
            else:
                pad = torch.full((t.shape[0], pad_len, t.shape[2]), pad_value, dtype=t.dtype)
            t = torch.cat([pad, t], dim=1)

        padded.append(t)

    return torch.cat(padded, dim=0)
