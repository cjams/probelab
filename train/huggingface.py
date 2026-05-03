import torch

from typing import Callable

from probelab.dataset.base import Example, ProbeDataset
from probelab.model import HFModelHandle, underlying_tokenizer
from probelab.train.activation import ActivationCollector, ActivationDataset, ActivationSpec


class HFActivationCollector(ActivationCollector):
    """
    Collects residual-stream activations from a HuggingFace causal LM.

    Uses output_hidden_states=True to capture the residual stream after each
    transformer block without requiring manual hooks. Only "resid_post" is
    supported as a component; use the transformer_lens backend for MLP/attn
    or for other residual-stream positions (resid_pre, resid_mid).

    The handle's tokenizer is expected to be configured for left-padding so
    that all real tokens are contiguous at the right edge of each sequence —
    a requirement for LastTokenSelector and OffsetSliceTokenSelector to work
    correctly. load_hf() sets this by default.

    Args:
        handle: Loaded model + tokenizer + model_id.
        spec:   Declares which layers (and component) to capture.
    """
    def __init__(
        self,
        handle: HFModelHandle,
        spec: ActivationSpec,
    ):
        if spec.component != "resid_post":
            raise ValueError(
                f"HFActivationCollector only supports component='resid_post', "
                f"got {spec.component!r}. Use the transformer_lens backend for "
                f"other components."
            )

        self.model = handle.model
        self.tokenizer = handle.tokenizer
        self.model_id = handle.model_id
        self.spec = spec

    def collect(
        self,
        dataset: ProbeDataset,
        batch_size: int = 16,
        prompt_fn: Callable[[Example], str] | None = None,
    ) -> ActivationDataset:
        """
        Run inference over `dataset` and return hidden states at the layers
        specified in self.spec.

        Args:
            dataset:    Source ProbeDataset.
            batch_size: Examples per forward pass.
            prompt_fn:  Optional formatter applied to each Example before
                        tokenization. When None, Example.text is used directly.

        Returns:
            ActivationDataset with shape (n_examples, seq_len, d_model) per layer.
        """
        texts = [
            prompt_fn(ex) if prompt_fn is not None else ex.text
            for ex in dataset
        ]

        labels = torch.tensor([ex.label for ex in dataset], dtype=torch.bool)
        example_ids = [ex.id for ex in dataset]

        # Resolve targets on the first batch once we know n_states.
        resolved_targets: list[int] | None = None
        layer_buffers: dict[int, list[torch.Tensor]] = {}
        all_input_ids: list[torch.Tensor] = []
        all_attention_masks: list[torch.Tensor] = []

        for batch_start in range(0, len(texts), batch_size):
            batch_texts = texts[batch_start : batch_start + batch_size]

            # Pass text by keyword: multimodal processors (e.g. Gemma 4)
            # bind the first positional arg to `images`, which leaves
            # `text=None` and crashes inside the processor.
            encoded = self.tokenizer(
                text=batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                output = self.model(
                    **encoded,
                    output_hidden_states=True,
                )

            # hidden_states: tuple of length n_states (embedding output +
            # one entry per transformer block for this backend).
            hidden_states = output.hidden_states
            n_states = len(hidden_states)

            if resolved_targets is None:
                resolved_targets = self.spec.resolve_targets(n_states)
                layer_buffers = {idx: [] for idx in resolved_targets}

            for idx in resolved_targets:
                layer_buffers[idx].append(hidden_states[idx].detach().cpu())

            all_input_ids.append(encoded["input_ids"].cpu())
            all_attention_masks.append(encoded["attention_mask"].cpu())

        # Stack batches. Sequences may differ in length across batches if the
        # longest sequence varies — pad to the global max before stacking.
        input_ids = _pad_and_stack(all_input_ids, pad_value=underlying_tokenizer(self.tokenizer).pad_token_id)
        attention_mask = _pad_and_stack(all_attention_masks, pad_value=0)

        activations: dict[int, torch.Tensor] = {}
        for layer_idx, chunks in layer_buffers.items():
            activations[layer_idx] = _pad_and_stack(chunks, pad_value=0.0)

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

    def _text_config(self):
        cfg = self.model.config
        return cfg.text_config if hasattr(cfg, "text_config") else cfg

    def d_model(self) -> int:
        return self._text_config().hidden_size

    def d_vocab(self) -> int:
        return self._text_config().vocab_size

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
            # Left-pad: prepend the padding.
            t = torch.cat([pad, t], dim=1)
        padded.append(t)

    return torch.cat(padded, dim=0)
