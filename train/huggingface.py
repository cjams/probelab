import torch

from typing import Callable
from transformers import AutoModelForCausalLM, AutoTokenizer

from probelab.dataset.base import Example, ProbeDataset
from train.activation import ActivationCollector, ActivationDataset, ActivationSpec

class HFActivationCollector(ActivationCollector):
    """
    Collects residual-stream activations from a HuggingFace causal LM.

    Uses output_hidden_states=True to capture the residual stream after each
    transformer block without requiring manual hooks. Only "residual" is
    supported as a component; use the transformer_lens backend for MLP/attn.

    The tokenizer is configured for left-padding by default so that all real
    tokens are contiguous at the right edge of each sequence — a requirement
    for LastTokenSelector and OffsetSliceTokenSelector to work correctly.

    Args:
        model_id:           HuggingFace repo ID or local path.
        spec:               Declares which layers (and component) to capture.
        device:             Device to run inference on.
        dtype:              Model weight dtype. bfloat16 for most modern GPUs.
        trust_remote_code:  Forwarded to from_pretrained.
        padding_side:       Tokenizer padding side. "left" (default) keeps real
                            tokens right-aligned, which token selectors assume.
    """
    def __init__(
        self,
        model_id: str,
        spec: ActivationSpec,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = False,
        padding_side: str = "left",
    ):
        if spec.component != "residual":
            raise ValueError(
                f"HFActivationCollector only supports component='residual', "
                f"got {spec.component!r}. Use the transformer_lens backend for "
                f"other components."
            )

        self.model_id = model_id
        self.spec = spec
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )

        self.tokenizer.padding_side = padding_side

        # Some tokenizers (e.g. LLaMA) have no pad token by default.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        ).to(device)

        self.model.eval()

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

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

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
        input_ids = _pad_and_stack(all_input_ids, pad_value=self.tokenizer.pad_token_id)
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

    def d_model(self) -> int:
        return self.model.config.hidden_size

    def d_vocab(self) -> int:
        return self.model.config.vocab_size

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