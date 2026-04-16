import torch

from abc import ABC, abstractmethod
from train.activation import ActivationDataset

class TokenSelector(ABC):
    """
    Selects which token positions from an ActivationDataset contribute to
    the probe training signal, and reduces them to (n_training, d_model) vectors.

    Kept separate from ActivationCollector because the right token positions
    depend on the research question, not on how the model is run.
    """

    @abstractmethod
    def select(
        self,
        act_dataset: ActivationDataset,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract training vectors for a single layer.

        Args:
            act_dataset: The full activation dataset.
            layer:       Which layer to select from. Must be a key in
                         act_dataset.activations.

        Returns:
            (activations, labels) where:
              activations — (n_training, d_model)
              labels      — (n_training,) bool

            n_training may exceed n_examples when the selector expands each
            example into multiple token positions (e.g. OffsetSliceTokenSelector).
        """
        ...


class LastTokenSelector(TokenSelector):
    """
    Takes the last real (non-padding) token from each example.

    Uses attention_mask to find the final non-padding position per row,
    so it works correctly regardless of padding side.
    """

    def select(
        self,
        act_dataset: ActivationDataset,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        acts = act_dataset.activations[layer]  # (n, seq, d)
        mask = act_dataset.attention_mask      # (n, seq)

        # cumsum then argmax gives the index of the last 1 in each row.
        last_indices = mask.long().cumsum(dim=1).argmax(dim=1)  # (n,)
        selected = acts[torch.arange(len(acts)), last_indices]  # (n, d)

        return selected, act_dataset.labels


class MeanTokenSelector(TokenSelector):
    """
    Mean-pools over non-padding tokens, optionally skipping a leading prefix.

    Args:
        skip_prefix: Number of leading real tokens to exclude before pooling.
                     Useful to drop the instruction portion of a prompt so
                     only response-side tokens contribute.
    """

    def __init__(self, skip_prefix: int = 0):
        self.skip_prefix = skip_prefix

    def select(
        self,
        act_dataset: ActivationDataset,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        acts = act_dataset.activations[layer]      # (n, seq, d)
        mask = act_dataset.attention_mask.float()   # (n, seq)

        # Skip over padding tokens and optional prefix tokens
        if self.skip_prefix > 0:
            # Zero out the first skip_prefix real tokens per row using cumsum.
            real_cumsum = mask.long().cumsum(dim=1)           # (n, seq)
            prefix_mask = (real_cumsum <= self.skip_prefix).float()
            mask = mask * (1.0 - prefix_mask)

        mask_3d = mask.unsqueeze(-1)                           # (n, seq, 1)
        summed = (acts * mask_3d).sum(dim=1)                   # (n, d)
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1)    # (n, 1)

        return summed / counts, act_dataset.labels


class OffsetSliceTokenSelector(TokenSelector):
    """
    Treats each token in a fixed range as its own independent training example,
    expanding the dataset from n_examples to n_examples * slice_len rows.

    Offsets are into the real (non-padding) token sequence, so padding side
    doesn't matter. This is the pattern from the refusal probe paper, where
    all post-instruction tokens contribute independently.

    Args:
        start: Start of the slice within real tokens (inclusive).
               When anchor="end", negative values count from the last real token.
        end:   End of the slice (exclusive). None = up to the last real token.
               When anchor="end", negative values count from the last real token.
        anchor: "end"   — offsets relative to the last real token (default).
                "start" — offsets relative to the first real token.
    """

    def __init__(
        self,
        start: int,
        end: int | None = None,
        anchor: str = "end",
    ):
        if anchor not in ("start", "end"):
            raise ValueError(f"anchor must be 'start' or 'end', got {anchor!r}")

        self.start = start
        self.end = end
        self.anchor = anchor

    def select(
        self,
        act_dataset: ActivationDataset,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            (activations, labels, positions) where:
              activations — (n_training, d_model)
              labels      — (n_training,) bool
              positions   — (n_training,) int, offset within the real-token
                            sequence (0 = first real token). Use this to filter
                            or group by token position during hyperparameter search.
        """
        acts = act_dataset.activations[layer]  # (n, seq, d)
        mask = act_dataset.attention_mask      # (n, seq)
        labels = act_dataset.labels            # (n,)

        n, seq_len, d = acts.shape
        real_lengths = mask.long().sum(dim=1)  # (n,)

        all_vecs: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        all_positions: list[torch.Tensor] = []

        for i in range(n):
            length = int(real_lengths[i].item())

            if self.anchor == "end":
                abs_start = length + self.start if self.start < 0 else self.start
                abs_end = length if self.end is None else (length + self.end if self.end < 0 else self.end)
            else:
                abs_start = self.start
                abs_end = length if self.end is None else self.end

            abs_start = max(0, abs_start)
            abs_end = min(length, abs_end)

            if abs_start >= abs_end:
                continue

            # Real tokens are in the rightmost `length` positions (left-padded).
            pad_offset = seq_len - length
            seq_start = pad_offset + abs_start
            seq_end = pad_offset + abs_end

            slice_len = seq_end - seq_start
            all_vecs.append(acts[i, seq_start:seq_end])                  # (slice_len, d)
            all_labels.append(labels[i].expand(slice_len))
            all_positions.append(torch.arange(abs_start, abs_end))       # (slice_len,)

        if not all_vecs:
            return torch.empty(0, d), torch.empty(0, dtype=torch.bool), torch.empty(0, dtype=torch.long)

        return torch.cat(all_vecs, dim=0), torch.cat(all_labels, dim=0), torch.cat(all_positions, dim=0)


class AssistantTokenSelector(TokenSelector):
    """
    Mean-pools over the assistant portion of each sequence.

    Finds the first occurrence of `assistant_token_id` in input_ids per row,
    then mean-pools over all real tokens from that position to the end of the
    real sequence. Falls back to pooling over all real tokens if the delimiter
    is not found with a masked mean.

    The pattern from strategic deception work.

    Args:
        assistant_token_id: Token ID that marks the start of the assistant turn.
                            Look this up from your tokenizer before constructing,
                            e.g. tokenizer.convert_tokens_to_ids("<|im_start|>").
    """

    def __init__(self, assistant_token_id: int):
        self.assistant_token_id = assistant_token_id

    def select(
        self,
        act_dataset: ActivationDataset,
        layer: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        acts = act_dataset.activations[layer]   # (n, seq, d)
        mask = act_dataset.attention_mask       # (n, seq)
        input_ids = act_dataset.input_ids       # (n, seq)

        n, _, d = acts.shape
        selected = torch.zeros(n, d)

        for i in range(n):
            positions = (input_ids[i] == self.assistant_token_id).nonzero(as_tuple=True)[0]

            if len(positions) == 0:
                real_mask = mask[i].float()
            else:
                start = int(positions[0].item())
                real_mask = mask[i].float().clone()
                real_mask[:start] = 0.0

            count = real_mask.sum().clamp(min=1)
            selected[i] = (acts[i] * real_mask.unsqueeze(-1)).sum(dim=0) / count

        return selected, act_dataset.labels
