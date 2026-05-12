import torch

from abc import ABC, abstractmethod
from typing import Literal
from probelab.train.activation import ActivationDataset

_SENTINEL = "XPROBESENTINEL"
LayerSelection = int | list[int] | Literal["all", "all_transformer"]


def get_post_instruction_tokens(tokenizer, tokenize: bool = True) -> list[int] | str:
    """
    Return the tokens (or text) that follow user content in a chat-formatted
    prompt, up to and including the generation prompt.

    Uses a sentinel as the user content so the exact character boundary where
    user content ends can be located in the formatted string. Everything after
    that boundary (closing user-turn tokens + assistant generation prompt) is
    returned as token IDs or raw text.

    Args:
        tokenizer: A HuggingFace tokenizer (or AutoProcessor for multimodal
                   models — unwrapped via `underlying_tokenizer`) with
                   apply_chat_template support.
        tokenize:  If True (default), return a list of token IDs.
                   If False, return the raw post-sentinel string.

    Returns:
        List of token IDs when tokenize=True, or the post-sentinel string
        when tokenize=False.

    Raises:
        ValueError: If the tokenizer has no chat template or the sentinel is
                    not found in the formatted output.
    """
    # Multimodal processors (Gemma 4, etc.) have an apply_chat_template that
    # adds image-token scaffolding the bare tokenizer doesn't, and reject
    # positional text= calls. Unwrap to the underlying tokenizer so this
    # path is identical for plain and multimodal models.
    from probelab.model import underlying_tokenizer
    tok = underlying_tokenizer(tokenizer)

    messages = [{"role": "user", "content": _SENTINEL}]

    try:
        formatted = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception as exc:
        raise ValueError(f"Tokenizer has no usable chat template: {exc}") from exc

    if _SENTINEL not in formatted:
        raise ValueError("Sentinel not found in formatted template output.")

    sentinel_end = formatted.index(_SENTINEL) + len(_SENTINEL)
    post_text = formatted[sentinel_end:]

    if not tokenize:
        return post_text

    # Pass text by keyword for the same reason the upstream collectors do —
    # processors bind the first positional arg to `images`.
    return tok(text=post_text, add_special_tokens=False)["input_ids"]


def _resolve_layers(act_dataset: ActivationDataset, layer: LayerSelection) -> list[int]:
    if layer == "all":
        return sorted(act_dataset.activations.keys())

    if layer == "all_transformer":
        return [k for k in sorted(act_dataset.activations.keys()) if k != 0]

    if isinstance(layer, int):
        return [layer]

    return sorted(set(layer))

# ---------------------------------------------------------------------------
# Selectors
#
# A TokenSelector identifies which positions in the sequence are relevant.
# select() always returns (activations, labels, mask) where:
#   activations  — dict[layer -> (n, seq_len, d_model)], full sequence activations
#                  for the requested layers
#   labels       — (n,) bool
#   mask         — (n, seq_len) bool, True at every selected position
#
# Selectors do not reduce. Apply a TokenReducer afterward when the selection
# implicates more than one position per example.
# ---------------------------------------------------------------------------

class TokenSelector(ABC):
    @abstractmethod
    def select(
        self,
        act_dataset: ActivationDataset,
        layer: LayerSelection,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Args:
            act_dataset: The full activation dataset.
            layer:       int, list[int], or "all".

        Returns:
            (activations, labels, mask) where activations maps each resolved
            layer to (n, seq_len, d_model), labels is (n,) bool, and mask is
            (n, seq_len) bool marking the selected positions.
        """
        ...

    @abstractmethod
    def positions(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return (batch, seq_len) bool mask of selected positions.

        Used by intervention hooks where no ActivationDataset is available.
        """
        ...


class AllTokenSelector(TokenSelector):
    """
    Selects all non-padding tokens. This is the default — other selectors
    narrow down from here.
    """

    def select(
        self,
        act_dataset: ActivationDataset,
        layer: LayerSelection,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        layers = _resolve_layers(act_dataset, layer)
        activations = {l: act_dataset.activations[l] for l in layers}

        return activations, act_dataset.labels, act_dataset.attention_mask.bool()

    def positions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return attention_mask.bool()


class LastNTokenSelector(TokenSelector):
    """
    Selects the last n real (non-padding) tokens per example.

    When n=1 (default) the mask has exactly one True per row and no reducer
    is needed. For n>1 apply a TokenReducer to aggregate the selected positions.

    Args:
        n: Number of tokens to select from the end of each real sequence.
    """

    def __init__(self, n: int = 1):
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        
        self.n = n

    def select(
        self,
        act_dataset: ActivationDataset,
        layer: LayerSelection,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        layers = _resolve_layers(act_dataset, layer)
        attn = act_dataset.attention_mask  # (n_ex, seq)
        n_ex, seq_len = attn.shape

        real_lengths = attn.long().sum(dim=1)  # (n_ex,)
        mask = torch.zeros(n_ex, seq_len, dtype=torch.bool)

        for i in range(n_ex):
            length = int(real_lengths[i].item())
            n_take = min(self.n, length)
            # Real tokens are right-aligned; last n_take are at seq_len - n_take.
            mask[i, seq_len - n_take :] = True

        activations = {l: act_dataset.activations[l] for l in layers}
        return activations, act_dataset.labels, mask

    def positions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        n, seq_len = input_ids.shape
        real_lengths = attention_mask.long().sum(dim=1)
        mask = torch.zeros(n, seq_len, dtype=torch.bool, device=input_ids.device)

        for i in range(n):
            length = int(real_lengths[i].item())
            n_take = min(self.n, length)
            mask[i, seq_len - n_take:] = True

        return mask


class PostInstructionTokenSelector(TokenSelector):
    """
    Selects the post-instruction tokens at the tail of each prompt.

    Post-instruction tokens are the fixed template tokens after the user's
    message — typically a closing user-turn marker plus the generation prompt
    prefix (e.g. "<|eot_id|><|start_header_id|>assistant<|end_header_id|>").
    Their count comes from get_post_instruction_tokens.

    Since these are always the last real tokens in a left-padded sequence,
    the mask marks the last n_post positions in every row.

    Args:
        tokenizer: HuggingFace tokenizer — used once at construction to count
                   post-instruction tokens via get_post_instruction_tokens.
    """

    def __init__(self, tokenizer):
        post_ids = get_post_instruction_tokens(tokenizer)
        
        if not post_ids:
            raise ValueError("get_post_instruction_tokens returned no tokens for this tokenizer.")

        self.n_post = len(post_ids)

    def select(
        self,
        act_dataset: ActivationDataset,
        layer: LayerSelection,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        layers = _resolve_layers(act_dataset, layer)
        n, seq_len, _ = act_dataset.activations[layers[0]].shape

        mask = torch.zeros(n, seq_len, dtype=torch.bool)
        mask[:, seq_len - self.n_post :] = True

        activations = {l: act_dataset.activations[l] for l in layers}
        return activations, act_dataset.labels, mask

    def positions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        n, seq_len = input_ids.shape
        mask = torch.zeros(n, seq_len, dtype=torch.bool, device=input_ids.device)
        mask[:, seq_len - self.n_post:] = True

        return mask


# ---------------------------------------------------------------------------
# Reducers
#
# A TokenReducer consumes the (activations, labels, mask) triple from a
# selector and produces training-ready vectors.
# ---------------------------------------------------------------------------

class TokenReducer(ABC):
    @abstractmethod
    def reduce(
        self,
        activations: dict[int, torch.Tensor],
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple:
        """
        Args:
            activations: dict[layer -> (n, seq_len, d_model)].
            labels:      (n,) bool.
            mask:        (n, seq_len) bool marking selected positions.
        """
        ...


class MeanReducer(TokenReducer):
    """
    Mean-pools selected positions into one vector per example.

    Returns (activations, labels) with shape (n, d_model) per layer.
    """

    def reduce(
        self,
        activations: dict[int, torch.Tensor],
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        m = mask.float().unsqueeze(-1)                          # (n, seq, 1)
        counts = mask.float().sum(dim=1, keepdim=True).clamp(min=1)  # (n, 1)

        result = {
            l: (acts * m).sum(dim=1) / counts
            for l, acts in activations.items()
        }

        return result, labels


class EachPositionReducer(TokenReducer):
    """
    Treats each selected position as its own independent training example.

    Returns (activations, labels, positions) where:
      activations  — dict[layer -> (n_selected, d_model)]
      labels       — (n_selected,) bool, one entry per selected position
      positions    — (n_selected,) int, flat sequence index of each row
    """

    def reduce(
        self,
        activations: dict[int, torch.Tensor],
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        # mask is (n, seq_len); nonzero gives (n_selected, 2) — [example_idx, seq_idx]
        selected = mask.nonzero(as_tuple=False)  # (n_selected, 2)
        ex_idx, pos_idx = selected[:, 0], selected[:, 1]

        result = {l: acts[ex_idx, pos_idx] for l, acts in activations.items()}
        return result, labels[ex_idx], pos_idx