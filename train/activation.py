import json
import torch

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Literal

from probelab.dataset.base import Example, ProbeDataset

"""
Declares which parts of the model to capture activations from.

Named targets:
  "all"              — every hidden state the backend exposes.
  "all_transformer"  — all transformer blocks, excluding pre-transformer
                       components (embedding, positional encoding).
  "embedding"        — token embedding output only.
  "positional"       — positional encoding output only (where the backend
                       exposes it as a separate state).

Integer list:
  Explicit positional indices into the sequence of hidden states returned by
  the backend. The exact mapping of index -> model component is
  backend-specific and may become more fine-grained as sub-component support
  is added. Negative indices are resolved at collection time.
"""
ActivationTarget = (
    list[int]
    | Literal["all", "all_transformer", "embedding", "positional"]
)

@dataclass
class ActivationSpec:
    """
    Declares which activations to capture during a collection run.

    Kept independent of any model or backend so it can be serialized,
    compared, and reused across collectors.
    """

    """Which parts of the model to capture. See ActivationTarget."""
    targets: ActivationTarget

    """
    Which internal signal to capture. Only "resid_post" is portable across
    backends; the rest require the transformer_lens backend.

    Layer-index convention for all components (matches the hidden_states
    convention used by HFActivationCollector):
      - idx 0         -> pre-block-0 state (embedding output).
      - idx i in 1..N -> the state "around" transformer block i-1. Which
                         specific state depends on the component.

    Supported values:
      "resid_post" — residual stream after block i-1.
                     idx 0 = embedding output. HF + TL.
      "resid_pre"  — residual entering block i-1.
                     idx 0 = embedding output (same as "resid_post" at 0).
                     TL only.
      "resid_mid"  — residual between attn and mlp of block i-1 (after attn_out
                     has been added). idx 0 invalid. TL only.
      "mlp_out"    — mlp's additive contribution from block i-1.
                     idx 0 invalid. TL only.
      "attn_out"   — attn's additive contribution from block i-1.
                     idx 0 invalid. TL only.
    """
    component: str = "resid_post"

    def resolve_targets(self, n_states: int) -> list[int]:
        """
        Expand targets to a concrete list of indices given the number of
        hidden states the backend returned (len(output.hidden_states)).

        Args:
            n_states: Total number of hidden states available, including any
                      pre-transformer states (e.g. embedding, positional).
                      Backend-specific; callers should pass the actual count.

        Returns:
            Sorted list of non-negative indices, deduplicated.
        """
        if isinstance(self.targets, list):
            return sorted(set(i % n_states for i in self.targets))
        if self.targets == "all":
            return list(range(n_states))
        if self.targets == "all_transformer":
            # Convention: index 0 = embedding, 1 = positional (if separate),
            # then transformer blocks. For backends that merge embedding +
            # positional into a single state, this skips only index 0.
            # Callers may need to override if their backend differs.
            return list(range(1, n_states))
        if self.targets == "embedding":
            return [0]
        if self.targets == "positional":
            # Only meaningful when the backend exposes positional encoding as a
            # distinct hidden state. Conventionally index 1; will need revisiting
            # for backends that fuse it with the embedding.
            return [1]
        raise ValueError(f"Unknown ActivationTarget: {self.targets!r}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ActivationSpec":
        return cls(**d)

@dataclass
class ActivationDataset:
    """
    Output of a collection run. Stores full per-token hidden states so that
    downstream TokenSelectors can apply their own position logic.
    """

    """
    Keyed by resolved layer index. Note the indices are absolute
    layer indices relative to the model (so they don't necessarily start
    at zero).

    Each tensor has shape (n_examples, seq_len, d_model).
    """
    activations: dict[int, torch.Tensor]

    """Shape (n_examples,), dtype bool. True = positive class."""
    labels: torch.Tensor

    """
    Shape (n_examples, seq_len).
    Included so TokenSelectors can reason about token identity
    (e.g. locating the assistant-turn delimiter).
    """
    input_ids: torch.Tensor

    """
    Shape (n_examples, seq_len). 1 = real token, 0 = padding.
    Used by selectors to exclude padding positions.
    """
    attention_mask: torch.Tensor

    """Original Example.id values, in collection order."""
    example_ids: list[int]

    """Forwarded from ProbeDataset"""
    concept: str

    """HF repo ID or local path used during collection."""
    model_id: str

    """The spec of activations collected"""
    spec: ActivationSpec

    def n_layers(self) -> int:
        """Return total number of layers collected"""
        return len(self.activations)

    def save(self, path: Path) -> None:
        """
        Serialize to disk. Tensors go to <path>/activations.pt;
        metadata goes to <path>/metadata.json.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "activations": self.activations,
                "labels": self.labels,
                "input_ids": self.input_ids,
                "attention_mask": self.attention_mask,
            },
            path / "activations.pt",
        )

        with open(path / "metadata.json", "w") as f:
            json.dump(
                {
                    "example_ids": self.example_ids,
                    "concept": self.concept,
                    "model_id": self.model_id,
                    "spec": self.spec.to_dict(),
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: Path) -> "ActivationDataset":
        """Load a previously saved ActivationDataset from disk."""
        path = Path(path)

        tensors = torch.load(path / "activations.pt", weights_only=True)

        with open(path / "metadata.json") as f:
            meta = json.load(f)

        return cls(
            activations=tensors["activations"],
            labels=tensors["labels"],
            input_ids=tensors["input_ids"],
            attention_mask=tensors["attention_mask"],
            example_ids=meta["example_ids"],
            concept=meta["concept"],
            model_id=meta["model_id"],
            spec=ActivationSpec.from_dict(meta["spec"]),
        )

class ActivationCollector(ABC):
    """
    Abstract base for gathering model activations over a ProbeDataset.

    Subclasses handle model loading and inference details.
    Call sites depend only on this interface.
    """

    @abstractmethod
    def collect(
        self,
        dataset: ProbeDataset,
        batch_size: int = 16,
        prompt_fn: Callable[[Example], str] | None = None,
    ) -> ActivationDataset:
        """
        Run inference over `dataset` and return hidden states for the layers
        specified in this collector's ActivationSpec.

        Args:
            dataset:    Source ProbeDataset.
            batch_size: Number of examples per forward pass.
            prompt_fn:  Optional function that maps an Example to the string
                        that gets tokenized. Use this to apply chat templates,
                        system prompts, or instruction wrapping. When None,
                        Example.text is tokenized directly.

        Returns:
            ActivationDataset with tensors of shape (n_examples, seq_len, d_model)
            per requested layer.
        """
        ...

    @abstractmethod
    def d_model() -> int:
        """
        Returns the dimension of the residual stream
        """
        ...

    @abstractmethod
    def d_vocab() -> int:
        """
        Returns the number of tokens
        """
        ...