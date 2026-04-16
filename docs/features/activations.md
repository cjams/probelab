# Probe Training: Activation Collection

This document covers the design for the `train/` module — everything needed
to go from a `ProbeDataset` to training data for a linear probe.

The two main responsibilities are:

1. **Activation collection** — run inference over a `ProbeDataset`, capturing
   hidden-state tensors at specified layers.
2. **Token selection** — choose which token positions from each example's
   sequence contribute to the training signal.

These are kept separate because the right token positions differ by research
question (e.g. refusal probes often use post-instruction tokens; deception
probes use assistant output tokens), while the collection machinery is the same.

---

## Module layout

```
train/
  base.py              # ActivationSpec, ActivationDataset, ActivationCollector, TokenSelector
  huggingface.py       # HFActivationCollector
  transformer_lens.py  # (later)
```

---

## Core types

### `ActivationSpec`

Declares *what* to capture. Kept independent of any model so it can be
serialized and reused across backends.

```python
@dataclass
class ActivationSpec:
    layers: list[int]
    # Layer indices to capture. 0 = embedding output, 1..N = transformer layers.
    # Negative indices resolve relative to total depth (e.g. -1 = last layer).

    component: str = "residual"
    # Which internal signal to hook.
    # "residual" — residual stream (post-layernorm output of each block).
    # HF backend captures this via output_hidden_states=True.
    # Other values ("mlp", "attn_out") are reserved for the transformer_lens backend.
```

### `ActivationDataset`

The output of a collection run. Stores the full sequence of hidden states so
that downstream token selectors can make their own position decisions.

```python
@dataclass
class ActivationDataset:
    activations: dict[int, torch.Tensor]
    # Keyed by resolved layer index.
    # Each tensor: (n_examples, seq_len, d_model).

    labels: torch.Tensor
    # Shape (n_examples,), dtype bool. True = positive class.

    input_ids: torch.Tensor
    # Shape (n_examples, seq_len). Needed by token selectors that reason
    # about token identity (e.g. finding the end of an instruction turn).

    attention_mask: torch.Tensor
    # Shape (n_examples, seq_len). 1 = real token, 0 = padding.
    # Used by selectors to exclude padding positions.

    example_ids: list[int]
    # Original Example.id values, in collection order.

    concept: str
    # Forwarded from the source ProbeDataset.

    model_id: str
    # HF repo ID or local path used during collection.

    spec: ActivationSpec
    # The spec used to produce this dataset.
```

### `ActivationCollector` (abstract)

All backends implement this interface. Call sites depend only on this type.

```python
class ActivationCollector(ABC):
    @abstractmethod
    def collect(
        self,
        dataset: ProbeDataset,
        batch_size: int = 16,
        prompt_fn: Callable[[Example], str] | None = None,
    ) -> ActivationDataset:
        """
        Run inference over `dataset` and capture hidden states at the layers
        specified in this collector's ActivationSpec.

        Args:
            dataset:    Source ProbeDataset.
            batch_size: Examples per forward pass.
            prompt_fn:  Optional function that formats an Example into the
                        string that gets tokenized. Use this to apply chat
                        templates, system prompts, or instruction wrapping.
                        When None, Example.text is tokenized directly.

        Returns:
            ActivationDataset with shape (n_examples, seq_len, d_model) per layer.
        """
        ...
```

`prompt_fn` lives on `collect()` rather than `__init__` so one loaded model
can be reused with different prompt designs without reloading weights.

### `TokenSelector` (abstract)

Separate concern: given an `ActivationDataset`, produce per-example vectors
for probe training. The selector decides which token positions matter and how
to reduce them.

```python
class TokenSelector(ABC):
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
            layer:       Which layer to select from (must be in act_dataset.activations).

        Returns:
            (activations, labels) where:
              activations — (n_training, d_model)
              labels      — (n_training,) bool

        The number of training examples may differ from n_examples when the
        selector expands each example into multiple token positions (e.g.
        treating each post-instruction token as its own training point).
        """
        ...
```

---

## Concrete token selectors

### `LastTokenSelector`

Takes the last real (non-padding) token from each example. The simplest
baseline for decoder-only models.

```python
class LastTokenSelector(TokenSelector):
    # Uses attention_mask to find the final non-padding position per row.
```

### `MeanTokenSelector`

Mean-pools over a contiguous slice of non-padding tokens per example,
optionally skipping the first `skip_prefix` tokens (useful to exclude the
instruction portion of a prompt).

```python
class MeanTokenSelector(TokenSelector):
    def __init__(self, skip_prefix: int = 0):
        ...
    # Mean over real tokens[skip_prefix:] per example.
```

### `OffsetSliceSelector`

Returns one vector per token position in a fixed range relative to the end
of the real sequence (or start). Each token becomes its own training example.
This is the pattern from the refusal probe paper: all post-instruction tokens
contribute independently.

```python
class OffsetSliceSelector(TokenSelector):
    def __init__(self, start: int, end: int | None = None, anchor: str = "end"):
        ...
    # anchor="end": slice relative to last real token (e.g. start=-20 to end=0).
    # anchor="start": slice relative to first real token.
    # Returns (n_examples * slice_len, d_model) and expanded labels.
```

### `AssistantTokenSelector`

Uses `input_ids` and a caller-supplied delimiter token ID (the assistant-turn
start token for the model's chat template) to locate the assistant portion of
each sequence, then mean-pools over those tokens. The pattern from strategic
deception work.

```python
class AssistantTokenSelector(TokenSelector):
    def __init__(self, assistant_token_id: int):
        ...
    # Finds first occurrence of assistant_token_id per example;
    # mean-pools over all tokens from that position to end of real sequence.
    # Caller is responsible for looking up the correct token ID from the tokenizer.
```

---

## HuggingFace backend: `HFActivationCollector`

```python
class HFActivationCollector(ActivationCollector):
    def __init__(
        self,
        model_id: str,
        spec: ActivationSpec,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = False,
    ):
        ...
```

Model and tokenizer are loaded in `__init__` so misconfiguration fails at
construction time. The tokenizer is configured for left-padding by default
(overridable via `padding_side`), since `LastTokenSelector` relies on the last
real token being at a fixed offset from the right edge of the sequence.

```python
class HFActivationCollector(ActivationCollector):
    def __init__(
        self,
        model_id: str,
        spec: ActivationSpec,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = False,
        padding_side: str = "left",
    ):
        ...
```

**`collect()` internals:**

1. Tokenize each batch: `padding=True`, `truncation=True`, `return_tensors="pt"`.
2. Forward pass under `torch.no_grad()` with `output_hidden_states=True`.
3. `model_output.hidden_states` is a tuple of length `n_layers + 1`;
   index 0 = embedding output, index `i` = output of transformer block `i`.
4. Resolve negative layer indices against `len(hidden_states)`.
5. Slice out only requested layers; detach and move to CPU to avoid OOM.
6. Accumulate across batches, then stack into `(n_examples, seq_len, d_model)`.
7. Return `ActivationDataset` with `input_ids` and `attention_mask` included.

**Components beyond `"residual"`** (`"mlp"`, `"attn_out"`, etc.) are not
supported in the HF backend — use the `transformer_lens` backend for those.
Note: when intervention/steering support is added, forward hooks will be
needed regardless of backend; that's a separate concern from probe training.

---

## Intended usage

```python
from train.base import ActivationSpec, LastTokenSelector, OffsetSliceSelector
from train.huggingface import HFActivationCollector
from dataset.loaders.harmbench import HarmBenchLoader
from dataset.loaders.alpaca import AlpacaLoader
import torch

spec = ActivationSpec(layers=list(range(16, 32)))

collector = HFActivationCollector(
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    spec=spec,
    device="cuda",
    dtype=torch.bfloat16,
)

dataset = HarmBenchLoader("standard").load().join(AlpacaLoader().load()).balance()

act_dataset = collector.collect(
    dataset,
    batch_size=8,
    prompt_fn=lambda ex: f"[INST] {ex.text} [/INST]",
)
# act_dataset.activations[28] -> Tensor (n_examples, seq_len, 4096)

# For probe training: pick a token selection strategy
selector = LastTokenSelector()
activations, labels = selector.select(act_dataset, layer=28)
# activations -> (n_examples, 4096)
# labels      -> (n_examples,) bool
```

---

## Decisions

1. **`ActivationDataset` save/load** — yes. `safetensors` for tensors + a
   sidecar JSON for metadata (`concept`, `model_id`, `spec`, `example_ids`).

2. **Padding side** — left-pad by default; overridable via `padding_side` on
   `HFActivationCollector.__init__`.

3. **`component` for HF backend** — only `"residual"` is supported via
   `output_hidden_states=True`. Non-residual hooks are deferred to the
   `transformer_lens` backend. Forward hook support will be added when
   intervention/steering is built out.

4. **`AssistantTokenSelector` delimiter** — caller-supplied `assistant_token_id`.
   Callers look it up from the tokenizer themselves.
