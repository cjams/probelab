import pytest
import torch

from probelab.train.activation import ActivationDataset, ActivationSpec
from probelab.train.token import (
    AllTokenSelector,
    EachPositionReducer,
    LastNTokenSelector,
    MeanReducer,
    PostInstructionTokenSelector,
    _resolve_layers,
    get_post_instruction_tokens,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_act_dataset(
    n: int = 2,
    seq_len: int = 5,
    d_model: int = 4,
    layers: list[int] = None,
    attention_mask: torch.Tensor = None,
) -> ActivationDataset:
    if layers is None:
        layers = [0, 1]

    # Layer l has values offset by l*1000 so layers are trivially distinguishable.
    activations = {
        l: (torch.arange(n * seq_len * d_model, dtype=torch.float).reshape(n, seq_len, d_model) + l * 1000)
        for l in layers
    }

    labels = torch.tensor([i % 2 == 0 for i in range(n)])

    if attention_mask is None:
        # Default: first column is padding, rest real — shape (n, seq_len).
        attention_mask = torch.ones(n, seq_len, dtype=torch.long)
        attention_mask[:, 0] = 0

    input_ids = torch.zeros(n, seq_len, dtype=torch.long)

    return ActivationDataset(
        activations=activations,
        labels=labels,
        input_ids=input_ids,
        attention_mask=attention_mask,
        example_ids=list(range(n)),
        concept="test",
        model_id="test-model",
        spec=ActivationSpec(targets=layers),
    )


class MockTokenizer:
    """Fake tokenizer whose template wraps content in <u>...</u><a>."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        content = messages[0]["content"]
        suffix = "<|assistant|>" if add_generation_prompt else ""
        
        return f"<|user|>{content}<|end_user|>{suffix}"

    def __call__(self, text, add_special_tokens=True):
        # Each character becomes one token ID (its ordinal).
        return {"input_ids": [ord(c) for c in text]}


class BrokenTokenizer:
    """Tokenizer that raises when apply_chat_template is called."""

    def apply_chat_template(self, *args, **kwargs):
        raise RuntimeError("no template here")


class SentinelStrippingTokenizer:
    """Template that silently drops the user content — sentinel never appears."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "<|user|><|end_user|><|assistant|>"

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(c) for c in text]}


# ---------------------------------------------------------------------------
# get_post_instruction_tokens
# ---------------------------------------------------------------------------

class TestGetPostInstructionTokens:
    def test_returns_token_ids_by_default(self):
        tok = MockTokenizer()
        ids = get_post_instruction_tokens(tok)

        assert isinstance(ids, list)
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)

    def test_token_ids_correspond_to_post_sentinel_text(self):
        tok = MockTokenizer()
        ids = get_post_instruction_tokens(tok, tokenize=True)
        text = get_post_instruction_tokens(tok, tokenize=False)

        # The two calls must be consistent: tokenizing the text gives the same IDs.
        expected = [ord(c) for c in text]
        assert ids == expected

    def test_returns_text_when_tokenize_false(self):
        tok = MockTokenizer()
        result = get_post_instruction_tokens(tok, tokenize=False)

        assert isinstance(result, str)
        # The post-sentinel portion is everything after the sentinel in the template.
        assert result == "<|end_user|><|assistant|>"

    def test_raises_if_no_chat_template(self):
        with pytest.raises(ValueError, match="chat template"):
            get_post_instruction_tokens(BrokenTokenizer())

    def test_raises_if_sentinel_not_in_output(self):
        with pytest.raises(ValueError, match="Sentinel not found"):
            get_post_instruction_tokens(SentinelStrippingTokenizer())


# ---------------------------------------------------------------------------
# _resolve_layers
# ---------------------------------------------------------------------------

class TestResolveLayers:
    def test_all_returns_sorted_keys(self):
        ds = make_act_dataset(layers=[2, 0, 5])
        result = _resolve_layers(ds, "all")

        assert result == [0, 2, 5]

    def test_int_returns_singleton(self):
        ds = make_act_dataset(layers=[0, 1, 2])
        result = _resolve_layers(ds, 1)

        assert result == [1]

    def test_list_returns_sorted_and_deduplicated(self):
        ds = make_act_dataset(layers=[0, 1, 2, 3])
        result = _resolve_layers(ds, [3, 1, 1, 2])

        assert result == [1, 2, 3]


# ---------------------------------------------------------------------------
# AllTokenSelector
# ---------------------------------------------------------------------------

class TestAllTokenSelector:
    def test_mask_matches_attention_mask_as_bool(self):
        attn = torch.tensor([[0, 1, 1, 1, 1], [0, 0, 1, 1, 1]], dtype=torch.long)
        ds = make_act_dataset(attention_mask=attn)
        selector = AllTokenSelector()

        _, _, mask = selector.select(ds, layer=0)

        assert mask.dtype == torch.bool
        assert torch.equal(mask, attn.bool())

    def test_padding_positions_are_false_real_tokens_are_true(self):
        attn = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]], dtype=torch.long)
        ds = make_act_dataset(attention_mask=attn)
        selector = AllTokenSelector()

        _, _, mask = selector.select(ds, layer=0)

        # First two positions of row 0 are padding.
        assert not mask[0, 0].item()
        assert not mask[0, 1].item()
        assert mask[0, 2].item()

        # First position of row 1 is padding.
        assert not mask[1, 0].item()
        assert mask[1, 1].item()

    def test_returns_only_requested_layers(self):
        ds = make_act_dataset(layers=[0, 1, 2])
        selector = AllTokenSelector()

        acts, _, _ = selector.select(ds, layer=[0, 2])

        assert set(acts.keys()) == {0, 2}
        assert 1 not in acts

    def test_labels_are_unchanged(self):
        ds = make_act_dataset()
        selector = AllTokenSelector()

        _, labels, _ = selector.select(ds, layer=0)

        assert torch.equal(labels, ds.labels)


# ---------------------------------------------------------------------------
# LastNTokenSelector
# ---------------------------------------------------------------------------

class TestLastNTokenSelector:
    def test_n_less_than_1_raises(self):
        with pytest.raises(ValueError):
            LastNTokenSelector(n=0)

    def test_n1_selects_only_last_position(self):
        # Left-padded: real tokens are right-aligned, so last real token is always at seq_len-1.
        attn = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]], dtype=torch.long)
        ds = make_act_dataset(attention_mask=attn)
        selector = LastNTokenSelector(n=1)

        _, _, mask = selector.select(ds, layer=0)

        # Exactly one True per row, at the last position.
        assert mask.sum(dim=1).tolist() == [1, 1]
        assert mask[0, 4].item()
        assert mask[1, 4].item()

        # All other positions are False.
        assert not mask[0, :4].any().item()
        assert not mask[1, :4].any().item()

    def test_n2_selects_last_two_positions(self):
        attn = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]], dtype=torch.long)
        ds = make_act_dataset(attention_mask=attn)
        selector = LastNTokenSelector(n=2)

        _, _, mask = selector.select(ds, layer=0)

        assert mask.sum(dim=1).tolist() == [2, 2]
        assert torch.equal(mask[0], torch.tensor([False, False, False, True, True]))
        assert torch.equal(mask[1], torch.tensor([False, False, False, True, True]))

    def test_n_exceeds_sequence_length_clips_to_actual_length(self):
        # Row 0 has only 1 real token; requesting n=3 should clip to 1.
        attn = torch.tensor([[0, 0, 0, 0, 1], [1, 1, 1, 1, 1]], dtype=torch.long)
        ds = make_act_dataset(attention_mask=attn)
        selector = LastNTokenSelector(n=3)

        _, _, mask = selector.select(ds, layer=0)

        # Row 0: clipped to 1.
        assert mask[0].sum().item() == 1
        assert mask[0, 4].item()

        # Row 1: all 5 real tokens, n=3 → last 3.
        assert mask[1].sum().item() == 3
        assert torch.equal(mask[1], torch.tensor([False, False, True, True, True]))


# ---------------------------------------------------------------------------
# PostInstructionTokenSelector
# ---------------------------------------------------------------------------

class TestPostInstructionTokenSelector:
    def _make_tokenizer_with_n_post(self, n_post: int):
        """Return a mock tokenizer whose post-instruction region has n_post tokens."""

        class _Tok:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                content = messages[0]["content"]
                # Post-sentinel text is exactly n_post characters 'X'.
                return f"<|user|>{content}" + "X" * n_post

            def __call__(self, text, add_special_tokens=True):
                return {"input_ids": list(range(len(text)))}

        return _Tok()

    def test_marks_last_n_post_positions_true_for_all_rows(self):
        n_post = 3
        tok = self._make_tokenizer_with_n_post(n_post)
        ds = make_act_dataset(n=2, seq_len=8, layers=[0])
        selector = PostInstructionTokenSelector(tok)

        _, _, mask = selector.select(ds, layer=0)

        assert selector.n_post == n_post

        # Last n_post columns True everywhere.
        assert torch.equal(mask[:, -n_post:], torch.ones(2, n_post, dtype=torch.bool))

        # Everything before is False.
        assert not mask[:, :-n_post].any().item()

    def test_raises_if_post_tokens_is_empty(self):
        class EmptyPostTok:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                # Sentinel is present but nothing follows it.
                content = messages[0]["content"]
                return f"<|user|>{content}"

            def __call__(self, text, add_special_tokens=True):
                return {"input_ids": []}

        with pytest.raises(ValueError, match="no tokens"):
            PostInstructionTokenSelector(EmptyPostTok())


# ---------------------------------------------------------------------------
# MeanReducer
# ---------------------------------------------------------------------------

class TestMeanReducer:
    def test_single_selected_position_returns_that_vector_exactly(self):
        n, seq_len, d_model = 2, 5, 4
        ds = make_act_dataset(n=n, seq_len=seq_len, d_model=d_model, layers=[0])

        # Select exactly position 3 for every example.
        mask = torch.zeros(n, seq_len, dtype=torch.bool)
        mask[:, 3] = True

        acts, labels, _ = AllTokenSelector().select(ds, layer=0)
        reduced_acts, reduced_labels = MeanReducer().reduce(acts, labels, mask)

        # With a single selected position, mean == that position's vector.
        for i in range(n):
            assert torch.allclose(reduced_acts[0][i], ds.activations[0][i, 3])

        assert torch.equal(reduced_labels, labels)

    def test_mean_pools_two_positions_correctly(self):
        n, seq_len, d_model = 1, 4, 2
        # Hand-craft activations for easy mental arithmetic.
        activations = {0: torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])}
        labels = torch.tensor([True])
        mask = torch.tensor([[False, True, True, False]])  # positions 1 and 2

        reduced_acts, _ = MeanReducer().reduce(activations, labels, mask)

        expected = torch.tensor([[4.0, 5.0]])  # mean of [3,4] and [5,6]: (3+5)/2=4, (4+6)/2=5
        assert torch.allclose(reduced_acts[0], expected)


# ---------------------------------------------------------------------------
# EachPositionReducer
# ---------------------------------------------------------------------------

class TestEachPositionReducer:
    def test_each_selected_position_becomes_its_own_row(self):
        n, seq_len, d_model = 2, 4, 3
        # Activations: example i, position j, dim k = i*100 + j*10 + k (float).
        raw = torch.zeros(n, seq_len, d_model)
        for i in range(n):
            for j in range(seq_len):
                for k in range(d_model):
                    raw[i, j, k] = float(i * 100 + j * 10 + k)

        activations = {0: raw}
        labels = torch.tensor([True, False])

        # Select position 2 from example 0, positions 1 and 3 from example 1.
        mask = torch.tensor([
            [False, False, True, False],
            [False, True, False, True],
        ])

        reduced_acts, reduced_labels, positions = EachPositionReducer().reduce(activations, labels, mask)

        # 3 positions selected total.
        assert reduced_acts[0].shape == (3, d_model)

        # Row 0: example 0, position 2.
        assert torch.allclose(reduced_acts[0][0], raw[0, 2])

        # Row 1: example 1, position 1.
        assert torch.allclose(reduced_acts[0][1], raw[1, 1])

        # Row 2: example 1, position 3.
        assert torch.allclose(reduced_acts[0][2], raw[1, 3])

    def test_labels_and_positions_match_selected_rows(self):
        n, seq_len, d_model = 2, 4, 2
        activations = {0: torch.zeros(n, seq_len, d_model)}
        labels = torch.tensor([True, False])

        # Select position 0 from example 0, position 2 from example 1.
        mask = torch.tensor([
            [True, False, False, False],
            [False, False, True, False],
        ])

        _, reduced_labels, positions = EachPositionReducer().reduce(activations, labels, mask)

        assert reduced_labels.tolist() == [True, False]
        assert positions.tolist() == [0, 2]