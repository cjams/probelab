from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ModelHandle:
    model: Any
    tokenizer: Any
    model_id: str


@dataclass
class HFModelHandle(ModelHandle):
    model: Any  # PreTrainedModel
    tokenizer: Any  # PreTrainedTokenizer


@dataclass
class TLModelHandle(ModelHandle):
    model: Any  # HookedTransformer
    tokenizer: Any  # PreTrainedTokenizer


# Model ID prefixes that require AutoProcessor + AutoModelForImageTextToText.
_IMAGE_TEXT_TO_TEXT_PREFIXES = (
    "google/gemma-4",
)


def load_hf(
    model_id: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    trust_remote_code: bool = False,
    padding_side: str = "left",
) -> HFModelHandle:
    """
    Load a HuggingFace model + tokenizer into a single handle.

    Centralises the load so the same handle can be reused across
    HFActivationCollector, HFResponseCollector, and HFInterventionBackend
    without paying the load cost more than once.

    The tokenizer is configured for left-padding by default so that all real
    tokens are contiguous at the right edge of each sequence — a requirement
    for LastTokenSelector and OffsetSliceTokenSelector to work correctly.

    Args:
        model_id:          HuggingFace repo ID or local path.
        dtype:             Model weight dtype. bfloat16 for most modern GPUs.
        device:            Device passed to device_map.
        trust_remote_code: Forwarded to from_pretrained.
        padding_side:      Tokenizer padding side. "left" (default) keeps real
                           tokens right-aligned, which token selectors assume.

    Returns:
        HFModelHandle holding the loaded model, tokenizer, and originating
        model_id.
    """
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
    )

    is_image_text = any(model_id.startswith(p) for p in _IMAGE_TEXT_TO_TEXT_PREFIXES)

    if is_image_text:
        tokenizer = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )

        underlying_tok = tokenizer.tokenizer
        underlying_tok.padding_side = padding_side

        if underlying_tok.pad_token is None:
            underlying_tok.pad_token = underlying_tok.eos_token

        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            device_map=device,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )
        tokenizer.padding_side = padding_side

        # Some tokenizers (e.g. LLaMA) have no pad token by default.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            device_map=device,
        )

    model.eval()

    return HFModelHandle(model=model, tokenizer=tokenizer, model_id=model_id)


def load_tl(
    model_id: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    padding_side: str = "left",
    fold_ln: bool = True,
    center_writing_weights: bool = True,
    center_unembed: bool = True,
    **from_pretrained_kwargs,
) -> TLModelHandle:
    """
    Load a TransformerLens HookedTransformer directly.

    HookedTransformer.from_pretrained pulls weights from the HF Hub (or your
    HF cache) using its own canonical model names. If TL's name for the model
    differs from the HF repo ID you have cached, it will re-download under
    the TL name — symlink the cache directories if you want to share weights.

    Args:
        model_id:               TL-canonical model name (e.g.
                                "meta-llama/Llama-3.1-8B-Instruct"). Stored
                                on the returned handle.
        dtype:                  Model weight dtype.
        device:                 Device the model is loaded onto.
        padding_side:           Tokenizer padding side. "left" keeps real
                                tokens right-aligned.
        fold_ln:                Fold LayerNorm weights into following linear
                                layers. TransformerLens default.
        center_writing_weights: Center the writing weights so the residual
                                stream has zero mean at each position.
        center_unembed:         Center the unembedding matrix rows.
        from_pretrained_kwargs: Additional kwargs forwarded to
                                HookedTransformer.from_pretrained.

    Returns:
        TLModelHandle holding the HookedTransformer, its tokenizer, and the
        TL model_id used for the load.
    """
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        model_id,
        dtype=dtype,
        device=device,
        fold_ln=fold_ln,
        center_writing_weights=center_writing_weights,
        center_unembed=center_unembed,
        **from_pretrained_kwargs,
    )

    tokenizer = model.tokenizer
    tokenizer.padding_side = padding_side

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Left-padding semantics in HookedTransformer are driven by
    # tokenizer.padding_side: when it is "left", forward() auto-infers an
    # attention_mask from pad_token_id on every call that doesn't pass one
    # explicitly. That matches HF, where generate() consumes our explicit
    # attention_mask.

    model.eval()

    return TLModelHandle(model=model, tokenizer=tokenizer, model_id=model_id)
