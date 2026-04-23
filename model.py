from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any
    model_id: str


@dataclass
class HFModelBundle(ModelBundle):
    model: Any  # PreTrainedModel
    tokenizer: Any  # PreTrainedTokenizer


@dataclass
class TLModelBundle(ModelBundle):
    model: Any  # HookedTransformer
    tokenizer: Any  # PreTrainedTokenizer


# Model ID prefixes that require AutoProcessor + AutoModelForImageTextToText.
_IMAGE_TEXT_TO_TEXT_PREFIXES = (
    "google/gemma-4",
)


def load_hf_bundle(
    model_id: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    trust_remote_code: bool = False,
    padding_side: str = "left",
) -> HFModelBundle:
    """
    Load a HuggingFace model + tokenizer into a single bundle.

    Centralises the load so the same bundle can be reused across
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
        HFModelBundle holding the loaded model, tokenizer, and originating
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

    return HFModelBundle(model=model, tokenizer=tokenizer, model_id=model_id)
