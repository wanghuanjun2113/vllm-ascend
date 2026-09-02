import gc
import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from safetensors import safe_open
from vllm.logger import logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead


COMPACT_OUTPUT_HEAD_CONFIG_ENV = "VLLM_ASCEND_COMPACT_OUTPUT_HEAD_CONFIG"


class CompactLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        original_vocab_size: int,
        compact_vocab_size: int,
        compact_to_original_ids: torch.Tensor,
    ) -> None:
        super().__init__(original_vocab_size)
        self.compact_vocab_size = compact_vocab_size
        self.register_buffer(
            "compact_to_original_ids",
            compact_to_original_ids.to(dtype=torch.int64),
            persistent=False,
        )

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
        embedding_bias: torch.Tensor | None,
    ) -> torch.Tensor | None:
        compact_logits = lm_head.quant_method.apply(
            lm_head,
            hidden_states,
            bias=embedding_bias,
        )
        compact_logits = self._gather_logits(compact_logits)
        if compact_logits is None:
            return None
        compact_logits = compact_logits[..., : self.compact_vocab_size]

        full_shape = (*compact_logits.shape[:-1], self.vocab_size)
        full_logits = torch.full(
            full_shape,
            float("-inf"),
            dtype=compact_logits.dtype,
            device=compact_logits.device,
        )
        full_logits.index_copy_(
            -1,
            self.compact_to_original_ids,
            compact_logits,
        )
        return full_logits


@dataclass
class CompactOutputVocabState:
    original_vocab_size: int
    compact_vocab_size: int
    compact_to_original_ids: torch.Tensor
    logits_processor: CompactLogitsProcessor
    lm_head: ParallelLMHead


def _get_language_model(model: nn.Module) -> nn.Module:
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        language_model = model
    if not hasattr(language_model, "lm_head"):
        raise TypeError(
            f"Model {type(model).__name__} does not expose a language-model lm_head."
        )
    return language_model


def apply_compact_output_head(
    model: nn.Module,
    device: torch.device,
) -> CompactOutputVocabState | None:
    config_path = os.getenv(COMPACT_OUTPUT_HEAD_CONFIG_ENV, "").strip()
    if not config_path:
        return None

    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    original_vocab_size = int(config["original_vocab_size"])
    compact_vocab_size = int(config["compact_vocab_size"])
    mapping = config["compact_to_original_ids"]
    weight_path = config["weight_path"]
    weight_key = config.get("weight_key", "lm_head.weight")

    if len(mapping) != compact_vocab_size:
        raise ValueError(
            f"Compact mapping has {len(mapping)} entries, expected {compact_vocab_size}."
        )
    if mapping != sorted(set(mapping)):
        raise ValueError("Compact-to-original token IDs must be sorted and unique.")
    if not mapping or mapping[0] < 0 or mapping[-1] >= original_vocab_size:
        raise ValueError(
            "Compact-to-original token IDs are outside the original vocabulary."
        )

    language_model = _get_language_model(model)
    old_head = language_model.lm_head
    if old_head.num_embeddings != original_vocab_size:
        raise ValueError(
            f"Model LM Head has {old_head.num_embeddings} rows, "
            f"expected {original_vocab_size}."
        )

    compact_head = ParallelLMHead(
        compact_vocab_size,
        old_head.embedding_dim,
        params_dtype=old_head.weight.dtype,
        quant_config=None,
        prefix="compact_lm_head",
    ).to(device)
    with safe_open(weight_path, framework="pt", device="cpu") as weights:
        compact_weight = weights.get_tensor(weight_key)
    expected_shape = (compact_vocab_size, old_head.embedding_dim)
    if tuple(compact_weight.shape) != expected_shape:
        raise ValueError(
            f"Compact LM Head weight shape is {tuple(compact_weight.shape)}, "
            f"expected {expected_shape}."
        )
    compact_head.weight_loader(compact_head.weight, compact_weight)
    compact_head.quant_method.process_weights_after_loading(compact_head)

    mapping_tensor = torch.tensor(mapping, dtype=torch.int64, device=device)
    processor = CompactLogitsProcessor(
        original_vocab_size,
        compact_vocab_size,
        mapping_tensor,
    ).to(device)
    language_model.lm_head = compact_head
    language_model.logits_processor = processor

    del old_head
    del compact_weight
    gc.collect()
    torch.npu.empty_cache()
    logger.info(
        "Enabled compact output LM Head from %s: compact=%d, original=%d",
        weight_path,
        compact_vocab_size,
        original_vocab_size,
    )
    return CompactOutputVocabState(
        original_vocab_size=original_vocab_size,
        compact_vocab_size=compact_vocab_size,
        compact_to_original_ids=mapping_tensor,
        logits_processor=processor,
        lm_head=compact_head,
    )


def apply_compact_output_processor_to_drafter(
    drafter,
    state: CompactOutputVocabState | None,
) -> None:
    if drafter is None or state is None:
        return
    draft_model = getattr(drafter, "model", None)
    if draft_model is None or not hasattr(draft_model, "lm_head"):
        raise TypeError("Compact output vocabulary requires a draft model with lm_head.")
    old_draft_head = draft_model.lm_head
    draft_model.lm_head = state.lm_head
    draft_model.logits_processor = state.logits_processor
    del old_draft_head
    gc.collect()
    torch.npu.empty_cache()
    logger.info(
        "Attached compact output logits processor to %s",
        type(draft_model).__name__,
    )
