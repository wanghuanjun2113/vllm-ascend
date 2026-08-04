# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Ascend optimizations for structured-output bitmasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from vllm.v1.structured_output.backend_types import StructuredOutputOptions

_FULL_MASK = np.int32(-1)
_TOOL_CALL_MASK_STATE_ATTR = "_vllm_ascend_tool_call_mask_state"


@dataclass
class _ToolCallMaskState:
    """Incremental tool-call boundary state for one request.

    The optimization is deliberately enabled only when both boundaries are
    represented by standalone tokenizer tokens. This makes transitions
    unambiguous without decoding the complete output on every scheduler step.
    """

    start_token_ids: frozenset[int]
    end_token_ids: frozenset[int]
    num_scanned_output_tokens: int = 0
    in_tool_call: bool = False

    def accept_token(self, token_id: int) -> None:
        if self.in_tool_call:
            if token_id in self.end_token_ids:
                self.in_tool_call = False
        elif token_id in self.start_token_ids:
            self.in_tool_call = True


def has_grammar_bitmask_constraints(
    grammar_bitmask: npt.NDArray[np.int32],
) -> bool:
    """Return whether at least one logits value is constrained."""

    return bool(grammar_bitmask.size and np.any(grammar_bitmask != _FULL_MASK))


def _encode(tokenizer: Any, text: str) -> list[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return [int(token_id) for token_id in token_ids]


def _create_tool_call_mask_state(
    structured_output_request: Any,
    tokenizer: Any,
) -> _ToolCallMaskState | None:
    try:
        request_type, grammar_spec = structured_output_request.structured_output_key
    except (AttributeError, TypeError, ValueError):
        return None
    if request_type is not StructuredOutputOptions.STRUCTURAL_TAG:
        return None

    try:
        structural_tag = json.loads(grammar_spec)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(structural_tag, dict) or structural_tag.get("type") != "structural_tag":
        return None
    triggered_tags = structural_tag.get("format")
    if (
        not isinstance(triggered_tags, dict)
        or triggered_tags.get("type") != "triggered_tags"
        or triggered_tags.get("at_least_one", False)
    ):
        # Required/forced tool calls are constrained from the first token and
        # must not use this optimization.
        return None

    triggers = triggered_tags.get("triggers")
    tags = triggered_tags.get("tags")
    if not isinstance(triggers, list) or not isinstance(tags, list):
        return None

    start_token_ids: set[int] = set()
    for trigger in triggers:
        if not isinstance(trigger, str) or "tool_call" not in trigger:
            return None
        encoded_trigger = _encode(tokenizer, trigger)
        if not encoded_trigger or tokenizer.decode([encoded_trigger[0]]) != "<tool_call>":
            return None
        start_token_ids.add(encoded_trigger[0])

    end_token_ids: set[int] = set()
    for tag in tags:
        if not isinstance(tag, dict):
            return None
        begin = tag.get("begin")
        end = tag.get("end")
        if (
            not isinstance(begin, str)
            or "tool_call" not in begin
            or not isinstance(end, str)
            or "/tool_call" not in end
        ):
            return None
        encoded_end = _encode(tokenizer, end)
        if not encoded_end or tokenizer.decode([encoded_end[-1]]) != "</tool_call>":
            return None
        end_token_ids.add(encoded_end[-1])

    if not start_token_ids or not end_token_ids:
        return None
    return _ToolCallMaskState(
        start_token_ids=frozenset(start_token_ids),
        end_token_ids=frozenset(end_token_ids),
    )


def _get_tool_call_mask_state(
    request: Any,
    tokenizer: Any,
) -> _ToolCallMaskState | None:
    structured_output_request = getattr(request, "structured_output_request", None)
    if structured_output_request is None:
        return None

    cached = getattr(
        structured_output_request,
        _TOOL_CALL_MASK_STATE_ATTR,
        None,
    )
    if cached is False:
        return None
    if isinstance(cached, _ToolCallMaskState):
        return cached

    state = _create_tool_call_mask_state(structured_output_request, tokenizer)
    setattr(
        structured_output_request,
        _TOOL_CALL_MASK_STATE_ATTR,
        state if state is not None else False,
    )
    return state


def _sync_accepted_output_tokens(
    state: _ToolCallMaskState,
    request: Any,
) -> bool:
    output_token_ids = request.output_token_ids
    num_output_tokens = len(output_token_ids)
    structured_output_request = request.structured_output_request

    reasoner = getattr(structured_output_request, "reasoner", None)
    reasoning_ended = getattr(structured_output_request, "reasoning_ended", None)
    if reasoner is not None and not reasoning_ended:
        # Ignore tool-like strings emitted in reasoning. Once the reasoning end
        # boundary is known, scanning resumes immediately after that boundary.
        state.num_scanned_output_tokens = num_output_tokens
        state.in_tool_call = False
        # Keep upstream's rows unchanged when the reasoning boundary appears
        # inside the current speculative window. It already makes reasoning
        # rows full masks and constrains any post-boundary rows correctly.
        return False

    if state.num_scanned_output_tokens > num_output_tokens:
        # Be conservative if a resumable request replaces its output history.
        state.num_scanned_output_tokens = 0
        state.in_tool_call = False

    scan_start = state.num_scanned_output_tokens
    reasoning_end_index = getattr(
        structured_output_request,
        "reasoning_end_token_index",
        None,
    )
    if reasoning_end_index is not None:
        output_reasoning_end = reasoning_end_index - request.num_prompt_tokens
        scan_start = max(scan_start, output_reasoning_end + 1)

    for token_id in output_token_ids[scan_start:]:
        state.accept_token(int(token_id))
    state.num_scanned_output_tokens = num_output_tokens
    return True


def mask_tool_call_free_text(
    manager: Any,
    requests: dict[str, Any],
    structured_output_request_ids: list[str],
    scheduled_spec_decode_tokens: dict[str, list[int]],
    grammar_bitmask: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    """Make structural-tag masks no-ops outside tool-call regions.

    The row layout mirrors ``StructuredOutputManager.grammar_bitmask``. Draft
    tokens are simulated locally so MTP can enter or leave a tool call within
    one speculative window without mutating the accepted request state.
    """

    is_diffusion = manager.vllm_config.model_config.is_diffusion
    expected_rows = sum(
        len(scheduled_spec_decode_tokens.get(req_id, ()))
        + (0 if is_diffusion and scheduled_spec_decode_tokens.get(req_id, ()) else 1)
        for req_id in structured_output_request_ids
    )
    if expected_rows != grammar_bitmask.shape[0]:
        # Upstream changed the row layout. Keep its original masks rather than
        # risk relaxing a constraint on the wrong logits row.
        return grammar_bitmask

    cumulative_index = 0
    for req_id in structured_output_request_ids:
        draft_token_ids = scheduled_spec_decode_tokens.get(req_id, ())
        has_bonus_row = not (is_diffusion and draft_token_ids)
        request = requests.get(req_id)
        state = _get_tool_call_mask_state(request, manager.tokenizer) if request is not None else None
        if state is None:
            cumulative_index += len(draft_token_ids) + int(has_bonus_row)
            continue

        if not _sync_accepted_output_tokens(state, request):
            cumulative_index += len(draft_token_ids) + int(has_bonus_row)
            continue
        in_tool_call = state.in_tool_call
        has_valid_draft = True

        for token_id in draft_token_ids:
            if not has_valid_draft or not in_tool_call:
                grammar_bitmask[cumulative_index].fill(_FULL_MASK)
            cumulative_index += 1
            if token_id == -1:
                has_valid_draft = False
                continue
            if not has_valid_draft:
                continue
            if in_tool_call:
                if token_id in state.end_token_ids:
                    in_tool_call = False
            elif token_id in state.start_token_ids:
                in_tool_call = True

        if has_bonus_row:
            if not has_valid_draft or not in_tool_call:
                grammar_bitmask[cumulative_index].fill(_FULL_MASK)
            cumulative_index += 1

    return grammar_bitmask
