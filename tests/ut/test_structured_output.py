# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
from vllm.v1.structured_output.backend_types import StructuredOutputOptions

from vllm_ascend.structured_output import (
    has_grammar_bitmask_constraints,
    mask_tool_call_free_text,
)

TOOL_START_TOKEN_ID = 100
TOOL_END_TOKEN_ID = 101


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        if text.startswith("<tool_call>"):
            return [TOOL_START_TOKEN_ID, 10, 11]
        if text.endswith("</tool_call>"):
            return [12, 13, TOOL_END_TOKEN_ID]
        raise AssertionError(f"Unexpected text: {text}")

    def decode(self, token_ids: list[int]) -> str:
        assert len(token_ids) == 1
        if token_ids[0] == TOOL_START_TOKEN_ID:
            return "<tool_call>"
        if token_ids[0] == TOOL_END_TOKEN_ID:
            return "</tool_call>"
        raise AssertionError(f"Unexpected token: {token_ids[0]}")


def make_structural_tag(*, at_least_one: bool = False) -> str:
    return json.dumps(
        {
            "type": "structural_tag",
            "format": {
                "type": "triggered_tags",
                "triggers": ["<tool_call>\n<function="],
                "tags": [
                    {
                        "type": "tag",
                        "begin": "<tool_call>\n<function=get_weather>\n",
                        "content": {
                            "type": "json_schema",
                            "json_schema": {"type": "object"},
                        },
                        "end": "\n</function>\n</tool_call>",
                    }
                ],
                "at_least_one": at_least_one,
            },
        }
    )


def make_request(
    output_token_ids: list[int],
    *,
    request_type: StructuredOutputOptions = StructuredOutputOptions.STRUCTURAL_TAG,
    grammar_spec: str | None = None,
    reasoner=None,
    reasoning_ended: bool | None = None,
    reasoning_end_token_index: int | None = None,
):
    structured_output_request = SimpleNamespace(
        structured_output_key=(
            request_type,
            grammar_spec if grammar_spec is not None else make_structural_tag(),
        ),
        reasoner=reasoner,
        reasoning_ended=reasoning_ended,
        reasoning_end_token_index=reasoning_end_token_index,
    )
    return SimpleNamespace(
        output_token_ids=output_token_ids,
        num_prompt_tokens=10,
        structured_output_request=structured_output_request,
    )


def make_manager():
    return SimpleNamespace(
        tokenizer=FakeTokenizer(),
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(is_diffusion=False)),
    )


def apply_mask(request, mask, draft_token_ids=None):
    req_id = "request"
    return mask_tool_call_free_text(
        make_manager(),
        {req_id: request},
        [req_id],
        {req_id: draft_token_ids or []},
        mask,
    )


def test_free_text_masks_are_full_before_and_after_tool_call():
    output_token_ids: list[int] = []
    request = make_request(output_token_ids)

    free_before = apply_mask(request, np.zeros((1, 4), dtype=np.int32))
    assert np.all(free_before == -1)

    output_token_ids.extend([7, TOOL_START_TOKEN_ID])
    in_tool_call = apply_mask(request, np.zeros((1, 4), dtype=np.int32))
    assert np.all(in_tool_call == 0)

    output_token_ids.extend([8, TOOL_END_TOKEN_ID, 9])
    free_after = apply_mask(request, np.zeros((1, 4), dtype=np.int32))
    assert np.all(free_after == -1)


def test_mtp_rows_follow_tool_call_boundaries_without_persisting_drafts():
    request = make_request([7])
    mask = np.zeros((4, 4), dtype=np.int32)

    result = apply_mask(
        request,
        mask,
        [TOOL_START_TOKEN_ID, 8, TOOL_END_TOKEN_ID],
    )

    assert np.all(result[0] == -1)
    assert np.all(result[1] == 0)
    assert np.all(result[2] == 0)
    assert np.all(result[3] == -1)

    next_step = apply_mask(request, np.zeros((1, 4), dtype=np.int32))
    assert np.all(next_step == -1)


def test_mtp_invalid_draft_disables_following_mask_rows():
    request = make_request([TOOL_START_TOKEN_ID])

    result = apply_mask(
        request,
        np.zeros((4, 4), dtype=np.int32),
        [8, -1, 9],
    )

    assert np.all(result[0] == 0)
    assert np.all(result[1] == 0)
    assert np.all(result[2] == -1)
    assert np.all(result[3] == -1)


def test_reasoning_tool_marker_does_not_activate_tool_mask():
    output_token_ids = [TOOL_START_TOKEN_ID]
    request = make_request(
        output_token_ids,
        reasoner=object(),
        reasoning_ended=False,
    )

    reasoning = apply_mask(request, np.zeros((1, 4), dtype=np.int32))
    assert np.all(reasoning == 0)

    output_token_ids.extend([77, TOOL_START_TOKEN_ID])
    request.structured_output_request.reasoning_ended = True
    request.structured_output_request.reasoning_end_token_index = 10
    after_reasoning = apply_mask(request, np.zeros((1, 4), dtype=np.int32))
    assert np.all(after_reasoning == 0)


def test_non_auto_or_non_structural_grammars_are_unchanged():
    forced_request = make_request(
        [],
        grammar_spec=make_structural_tag(at_least_one=True),
    )
    forced = apply_mask(forced_request, np.zeros((1, 4), dtype=np.int32))
    assert np.all(forced == 0)

    json_request = make_request(
        [],
        request_type=StructuredOutputOptions.JSON,
        grammar_spec='{"type": "object"}',
    )
    json_mask = apply_mask(json_request, np.zeros((1, 4), dtype=np.int32))
    assert np.all(json_mask == 0)

    nested_triggered_tags = json.loads(make_structural_tag())["format"]
    unknown_structural_request = make_request(
        [],
        grammar_spec=json.dumps(
            {
                "type": "structural_tag",
                "format": {
                    "type": "sequence",
                    "elements": [nested_triggered_tags],
                },
            }
        ),
    )
    unknown_mask = apply_mask(
        unknown_structural_request,
        np.zeros((1, 4), dtype=np.int32),
    )
    assert np.all(unknown_mask == 0)


def test_bitmask_constraint_detection():
    full_mask = np.full((2, 4), -1, dtype=np.int32)
    assert not has_grammar_bitmask_constraints(full_mask)

    full_mask[1, 2] = 0
    assert has_grammar_bitmask_constraints(full_mask)
