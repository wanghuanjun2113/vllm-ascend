#!/usr/bin/env python3
"""Benchmark strict auto tool calls with long free-text regions.

The request-specific chat template intentionally permits this output shape::

    reasoning -> long free text -> tool call -> long free text

Start ``vllm serve`` with ``--trust-request-chat-template`` before running this
benchmark. The script rejects samples that do not contain both free-text
regions or whose free-text token count is not much larger than the tool call.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

SCOPED_TOOL_CHAT_TEMPLATE = r"""
{%- if tools %}
{{- '<|im_start|>system\n# Tools\n\nYou have access to these functions:\n<tools>' }}
{%- for tool in benchmark_tools %}
{{- '\n' + (tool | tojson) }}
{%- endfor %}
{{- '\n</tools>\n\nA tool call must use this exact XML form:\n'
    + '<tool_call>\n<function=FUNCTION_NAME>\n'
    + '<parameter=PARAMETER_NAME>\nVALUE\n</parameter>\n'
    + '</function>\n</tool_call>\n'
    + 'You may write normal text before and after a tool call. '
    + 'When the user asks for trailing text, you must continue after '
    + '</tool_call>.<|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
{{- '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n<think>\nWe need follow the exact sequence. '
    + 'We have enough information. Now provide the final response' }}
{%- endif %}
""".strip()

DEFAULT_PROMPT = """Write a Reliability section with 20 numbered one-line
tips, invoke get_weather exactly once for Beijing in celsius using the required
function-call format, then continue in the same response with a Testing section
with 20 numbered one-line tips. Invoke the function rather than describing
it."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                    },
                },
                "required": ["city", "unit"],
                "additionalProperties": False,
            },
        },
    }
]


@dataclass(frozen=True)
class RequestMetrics:
    strict: bool
    ttft_ms: float
    tpot_ms: float
    e2el_ms: float
    completion_tokens: int
    reasoning_tokens: int
    free_before_tokens: int
    tool_tokens: int
    free_after_tokens: int

    @property
    def free_tokens(self) -> int:
        return self.free_before_tokens + self.free_after_tokens


@dataclass(frozen=True)
class BoundaryTokenIds:
    reasoning_end: int
    tool_start: int
    tool_end: int


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def load_boundary_token_ids(tokenizer_path: str) -> BoundaryTokenIds:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def single_token(text: str) -> int:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"Expected {text!r} to be one token, got {token_ids}. "
                "This benchmark is specific to standalone Qwen boundaries."
            )
        return int(token_ids[0])

    return BoundaryTokenIds(
        reasoning_end=single_token("</think>"),
        tool_start=single_token("<tool_call>"),
        tool_end=single_token("</tool_call>"),
    )


def _has_output(delta: dict[str, Any]) -> bool:
    return bool(
        delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning") or delta.get("tool_calls")
    )


def _find_after(token_ids: list[int], token_id: int, start: int = 0) -> int:
    try:
        return token_ids.index(token_id, start)
    except ValueError as error:
        raise RuntimeError(f"Required boundary token {token_id} was not generated: {token_ids}") from error


def _segment_counts(
    token_ids: list[int],
    boundary_ids: BoundaryTokenIds,
) -> tuple[int, int, int, int]:
    reasoning_end = _find_after(token_ids, boundary_ids.reasoning_end)
    tool_start = _find_after(
        token_ids,
        boundary_ids.tool_start,
        reasoning_end + 1,
    )
    tool_end = _find_after(token_ids, boundary_ids.tool_end, tool_start + 1)
    reasoning_tokens = reasoning_end + 1
    free_before_tokens = tool_start - reasoning_end - 1
    tool_tokens = tool_end - tool_start + 1
    free_after_tokens = len(token_ids) - tool_end - 1
    return (
        reasoning_tokens,
        free_before_tokens,
        tool_tokens,
        free_after_tokens,
    )


def _validate_tool_call(
    tool_names: dict[int, str],
    tool_arguments: dict[int, str],
) -> None:
    indexes = set(tool_names) | set(tool_arguments)
    if indexes != {0} or tool_names.get(0) != "get_weather":
        raise RuntimeError(
            f"Expected exactly one parsed get_weather call, got names={tool_names}, arguments={tool_arguments}"
        )
    try:
        arguments = json.loads(tool_arguments[0])
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Tool arguments are not valid JSON: {tool_arguments[0]!r}") from error
    if arguments != {"city": "Beijing", "unit": "celsius"}:
        raise RuntimeError(f"Unexpected tool arguments: {arguments}")


def run_request(
    args: argparse.Namespace,
    strict: bool,
    boundary_ids: BoundaryTokenIds,
) -> RequestMetrics:
    tools = json.loads(json.dumps(TOOLS))
    tools[0]["function"]["strict"] = strict
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,
        "seed": 0,
        "max_tokens": args.max_tokens,
        "min_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
        "skip_special_tokens": False,
        "chat_template": SCOPED_TOOL_CHAT_TEMPLATE,
        "chat_template_kwargs": {
            "enable_thinking": True,
            # Keep the rendered prompt identical between strict modes. The
            # request's actual ``tools`` object above still carries strict.
            "benchmark_tools": TOOLS,
        },
    }
    request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    first_output_at: float | None = None
    completion_tokens: int | None = None
    token_ids: list[int] = []
    tool_names: dict[int, str] = {}
    tool_arguments: dict[int, str] = {}
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage")
                if usage is not None:
                    completion_tokens = usage.get("completion_tokens")
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    token_ids.extend(choice.get("token_ids") or [])
                    for tool_call in delta.get("tool_calls") or []:
                        index = int(tool_call.get("index", 0))
                        function = tool_call.get("function") or {}
                        tool_names[index] = tool_names.get(index, "") + (function.get("name") or "")
                        tool_arguments[index] = tool_arguments.get(index, "") + (function.get("arguments") or "")
                    if first_output_at is None and _has_output(delta):
                        first_output_at = time.perf_counter()
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {body}") from error

    end = time.perf_counter()
    if first_output_at is None or completion_tokens is None:
        raise RuntimeError("Streaming response omitted output or usage")
    if len(token_ids) != completion_tokens:
        raise RuntimeError(
            f"Collected {len(token_ids)} token IDs, usage reported {completion_tokens} completion tokens"
        )
    _validate_tool_call(tool_names, tool_arguments)

    reasoning, free_before, tool, free_after = _segment_counts(
        token_ids,
        boundary_ids,
    )
    free_tokens = free_before + free_after
    if free_before <= 0 or free_after <= 0:
        raise RuntimeError("The response did not contain free text on both sides of the tool call")
    if free_tokens < args.min_free_to_tool_ratio * tool:
        raise RuntimeError(f"Free/tool token ratio {free_tokens / tool:.2f} is below {args.min_free_to_tool_ratio:.2f}")

    ttft = first_output_at - start
    return RequestMetrics(
        strict=strict,
        ttft_ms=ttft * 1000,
        tpot_ms=(end - first_output_at) / max(completion_tokens - 1, 1) * 1000,
        e2el_ms=(end - start) * 1000,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning,
        free_before_tokens=free_before,
        tool_tokens=tool,
        free_after_tokens=free_after,
    )


def summarize(samples: list[RequestMetrics], strict: bool) -> dict[str, Any]:
    selected = [sample for sample in samples if sample.strict is strict]
    tpots = [sample.tpot_ms for sample in selected]
    e2els = [sample.e2el_ms for sample in selected]
    return {
        "strict": strict,
        "requests": len(selected),
        "median_tpot_ms": statistics.median(tpots),
        "p90_tpot_ms": percentile(tpots, 0.9),
        "median_e2el_ms": statistics.median(e2els),
        "median_ttft_ms": statistics.median(sample.ttft_ms for sample in selected),
        "mean_completion_tokens": statistics.mean(sample.completion_tokens for sample in selected),
        "mean_reasoning_tokens": statistics.mean(sample.reasoning_tokens for sample in selected),
        "mean_free_before_tokens": statistics.mean(sample.free_before_tokens for sample in selected),
        "mean_free_after_tokens": statistics.mean(sample.free_after_tokens for sample in selected),
        "mean_free_tokens": statistics.mean(sample.free_tokens for sample in selected),
        "mean_tool_tokens": statistics.mean(sample.tool_tokens for sample in selected),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Served model name")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer path")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--min-free-to-tool-ratio", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1 or args.max_tokens < 2:
        parser.error("warmup must be >= 0, runs >= 1, and max-tokens >= 2")
    return args


def main() -> None:
    args = parse_args()
    boundary_ids = load_boundary_token_ids(args.tokenizer)

    for _ in range(args.warmup):
        run_request(args, False, boundary_ids)
        run_request(args, True, boundary_ids)

    samples: list[RequestMetrics] = []
    for index in range(args.runs):
        order = (False, True) if index % 2 == 0 else (True, False)
        for strict in order:
            sample = run_request(args, strict, boundary_ids)
            samples.append(sample)
            print(
                f"run={index + 1:02d} strict={str(strict).lower():5s} "
                f"tokens={sample.completion_tokens:4d} "
                f"free/tool={sample.free_tokens}/{sample.tool_tokens} "
                f"E2EL={sample.e2el_ms:8.2f} ms "
                f"TPOT={sample.tpot_ms:7.2f} ms"
            )

    strict_off = summarize(samples, False)
    strict_on = summarize(samples, True)
    result = {
        "config": {
            "model": args.model,
            "tokenizer": args.tokenizer,
            "warmup": args.warmup,
            "runs": args.runs,
            "max_tokens": args.max_tokens,
            "min_free_to_tool_ratio": args.min_free_to_tool_ratio,
        },
        "strict_off": strict_off,
        "strict_on": strict_on,
        "median_tpot_regression_percent": (strict_on["median_tpot_ms"] / strict_off["median_tpot_ms"] - 1) * 100,
        "median_e2el_regression_percent": (strict_on["median_e2el_ms"] / strict_off["median_e2el_ms"] - 1) * 100,
        "samples": [asdict(sample) for sample in samples],
    }
    print("\nSummary")
    print(json.dumps(result | {"samples": "omitted"}, indent=2))
    if args.output_json:
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
