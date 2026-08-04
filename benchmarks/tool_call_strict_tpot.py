#!/usr/bin/env python3
"""Compare TPOT with tool-call ``strict`` disabled and enabled.

This benchmark targets the auto-tool path while deliberately asking the model
for a normal text response. A strict structural-tag grammar should remain in
its unconstrained state for the whole response, so applying a logits mask in
this case is unnecessary overhead.

Start Qwen3.6-27B on two Ascend NPUs before running the benchmark, for example::

    vllm serve /models/Qwen3.6-27B \
      --served-model-name Qwen3.6-27B \
      --tensor-parallel-size 2 \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_coder

The parser name is model/checkpoint dependent. Use the parser recommended by
the Qwen3.6 checkpoint when it differs from ``qwen3_coder``.
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

DEFAULT_PROMPT = (
    "Do not call any tool. Answer directly in plain text. Write a numbered "
    "list of 100 concise recommendations for improving Python service "
    "reliability. Do not stop before item 100."
)

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
    used_tool: bool


def percentile(values: list[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sample."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _has_output(delta: dict[str, Any]) -> bool:
    return bool(delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"))


def run_request(args: argparse.Namespace, strict: bool) -> RequestMetrics:
    tools = json.loads(json.dumps(TOOLS))
    tools[0]["function"]["strict"] = strict
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
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
    used_tool = False
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
                    used_tool |= bool(delta.get("tool_calls"))
                    if first_output_at is None and _has_output(delta):
                        first_output_at = time.perf_counter()
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {body}") from error

    end = time.perf_counter()
    if first_output_at is None:
        raise RuntimeError("The response contained no streamed output token")
    if completion_tokens is None:
        raise RuntimeError(
            "The server did not return completion_tokens; ensure stream_options include_usage is supported"
        )

    ttft = first_output_at - start
    # Standard serving benchmark definition: decoding time after the first
    # token divided by the number of remaining output tokens.
    tpot = (end - first_output_at) / max(completion_tokens - 1, 1)
    return RequestMetrics(
        strict=strict,
        ttft_ms=ttft * 1000,
        tpot_ms=tpot * 1000,
        e2el_ms=(end - start) * 1000,
        completion_tokens=completion_tokens,
        used_tool=used_tool,
    )


def summarize(samples: list[RequestMetrics], strict: bool) -> dict[str, Any]:
    selected = [sample for sample in samples if sample.strict is strict]
    tpots = [sample.tpot_ms for sample in selected]
    ttfts = [sample.ttft_ms for sample in selected]
    return {
        "strict": strict,
        "requests": len(selected),
        "median_tpot_ms": statistics.median(tpots),
        "mean_tpot_ms": statistics.mean(tpots),
        "p90_tpot_ms": percentile(tpots, 0.90),
        "median_ttft_ms": statistics.median(ttfts),
        "mean_completion_tokens": statistics.mean(sample.completion_tokens for sample in selected),
        "unexpected_tool_calls": sum(sample.used_tool for sample in selected),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Served model name")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1 or args.max_tokens < 2:
        parser.error("warmup must be >= 0, runs >= 1, and max-tokens >= 2")
    return args


def main() -> None:
    args = parse_args()

    # Warm both structural-output paths before measurement. Alternating the
    # order reduces bias from temperature, clock, and background-load drift.
    for _ in range(args.warmup):
        run_request(args, strict=False)
        run_request(args, strict=True)

    samples: list[RequestMetrics] = []
    for index in range(args.runs):
        order = (False, True) if index % 2 == 0 else (True, False)
        for strict in order:
            sample = run_request(args, strict)
            samples.append(sample)
            print(
                f"run={index + 1:02d} strict={str(strict).lower():5s} "
                f"tokens={sample.completion_tokens:4d} "
                f"TTFT={sample.ttft_ms:8.2f} ms "
                f"TPOT={sample.tpot_ms:8.2f} ms "
                f"tool_called={sample.used_tool}"
            )

    strict_off = summarize(samples, strict=False)
    strict_on = summarize(samples, strict=True)
    regression = (strict_on["median_tpot_ms"] / strict_off["median_tpot_ms"] - 1) * 100
    result = {
        "config": {
            "model": args.model,
            "base_url": args.base_url,
            "warmup": args.warmup,
            "runs": args.runs,
            "max_tokens": args.max_tokens,
            "prompt": args.prompt,
        },
        "strict_off": strict_off,
        "strict_on": strict_on,
        "median_tpot_regression_percent": regression,
        "samples": [asdict(sample) for sample in samples],
    }

    print("\nSummary")
    print(json.dumps(result | {"samples": "omitted"}, indent=2))
    if strict_off["unexpected_tool_calls"] or strict_on["unexpected_tool_calls"]:
        print(
            "WARNING: at least one normal-text request called a tool; inspect "
            "the per-request samples before comparing TPOT."
        )
    if args.output_json:
        args.output_json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
