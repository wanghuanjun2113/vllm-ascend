#!/usr/bin/env python3
"""Fixed public examples; natural EOS; streamed client latency boundary."""
import argparse
import asyncio
import csv
import hashlib
import json
import re
import time
from pathlib import Path

import httpx


def cases(root, n):
    root = Path(root)
    zh = list(csv.reader((root / "mgsm_zh.tsv").read_text().splitlines(), delimiter="\t"))
    en = [json.loads(s) for s in (root / "gsm8k_test.jsonl").read_text().splitlines()]
    result = []
    for i, (q, a) in enumerate(zh[:n]):
        result.append({"id": f"mgsm-zh-{i}", "question": q, "answer": a,
                       "instruction": "请用中文简要说明解题步骤，最后一行写 #### 答案数值。"})
    for i, obj in enumerate(en[:n]):
        result.append({"id": f"gsm8k-en-{i}", "question": obj["question"],
                       "answer": obj["answer"].split("####")[-1].strip(),
                       "instruction": "Briefly explain the solution. End with #### followed by the numeric answer."})
    return result


async def request(client, case, run, temperature=0.0, max_tokens=256):
    body = {"model": "qwen-logits", "messages": [
        {"role": "system", "content": case["instruction"]},
        {"role": "user", "content": case["question"]}],
        "temperature": temperature, "top_p": 1.0 if temperature == 0 else 0.9,
        "top_k": -1 if temperature == 0 else 20, "seed": 42,
        "max_tokens": max_tokens, "stream": True,
        "stream_options": {"include_usage": True}, "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False}}
    start = time.perf_counter()
    first = last = None
    tokens, chunks, text = [], [], ""
    usage = {}
    rid = finish = stop = None
    async with client.stream("POST", "/v1/chat/completions", json=body,
                             headers={"X-Request-Id": run + "-" + case["id"]}) as response:
        if response.status_code != 200:
            raise RuntimeError((await response.aread()).decode())
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            obj = json.loads(line[6:])
            if "error" in obj:
                raise RuntimeError(obj["error"])
            rid = obj.get("id", rid)
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices", []):
                delta = choice.get("delta", {})
                ids = choice.get("token_ids") or []
                content = delta.get("content") or delta.get("reasoning") or ""
                if ids or content:
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    chunks.append({"t": now-start, "ids": ids, "text": content})
                tokens.extend(ids)
                text += content
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                    stop = choice.get("stop_reason")
    elapsed = time.perf_counter() - start
    count = usage.get("completion_tokens", len(tokens))
    match = re.findall(r"####\s*([-+]?\d[\d,.]*)", text)
    prediction = match[-1].replace(",", "").rstrip(".") if match else None
    expected = str(case["answer"]).replace(",", "").strip()
    return {"case": case, "response_id": rid, "request": body, "text": text,
            "token_ids": tokens, "chunks": chunks, "usage": usage,
            "finish_reason": finish, "stop_reason": stop,
            "e2e_s": elapsed, "ttft_s": None if first is None else first-start,
            "effective_tpot_s": ((last-first)/(count-1) if first and last and count > 1 else None),
            "prediction": prediction, "correct": prediction == expected}


async def main(args):
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    data = cases(args.datasets, args.n)
    (root / "cases.json").write_text(json.dumps(data, ensure_ascii=False, indent=2))
    async with httpx.AsyncClient(base_url=args.url, timeout=600.0, trust_env=False) as client:
        for i in range(args.warmup):
            await request(client, data[i % len(data)], args.name+"-warm", args.temperature, args.max_tokens)
        for concurrency in args.concurrency:
            # Warm the measured concurrency, including speculative kernels.
            await asyncio.gather(*(request(client, c, args.name+f"-warm-c{concurrency}",
                                           args.temperature, args.max_tokens)
                                   for c in data[:concurrency]))
            sem = asyncio.Semaphore(concurrency)
            async def one(c):
                async with sem:
                    r = await request(client, c, args.name+f"-c{concurrency}", args.temperature, args.max_tokens)
                    print(args.name, concurrency, c["id"], r["usage"].get("completion_tokens"),
                          r["finish_reason"], round(r["e2e_s"],3),flush=True)
                    return r
            before = (await client.get("/metrics")).text
            start = time.perf_counter()
            rows = await asyncio.gather(*(one(c) for c in data))
            duration = time.perf_counter()-start
            after = (await client.get("/metrics")).text
            (root/f"c{concurrency}.json").write_text(json.dumps(
                {"name": args.name, "concurrency": concurrency, "wall_s": duration,
                 "results": rows}, ensure_ascii=False))
            (root/f"c{concurrency}-metrics-before.txt").write_text(before)
            (root/f"c{concurrency}-metrics-after.txt").write_text(after)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--url", default="http://127.0.0.1:18327")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--concurrency", nargs="+", type=int, default=[1,4])
    p.add_argument("--temperature", type=float, default=0.0)
    asyncio.run(main(p.parse_args()))
