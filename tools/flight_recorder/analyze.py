#!/usr/bin/env python3
"""Offline trace inspection. Does not import vLLM or require an NPU."""
import argparse
import hashlib
import io
import json
import math
import sqlite3
from pathlib import Path

import numpy as np


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


class TraceStore:
    def __init__(self, root):
        self.root = Path(root)
        self.shards = [(p.stem, sqlite3.connect(p.as_uri() + "?mode=ro", uri=True))
                       for p in sorted(self.root.resolve().glob("*.sqlite"))]
        self.cache = {}

    def events(self, rid=None):
        rows = []
        for shard, db in self.shards:
            query = "SELECT data FROM events"
            args = ()
            if rid is not None:
                query += " WHERE request_id=?"
                args = (rid,)
            rows.extend(json.loads(x[0]) for x in db.execute(query + " ORDER BY id", args))
        return rows

    def requests(self):
        result = {}
        for e in self.events():
            rid = e["request_id"]
            r = result.setdefault(rid, {"request_id": rid, "output_token_ids": []})
            if e["kind"] == "request":
                r.update(e)
            elif e["kind"] == "output":
                r["output_token_ids"].extend(e["token_ids"])
                if e["finish_reason"] is not None:
                    r["finish_reason"] = e["finish_reason"]
                    r["stop_reason"] = e["stop_reason"]
            elif e["kind"] == "abort":
                r["finish_reason"] = "abort"
        defaults_path = self.root.parent.parent / "sampling-defaults.json"
        defaults = json.loads(defaults_path.read_text()) if defaults_path.exists() else None
        for request in result.values():
            if request.get("kind") == "request":
                if request.get("sampling_params_complete"):
                    request["sampling_params_encoding"] = "explicit runtime fields"
                elif defaults is not None:
                    request["sampling_params"] = {
                        **defaults["values"], **request["sampling_params"]}
                    request["sampling_params_encoding"] = "non-default fields + pinned class defaults"
                else:
                    request["sampling_params_encoding"] = "non-default fields only"
        return result

    def rows(self, rid):
        result = []
        for shard, db in self.shards:
            for pos, token, chunk, row in db.execute(
                    "SELECT position,token_id,chunk,row FROM tokens WHERE request_id=? ORDER BY position", (rid,)):
                result.append((pos, token, shard, chunk, row))
        return sorted(result)

    def chunk(self, shard, chunk):
        key = (shard, chunk)
        if key not in self.cache:
            db = dict(self.shards)[shard]
            offset, size, digest, meta = db.execute(
                "SELECT offset,size,sha256,meta FROM chunks WHERE id=?", (chunk,)).fetchone()
            with (self.root / (shard + ".bin")).open("rb") as f:
                f.seek(offset)
                raw = f.read(size)
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("chunk checksum mismatch: " + str(key))
            with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                arrays = {k: data[k] for k in data.files}
            # A small cache keeps sequential access fast without retaining a run.
            if len(self.cache) >= 16:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = (arrays, json.loads(meta))
        return self.cache[key]

    def token(self, item):
        pos, token, shard, chunk, row = item
        a, meta = self.chunk(shard, chunk)
        if int(a["selected"][row]) != token:
            raise ValueError("token index and payload disagree")
        out = {"position": pos, "token_id": token, "top_k": meta["top_k"],
               "probe_ids": [token] + meta["probe_ids"],
               "final_seen": bool(a["final_seen"][row])}
        slot = int(a["flat_slots"][row])
        reqidx, step = divmod(slot, meta["width"])
        counts = meta["num_draft_tokens"]
        if counts:
            n = counts[reqidx]
            draft_start = sum(counts[:reqidx])
            draft = meta["draft_ids"][draft_start:draft_start+n]
            out["draft_ids"] = draft
            # Infer only from accepted-prefix length, not from accidental token equality.
            same_req = a["flat_slots"] // meta["width"] == reqidx
            accepted_length = int(same_req.sum())
            out["source"] = ("bonus" if step == n else
                             "accepted_draft" if step < accepted_length-1 else "replacement")
        else:
            out["source"] = "ordinary"
        out["graph_mode"] = meta.get("graph_mode", "unknown")
        out["selection_mode"] = ("greedy" if meta["all_greedy"] else
                                 "random" if meta["all_random"] else "mixed")
        out["block_verify"] = meta["block_verify"]
        out["entropy_verify"] = meta["entropy_verify"]
        for stage in ("raw", "final"):
            vals = a[stage + "_values"][row].astype(float)
            lse = float(a[stage + "_lse"][row])
            mass = float(np.exp(vals - lse).sum()) if math.isfinite(lse) else float("nan")
            out[stage] = {
                "ids": a[stage + "_ids"][row].tolist(), "logits": vals.tolist(),
                "logsumexp": lse, "topk_mass": mass,
                "counts": a[stage + "_counts"][row].tolist(),
                "probe_logits": a[stage + "_probe_values"][row].tolist(),
                "probe_ranks": a[stage + "_probe_ranks"][row].tolist(),
            }
        return out

    def audit(self):
        report = []
        failures = [p.name for p in self.root.glob("FAILED-*")]
        requests = self.requests()
        for _, db in self.shards:
            for (rid,) in db.execute("SELECT DISTINCT request_id FROM tokens"):
                requests.setdefault(rid, {"request_id": rid, "output_token_ids": []})
        for rid, request in requests.items():
            rows = self.rows(rid)
            ids = [x[1] for x in rows]
            expected = request["output_token_ids"]
            contiguous = [x[0] for x in rows] == list(range(len(rows)))
            matched = ids[:len(expected)] == expected
            checked = True
            for _, token_id, shard, chunk, row in rows:
                arrays, _ = self.chunk(shard, chunk)
                if int(arrays["selected"][row]) != token_id:
                    raise ValueError("token index and payload disagree")
                checked = checked and bool(arrays["final_seen"][row])
            complete = (request.get("kind") == "request" and not failures and contiguous and matched and checked
                        and len(ids) >= len(expected) and "finish_reason" in request)
            report.append({
                "request_id": rid, "external_req_id": request.get("external_req_id"),
                "expected_tokens": len(expected), "recorded_tokens": len(ids),
                "unused_after_stop": max(0, len(ids)-len(expected)),
                "contiguous": contiguous, "output_match": matched,
                "finish_reason": request.get("finish_reason"),
                "trace_complete": complete,
            })
        return {"writer_failures": failures, "requests": report,
                "complete": bool(report) and not failures and all(r["trace_complete"] for r in report)}

    def export(self, rid, tokenizer):
        request = self.requests()[rid]
        rows = [self.token(x) for x in self.rows(rid)]
        for row in rows:
            row["committed_to_output"] = row["position"] < len(request["output_token_ids"])
        manifest_path = self.root.parent / "manifest.json"
        if manifest_path.exists():
            expected = json.loads(manifest_path.read_text()).get("model_files", {})
            for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json"):
                if name in expected:
                    path = Path(tokenizer) / name
                    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected[name]["sha256"]:
                        raise ValueError("tokenizer snapshot mismatch: " + name)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
        ids = set(request.get("prompt_token_ids", []))
        for r in rows:
            ids.update(r["probe_ids"])
            for stage in ("raw", "final"):
                ids.update(r[stage]["ids"])
        labels = {str(i): tok.decode([i], skip_special_tokens=False) for i in ids}
        return clean({"request": request, "tokens": rows, "labels": labels,
                      "prompt": tok.decode(request.get("prompt_token_ids", [])),
                      "output": tok.decode(request["output_token_ids"], skip_special_tokens=False),
                      "audit": next(x for x in self.audit()["requests"] if x["request_id"] == rid)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    sub.add_parser("audit")
    q = sub.add_parser("query")
    q.add_argument("--request", required=True)
    q.add_argument("--position", type=int, required=True)
    q = sub.add_parser("html")
    q.add_argument("--request", required=True)
    q.add_argument("--tokenizer", required=True)
    q.add_argument("--out", required=True)
    args = p.parse_args()
    store = TraceStore(args.root)
    if args.command == "list":
        result = [{"request_id": k, "external_req_id": v.get("external_req_id"),
                   "tokens": len(v["output_token_ids"]), "finish_reason": v.get("finish_reason")}
                  for k, v in store.requests().items()]
    elif args.command == "audit":
        result = store.audit()
    elif args.command == "query":
        matches = [r for r in store.rows(args.request) if r[0] == args.position]
        if len(matches) != 1:
            raise ValueError("missing or ambiguous token")
        result = store.token(matches[0])
        result["committed_to_output"] = args.position < len(
            store.requests()[args.request]["output_token_ids"])
    else:
        result = store.export(args.request, args.tokenizer)
        template = Path(__file__).with_name("viewer.html").read_text()
        data = json.dumps(result, ensure_ascii=False).replace("<", "\\u003c")
        Path(args.out).write_text(template.replace("/*TRACE_DATA*/null", data))
        Path(args.out).with_suffix(".json").write_text(json.dumps(result, ensure_ascii=False))
        print(args.out)
        return
    print(json.dumps(clean(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
