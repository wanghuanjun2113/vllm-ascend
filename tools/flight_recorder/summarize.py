#!/usr/bin/env python3
import json
import re
import statistics
from collections import Counter
from pathlib import Path
import numpy as np
from analyze import TraceStore, clean

ART=Path("/home/w00498770/dev/artifacts/vllm-023-logits-save")


def metric(text, name):
    return sum(float(line.rsplit(" ",1)[1]) for line in text.splitlines()
               if line.startswith(name+"{"))


def summarize():
    results={"performance":[],"storage":[],"output_comparison":[]}
    for name in ("baseline-final","top128-final","top1024-final","pooled128","pooled1024"):
        root=ART/name
        for c in (1,4):
            file=root/f"c{c}.json"
            if not file.exists():continue
            data=json.loads(file.read_text());rows=data["results"]
            tpot=[r["effective_tpot_s"]*1000 for r in rows if r["effective_tpot_s"] is not None]
            before=(root/f"c{c}-metrics-before.txt").read_text()
            after=(root/f"c{c}-metrics-after.txt").read_text()
            accepted=metric(after,"vllm:spec_decode_num_accepted_tokens_total")-metric(before,"vllm:spec_decode_num_accepted_tokens_total")
            proposed=metric(after,"vllm:spec_decode_num_draft_tokens_total")-metric(before,"vllm:spec_decode_num_draft_tokens_total")
            tokens=sum(r["usage"]["completion_tokens"] for r in rows)
            results["performance"].append({
                "run":name,"concurrency":c,"requests":len(rows),"output_tokens":tokens,
                "wall_s":data["wall_s"],"throughput_token_s":tokens/data["wall_s"],
                "ttft_median_ms":statistics.median(r["ttft_s"]*1000 for r in rows),
                "effective_tpot_median_ms":statistics.median(tpot),
                "effective_tpot_p95_ms":float(np.percentile(tpot,95)),
                "stops":dict(Counter(r["finish_reason"] for r in rows)),
                "exact_answer_matches":sum(r["correct"] for r in rows),
                "mtp_accepted":accepted,"mtp_proposed":proposed,
                "mtp_acceptance":accepted/proposed if proposed else None})
            if name!="baseline-final" and (ART/"baseline-final"/f"c{c}.json").exists():
                base=json.loads((ART/"baseline-final"/f"c{c}.json").read_text())["results"]
                pairs=list(zip(base,rows))
                results["output_comparison"].append({
                    "run":name,"concurrency":c,
                    "identical_token_sequences":sum(a["token_ids"]==b["token_ids"] for a,b in pairs),
                    "pairs":len(pairs)})
        if (root/"trace").exists():
            store=TraceStore(root/"trace");audit=store.audit()
            (root/"audit.json").write_text(json.dumps(audit,indent=2))
            masses={"raw":[],"final":[]};sources=Counter();graphs=Counter();steps=0
            for rid in store.requests():
                for row in store.rows(rid):
                    t=store.token(row);steps+=1;sources[t["source"]]+=1;graphs[t["graph_mode"]]+=1
                    for stage in masses:
                        if np.isfinite(t[stage]["topk_mass"]):masses[stage].append(t[stage]["topk_mass"])
            pool_info=[json.loads(p.read_text()) for p in (root/"trace").glob("pool-*.json")]
            frame_meta=[]
            for _,db in store.shards:
                frame_meta.extend(json.loads(row[0]) for row in db.execute("SELECT meta FROM chunks"))
            profile={}
            for key in ("pool_wait_ns","capture_cpu_ns","cpu_statistics_ns"):
                values=[m[key]/1e6 for m in frame_meta if key in m and all("-warm" not in r for r in m["req_ids"])]
                if values:
                    profile[key.replace("_ns","_ms")]={"median":float(np.median(values)),"p95":float(np.percentile(values,95)),"max":max(values)}
            bytes_=sum(p.stat().st_size for p in (root/"trace").iterdir() if p.is_file())
            results["storage"].append({
                "run":name,"recorded_tokens":steps,"bytes":bytes_,
                "bytes_per_token":bytes_/steps if steps else None,
                "trace_complete":audit["complete"],"requests":len(audit["requests"]),
                "sources":dict(sources),"graph_modes":dict(graphs),
                "pool_info":pool_info,"recording_profile":profile,
                "topk_mass":{s:{"min":min(m),"median":float(np.median(m)),
                                "p05":float(np.percentile(m,5))} if m else {} for s,m in masses.items()}})
    result=clean(results)
    (ART/"results.json").write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__":
    summarize()
