#!/usr/bin/env python3
"""Capture exact-length single-request prefills; raw BF16, no NVFP4 conversion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path("/home/w00498770/dev/artifacts/vllm-023-qwen35-qkv")
MODEL = ROOT/"weights/Qwen3.5-35B-A3B"
CONTROL = ROOT/"active_capture.json"
os.environ["QKV_CAPTURE_CONTROL"] = str(CONTROL)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--lengths", default="10,128,256,512,1024,2048,4096")
    p.add_argument("--families", default="alice,holmes,frankenstein,python_ast,chinese_systems,arithmetic")
    args=p.parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import torch
    out=ROOT/"captures"/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    CONTROL.unlink(missing_ok=True)
    tok=AutoTokenizer.from_pretrained(str(MODEL))
    prompts=[]
    for family in args.families.split(","):
        text=(ROOT/"corpus"/(family+".txt")).read_text()
        ids=tok.encode(text,add_special_tokens=False)
        for length in map(int,args.lengths.split(",")):
            if len(ids)<length:
                raise RuntimeError(f"Corpus too short: {family}, {len(ids)} < {length}")
            tokens=ids[:length]
            digest=hashlib.sha256(json.dumps(tokens,separators=(",",":")).encode()).hexdigest()
            prompts.append({"sample_id":f"{family}_t{length}","seq_len":length,"token_ids":tokens,
                            "token_ids_sha256":digest,"family":family})
    (out/"prompts.json").write_text(json.dumps(prompts,ensure_ascii=False,indent=2))
    config=dict(model=str(MODEL),tensor_parallel_size=2,dtype="bfloat16",
                max_model_len=8192,max_num_seqs=1,max_num_batched_tokens=8192,
                gpu_memory_utilization=0.82,enforce_eager=True,
                enable_prefix_caching=False,enable_chunked_prefill=False,
                async_scheduling=False,language_model_only=True,
                additional_config={"enable_cpu_binding":False},
                limit_mm_per_prompt={"image":0,"video":0},
                seed=20260908)
    (out/"run_config.json").write_text(json.dumps(config,indent=2))
    started=time.monotonic()
    llm=LLM(**config)
    params=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True,logprobs=5)
    # Control/capture paired request: no persistent prefix cache is enabled.
    first=prompts[0]
    baseline=llm.generate([{"prompt_token_ids":first["token_ids"]}],params,use_tqdm=False)[0]
    results=[]
    try:
        for index,record in enumerate(prompts):
            control={k:v for k,v in record.items() if k!="token_ids"}
            control["output_dir"]=str(out/"raw")
            tmp=CONTROL.with_suffix(".tmp")
            tmp.write_text(json.dumps(control))
            os.replace(tmp,CONTROL)
            t=time.monotonic()
            response=llm.generate([{"prompt_token_ids":record["token_ids"]}],params,use_tqdm=False)[0]
            CONTROL.unlink()
            actual=response.outputs[0]
            result={"sample_id":record["sample_id"],"prompt_tokens":len(response.prompt_token_ids),
                    "output_token_ids":actual.token_ids,"text":actual.text,
                    "elapsed_s":time.monotonic()-t}
            if index==0:
                result["capture_disabled_token_ids"]=baseline.outputs[0].token_ids
                result["capture_on_off_tokens_equal"]=actual.token_ids==baseline.outputs[0].token_ids
                bp=baseline.outputs[0].logprobs[0]
                ap=actual.logprobs[0]
                shared=set(bp)&set(ap)
                result["capture_on_off_max_logprob_delta"]=max(abs(bp[k].logprob-ap[k].logprob) for k in shared)
                if not result["capture_on_off_tokens_equal"] or result["capture_on_off_max_logprob_delta"]>1e-5:
                    raise RuntimeError("Capture on/off regression")
            results.append(result)
            (out/"generation_results.json").write_text(json.dumps(results,ensure_ascii=False,indent=2))
            print(json.dumps(result,ensure_ascii=False),flush=True)
    finally:
        CONTROL.unlink(missing_ok=True)
        llm.llm_engine.engine_core.shutdown()
    (out/"COMPLETE.json").write_text(json.dumps({"requests":len(results),"elapsed_s":time.monotonic()-started}))
    print("CAPTURE_COMPLETE",flush=True)

if __name__=="__main__":
    main()
