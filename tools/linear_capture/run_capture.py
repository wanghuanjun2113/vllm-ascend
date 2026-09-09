#!/usr/bin/env python3
"""Capture 50 distinct checkpoint Linear matrices and real prefill activations."""
import hashlib,json,os,time
from pathlib import Path

ART=Path("/home/w00498770/dev/artifacts/vllm-023-qwen35-qkv")
ROOT=ART/"linear_capture_v1"
CONTROL=ART/"active_linear_capture.json"
os.environ["LINEAR_CAPTURE_CONTROL"]=str(CONTROL)
os.environ.pop("QKV_CAPTURE_CONTROL",None)
FULL=list(range(3,40,4))
FAMILIES=("alice","python_ast","chinese_systems","holmes","frankenstein","arithmetic")


def specifications():
    specs=[]
    def add(layer,module,weight,segments,shape,mode="column"):
        specs.append({"id":f"{len(specs):03d}","layer":layer,"module":f"layers.{layer}.{module}",
                      "weight_key":f"model.language_model.layers.{layer}.{weight}.weight",
                      "segments":segments,"shape":shape,"mode":mode})
    for layer in FULL:add(layer,"self_attn.qkv_proj","self_attn.q_proj",[0],[8192,2048])
    for layer in (0,8,16,24,32):add(layer,"linear_attn.in_proj_qkvz","linear_attn.in_proj_qkv",[0,1,2],[8192,2048])
    for layer in FULL:add(layer,"self_attn.qkv_proj","self_attn.k_proj",[1],[512,2048])
    for layer in (3,11,19,27,35):add(layer,"self_attn.qkv_proj","self_attn.v_proj",[2],[512,2048])
    for layer in (3,11,19,27,35):add(layer,"mlp.shared_expert.gate_up_proj","mlp.shared_expert.up_proj",[1],[512,2048])
    for layer in (3,19,35):add(layer,"self_attn.o_proj","self_attn.o_proj",[],[2048,4096],"row")
    for layer in (0,16):add(layer,"linear_attn.out_proj","linear_attn.out_proj",[],[2048,4096],"row")
    for layer in FULL:add(layer,"mlp.shared_expert.down_proj","mlp.shared_expert.down_proj",[],[2048,512],"row")
    assert len(specs)==50 and len({s["weight_key"] for s in specs})==50
    return specs


def main():
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    ROOT.mkdir(exist_ok=False)
    specs=specifications()
    (ROOT/"SPECIFICATIONS.json").write_text(json.dumps(specs,indent=2))
    model=ART/"weights/Qwen3.5-35B-A3B"
    tok=AutoTokenizer.from_pretrained(str(model))
    prompts=[]
    for family in FAMILIES:
        tokens=tok.encode((ART/"corpus"/(family+".txt")).read_text(),add_special_tokens=False)[:4096]
        assert len(tokens)==4096
        prompts.append({"sample_id":family,"seq_len":4096,"token_ids":tokens,
                        "token_ids_sha256":hashlib.sha256(json.dumps(tokens,separators=(",",":")).encode()).hexdigest()})
    (ROOT/"prompts.json").write_text(json.dumps(prompts,ensure_ascii=False))
    config=dict(model=str(model),tensor_parallel_size=2,dtype="bfloat16",max_model_len=8192,
                max_num_seqs=1,max_num_batched_tokens=8192,gpu_memory_utilization=.82,
                enforce_eager=True,enable_prefix_caching=False,enable_chunked_prefill=True,
                async_scheduling=False,language_model_only=True,seed=20260909,
                limit_mm_per_prompt={"image":0,"video":0},
                additional_config={"enable_cpu_binding":False,"enable_flashcomm1":False,
                "enable_matmul_allreduce":False,"enable_shared_expert_dp":False,
                "multistream_overlap_shared_expert":False})
    (ROOT/"run_config.json").write_text(json.dumps(config,indent=2))
    CONTROL.unlink(missing_ok=True)
    llm=LLM(**config);params=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True,logprobs=5)
    results=[]
    try:
        baseline=llm.generate([{"prompt_token_ids":prompts[0]["token_ids"]}],params,use_tqdm=False)[0]
        for index,prompt in enumerate(prompts):
            control={k:v for k,v in prompt.items() if k!="token_ids"}
            control.update(output_dir=str(ROOT/"raw"),modules=sorted({s["module"] for s in specs}))
            tmp=CONTROL.with_suffix(".tmp");tmp.write_text(json.dumps(control));os.replace(tmp,CONTROL)
            start=time.monotonic()
            response=llm.generate([{"prompt_token_ids":prompt["token_ids"]}],params,use_tqdm=False)[0]
            CONTROL.unlink()
            row={"sample_id":prompt["sample_id"],"tokens":response.outputs[0].token_ids,"elapsed_s":time.monotonic()-start}
            if index==0:
                a=response.outputs[0];b=baseline.outputs[0]
                row["control_token_equal"]=a.token_ids==b.token_ids
                row["control_max_logprob_delta"]=max(abs(a.logprobs[0][k].logprob-b.logprobs[0][k].logprob) for k in set(a.logprobs[0])&set(b.logprobs[0]))
                assert row["control_token_equal"] and row["control_max_logprob_delta"]<=1e-5
            paths=list((ROOT/"raw/activations"/prompt["sample_id"]).glob("*.pt"))
            assert len(paths)==2*len(control["modules"]), (len(paths),len(control["modules"]))
            results.append(row);(ROOT/"generation_results.json").write_text(json.dumps(results,indent=2))
            print(json.dumps(row),flush=True)
    finally:
        CONTROL.unlink(missing_ok=True);llm.llm_engine.engine_core.shutdown()
    (ROOT/"COMPLETE.json").write_text(json.dumps({"requests":len(results),"logical_matrices":50,"physical_modules":len(control["modules"])}))
    print("LINEAR_CAPTURE_COMPLETE",flush=True)

if __name__=="__main__":main()
