#!/usr/bin/env python3
"""Run a frozen three-corpus plan with one BF16 TP2 model instance."""
import json,os,time
from pathlib import Path
ART=Path("/home/w00498770/dev/artifacts/vllm-023-qwen35-qkv/versions_v234")
BASE=ART.parent
QC=ART/"active_qkv.json";LC=ART/"active_linear.json"
os.environ["QKV_CAPTURE_CONTROL"]=str(QC)
os.environ["LINEAR_CAPTURE_CONTROL"]=str(LC)
def write(path,obj):
 tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(obj));os.replace(tmp,path)
def main():
 from vllm import LLM,SamplingParams
 plans=[json.loads((ART/f"plan_v{v}.json").read_text()) for v in (2,3,4)]
 for p in plans:
  out=ART/f"capture_v{p['version']}"
  if out.exists():raise FileExistsError(out)
  out.mkdir()
 QC.unlink(missing_ok=True);LC.unlink(missing_ok=True)
 config=dict(model=str(BASE/"weights/Qwen3.5-35B-A3B"),tensor_parallel_size=2,dtype="bfloat16",
 max_model_len=8192,max_num_seqs=1,max_num_batched_tokens=8192,gpu_memory_utilization=.82,
 enforce_eager=True,enable_prefix_caching=False,enable_chunked_prefill=True,async_scheduling=False,
 language_model_only=True,seed=20260909,limit_mm_per_prompt={"image":0,"video":0},
 additional_config={"enable_cpu_binding":False,"enable_flashcomm1":False,"enable_matmul_allreduce":False,
 "enable_shared_expert_dp":False,"multistream_overlap_shared_expert":False})
 (ART/"run_config.json").write_text(json.dumps(config,indent=2))
 llm=LLM(**config);params=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True,logprobs=5)
 try:
  for plan in plans:
   version=plan["version"];out=ART/f"capture_v{version}";results=[]
   first=plan["requests"][0]
   baseline=llm.generate([{"prompt_token_ids":first["token_ids"]}],params,use_tqdm=False)[0]
   for i,req in enumerate(plan["requests"]):
    root=out/req["bank"]
    common={"sample_id":req["id"],"seq_len":req["seq_len"],"token_ids_sha256":req["token_ids_sha256"]}
    if req["qkv_layers"]:write(QC,{**common,"output_dir":str(root/"qkv"),"layers":req["qkv_layers"]})
    else:QC.unlink(missing_ok=True)
    if req["linear_modules"]:write(LC,{**common,"output_dir":str(root/"linear"),"modules":req["linear_modules"]})
    else:LC.unlink(missing_ok=True)
    t=time.monotonic()
    response=llm.generate([{"prompt_token_ids":req["token_ids"]}],params,use_tqdm=False)[0]
    QC.unlink(missing_ok=True);LC.unlink(missing_ok=True)
    row={"request":req["id"],"seq_len":len(response.prompt_token_ids),"tokens":response.outputs[0].token_ids,"seconds":time.monotonic()-t}
    assert row["seq_len"]==req["seq_len"]
    if i==0:
     a=response.outputs[0];b=baseline.outputs[0]
     row["control_tokens_equal"]=a.token_ids==b.token_ids
     row["control_max_logprob_delta"]=max(abs(a.logprobs[0][k].logprob-b.logprobs[0][k].logprob) for k in set(a.logprobs[0])&set(b.logprobs[0]))
     assert row["control_tokens_equal"] and row["control_max_logprob_delta"]<=1e-5
    if req["qkv_layers"]:
     paths=list((root/"qkv"/req["id"]).glob("*.pt"))
     assert len(paths)==2*len(req["qkv_layers"]),(req["id"],len(paths))
    if req["linear_modules"]:
     paths=list((root/"linear/activations"/req["id"]).glob("*.pt"))
     assert len(paths)==2*len(req["linear_modules"]),(req["id"],len(paths))
    results.append(row)
    (out/"generation_results.json").write_text(json.dumps(results,indent=2))
    if i%20==0 or i+1==len(plan["requests"]):print(json.dumps({"version":version,"requests_done":i+1,"total":len(plan["requests"]),"last":req["id"]}),flush=True)
   (out/"COMPLETE.json").write_text(json.dumps({"requests":len(results)}))
 finally:
  QC.unlink(missing_ok=True);LC.unlink(missing_ok=True);llm.llm_engine.engine_core.shutdown()
 print("ALL_CAPTURES_COMPLETE",flush=True)
if __name__=="__main__":main()
