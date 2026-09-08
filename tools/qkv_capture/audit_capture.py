#!/usr/bin/env python3
"""Merge TP2 head shards and verify captured GQA against backend output rows."""
import argparse,hashlib,json,re
from pathlib import Path
import torch
torch.set_num_threads(4)

def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument("run_dir");args=p.parse_args()
 root=Path(args.run_dir);prompts=json.loads((root/"prompts.json").read_text())
 model_config=json.loads((Path("/home/w00498770/dev/artifacts/vllm-023-qwen35-qkv/weights/Qwen3.5-35B-A3B/config.json")).read_text())["text_config"]
 layers=[i for i,t in enumerate(model_config["layer_types"]) if t=="full_attention"]
 entries=[]
 for record in prompts:
  sample=record["sample_id"];T=record["seq_len"]
  files=list((root/"raw"/sample).glob("*.pt"));groups={}
  for f in files:
   d=torch.load(f,weights_only=True,map_location="cpu")
   layer=int(re.search(r"layers\.(\d+)\.",d["layer_name"]).group(1))
   groups.setdefault(layer,[]).append((d,f))
  if set(groups)!=set(layers):raise RuntimeError(f"Missing or extra layers: {sample}, {list(groups)}")
  for layer,shards in sorted(groups.items()):
   shards.sort(key=lambda x:x[0]["tp_rank"])
   if [d["tp_rank"] for d,_ in shards]!=[0,1]:raise RuntimeError("TP coverage")
   first=shards[0][0];rows=first["output_query_rows"]
   for d,_ in shards:
    assert d["tp_size"]==2 and d["seq_len"]==T and d["token_ids_sha256"]==record["token_ids_sha256"]
    assert d["head_dim"]==256 and d["q_heads_local"]==8 and d["kv_heads_local"]==1
    assert torch.equal(d["positions"],first["positions"]) and torch.equal(d["output_query_rows"],rows)
   tensors={name:torch.cat([d[name].reshape(T,-1) for d,_ in shards],dim=-1).contiguous() for name in ("q","k","v")}
   for name,x in tensors.items():
    assert x.shape==(T,4096 if name=="q" else 512) and x.dtype==torch.bfloat16
    assert bool(torch.isfinite(x).all())
   q=tensors["q"].reshape(T,16,256)[rows].transpose(0,1).float()
   k=tensors["k"].reshape(T,2,256).transpose(0,1).float().repeat_interleave(8,0)
   v=tensors["v"].reshape(T,2,256).transpose(0,1).float().repeat_interleave(8,0)
   mask=torch.arange(T)[None,:]>rows[:,None]
   scores=(q@k.transpose(-1,-2))*float(first["scaling"])
   probability=torch.softmax(scores.masked_fill(mask[None],-torch.inf),dim=-1)
   reference=(probability@v).transpose(0,1).reshape(len(rows),4096)
   observed=torch.cat([d["attention_output_rows"].reshape(len(rows),-1) for d,_ in shards],-1).float()
   error=observed-reference
   nrmse=float(error.square().mean().sqrt()/reference.square().mean().sqrt().clamp_min(1e-12))
   if nrmse>0.02:raise RuntimeError(f"Attention replay mismatch {sample} layer={layer}: {nrmse}")
   payload={**tensors,"positions":first["positions"],"layer_index":layer,
            "sample_id":sample,"seq_len":T,"num_q_heads":16,"num_kv_heads":2,"head_dim":256,
            "scaling":first["scaling"],"capture_stage":first["capture_stage"],
            "token_ids_sha256":record["token_ids_sha256"],
            "output_query_rows":rows,"attention_output_rows":observed.to(torch.bfloat16)}
   out=root/"merged"/sample;out.mkdir(parents=True,exist_ok=True)
   target=out/f"layer_{layer:02d}.pt"
   torch.save(payload,target)
   entry={"sample_id":sample,"layer_index":layer,"seq_len":T,
          "path":str(target.relative_to(root)),"bytes":target.stat().st_size,"sha256":sha(target),
          "replay_rows":len(rows),"replay_nrmse":nrmse,"replay_max_abs":float(error.abs().max()),
          "shapes":{n:list(x.shape) for n,x in tensors.items()},
          "raw_shards":[{"path":str(f.relative_to(root)),"sha256":sha(f)} for _,f in shards]}
   entries.append(entry)
  print(json.dumps({"sample":sample,"layers":len(groups),"max_nrmse":max(e["replay_nrmse"] for e in entries if e["sample_id"]==sample)}),flush=True)
 report={"requests":len(prompts),"layers":layers,"records":len(entries),
         "max_replay_nrmse":max(e["replay_nrmse"] for e in entries),"entries":entries,
         "dtype":"bfloat16","nvfp4_conversion":False,"replay":"FP32 causal GQA on up to 64 query rows per sample/layer"}
 (root/"audit.json").write_text(json.dumps(report,indent=2))
 print("AUDIT_PASSED",len(entries),report["max_replay_nrmse"],flush=True)

if __name__=="__main__":main()
