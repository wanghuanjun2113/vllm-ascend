#!/usr/bin/env python3
"""Audit actual model tensors and construct three fixed-size NVFP4 datasets."""
import hashlib,json,os,sys,time
from pathlib import Path
from collections import Counter
import torch
from safetensors import safe_open
WORK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(WORK/"tools/qkv_capture"))
from export_nvfp4 import quantize,unit_tests
sys.path.insert(0,str(WORK/"tools/linear_capture"))
from audit_and_export import assemble
sys.path.insert(0,"/home/w00498770/algo/algorithm_9_7_2/scripts")
import self_check
ART=Path("/home/w00498770/dev/artifacts/vllm-023-qwen35-qkv/versions_v234")
DATA=Path("/home/w00498770/algo/algorithm_9_7_2/data")
MODEL=ART.parent/"weights/Qwen3.5-35B-A3B"
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def qkv(req,layer,raw):
 folder=raw/"qkv"/req["id"]
 paths=[next(folder.glob(f"*layers.{layer}.self_attn.attn.tp{rank}.pt")) for rank in (0,1)]
 shards=[torch.load(p,weights_only=True,map_location="cpu") for p in paths]
 n=req["seq_len"];first=shards[0];rows=first["output_query_rows"]
 for rank,d in enumerate(shards):
  assert d["tp_rank"]==rank and d["tp_size"]==2 and d["seq_len"]==n
  assert d["token_ids_sha256"]==req["token_ids_sha256"]
  assert torch.equal(d["positions"],first["positions"]) and torch.equal(d["output_query_rows"],rows)
 tensors={r:torch.cat([d[r].reshape(n,-1) for d in shards],-1).contiguous() for r in ("q","k","v")}
 for r,x in tensors.items():
  assert x.shape==(n,4096 if r=="q" else 512) and bool(torch.isfinite(x).all())
 q=tensors["q"].reshape(n,16,256)[rows].transpose(0,1).float()
 k=tensors["k"].reshape(n,2,256).transpose(0,1).float().repeat_interleave(8,0)
 v=tensors["v"].reshape(n,2,256).transpose(0,1).float().repeat_interleave(8,0)
 scores=q@k.transpose(-1,-2)/16
 mask=torch.arange(n)[None,:]>rows[:,None]
 reference=(torch.softmax(scores.masked_fill(mask[None],-torch.inf),-1)@v).transpose(0,1).reshape(len(rows),4096)
 observed=torch.cat([d["attention_output_rows"].reshape(len(rows),-1) for d in shards],-1).float()
 err=float((observed-reference).square().mean().sqrt()/reference.square().mean().sqrt().clamp_min(1e-12))
 assert err<.02,(req["id"],layer,err)
 sample={};stats={}
 for r,x in tensors.items():sample[r],stats[r]=quantize(x)
 return sample,{"request":req["id"],"layer":layer,"nrmse":err,"quantization":stats,"source_sha256":[sha(p) for p in paths]}
def linear_x(req,spec,weights,raw,checkpoint,rows_count):
 paths=[raw/"linear/activations"/req["id"]/f"{spec['module']}.tp{rank}.pt" for rank in (0,1)]
 inputs=[torch.load(p,weights_only=True,map_location="cpu") for p in paths]
 for rank,d in enumerate(inputs):
  assert d["rank"]==rank and d["seq_len"]==req["seq_len"] and d["token_ids_sha256"]==req["token_ids_sha256"]
 w,x,y,mode=assemble(spec,weights,inputs)
 assert torch.equal(w,checkpoint) and list(w.shape)==spec["shape"]
 assert x.shape==(req["seq_len"],spec["shape"][1]) and bool(torch.isfinite(x).all())
 rr=inputs[0]["output_rows"];reference=x[rr].float()@w.float().T
 err=float((y-reference).square().mean().sqrt()/reference.square().mean().sqrt().clamp_min(1e-12))
 assert err<.015,(req["id"],err)
 rows=torch.floor((torch.arange(rows_count)+.5)*len(x)/rows_count).long()
 assert rows.unique().numel()==rows_count
 pair,stats=quantize(x[rows])
 return pair,{"request":req["id"],"nrmse":err,"mode":mode,"rows":rows.tolist(),"quantization":stats,"source_sha256":[sha(p) for p in paths]}
def save_group(group,path,kind):
 (self_check._normalize_attention_dataset if kind=="attention" else self_check._normalize_linear_dataset)([group])
 torch.save([group],path)
 loaded=torch.load(path,weights_only=True,map_location="cpu")
 (self_check._normalize_attention_dataset if kind=="attention" else self_check._normalize_linear_dataset)(loaded)
 return {"path":str(path),"bytes":path.stat().st_size,"sha256":sha(path)}
def main():
 torch.set_num_threads(8);unit_tests()
 wi=json.loads((MODEL/"model.safetensors.index.json").read_text())["weight_map"]
 summary={}
 all_token_hashes=set()
 for version in (2,3,4):
  plan_path=ART/f"plan_v{version}.json";plan=json.loads(plan_path.read_text())
  capture=ART/f"capture_v{version}";assert (capture/"COMPLETE.json").is_file()
  dest=DATA/f"qwen35_35B_v{version}"
  if dest.exists():raise FileExistsError(dest)
  dest.mkdir()
  reqs={r["id"]:r for r in plan["requests"]}
  for r in reqs.values():
   assert r["token_ids_sha256"] not in all_token_hashes,("duplicate input across versions",r["id"])
   all_token_hashes.add(r["token_ids_sha256"])
  # Source identity and raw token intervals must not cross bank or role.
  used={}
  for r in reqs.values():
   for span in r["source_spans"]:
    key=span["source_group"];tag=(r["bank"],r["role"])
    assert key not in used or used[key]==tag
    used[key]=tag
  version_result={}
  for bank,groups in plan["banks"].items():
   target=dest/bank;stage=target/".building"
   (stage/"attention_shards").mkdir(parents=True);(stage/"linear_shards").mkdir()
   raw=capture/bank;attention=[];linears=[];checks=[]
   for g in groups["attention"]:
    cal=[];tests=[]
    for kind,result in (("calib",cal),("test",tests)):
     for rid in g[kind]:
      sample,audit=qkv(reqs[rid],g["layer"],raw);result.append(sample);checks.append({"kind":"attention",**audit})
    assert len(cal)==len(tests)==5
    data={"q_num_heads":16,"kv_num_heads":2,"head_dim":256,"calib":cal,"test":tests}
    file=stage/"attention_shards"/f"{g['id']:03d}.pt"
    saved=save_group(data,file,"attention");saved["path"]=f"attention_shards/{file.name}"
    attention.append({**g,**saved})
   for gi,g in enumerate(groups["linear"]):
    spec=g["spec"]
    wpaths=[raw/"linear/weights"/f"{spec['module']}.tp{rank}.pt" for rank in (0,1)]
    weights=[torch.load(p,weights_only=True,map_location="cpu") for p in wpaths]
    with safe_open(str(MODEL/wi[spec["weight_key"]]),framework="pt") as f:checkpoint=f.get_tensor(spec["weight_key"])
    wp,wstats=quantize(checkpoint)
    cal=[];tests=[];la=[]
    for kind,counts,result in (("calib",g["calib_rows"],cal),("test",g["test_rows"],tests)):
     for rid,count in zip(g[kind],counts):
      pair,audit=linear_x(reqs[rid],spec,weights,raw,checkpoint,count);result.append(pair)
      checks.append({"kind":"linear","group":gi,**audit});la.append(audit)
    assert len(cal)==len(tests)==5
    data={"weight":wp,"calib_activation_list":cal,"test_activation_list":tests}
    file=stage/"linear_shards"/f"{gi:03d}.pt"
    saved=save_group(data,file,"linear");saved["path"]=f"linear_shards/{file.name}"
    linears.append({**g,**saved,"weight_quantization":wstats,"input_audit":la,"optional_shape_weight":[1.5,1,1,.5,1][gi]})
   assert len(attention)==25 and len(linears)==5
   hist=Counter(t for g in attention for t in g["test_lengths"])
   assert (hist[10],hist[128],hist[512])==(1,3,4)
   for g in attention:assert set((256,1024,2048,4096)).issubset(g["test_lengths"])
   shape_hist=Counter(tuple(g["spec"]["shape"]) for g in linears)
   assert shape_hist==Counter({(8192,2048):1,(512,2048):2,(2048,4096):1,(2048,512):1})
   for kind in ("attention","linear"):os.replace(stage/f"{kind}_shards",target/f"{kind}_shards")
   stage.rmdir()
   manifest={"version":version,"split":bank,"corpus":plan["corpus"],"plan_sha256":sha(plan_path),
    "attention_groups":attention,"linear_groups":linears,"group_counts":{"linear":5,"attention":25},
    "calibration_per_group":5,"test_per_group":5,"test_samples":{"linear":25,"attention":125},
    "attention_test_T_histogram":dict(hist),"linear_shape_histogram":{str(k):v for k,v in shape_hist.items()},
    "linear_optional_weights_note":"Explicit shape-proportion correction only; test_machine scores remain unweighted.",
    "calibration_lengths_are_local_choice":True,"fifth_common_attention_sample_is_construction_choice":True,
    "source_bank_and_role_groups_disjoint":True,"shared_panel_across_different_layers":True}
   (target/"MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
   (target/"AUDIT.json").write_text(json.dumps(checks,indent=2))
   version_result[bank]={"groups":30,"calibration_samples":150,"test_samples":150,
    "bytes":sum(g["bytes"] for g in attention+linears),"attention_T_histogram":dict(hist),
    "max_qkv_nrmse":max(c["nrmse"] for c in checks if c["kind"]=="attention"),
    "max_linear_nrmse":max(c["nrmse"] for c in checks if c["kind"]=="linear")}
   print("SPLIT_DATA_READY",version,bank,json.dumps(version_result[bank]),flush=True)
  (dest/"SOURCE_DOCUMENTS.json").write_text(json.dumps(plan["documents"],ensure_ascii=False,indent=2))
  (dest/"CAPTURE_PLAN.json").write_text(json.dumps(plan,ensure_ascii=False))
  (dest/"BUILD_SUMMARY.json").write_text(json.dumps(version_result,indent=2))
  summary[version]=version_result
 (ART/"BUILD_SUMMARY.json").write_text(json.dumps(summary,indent=2))
 print("ALL_DATASETS_BUILT",flush=True)
if __name__=="__main__":main()
