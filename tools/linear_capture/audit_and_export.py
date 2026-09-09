#!/usr/bin/env python3
"""Audit captured Linear operands against checkpoint, then build NVFP4 banks."""
import hashlib,json,os,sys,time
from pathlib import Path
from collections import Counter
import torch
from safetensors import safe_open

ART=Path("/home/w00498770/dev/artifacts/vllm-023-qwen35-qkv")
CAP=ART/"linear_capture_v1"
DATA=Path("/home/w00498770/algo/algorithm_9_7_2/data/qwen35_35B")
MODEL=ART/"weights/Qwen3.5-35B-A3B"
BANKS={"development":["alice","python_ast","chinese_systems"],
       "validation":["holmes","frankenstein","arithmetic"]}
SUPPORT=[1,2,7,8,10,14,15,16,18,22,25,26,31,39,42,56,58,60,128,512,1024]
CALIB=[64,128,256,512,1024]
COUNTS={(8192,2048):15,(512,2048):20,(2048,4096):5,(2048,512):10}

def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()

def pieces(tensor,sizes,segments,axis):
 result=[]
 offset=0
 for i,size in enumerate(sizes):
  if i in segments:result.append(tensor.narrow(axis,offset,size))
  offset+=size
 assert offset==tensor.shape[axis]
 return result

def assemble(spec,weights,inputs):
 shape=spec["shape"]
 ws=[d["weight"] for d in weights]
 xs=[d["x"] for d in inputs]
 ys=[d["y_local_rows"] for d in inputs]
 assert torch.equal(inputs[0]["output_rows"],inputs[1]["output_rows"])
 if spec["mode"]=="row":
  assert ws[0].shape[0]==shape[0]
  if ws[0].shape[1]==shape[1]:
   assert torch.equal(ws[0],ws[1]) and torch.equal(xs[0],xs[1])
   return ws[0],xs[0],ys[0].float(),"replicated"
  assert ws[0].shape[1]*2==shape[1]
  return torch.cat(ws,1),torch.cat(xs,1),ys[0].float()+ys[1].float(),"row_tp2_partial_sum"
 output_sizes=weights[0]["output_sizes"]
 assert output_sizes==weights[1]["output_sizes"]
 factor=sum(output_sizes)//ws[0].shape[0]
 assert factor in (1,2) and sum(output_sizes)==factor*ws[0].shape[0]
 if factor==1:
  assert torch.equal(ws[0],ws[1]) and torch.equal(xs[0],xs[1])
  wp=pieces(ws[0],output_sizes,spec["segments"],0)
  yp=pieces(ys[0],output_sizes,spec["segments"],1)
  return torch.cat(wp,0),xs[0],torch.cat(yp,1).float(),"column_replicated"
 sizes=[s//2 for s in output_sizes]
 wp=[pieces(w,sizes,spec["segments"],0) for w in ws]
 yp=[pieces(y,sizes,spec["segments"],1) for y in ys]
 assert torch.equal(xs[0],xs[1]),spec["id"]
 w=torch.cat([torch.cat([wp[0][i],wp[1][i]],0) for i in range(len(wp[0]))],0)
 y=torch.cat([torch.cat([yp[0][i],yp[1][i]],1) for i in range(len(yp[0]))],1)
 return w,xs[0],y.float(),"column_tp2_component_concat"

def main():
 torch.set_num_threads(8)
 sys.path.insert(0,str(Path(__file__).parents[1]/"qkv_capture"))
 from export_nvfp4 import quantize,unit_tests
 sys.path.insert(0,"/home/w00498770/algo/algorithm_9_7_2/scripts")
 import self_check
 unit_tests()
 assert (CAP/"COMPLETE.json").is_file()
 specs=json.loads((CAP/"SPECIFICATIONS.json").read_text())
 assert Counter(tuple(s["shape"]) for s in specs)==Counter(COUNTS)
 index=json.loads((MODEL/"model.safetensors.index.json").read_text())["weight_map"]
 assert not set(BANKS["development"])&set(BANKS["validation"])
 original_attention={}
 for bank in BANKS:
  original_attention[bank]={"manifest":sha(DATA/bank/"MANIFEST.json"),
                           "shards":{p.name:sha(p) for p in (DATA/bank/"attention_shards").glob("*.pt")}}
  if (DATA/bank/"linear_shards").exists():raise FileExistsError(DATA/bank/"linear_shards")
  (DATA/bank/".linear_build").mkdir()
 records={bank:[] for bank in BANKS};audits=[];bucket_index=Counter()
 prompts={p["sample_id"]:p for p in json.loads((CAP/"prompts.json").read_text())}
 started=time.monotonic()
 for spec in specs:
  weight_files=[CAP/"raw/weights"/f"{spec['module']}.tp{rank}.pt" for rank in (0,1)]
  weights=[torch.load(p,weights_only=True,map_location="cpu") for p in weight_files]
  for rank,d in enumerate(weights):assert d["rank"]==rank and d["tp_size"]==2
  with safe_open(str(MODEL/index[spec["weight_key"]]),framework="pt") as f:
   checkpoint=f.get_tensor(spec["weight_key"])
  pools={}
  for family in [f for fs in BANKS.values() for f in fs]:
   paths=[CAP/"raw/activations"/family/f"{spec['module']}.tp{rank}.pt" for rank in (0,1)]
   inputs=[torch.load(p,weights_only=True,map_location="cpu") for p in paths]
   for rank,d in enumerate(inputs):
    assert d["rank"]==rank and d["seq_len"]==4096 and d["token_ids_sha256"]==prompts[family]["token_ids_sha256"]
   w,x,y,mode=assemble(spec,weights,inputs)
   assert list(w.shape)==spec["shape"] and torch.equal(w,checkpoint),spec
   assert x.shape==(4096,spec["shape"][1]) and x.dtype==torch.bfloat16
   assert bool(torch.isfinite(x).all()) and bool(torch.isfinite(w).all())
   rows=inputs[0]["output_rows"]
   reference=x[rows].float()@w.float().T
   err=y-reference
   nrmse=float(err.square().mean().sqrt()/reference.square().mean().sqrt().clamp_min(1e-12))
   assert nrmse<.015,(spec["id"],family,nrmse)
   audits.append({"id":spec["id"],"family":family,"mode":mode,"checkpoint_weight_equal":True,
                  "replay_rows":len(rows),"nrmse":nrmse,"max_abs":float(err.abs().max()),
                  "input_files":[{"path":str(p.relative_to(CAP)),"sha256":sha(p)} for p in paths]})
   pools[family]=x.contiguous()
  wp,wstats=quantize(checkpoint)
  shape=tuple(spec["shape"]);j=bucket_index[shape];bucket_index[shape]+=1
  M=SUPPORT[round(j*(len(SUPPORT)-1)/(COUNTS[shape]-1))]
  test_rows=torch.floor((torch.arange(M)+.5)*4096/M).long()
  for bank,families in BANKS.items():
   held=families[int(spec["id"])%3];donors=[f for f in families if f!=held]
   offset={f:0 for f in donors};panel=[];calib=[]
   for i,length in enumerate(CALIB):
    donor=donors[i%2];start=offset[donor];offset[donor]+=length
    values=pools[donor][start:start+length]
    pair,stats=quantize(values);calib.append(pair)
    panel.append({"document":donor,"start":start,"rows":length,"quantization":stats})
   xp,xstats=quantize(pools[held][test_rows])
   group={"weight":wp,"calib_activation_list":calib,"test_activation_list":[xp]}
   self_check._normalize_linear_dataset([group])
   name=f"{spec['id']}_{spec['shape'][0]}x{spec['shape'][1]}.pt"
   path=DATA/bank/".linear_build"/name
   torch.save([group],path)
   loaded=torch.load(path,weights_only=True,map_location="cpu")
   self_check._normalize_linear_dataset(loaded)
   for before,after in zip([wp,*calib,xp],[loaded[0]["weight"],*loaded[0]["calib_activation_list"],loaded[0]["test_activation_list"][0]]):
    assert all(torch.equal(a,b) for a,b in zip(before,after))
   records[bank].append({**spec,"path":"linear_shards/"+name,"test_rows":M,"test_document":held,
        "test_token_positions":test_rows.tolist(),"calib":panel,"weight_quantization":wstats,
        "test_quantization":xstats,"bytes":path.stat().st_size,"sha256":sha(path)})
  print("MATRIX_VERIFIED_AND_EXPORTED",spec["id"],spec["shape"],"M",M,flush=True)
 for bank in BANKS:
  assert Counter(tuple(r["shape"]) for r in records[bank])==Counter(COUNTS)
  assert set(r["test_rows"] for r in records[bank])==set(SUPPORT)
  assert sha(DATA/bank/"MANIFEST.json")==original_attention[bank]["manifest"]
  assert {p.name:sha(p) for p in (DATA/bank/"attention_shards").glob("*.pt")}==original_attention[bank]["shards"]
  os.replace(DATA/bank/".linear_build",DATA/bank/"linear_shards")
  manifest={"dataset":"qwen35_35B_linear_v1","bank":bank,"groups":records[bank],
      "group_count":50,"test_samples":50,"shape_counts":{str(k):v for k,v in COUNTS.items()},
      "test_row_support":SUPPORT,"calib_lengths":CALIB,
      "documents":BANKS[bank],"source_capture":str(CAP),
      "row_sampling":"stratified real activation rows from one 4096-token prefill; not independent short requests or decode",
      "weight_role_counts":"coverage choice, not probed role frequencies",
      "calibration_note":"local panel; nonoverlapping row slices within each donor, two donor documents per group",
      "quantization":"same E2M1/E4M3/global FP32 exporter as Attention",
      "source_script_sha256":sha(Path(__file__)),"attention_unchanged":True}
  (DATA/bank/"LINEAR_MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
 report={"matrices":50,"unique_weight_keys":len({s["weight_key"] for s in specs}),"activation_replays":len(audits),
         "max_nrmse":max(a["nrmse"] for a in audits),"checks":audits,
         "banks":{b:{"groups":50,"test_samples":50,"bytes":sum(r["bytes"] for r in rs)} for b,rs in records.items()},
         "attention_unchanged":True,"elapsed_s":time.monotonic()-started}
 (CAP/"AUDIT_AND_EXPORT.json").write_text(json.dumps(report,indent=2))
 print("LINEAR_EXPORT_PASSED",json.dumps({k:v for k,v in report.items() if k!="checks"}),flush=True)

if __name__=="__main__":main()
