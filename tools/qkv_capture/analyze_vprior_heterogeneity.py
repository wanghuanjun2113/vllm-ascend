#!/usr/bin/env python3
"""Diagnose V-stat heterogeneity and a fixed two-type calibration-only predictor."""
import hashlib,importlib.util,json,sys
from pathlib import Path
import torch

DATA=Path("/home/w00498770/algo/algorithm_9_7_2/data/qwen35_35B")
OUT=DATA/"prior_heterogeneity"
PARENT=Path("/tmp/qwen35_online_replay5/sources/sota21690/solution.py")
TS=[10,128,256,512,1024,2048,4096]
CALIB_TS=[128,256,512,1024,2048]

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()

def extract(bank,m):
 mf=DATA/bank/"MANIFEST.json";manifest=json.loads(mf.read_text())
 records=[]
 for g in manifest["groups"]:
  p=DATA/bank/g["path"];assert sha(p)==g["sha256"]
  group=torch.load(p,map_location="cpu",weights_only=True)[0]
  for sample in group["test"]:
   t=sample["q"][0].shape[0]
   q=m._dequantize_nvfp4_fp32(*sample["q"]);k=m._dequantize_nvfp4_fp32(*sample["k"])
   lag,_=m._v_local_window2_mass(q,k,16,2,256)
   tail=m._derive_v_tail([sample],16,2,256)
   ratio=tail[str(t)]/t if str(t) in tail else torch.zeros(2)
   for head in range(2):
    records.append({"bank":bank,"layer":g["layer"],"document":g["test_family"],
                    "head":head,"T":t,"lags":lag[head].tolist(),"tail_ratio":float(ratio[head])})
  print(bank,"layer",g["layer"],g["test_family"],flush=True)
 return manifest,records

def eta(x,labels):
 mean=x.mean();ss=(x-mean).square().sum()
 if ss<=1e-25:return None
 between=torch.zeros((),dtype=x.dtype)
 for label in set(labels):
  v=x[torch.tensor([z==label for z in labels])]
  between+=len(v)*(v.mean()-mean).square()
 return float(between/ss)

def diagnostic(records):
 result={}
 for t in TS:
  rows=[r for r in records if r["T"]==t]
  metrics={"lag1":[r["lags"][1] for r in rows],
           "band_offdiag_mass":[2*sum(r["lags"][1:]) for r in rows],
           "tail_ratio":[r["tail_ratio"] for r in rows]}
  item={}
  for name,values in metrics.items():
   x=torch.tensor(values,dtype=torch.float64)
   layer=[r["layer"] for r in rows]
   head=[r["head"] for r in rows]
   pair=[(r["layer"],r["head"]) for r in rows]
   document=[r["document"] for r in rows]
   # Predict a document's statistic from the same layer/head in other documents.
   exact=global_loss=0.
   for i,row in enumerate(rows):
    donors=[j for j,r in enumerate(rows) if r["document"]!=row["document"]]
    matched=[j for j in donors if pair[j]==pair[i]]
    exact+=(x[i]-x[matched].mean()).square().item()
    global_loss+=(x[i]-x[donors].mean()).square().item()
   item[name]={"mean":float(x.mean()),"p10":float(x.quantile(.1)),"median":float(x.median()),
               "p90":float(x.quantile(.9)),"std":float(x.std(unbiased=False)),
               "layer_eta2":eta(x,layer),"head_global_eta2":eta(x,head),
               "layer_head_eta2":eta(x,pair),"document_eta2":eta(x,document),
               "leave_document_out_matched_head_MSE_over_global":exact/global_loss if global_loss>1e-25 else None}
  result[t]=item
 return result

def arrays(manifest,records):
 lookup={(r["layer"],r["document"],r["T"],r["head"]):r for r in records}
 X=[];Y=[];C=[];meta=[]
 for g in manifest["groups"]:
  for h in range(2):
   observed=[]
   for p in g["calib"]:
    r=lookup[g["layer"],p["family"],p["seq_len"],h]
    observed.append((p["seq_len"],r))
   # Input: statistics of this group's calibration Q/K, no layer number.
   x=[v for t,r in sorted(observed) for v in r["lags"][1:]+[r["tail_ratio"]]]
   y=[];cal=[]
   pool=torch.tensor([r["lags"][1:] for _,r in observed]).mean(0).tolist()
   for t in TS:
    r=lookup[g["layer"],g["test_family"],t,h]
    y.append(r["lags"][1:]+[r["tail_ratio"]])
    exact=[r for tt,r in observed if tt==t]
    lags=exact[0]["lags"][1:] if exact else pool
    near_t,near=min(observed,key=lambda pair:abs(pair[0]-t))
    tail=(near["tail_ratio"]*near_t/t if t>=128 else 0.)
    cal.append(lags+[tail])
   X.append(x);Y.append(y);C.append(cal)
   meta.append({"layer":g["layer"],"head":h,"document":g["test_family"]})
 return torch.tensor(X,dtype=torch.float64),torch.tensor(Y,dtype=torch.float64),torch.tensor(C,dtype=torch.float64),meta

def kmeans2(x):
 first=int(x.square().sum(1).argmax())
 second=int((x-x[first]).square().sum(1).argmax())
 centers=x[[first,second]].clone()
 for _ in range(50):
  labels=torch.cdist(x,centers).argmin(1)
  assert all(bool((labels==k).any()) for k in range(2))
  update=torch.stack([x[labels==k].mean(0) for k in range(2)])
  if torch.allclose(update,centers,atol=1e-12,rtol=0):break
  centers=update
 return centers,torch.cdist(x,centers).argmin(1)

def main():
 torch.set_num_threads(8)
 assert sha(PARENT)=="9153c980e6ce318f193c73110ac61ed660acbad48f181cd1eff72598849ae98c"
 OUT.mkdir(exist_ok=False)
 plan={"K":2,"fit":"development only","features":"5 calibration lengths x (lag1..8,tail/T), no layer/head/document ID",
       "validation":"frozen centers and target templates; no refit",
       "metric":"statistic reconstruction MSE, standardized by development target std per T/feature; NOT Attention score",
       "limits":"3 documents per bank; layer-head eta2 is descriptive; identity-matched prediction is an oracle diagnostic, not deployable routing"}
 (OUT/"PLAN.json").write_text(json.dumps(plan,indent=2))
 spec=importlib.util.spec_from_file_location("vprior_stats_parent",PARENT)
 m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 dev_manifest,dev=extract("development",m)
 (OUT/"development_records.json").write_text(json.dumps(dev))
 dX,dY,dC,dmeta=arrays(dev_manifest,dev)
 mean=dX.mean(0);scale=dX.std(0,unbiased=False).clamp_min(1e-9)
 centers,labels=kmeans2((dX-mean)/scale)
 templates=torch.stack([dY[labels==k].mean(0) for k in range(2)])
 global_template=dY.mean(0);yscale=dY.std(0,unbiased=False).clamp_min(1e-9)
 frozen={"mean":mean.tolist(),"scale":scale.tolist(),"centers":centers.tolist(),"templates":templates.tolist(),
         "global_template":global_template.tolist(),"target_scale":yscale.tolist(),"cluster_counts":[int((labels==k).sum()) for k in range(2)]}
 (OUT/"FROZEN_TWO_TYPE_MODEL.json").write_text(json.dumps(frozen,indent=2))
 frozen_sha=sha(OUT/"FROZEN_TWO_TYPE_MODEL.json")
 val_manifest,val=extract("validation",m)
 (OUT/"validation_records.json").write_text(json.dumps(val))
 vX,vY,vC,vmeta=arrays(val_manifest,val)
 vl=torch.cdist((vX-mean)/scale,centers).argmin(1)
 scores={}
 for bank,Y,C,L,meta in (("development",dY,dC,labels,dmeta),("validation",vY,vC,vl,vmeta)):
  predictions={"global_prior":global_template.expand_as(Y),"two_type_prior":templates[L],"calibration_statistics":C}
  err={k:((pred-Y)/yscale).square() for k,pred in predictions.items()}
  # Zero-variance dimensions (T10 tail) agree exactly and contribute zero.
  score={k:float(v.mean()) for k,v in err.items()}
  perT={t:{k:float(v[:,i].mean()) for k,v in err.items()} for i,t in enumerate(TS)}
  scores[bank]={"standardized_statistic_MSE":score,"per_T":perT,
                "cluster_counts":[int((L==k).sum()) for k in range(2)],
                "assignments":[dict(r,type=int(l)) for r,l in zip(meta,L)]}
 # Stability of type labels across documents of the same layer/head.
 stable={}
 for bank,meta,L in (("development",dmeta,labels),("validation",vmeta,vl)):
  groups={}
  for row,label in zip(meta,L):groups.setdefault((row["layer"],row["head"]),[]).append(int(label))
  stable[bank]={"unanimous_layer_heads":sum(len(set(v))==1 for v in groups.values()),"total_layer_heads":len(groups)}
 result={"development":diagnostic(dev),"validation":diagnostic(val),"type_predictor":scores,"type_stability":stable,
         "frozen_model_sha256":frozen_sha,"parent_source_sha256":sha(PARENT),
         "data_manifest_sha256":{b:sha(DATA/b/"MANIFEST.json") for b in ("development","validation")},
         "script_sha256":sha(Path(__file__))}
 assert sha(OUT/"FROZEN_TWO_TYPE_MODEL.json")==frozen_sha
 (OUT/"SUMMARY.json").write_text(json.dumps(result,indent=2))
 print("HETEROGENEITY_ANALYSIS_COMPLETE",json.dumps({"predictor":{b:v["standardized_statistic_MSE"] for b,v in scores.items()},"stability":stable}),flush=True)

if __name__=="__main__":main()
