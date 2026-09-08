#!/usr/bin/env python3
"""Export captured BF16 QKV as test_machine-compatible NVFP4 numeric pairs."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import torch

LEVELS=(0.,.5,1.,1.5,2.,3.,4.,6.)
EDGES=(.25,.75,1.25,1.75,2.5,3.5,5.)
FAMILIES=("alice","holmes","frankenstein","python_ast","chinese_systems","arithmetic")
LENGTHS=(10,128,256,512,1024,2048,4096)
CALIB_LENGTHS=(128,256,512,1024,2048)

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()

def e2m1(x):
 edges=torch.tensor(EDGES,dtype=torch.float32)
 levels=torch.tensor(LEVELS,dtype=torch.float32)
 a=x.abs().contiguous()
 code=torch.bucketize(a,edges,right=False)
 # At an exact midpoint choose the even E2M1 encoding.
 ties=a==edges[code.clamp_max(6)]
 code+=((code<7)&ties&((code%2)==1)).long()
 return (levels[code]*torch.sign(x)).to(torch.bfloat16)

def quantize(x):
 x=x.float().contiguous()
 assert x.ndim==2 and x.shape[-1]%16==0 and bool(torch.isfinite(x).all())
 blocks=x.reshape(x.shape[0],-1,16)
 amax=x.abs().max()
 global_scale=amax/(6.*448.) if float(amax)>0 else torch.tensor(1.)
 raw_scale=blocks.abs().amax(-1)/(6.*global_scale)
 local=raw_scale.clamp(2.**-9,448.).to(torch.float8_e4m3fn).float()
 effective=(local*global_scale).contiguous()
 carrier=e2m1(blocks/effective[...,None]).reshape_as(x).contiguous()
 reconstructed=(carrier.float().reshape_as(blocks)*effective[...,None]).reshape_as(x)
 mse=float((reconstructed-x).square().mean())
 stats={"global_scale_fp32":float(global_scale),"mse":mse,
        "relative_mse":mse/max(float(x.square().mean()),1e-30),
        "max_abs_error":float((reconstructed-x).abs().max()),
        "e4m3_min_clamped_blocks":int((raw_scale<2.**-9).sum()),
        "e4m3_max_clamped_blocks":int((raw_scale>448.).sum())}
 assert bool(torch.isfinite(effective).all()) and bool((effective>0).all())
 assert torch.equal(local,local.to(torch.float8_e4m3fn).float())
 return [carrier,effective],stats

def unit_tests():
 mid=torch.tensor(EDGES)
 expected=torch.tensor([0.,1.,1.,2.,2.,4.,4.],dtype=torch.bfloat16)
 assert torch.equal(e2m1(mid),expected)
 assert torch.equal(e2m1(-mid),-expected)
 lo=torch.nextafter(mid,torch.full_like(mid,-torch.inf))
 hi=torch.nextafter(mid,torch.full_like(mid,torch.inf))
 assert torch.equal(e2m1(lo),torch.tensor(LEVELS[:-1],dtype=torch.bfloat16))
 assert torch.equal(e2m1(hi),torch.tensor(LEVELS[1:],dtype=torch.bfloat16))
 for x in (torch.zeros(2,64),torch.arange(-64,64).reshape(2,64).float(),
           torch.full((2,64),1e-12),torch.full((2,64),1e6)):
  pair,stats=quantize(x)
  assert pair[0].shape==x.shape and pair[1].shape==(2,4)
  assert stats["relative_mse"]<.1
 # Independent brute-force nearest code with explicit even-code tie breaking.
 torch.manual_seed(35);x=torch.randn(4,64)
 pair,stats=quantize(x)
 normalized=x.reshape(4,4,16)/pair[1][...,None]
 levels=torch.tensor(LEVELS)
 dist=(normalized.abs()[...,None]-levels).abs()
 best=dist.min(-1,keepdim=True).values
 order=torch.tensor([0,2,4,6,1,3,5,7])
 idx=(dist[...,order]==best).int().argmax(-1)
 oracle=levels[order[idx]]*normalized.sign()
 assert torch.equal(pair[0],oracle.reshape_as(x).to(torch.bfloat16))
 print("NVFP4_NUMERIC_TESTS_PASSED",flush=True)

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--source",type=Path,required=True)
 p.add_argument("--output",type=Path,required=True)
 p.add_argument("--project",type=Path,required=True)
 args=p.parse_args()
 torch.set_num_threads(8);unit_tests()
 sys.path.insert(0,str(args.project/"scripts"))
 import self_check
 audit=json.loads((args.source/"audit.json").read_text())
 expected={e["path"]:e["sha256"] for e in audit["entries"]}
 if args.output.exists():raise FileExistsError(args.output)
 args.output.mkdir(parents=True)
 stage=args.output/".building";stage.mkdir()
 groups=[];statistics=[]
 t=time.monotonic()
 layers=audit["layers"]
 for layer in layers:
  data={}
  for family in FAMILIES:
   for length in LENGTHS:
    rel=f"merged/{family}_t{length}/layer_{layer:02d}.pt"
    path=args.source/rel
    assert sha(path)==expected[rel],rel
    raw=torch.load(path,weights_only=True,map_location="cpu")
    sample={}
    for role in ("q","k","v"):
     pair,stats=quantize(raw[role]);sample[role]=pair
     statistics.append({"layer":layer,"family":family,"seq_len":length,"role":role,**stats})
     assert pair[0].shape==(length,4096 if role=="q" else 512)
     assert pair[1].shape==(length,256 if role=="q" else 32)
    data[(family,length)]=sample
  for held in FAMILIES:
   donors=[f for f in FAMILIES if f!=held]
   calibration=[{"family":f,"seq_len":length} for f,length in zip(donors,CALIB_LENGTHS)]
   group={"q_num_heads":16,"kv_num_heads":2,"head_dim":256,
          "calib":[data[(f,length)] for f,length in zip(donors,CALIB_LENGTHS)],
          "test":[data[(held,length)] for length in LENGTHS]}
   self_check._normalize_attention_dataset([group])
   target=stage/f"{len(groups):03d}_layer{layer:02d}_holdout_{held}.pt"
   torch.save([group],target)
   # Validate what the tester will actually deserialize.
   self_check._normalize_attention_dataset(torch.load(target,weights_only=True,map_location="cpu"))
   groups.append({"index":len(groups),"path":"attention_shards/"+target.name,
                  "layer":layer,"test_family":held,"calib":calibration,
                  "test_lengths":list(LENGTHS),"bytes":target.stat().st_size,"sha256":sha(target)})
  print(json.dumps({"layer":layer,"groups_done":len(groups)}),flush=True)
 os.replace(stage,args.output/"attention_shards")
 manifest={"version":"qwen35_35B_nvfp4_v1","model":"Qwen/Qwen3.5-35B-A3B",
           "source":str(args.source),"source_audit_sha256":sha(args.source/"audit.json"),
           "source_manifest_sha256":sha(args.source/"source_manifest.json"),
           "converter_sha256":sha(Path(__file__)),"quantization":"E2M1 RNE + E4M3FN per16 + FP32 per-tensor global scale",
           "global_scale":"amax(tensor)/(6*448), or 1 for all-zero tensor",
           "interface_scale":"E4M3 block scale multiplied by global scale, stored FP32",
           "carrier_dtype":"bfloat16 numeric E2M1 values; NOT packed nibbles",
           "groups":groups,"group_count":len(groups),"calib_per_group":5,"test_per_group":7,
           "unique_qkv_samples":len(layers)*len(FAMILIES)*len(LENGTHS),
           "test_samples":len(groups)*7,"linear_present":False,
           "split":"leave one document out within each layer; folds share calibration data",
           "elapsed_s":time.monotonic()-t}
 (args.output/"MANIFEST.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
 (args.output/"QUANTIZATION_STATS.json").write_text(json.dumps(statistics,indent=2))
 summary={"groups":len(groups),"test_samples":len(groups)*7,"quantized_tensors":len(statistics),
          "total_shard_bytes":sum(g["bytes"] for g in groups),
          "max_relative_mse":max(s["relative_mse"] for s in statistics),
          "mean_relative_mse":sum(s["relative_mse"] for s in statistics)/len(statistics),
          "elapsed_s":manifest["elapsed_s"]}
 (args.output/"CONVERSION_PASSED.json").write_text(json.dumps(summary,indent=2))
 print("CONVERSION_PASSED",json.dumps(summary),flush=True)

if __name__=="__main__":main()
