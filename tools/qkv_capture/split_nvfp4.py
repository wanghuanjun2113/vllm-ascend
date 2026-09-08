#!/usr/bin/env python3
"""Freeze document-disjoint banks, rebuilding calibration within each bank."""
import hashlib,json,os,sys
from pathlib import Path
import torch

ROOT=Path("/home/w00498770/algo/algorithm_9_7_2/data/qwen35_35B")
BANKS={"development":["alice","python_ast","chinese_systems"],
       "validation":["holmes","frankenstein","arithmetic"]}
LENGTHS=[10,128,256,512,1024,2048,4096]
CALIB=[128,256,512,1024,2048]

def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()

def main():
 torch.set_num_threads(8)
 sys.path.insert(0,"/home/w00498770/algo/algorithm_9_7_2/scripts")
 import self_check
 original=json.loads((ROOT/"MANIFEST.json").read_text())
 lookup={(g["layer"],g["test_family"]):g for g in original["groups"]}
 assert not set(BANKS["development"])&set(BANKS["validation"])
 for bank in BANKS:
  if (ROOT/bank).exists():raise FileExistsError(ROOT/bank)
  (ROOT/bank/".building").mkdir(parents=True)
 split={"version":"qwen35_document_split_v1","banks":{},"source_manifest_sha256":sha(ROOT/"MANIFEST.json"),
        "script_sha256":sha(Path(__file__)),"note":"Frozen before five-scheme replay; all documents already exposed by raw capture. No claim of a blind independent dataset."}
 for bank,families in BANKS.items():
  groups=[]
  for layer in sorted({x[0] for x in lookup}):
   samples={}
   sources={}
   for family in families:
    entry=lookup[layer,family];path=ROOT/entry["path"]
    assert sha(path)==entry["sha256"]
    g=torch.load(path,weights_only=True,map_location="cpu")[0]
    assert len(g["test"])==7
    samples[family]=dict(zip(LENGTHS,g["test"]))
    sources[family]=entry["sha256"]
   for held_index,held in enumerate(families):
    donors=[x for x in families if x!=held]
    panel=[{"family":donors[i%2],"seq_len":t} for i,t in enumerate(CALIB)]
    group={"q_num_heads":16,"kv_num_heads":2,"head_dim":256,
           "calib":[samples[x["family"]][x["seq_len"]] for x in panel],
           "test":[samples[held][t] for t in LENGTHS]}
    self_check._normalize_attention_dataset([group])
    name=f"{len(groups):03d}_layer{layer:02d}_holdout_{held}.pt"
    out=ROOT/bank/".building"/name
    torch.save([group],out)
    loaded=self_check._normalize_attention_dataset(torch.load(out,weights_only=True,map_location="cpu"))[0]
    for before,after in zip(group["calib"]+group["test"],loaded["calib"]+loaded["test"]):
     for role in ("q","k","v"):
      assert all(torch.equal(a,b) for a,b in zip(before[role],after[role]))
    groups.append({"index":len(groups),"path":"attention_shards/"+name,"layer":layer,
                   "test_family":held,"test_lengths":LENGTHS,"calib":panel,
                   "bytes":out.stat().st_size,"sha256":sha(out),"source_shard_sha256":sources})
   print(bank,"layer",layer,flush=True)
  os.replace(ROOT/bank/".building",ROOT/bank/"attention_shards")
  manifest={"bank":bank,"documents":families,"group_count":30,"test_samples":210,"calib_per_group":5,
            "groups":groups,"parent_manifest_sha256":split["source_manifest_sha256"],
            "quantization":"unchanged NVFP4 pairs; no requantization",
            "folds_independent":False,"calib_and_test_documents_disjoint_within_group":True}
  (ROOT/bank/"MANIFEST.json").write_text(json.dumps(manifest,indent=2))
  split["banks"][bank]={"documents":families,"manifest_sha256":sha(ROOT/bank/"MANIFEST.json"),
                         "groups":30,"test_samples":210,"bytes":sum(g["bytes"] for g in groups)}
 (ROOT/"SPLIT.json").write_text(json.dumps(split,indent=2))
 print("SPLIT_PASSED",json.dumps(split["banks"]),flush=True)

if __name__=="__main__":main()
