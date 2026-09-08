import importlib.util,json,hashlib,sys
from pathlib import Path
import torch
torch.set_num_threads(8)
root=Path("/tmp/qwen35_vprior_pilot")
spec=importlib.util.spec_from_file_location("frozen_prior_parent",root/"baseline/solution.py")
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
data=Path("/home/w00498770/algo/algorithm_9_7_2/data/qwen35_35B")
manifest=json.loads((data/"development/MANIFEST.json").read_text())
collected={}
for g in manifest["groups"]:
 p=data/"development"/g["path"]
 h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==g["sha256"]
 group=torch.load(p,weights_only=True,map_location="cpu")[0]
 for sample in group["test"]:
  t=int(sample["q"][0].shape[0])
  q=m._dequantize_nvfp4_fp32(*sample["q"]);k=m._dequantize_nvfp4_fp32(*sample["k"])
  lags,_=m._v_local_window2_mass(q,k,16,2,256)
  tails=m._derive_v_tail([sample],16,2,256)
  ratio=tails[str(t)]/t if str(t) in tails else torch.zeros(2)
  collected.setdefault(t,[]).append(torch.cat((lags,ratio[:,None]),-1))
prior={}
for t,values in sorted(collected.items()):
 stack=torch.cat(values)
 v=stack.mean(0)
 prior[t]={"lags":v[:9].tolist(),"tail_ratio":float(v[9]),"observations":len(stack)}
assert len(prior)==7
out=data/"prior_pilot"
(out/"PRIOR.json").write_text(json.dumps(prior,indent=2))
(root/"prior.json").write_text(json.dumps(prior))
print("PRIOR_FROZEN",json.dumps(prior),flush=True)
