import json,time,statistics,os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import torch_npu
torch.npu.set_device(0)
R,V=4,248320
x=torch.randn((R,V),device="npu",dtype=torch.float32)
dst=torch.empty_like(x)
host=torch.empty((R,V),dtype=torch.float32,pin_memory=True)
chosen=torch.tensor([1,2,3,4],device="npu")
rmap=torch.arange(R,device="npu")
result={}
def timed(name,fn,reps=30):
    for _ in range(3):fn()
    torch.npu.synchronize()
    times=[]
    for _ in range(reps):
        start=time.perf_counter();fn();torch.npu.synchronize()
        times.append((time.perf_counter()-start)*1000)
    result[name]={"median_ms":statistics.median(times),"mean_ms":statistics.mean(times)}
timed("npu_full_snapshot_copy",lambda:dst.copy_(x))
timed("npu_raw_float_clone",lambda:x.to(torch.float32).clone())
timed("pinned_d2h_full_4xV",lambda:host.copy_(x,non_blocking=True))
timed("top128",lambda:torch.topk(x,128,dim=-1))
timed("top1024",lambda:torch.topk(x,1024,dim=-1))
timed("logsumexp",lambda:torch.logsumexp(x,-1))
timed("finite_counts",lambda:torch.stack([
    torch.isfinite(x).sum(-1),torch.isnan(x).sum(-1),
    torch.isposinf(x).sum(-1),torch.isneginf(x).sum(-1)],dim=-1))
def ranks():
    val=x[rmap].gather(1,chosen[:,None])
    return [(x[rmap]>val).sum(-1)+1 for _ in range(3)]
timed("three_probe_ranks_repeated_gather",ranks)
raw=x.cpu().numpy()
def rowstats(row,k):
    ids=np.argpartition(row,len(row)-k)[-k:]
    vals=row[ids]
    order=np.lexsort((ids,-vals))
    maximum=float(row.max())
    lse=maximum+np.log(np.exp(row.astype(np.float64)-maximum).sum())
    counts=[np.isfinite(row).sum(),np.isnan(row).sum(),np.isposinf(row).sum(),np.isneginf(row).sum()]
    ranks=[np.count_nonzero(row>row[i])+1 for i in (1,248044,248046)]
    return ids[order],vals[order],lse,counts,ranks
with ThreadPoolExecutor(max_workers=4) as pool:
    for k in (128,1024):
        for _ in range(3):list(pool.map(lambda row:rowstats(row,k),raw))
        times=[]
        for _ in range(30):
            start=time.perf_counter();list(pool.map(lambda row:rowstats(row,k),raw))
            times.append((time.perf_counter()-start)*1000)
        result[f"cpu_4threads_4rows_top{k}"]={"median_ms":statistics.median(times),"mean_ms":statistics.mean(times)}
result["shape"]=[R,V]
result["note"]="Synthetic standalone component timings with synchronization; not E2E overhead."
path=Path("/home/w00498770/dev/artifacts/vllm-023-logits-save/component-profile.json")
path.write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
