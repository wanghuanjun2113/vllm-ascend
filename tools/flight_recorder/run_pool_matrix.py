import json,subprocess,time,hashlib
from pathlib import Path
import manage
from run_matrix import wait_ready,bench,audit
ART=manage.ART
BASE=manage.BASE
def state(stage,**kw):
    obj={"stage":stage,"time":time.time(),**kw}
    (ART/"pool-matrix-status.json").write_text(json.dumps(obj,indent=2))
    print(obj,flush=True)
def snapshot(name):
    path=ART/name/"manifest.json";meta=json.loads(path.read_text())
    meta["recording_backend"]="cpu_pool"
    meta["first_request_source"]={}
    for repo in ("vllm","vllm-ascend"):
        root=BASE/repo
        meta["first_request_source"][repo]={
            "head":subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"]).decode().strip(),
            "diff_sha256":hashlib.sha256(subprocess.check_output(["git","-C",str(root),"diff","HEAD"])).hexdigest()}
    path.write_text(json.dumps(meta,indent=2))
try:
    for name,k in (("pooled128",128),("pooled1024",1024)):
        if name=="pooled1024":
            state("starting",name=name);manage.stop();manage.start(name,k)
        state("waiting_ready",name=name);wait_ready();snapshot(name)
        state("benchmark",name=name);bench(name);audit(name)
    name="pooled-random1024";(ART/name).mkdir()
    state("random");bench(name,n=2,concurrency=(1,4),temperature=0.7,warmup=0)
    audit("pooled1024")
    manage.stop()
    audit("pooled128");audit("pooled1024")
    state("done")
except Exception as exc:
    state("failed",error=repr(exc))
    raise
