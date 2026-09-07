#!/usr/bin/env python3
import json
import subprocess
import time
import urllib.request
from pathlib import Path
import manage
from analyze import TraceStore

ART=manage.ART
TOOLS=Path(__file__).parent


def status(stage, **extra):
    result={"stage":stage,"timestamp":time.time(),**extra}
    (ART/"matrix-status.json").write_text(json.dumps(result,indent=2))
    print(result,flush=True)


def wait_ready():
    deadline=time.monotonic()+900
    while time.monotonic()<deadline:
        try:
            if urllib.request.urlopen("http://127.0.0.1:18327/health",timeout=2).status==200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError("model did not become ready")


def bench(name, n=8, concurrency=(1,4), temperature=0, warmup=4):
    run=ART/name
    cmd=["python",str(TOOLS/"benchmark.py"),"--datasets",str(ART/"datasets"),
         "--out",str(run),"--name",name,"--n",str(n),
         "--concurrency",*[str(c) for c in concurrency],
         "--temperature",str(temperature),"--warmup",str(warmup)]
    with (run/"benchmark.log").open("w") as f:
        subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=1800)


def audit(name):
    time.sleep(2)
    result=TraceStore(ART/name/"trace").audit()
    (ART/name/"audit.json").write_text(json.dumps(result,indent=2))
    if not result["complete"]:
        raise RuntimeError("trace audit failed: "+name)
    status("audit_pass",name=name,requests=len(result["requests"]))


def main():
    for name,k in (("baseline-final",0),("top128-final",128),("top1024-final",1024)):
        status("starting",name=name)
        manage.stop()
        manage.start(name,k)
        wait_ready()
        status("benchmark",name=name)
        bench(name)
        if k:
            audit(name)
    status("random_sampling")
    extra=ART/"random1024-final";extra.mkdir()
    bench("random1024-final",n=2,concurrency=(1,4),temperature=0.7,warmup=0)
    audit("top1024-final")
    status("finished_matrix")
    manage.stop()
    for name in ("top128-final","top1024-final"):
        audit(name)
    status("done")


if __name__=="__main__":
    try:main()
    except Exception as exc:
        status("failed",error=repr(exc))
        raise
