#!/usr/bin/env python3
"""Test frozen catastrophic-negative submissions on the fixed development bank."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from replay_online_cohort import parse_log,sha

PROJECT=Path("/home/w00498770/algo/algorithm_9_7_2")
DATA=PROJECT/"data/qwen35_35B"
OUT=DATA/"replay_negative"
TMP=Path("/tmp/qwen35_negative_replay")

def main():
 p=argparse.ArgumentParser();p.add_argument("--id",required=True);p.add_argument("--cpus",required=True);a=p.parse_args()
 cohort=json.loads((OUT/"COHORT.json").read_text())
 c=next(x for x in cohort["schemes"] if x["id"]==a.id)
 source=TMP/a.id/"solution.py";assert sha(source)==c["source_sha256"]
 manifest_path=DATA/"development/MANIFEST.json";mh=sha(manifest_path)
 manifest=json.loads(manifest_path.read_text())
 for g in manifest["groups"]:assert sha(DATA/"development"/g["path"])==g["sha256"]
 tester=PROJECT/"scripts/test_machine.py";assert sha(tester)==cohort["tester_sha256"]
 cmd=[sys.executable,str(tester),"--owner","qwen35-negative-"+a.id,"--cpus",a.cpus,
      "--solution",str(source),"--data-dir",str(DATA/"development"),"--skip-linear",
      "--attention-runtime-scope","role-isolated","--attention-call-order","interleaved",
      "--attention-query-row-limit","128","--attention-full-query-max-t","512","--timeout-seconds","0"]
 log=TMP/(a.id+".log")
 t=time.monotonic()
 with log.open("w") as f:
  run=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1"))
 text=log.read_text();samples,total,dynamic=parse_log(text)
 complete=run.returncode==0 and len(samples)==210 and total is not None
 result={"id":a.id,"bank":"development","source":c,"bank_manifest_sha256":mh,
         "exit_code":run.returncode,"status":"complete" if complete else "failed",
         "cases":len(samples),"score_100pt":total,"dynamic_seconds":dynamic,
         "wall_seconds":time.monotonic()-t,"command":cmd,"log":str(log),"log_sha256":sha(log),
         "samples":samples,"negative_cases":sum(x["score_100pt"]<0 for x in samples),
         "total_negative_reproduced":bool(total<0) if total is not None else None}
 assert sha(source)==c["source_sha256"] and sha(manifest_path)==mh and sha(tester)==cohort["tester_sha256"]
 if samples:
  result["worst_case"]=min(samples,key=lambda x:x["score_100pt"])
  buckets={}
  for x in samples:
   g=manifest["groups"][x["group"]];t=g["test_lengths"][x["sample"]]
   for key in (f"T={t}",f"layer={g['layer']}",f"document={g['test_family']}"):
    buckets.setdefault(key,[]).append(x["score_100pt"])
  result["buckets"]={k:{"n":len(v),"sum":sum(v),"mean":sum(v)/len(v),"negative_cases":sum(x<0 for x in v)} for k,v in buckets.items()}
 (OUT/(a.id+".json")).write_text(json.dumps(result,indent=2))
 print(json.dumps({k:v for k,v in result.items() if k not in ("samples","buckets","command")}),flush=True)
 if not complete:print(text[-7000:],flush=True)
 sys.exit(0 if complete else 1)
if __name__=="__main__":main()
