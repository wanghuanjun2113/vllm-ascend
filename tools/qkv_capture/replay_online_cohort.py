#!/usr/bin/env python3
"""Run a frozen online cohort with the project's normal CPU-lease evaluator."""
import argparse,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path

PROJECT=Path("/home/w00498770/algo/algorithm_9_7_2")
DATA=PROJECT/"data/qwen35_35B"
RESULT=DATA/"replay_5"
TEMP=Path("/tmp/qwen35_online_replay5")

def sha(p):
 return hashlib.sha256(p.read_bytes()).hexdigest()

def parse_log(text):
 samples=[]
 pattern=r"\[Attention\]\[Group (\d+)\]\[Sample (\d+)\] player_mse=(\S+) standard_mse=(\S+) score=(\S+)"
 for m in re.finditer(pattern,text):
  g,s,pm,sm,score=m.groups()
  samples.append({"group":int(g),"sample":int(s),"player_mse":float(pm),"standard_mse":float(sm),"score_100pt":100*float(score)})
 total=re.findall(r"Attention 100-point score=(\S+)",text)
 dynamic=re.findall(r"Attention player_dynamic=(\S+) seconds",text)
 return samples,(float(total[-1]) if total else None),(float(dynamic[-1]) if dynamic else None)

def main():
 p=argparse.ArgumentParser();p.add_argument("--bank",choices=("development","validation"),required=True)
 p.add_argument("--cpus",required=True);args=p.parse_args()
 cohort=json.loads((RESULT/"COHORT.json").read_text())
 RESULT.mkdir(exist_ok=True);(TEMP/"logs").mkdir(exist_ok=True)
 bm=DATA/args.bank/"MANIFEST.json"
 expected_bank=sha(bm)
 manifest=json.loads(bm.read_text())
 assert manifest["test_samples"]==210
 for g in manifest["groups"]:
  assert sha(DATA/args.bank/g["path"])==g["sha256"]
 for candidate in cohort["schemes"]:
  sid=candidate["id"];source=TEMP/"sources"/sid/"solution.py"
  assert sha(source)==candidate["source_sha256"]
  log=TEMP/"logs"/f"{sid}_{args.bank}.log"
  cmd=[sys.executable,str(PROJECT/"scripts/test_machine.py"),
       "--owner",f"qwen35-{sid}-{args.bank}","--cpus",args.cpus,
       "--solution",str(source),"--data-dir",str(DATA/args.bank),
       "--skip-linear","--attention-runtime-scope","role-isolated",
       "--attention-call-order","interleaved","--attention-query-row-limit","128",
       "--attention-full-query-max-t","512","--timeout-seconds","0"]
  started=time.monotonic()
  with log.open("w") as out:
   process=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1"))
  text=log.read_text()
  samples,total,dynamic=parse_log(text)
  result={"id":sid,"bank":args.bank,"online_score":candidate["score"],
          "source_commit":candidate["sha"],"source_sha256":sha(source),
          "bank_manifest_sha256":expected_bank,"command":cmd,"cpus":args.cpus,
          "exit_code":process.returncode,"cases":len(samples),"score_100pt":total,
          "dynamic_seconds":dynamic,"wall_seconds":time.monotonic()-started,
          "log":str(log),"log_sha256":sha(log),"samples":samples,
          "status":"complete" if process.returncode==0 and len(samples)==210 and total is not None else "failed"}
  assert sha(bm)==expected_bank and sha(source)==candidate["source_sha256"]
  (RESULT/f"{sid}_{args.bank}.json").write_text(json.dumps(result,indent=2))
  print(json.dumps({k:v for k,v in result.items() if k not in ("samples","command")}),flush=True)
  if result["status"]!="complete":
   print(text[-6000:],flush=True)
 print("BANK_REPLAY_FINISHED",args.bank,flush=True)

if __name__=="__main__":main()
