import sys,json,time,subprocess,os,hashlib
from pathlib import Path
sys.path.insert(0,"/home/w00498770/dev/worktrees/vllm-023-qwen35-qkv/vllm-ascend/tools/qkv_capture")
from replay_online_cohort import parse_log
root=Path("/tmp/qwen35_vprior_pilot")
out=Path("/home/w00498770/algo/algorithm_9_7_2/data/qwen35_35B/prior_pilot")
mode,bank,cpus=sys.argv[1:]
build=json.loads((out/"BUILD.json").read_text());source=root/mode/"solution.py"
assert hashlib.sha256(source.read_bytes()).hexdigest()==build["arms"][mode]["sha256"]
cmd=[sys.executable,"/home/w00498770/algo/algorithm_9_7_2/scripts/test_machine.py",
"--owner",f"vprior-{mode}-{bank}","--cpus",cpus,"--solution",str(source),
"--data-dir",build["subsets"][bank]["path"],"--skip-linear",
"--attention-runtime-scope","role-isolated","--attention-call-order","interleaved",
"--attention-query-row-limit","128","--attention-full-query-max-t","512","--timeout-seconds","0"]
log=root/(mode+"_"+bank+".log");start=time.monotonic()
with log.open("w") as f:
 p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1"))
text=log.read_text();samples,score,dynamic=parse_log(text)
result={"arm":mode,"bank":bank,"cases":len(samples),"score":score,
"baseline":build["subsets"][bank]["baseline_score_100pt"],
"delta":score-build["subsets"][bank]["baseline_score_100pt"] if score is not None else None,
"dynamic_s":dynamic,"wall_s":time.monotonic()-start,"exit_code":p.returncode,
"status":"complete" if p.returncode==0 and len(samples)==63 else "failed",
"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"samples":samples,"command":cmd,
"log":str(log),"log_sha256":hashlib.sha256(log.read_bytes()).hexdigest()}
(out/(mode+"_"+bank+".json")).write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ("samples","command")}),flush=True)
if result["status"]!="complete":print(text[-6000:],flush=True)
