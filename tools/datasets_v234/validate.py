#!/usr/bin/env python3
"""Validate all six datasets with the unchanged standard evaluator and real CPU leases."""
import json,hashlib,os,re,subprocess,sys
from pathlib import Path
PROJECT=Path("/home/w00498770/algo/algorithm_9_7_2")
ART=Path("/home/w00498770/dev/artifacts/vllm-023-qwen35-qkv/versions_v234")
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
def main():
 results=[]
 tester=PROJECT/"scripts/test_machine.py";wrapper=PROJECT/"scripts/run_diagnostic_eval.py"
 tester_hash=sha(tester);wrapper_hash=sha(wrapper)
 for version in (2,3,4):
  root=PROJECT/f"data/qwen35_35B_v{version}"
  for bank in ("development","test"):
   manifest=json.loads((root/bank/"MANIFEST.json").read_text())
   for g in manifest["attention_groups"]+manifest["linear_groups"]:
    assert sha(root/bank/g["path"])==g["sha256"]
   owner=f"v{version}-{bank}-standard-zero";log=Path(f"/tmp/qwen35_v{version}_{bank}_zero.log")
   cmd=[sys.executable,str(wrapper),"--owner",owner,"--cpus","0-95","--threads","8","--",
        sys.executable,str(tester),"--owner",owner,"--cpus","0-95","--solution",str(PROJECT/"scripts/standard_hif4.py"),
        "--data-dir",str(root/bank),"--expect-zero","--attention-runtime-scope","role-isolated",
        "--attention-call-order","interleaved","--attention-query-row-limit","128",
        "--attention-full-query-max-t","512","--timeout-seconds","0"]
   with log.open("w") as out:p=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1"))
   text=log.read_text()
   passed=p.returncode==0 and "EXPECT-ZERO: PASSED" in text and bool(re.search(r"Linear\s+cases=\s*25 score=0",text)) and bool(re.search(r"Attention\s+cases=\s*125 score=0",text))
   result={"version":version,"split":bank,"passed":passed,"exit_code":p.returncode,
      "linear_cases":25,"attention_cases":125,"total_cases":150,"command":cmd,"log":str(log),"log_sha256":sha(log),
      "tester_sha256":tester_hash,"lease_wrapper_sha256":wrapper_hash,"threads":8,
      "scope":"format and zero-control, not performance or candidate ranking",
      "attention_output":"T<=512 complete; T>512 sampled128 query rows; full QKV quantization"}
   (root/bank/"TEST_VALIDATION.json").write_text(json.dumps(result,indent=2))
   results.append(result);print(json.dumps({k:v for k,v in result.items() if k not in ("command",)}),flush=True)
   if not passed:print(text[-6000:],flush=True);raise RuntimeError("Zero control failed")
 assert sha(tester)==tester_hash and sha(wrapper)==wrapper_hash
 (ART/"TEST_VALIDATION.json").write_text(json.dumps(results,indent=2))
 print("ALL_SIX_ZERO_CONTROLS_PASSED",flush=True)
if __name__=="__main__":main()
