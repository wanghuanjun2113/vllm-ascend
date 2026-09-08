#!/usr/bin/env python3
"""Partition development into three exact group copies without changing its files."""
import hashlib,json,shutil,sys
from pathlib import Path

ROOT=Path("/home/w00498770/algo/algorithm_9_7_2/data/qwen35_35B")
SOURCE=ROOT/"development"
OUTPUT=ROOT/"development_subsets"

def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()

def snapshot():
 return {str(p.relative_to(SOURCE)):sha(p) for p in SOURCE.rglob("*") if p.is_file()}

def main():
 import torch
 torch.set_num_threads(4)
 sys.path.insert(0,"/home/w00498770/algo/algorithm_9_7_2/scripts")
 import self_check
 before=snapshot()
 manifest=json.loads((SOURCE/"MANIFEST.json").read_text())
 if OUTPUT.exists():raise FileExistsError(OUTPUT)
 OUTPUT.mkdir()
 assigned=[];subsets=[]
 for number,family in enumerate(manifest["documents"],1):
  name=f"subset_{number}";dest=OUTPUT/name
  (dest/"attention_shards").mkdir(parents=True)
  groups=[]
  for old in manifest["groups"]:
   if old["test_family"]!=family:continue
   src=SOURCE/old["path"]
   assert before[old["path"]]==old["sha256"]
   target=dest/"attention_shards"/f"{len(groups):03d}_layer{old['layer']:02d}_holdout_{family}.pt"
   shutil.copyfile(src,target)
   assert src.stat().st_ino!=target.stat().st_ino
   assert sha(target)==old["sha256"]
   loaded=self_check._normalize_attention_dataset(torch.load(target,weights_only=True,map_location="cpu"))
   assert len(loaded)==1 and len(loaded[0]["calib"])==5 and len(loaded[0]["test"])==7
   new=dict(old,index=len(groups),parent_index=old["index"],parent_path=old["path"],
            path=str(target.relative_to(dest)))
   groups.append(new);assigned.append(old["index"])
  assert len(groups)==10 and len({g["layer"] for g in groups})==10
  child={**manifest,"bank":name,"group_count":10,"test_samples":70,"groups":groups,
         "test_documents":[family],"parent_bank":str(SOURCE),
         "parent_manifest_sha256":before["MANIFEST.json"],
         "calibration":"unchanged from each original development group",
         "storage":"independent copies, not symlinks or hardlinks"}
  (dest/"MANIFEST.json").write_text(json.dumps(child,ensure_ascii=False,indent=2))
  (dest/"README.md").write_text(
   f"# {name}\n\n测试文档：{family}。10组、70个测试样本；全部10层和7种T。\n"
   "每组5份校准样本与原开发集完全一致，仍来自另外两个开发文档。\n"
   "源分片逐字节复制，parent_index记录原组编号；未重新量化或修改方案。\n"
   "这是快速开发子集，不是独立验证集。不同子集会共享校准文档。\n"
   "运行原test_machine.py时使用 --skip-linear 和 role-isolated；\n"
   "推荐T<=512完整输出、长T抽样128个query行，与完整开发集保持相同口径。\n")
  subsets.append({"name":name,"test_document":family,"groups":10,"test_samples":70,
                  "parent_indices":[g["parent_index"] for g in groups],
                  "bytes":sum(g["bytes"] for g in groups),"manifest_sha256":sha(dest/"MANIFEST.json")})
 assert sorted(assigned)==list(range(30)) and len(set(assigned))==30
 after=snapshot();assert before==after
 report={"subsets":subsets,"disjoint_group_partition":True,"union_equals_original_groups":True,
         "original_files_unchanged":True,"original_files_checked":len(before),
         "parent_files_sha256":before,"all_copied_shards_sha256_match":True,
         "all_shards_deserialized_and_normalized":30,"algorithm_tests_rerun":False,"npu_used":False}
 (OUTPUT/"VERIFICATION.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
 (OUTPUT/"README.md").write_text(
  "# 开发集快速子集\n\n原development目录保持原样。三个子集各10组70样本，合并恰好是原30组210样本。\n"
  "subset_1：Alice；subset_2：Python源码；subset_3：中文技术文本。\n"
  "每个子集覆盖10个Full Attention层和全部7种T，并完整沿用原校准样本。\n"
  "后续先选一个相关子集做父子配对，出现收益后检查另外两个子集，再回完整开发集和验证集。\n"
  "样本数为原来的1/3，但没有重测实际耗时，不承诺严格3倍加速。\n"
  "子集为独立文件副本，避免通过链接意外修改原开发集。详情见各MANIFEST与VERIFICATION.json。\n")
 print(json.dumps({k:v for k,v in report.items() if k!="parent_files_sha256"},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
