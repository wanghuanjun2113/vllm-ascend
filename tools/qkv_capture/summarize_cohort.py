#!/usr/bin/env python3
"""Summarize ten complete frozen-cohort replays without fitting online scores."""
import json,math
from pathlib import Path
ROOT=Path("/home/w00498770/algo/algorithm_9_7_2/data/qwen35_35B")
R=ROOT/"replay_5"
BANKS=("development","validation")
def ranks(values):
 return [1+sum(x>v for x in values)+(sum(x==v for x in values)-1)/2 for v in values]
def corr(a,b):
 ma=sum(a)/len(a);mb=sum(b)/len(b)
 den=math.sqrt(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))
 return sum((x-ma)*(y-mb) for x,y in zip(a,b))/den if den else None
def main():
 cohort=json.loads((R/"COHORT.json").read_text())["schemes"]
 summary={"rows":[],"banks":{}}
 for c in cohort:
  row={k:c[k] for k in ("id","name","score","sha","source_url")}
  for bank in BANKS:
   d=json.loads((R/f"{c['id']}_{bank}.json").read_text())
   if d["status"]!="complete" or d["cases"]!=210:raise RuntimeError(f"Incomplete {c['id']} {bank}")
   row[bank]={"score_100pt":d["score_100pt"],"mean_score_100pt":d["score_100pt"]/210,
              "dynamic_seconds":d["dynamic_seconds"],"wall_seconds":d["wall_seconds"]}
  summary["rows"].append(row)
 for bank in BANKS:
  scores=[r[bank]["score_100pt"] for r in summary["rows"]]
  online=[r["score"] for r in summary["rows"]]
  pairs=[]
  for i in range(5):
   for j in range(i+1,5):
    pairs.append({"higher_online":cohort[i]["id"],"lower_online":cohort[j]["id"],
                  "local_delta":scores[i]-scores[j],"same_direction":scores[i]>scores[j],
                  "tie":scores[i]==scores[j]})
  per_case={}
  manifest=json.loads((ROOT/bank/"MANIFEST.json").read_text())
  for c in cohort:
   result=json.loads((R/f"{c['id']}_{bank}.json").read_text())
   buckets={}
   for s in result["samples"]:
    g=manifest["groups"][s["group"]]
    t=g["test_lengths"][s["sample"]]
    for key in (f"T={t}",f"layer={g['layer']}",f"document={g['test_family']}"):
     buckets.setdefault(key,[]).append(s["score_100pt"])
   per_case[c["id"]]={k:{"samples":len(v),"sum":sum(v),"mean":sum(v)/len(v)} for k,v in buckets.items()}
  summary["banks"][bank]={"ranking":[c["id"] for _,c in sorted(zip(scores,cohort),key=lambda x:x[0],reverse=True)],
                         "spearman":corr(ranks(online),ranks(scores)),
                         "pairwise_same_direction":sum(p["same_direction"] for p in pairs),
                         "pairwise_ties":sum(p["tie"] for p in pairs),"pairs":pairs,"buckets":per_case}
 summary["notes"]=["Online scores include Linear; local scores include Attention only.",
   "Each bank has 210 scored samples, with full Q/K/V quantization; T>512 uses128 sampled query rows.",
   "Documents are disjoint across all calibration and test samples between banks; folds within each bank share calibration.",
   "Domain mix differs between banks. Timing is shared-host local dynamic cost, not online time.",
   "Five historical scored solutions are a retrospective cohort, not a prospective predictor validation."]
 (R/"SUMMARY.json").write_text(json.dumps(summary,indent=2))
 text=["# Qwen3.5 NVFP4 5方案开发/验证回放","","全部10次回放完成：每次30组、210样本。",
 "| 方案 | 线上总分 | 开发Attention分 | 验证Attention分 | 开发dynamic秒 | 验证dynamic秒 |",
 "|---|---:|---:|---:|---:|---:|"]
 for row in summary["rows"]:
  d=row["development"];v=row["validation"]
  text.append(f"| {row['id']} | {row['score']:.2f} | {d['score_100pt']:.6f} | {v['score_100pt']:.6f} | {d['dynamic_seconds']:.3f} | {v['dynamic_seconds']:.3f} |")
 text.extend(["","分数为test_machine的100-point累计Attention分；本地不包含Linear。",
              "长T抽样128个query行，T<=512完整输出；量化覆盖全量Q/K/V。"])
 for bank in BANKS:
  v=summary["banks"][bank]
  text.append(f"\n{bank}：排序 {v['ranking']}；Spearman={v['spearman']}；10个方案对同向{v['pairwise_same_direction']}，持平{v['pairwise_ties']}。")
 text.extend(["","只报告本次固定数据上的关联，不把排序相关性解释为未来提分预测。",
              "完整源码与线上回执见COHORT.json；逐样本分数、执行参数、原始日志路径与哈希见各回放JSON。",
              "每层、每T、每文档分桶见SUMMARY.json。未进行线上提交，也未使用NPU。"])
 (R/"RESULTS.md").write_text("\n".join(text)+"\n")
 print(json.dumps({"rows":summary["rows"],"banks":{k:{a:v[a] for a in ("ranking","spearman","pairwise_same_direction","pairwise_ties")} for k,v in summary["banks"].items()}}),flush=True)
if __name__=="__main__":main()
