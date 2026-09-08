#!/usr/bin/env python3
"""Small public-domain/code/authored diagnostic corpus, not an independent benchmark."""
import hashlib,json,urllib.request,time
from pathlib import Path
ROOT=Path("/home/w00498770/dev/artifacts/vllm-023-qwen35-qkv/corpus")
ROOT.mkdir(parents=True,exist_ok=True)
sources={}
for name,url in {
 "alice":"https://www.gutenberg.org/ebooks/11.txt.utf-8",
 "holmes":"https://www.gutenberg.org/ebooks/1661.txt.utf-8",
 "frankenstein":"https://www.gutenberg.org/ebooks/84.txt.utf-8"}.items():
 for attempt in range(5):
  try:
   with urllib.request.urlopen(url+f"?qkv={attempt}",timeout=60) as r:text=r.read().decode("utf-8-sig")
   break
  except Exception:
   if attempt==4:raise
   time.sleep(2)

 if "*** START OF" in text:
  text=text.split("*** START OF",1)[1].split("***",1)[1]
 (ROOT/(name+".txt")).write_text(text)
 sources[name]={"source":url,"kind":"public_domain_english"}
path=Path("/usr/local/python3.12.13/lib/python3.12/ast.py")
(ROOT/"python_ast.txt").write_text(path.read_text())
sources["python_ast"]={"source":str(path),"kind":"python_standard_library_PSF"}
paragraphs=[]
for i in range(160):
 paragraphs.append(f"第{i+1}个系统设计案例：某推理服务收到{i+3}个请求，输入长度为{128+i*17}个词元。调度器需要区分等待队列、正在执行的预填充请求和逐步生成的请求。首先记录请求到达时间、分配的缓存块和已计算的位置，再检查共享前缀是否满足一致性约束。若缓存不足，应依据明确的优先级选择等待或重新计算，不能把尚未完成的计算误认为可复用状态。实验采用固定文本与采样设置，对照开启和关闭缓存两种情况，分别记录首词元延迟、输出词元间隔及最终文本。分析时应把测量事实、代码行为和待检验的解释分开，不应将局部函数加速直接换算为整个服务的收益。案例中的业务对象编号为{i*31+7}，服务端口为{18000+i}。\n")
(ROOT/"chinese_systems.txt").write_text("\n".join(paragraphs))
sources["chinese_systems"]={"source":"authored deterministic technical cases","kind":"synthetic_chinese_diagnostic"}
paragraphs=[]
for i in range(180):
 a=13+i*7;b=19+i*3
 paragraphs.append(f"Problem {i+1}. A warehouse receives {a} packages in the morning and {b} in the afternoon. It sends out {i+5} packages. Compute the remaining count and explain why the subtraction is performed after combining arrivals. For a second experiment, arrange the remaining packages into batches of {i%7+2}, identifying the quotient and remainder. Check the result by reconstructing the original number. Now compare this calculation with a queue that receives the same number of requests but processes them concurrently. Describe which quantities are counts, which are rates, and which require measurement.\n")
(ROOT/"arithmetic.txt").write_text("\n".join(paragraphs))
sources["arithmetic"]={"source":"authored deterministic word problems","kind":"synthetic_english_diagnostic"}
for name,row in sources.items():
 data=(ROOT/(name+".txt")).read_bytes()
 row.update(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
(ROOT/"sources.json").write_text(json.dumps(sources,indent=2))
print(json.dumps(sources),flush=True)
