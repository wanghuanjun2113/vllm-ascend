#!/usr/bin/env python3
"""Prepare three disjoint open corpora and freeze all capture/group assignments."""
import concurrent.futures as cf,hashlib,json,os,re,sys,sysconfig,time,urllib.request
from pathlib import Path
from collections import defaultdict
import torch
from transformers import AutoTokenizer

ART=Path("/home/w00498770/dev/artifacts/vllm-023-qwen35-qkv/versions_v234")
MODEL=Path("/home/w00498770/dev/artifacts/vllm-023-qwen35-qkv/weights/Qwen3.5-35B-A3B")
WIKI_REF="acc295dc7b90714f1bf47f06004fc19a7fe235c4"
BOOKS=[1342,98,2701,2600,174,345,76,74,1400,1260,120,1232,5200,2591,4300,768,161,158,16328,219]
LENGTHS=[256,1024,2048,4096]
CALIB=[128,256,512,1024,2048]
LINEAR_M=[1,2,7,8,10,14,15,16,18,22,25,26,31,39,42,56,58,60,128,512,1024]
SOURCES={2:("WikiText-2","https://github.com/pytorch/examples/tree/"+WIKI_REF+"/word_language_model/data/wikitext-2","CC-BY-SA-3.0/GFDL; processed WikiText-2"),
3:("Project Gutenberg selected public-domain books","https://www.gutenberg.org/policy/license.html","Per-book Project Gutenberg notices; public-domain source texts"),
4:("CPython standard-library source","https://docs.python.org/3/license.html","PSF-2.0 and embedded notices")}
def sha(b):return hashlib.sha256(b).hexdigest()
def fetch(url,path):
 if path.exists():return path.read_bytes()
 for attempt in range(5):
  try:
   request=urllib.request.Request(url,headers={"User-Agent":"Qwen35-data-research/1.0"})
   with urllib.request.urlopen(request,timeout=60) as response:data=response.read()
   path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
   return data
  except Exception:
   if attempt==4:raise
   time.sleep(2**attempt)
def wiki_docs():
 docs=[]
 for split,filename in (("development","train.txt"),("test","test.txt")):
  url=f"https://raw.githubusercontent.com/pytorch/examples/{WIKI_REF}/word_language_model/data/wikitext-2/{filename}"
  path=ART/"corpus/wikitext"/filename
  text=fetch(url,path).decode()
  title=None;parts=[]
  def finish():
   if title and parts:
    text="".join(parts);docs.append({"id":"wiki:"+title,"group":"wiki:"+title,"bank":split,"text":text,"url":url,"file_sha256":sha(path.read_bytes())})
  for line in text.splitlines(keepends=True):
   if re.match(r"^\s*= [^=].*[^=] =\s*$",line):
    finish();title=line.strip().strip("= ").strip();parts=[line]
   else:parts.append(line)
  finish()
 return docs
def book_docs():
 def one(i):
  url=f"https://www.gutenberg.org/ebooks/{i}.txt.utf-8"
  raw=fetch(url,ART/"corpus/gutenberg"/f"{i}.txt")
  text=raw.decode("utf-8-sig")
  text=re.split(r"\*\*\* START OF[^\n]*\n",text,maxsplit=1)[-1]
  text=re.split(r"\*\*\* END OF",text,maxsplit=1)[0]
  return {"id":f"pg:{i}","group":f"pg:{i}","text":text,"url":url,"file_sha256":sha(raw)}
 with cf.ThreadPoolExecutor(max_workers=6) as pool:return list(pool.map(one,BOOKS))
def python_docs():
 base=Path(sysconfig.get_path("stdlib"));docs=[]
 for p in sorted(base.rglob("*.py")):
  rel=p.relative_to(base)
  if rel.parts[0].startswith(("config-", "_sysconfigdata", "_sysconfig_vars")) or p.name in ("sitecustomize.py", "usercustomize.py"):continue
  if any(x in ("site-packages","test","tests","__pycache__","ensurepip","idlelib","turtledemo") for x in rel.parts):continue
  raw=p.read_bytes()
  try:text=raw.decode("utf-8")
  except UnicodeDecodeError:continue
  if len(text)<300:continue
  group=rel.parts[0].removesuffix(".py")
  docs.append({"id":"python:"+str(rel),"group":"python:"+group,"text":text,"local_source":str(p),
               "url":"https://github.com/python/cpython/blob/v"+sys.version.split()[0]+"/Lib/"+str(rel),"file_sha256":sha(raw)})
 return docs
def assign(docs,version,tokenizer):
 seen=set();filtered=[]
 for d in docs:
  if d["id"] in seen:continue
  seen.add(d["id"]);d=dict(d);d["tokens"]=tokenizer.encode(d["text"],add_special_tokens=False)
  if len(d["tokens"])<8:continue
  d["text_sha256"]=sha(d["text"].encode());d.pop("text")
  filtered.append(d)
 groups=defaultdict(list)
 for d in filtered:groups[d["group"]].append(d)
 keys=sorted(groups,key=lambda k:sha(f"{version}:{k}".encode()))
 totals={"development":0,"test":0}
 for key in keys:
  if "bank" in groups[key][0]:bank=groups[key][0]["bank"]
  else:bank=min(totals,key=totals.get)
  for d in groups[key]:d["bank"]=bank
  totals[bank]+=sum(len(d["tokens"]) for d in groups[key])
 pools={}
 for bank in ("development","test"):
  own=[k for k in keys if groups[k][0]["bank"]==bank]
  total=sum(sum(len(d["tokens"]) for d in groups[k]) for k in own)
  amount=0;cal=set()
  for k in own:
   if amount<total*.25 or len(cal)<min(3,max(1,len(own)//3)):
    cal.add(k);amount+=sum(len(d["tokens"]) for d in groups[k])
  for role in ("calib","test"):
   selected=[d for k in own if (k in cal)==(role=="calib") for d in groups[k]]
   pools[bank,role]=selected
   print("POOL",version,bank,role,len(selected),sum(len(d["tokens"]) for d in selected),flush=True)
 group_sets={key:{d["group"] for d in ds} for key,ds in pools.items()}
 for a,sa in group_sets.items():
  for b,sb in group_sets.items():
   if a!=b:assert not sa&sb,(a,b,sa&sb)
 return pools,filtered
class Pool:
 def __init__(self,docs):
  self.docs=docs;self.offset=[0]*len(docs);self.cursor=0
 def take(self,n):
  tokens=[];spans=[];examined=0
  while len(tokens)<n:
   i=self.cursor%len(self.docs);self.cursor+=1;d=self.docs[i]
   available=len(d["tokens"])-self.offset[i]
   if available<=0:
    examined+=1
    if examined>len(self.docs):raise RuntimeError("Corpus pool exhausted")
    continue
   count=min(n-len(tokens),available);start=self.offset[i]
   tokens+=d["tokens"][start:start+count];self.offset[i]+=count
   spans.append({"document":d["id"],"source_group":d["group"],"start":start,"end":start+count})
   examined=0
  return tokens,spans
def linear_specs(version):
 a,b,c={2:(3,19,35),3:(15,31,7),4:(27,11,23)}[version]
 specs=[]
 def add(layer,module,weight,segments,shape,mode="column"):
  specs.append({"id":str(len(specs)),"layer":layer,"module":f"layers.{layer}.{module}",
  "weight_key":f"model.language_model.layers.{layer}.{weight}.weight","segments":segments,"shape":shape,"mode":mode})
 add(a,"self_attn.qkv_proj","self_attn.q_proj",[0],[8192,2048])
 add(a,"self_attn.qkv_proj","self_attn.k_proj",[1],[512,2048])
 add(b,"mlp.shared_expert.gate_up_proj","mlp.shared_expert.up_proj",[1],[512,2048])
 add(c,"self_attn.o_proj","self_attn.o_proj",[],[2048,4096],"row")
 add(b,"mlp.shared_expert.down_proj","mlp.shared_expert.down_proj",[],[2048,512],"row")
 return specs
def make_plan(version,docs,tokenizer):
 pools,filtered=assign(docs,version,tokenizer);requests=[];banks={};seen_tokens=set()
 specs=linear_specs(version)
 for bank in ("development","test"):
  cp,tp=Pool(pools[bank,"calib"]),Pool(pools[bank,"test"])
  groups=[];linears=[]
  def request(name,n,role,layers,module=None):
   for attempt in range(64):
    tokens,spans=(cp if role=="calib" else tp).take(n)
    digest=sha(json.dumps(tokens,separators=(",",":")).encode())
    if digest not in seen_tokens:break
   else:raise RuntimeError("Duplicate token windows")
   seen_tokens.add(digest)
   rid=f"v{version}_{bank}_{name}"
   requests.append({"id":rid,"version":version,"bank":bank,"role":role,"seq_len":n,
    "token_ids":tokens,"token_ids_sha256":sha(json.dumps(tokens,separators=(",",":")).encode()),
    "source_spans":spans,"qkv_layers":layers,"linear_modules":[module] if module else []})
   return rid
  for panel in range(5):
   gids=list(range(panel*5,panel*5+5))
   layers=[3+4*((g+(version-2)*3)%10) for g in gids]
   cal=[request(f"p{panel}_c{i}",n,"calib",layers,specs[panel]["module"]) for i,n in enumerate(CALIB)]
   base={n:request(f"p{panel}_base{n}",n,"test",layers) for n in LENGTHS}
   extra_lengths={}
   for g in gids:
    rare={0:10,1:128,5:128,6:128,2:512,9:512,10:512,11:512}
    ordinary=[i for i in range(25) if i not in rare]
    n=rare[g] if g in rare else LENGTHS[ordinary.index(g)%4]
    if n not in extra_lengths:extra_lengths[n]=request(f"p{panel}_extra{n}",n,"test",layers)
    groups.append({"id":g,"layer":3+4*((g+(version-2)*3)%10),"calib":cal,
                   "test":[base[t] for t in LENGTHS]+[extra_lengths[n]],"test_lengths":LENGTHS+[n]})
   ltests=[request(f"p{panel}_lt{i}",1024,"test",[],specs[panel]["module"]) for i in range(5)]
   linears.append({"spec":specs[panel],"calib":cal,"calib_rows":[64,128,256,512,1024],
                   "test":ltests,"test_rows":[LINEAR_M[(panel*5+i+version-2)%len(LINEAR_M)] for i in range(5)]})
  banks[bank]={"attention":groups,"linear":linears}
 metadata=[{k:v for k,v in d.items() if k!="tokens"}|{"tokens":len(d["tokens"])} for d in filtered]
 return {"version":version,"corpus":SOURCES[version],"banks":banks,"requests":requests,"documents":metadata}
def main():
 ART.mkdir(parents=True,exist_ok=True)
 tokenizer=AutoTokenizer.from_pretrained(str(MODEL))
 for version,reader in ((2,wiki_docs),(3,book_docs),(4,python_docs)):
  out=ART/f"plan_v{version}.json"
  if out.exists():print("PLAN_EXISTS",version,flush=True);continue
  plan=make_plan(version,reader(),tokenizer)
  (ART/f"plan_v{version}.json").write_text(json.dumps(plan,ensure_ascii=False))
  print("PLAN_READY",version,len(plan["requests"]),flush=True)
if __name__=="__main__":main()
