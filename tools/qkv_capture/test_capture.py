#!/usr/bin/env python3
import importlib.util,json,os,sys,tempfile,types,unittest
from pathlib import Path
import torch

class CaptureTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
  self.control=self.root/"active.json";os.environ["QKV_CAPTURE_CONTROL"]=str(self.control)
  path=Path(__file__).parents[2]/"vllm_ascend/patch/worker/qkv_capture.py"
  spec=importlib.util.spec_from_file_location("capture_unit",path)
  self.capture=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.capture)
  dist=types.ModuleType("vllm.distributed")
  dist.get_tensor_model_parallel_rank=lambda:0;dist.get_tensor_model_parallel_world_size=lambda:2
  self.metadata=types.SimpleNamespace(num_actual_tokens=3,num_prefills=1,num_decodes=0,seq_lens_list=[3])
  ctx=types.ModuleType("vllm.forward_context")
  ctx.get_forward_context=lambda:types.SimpleNamespace(attn_metadata={"model.layers.3.self_attn.attn":self.metadata})
  self.saved={k:sys.modules.get(k) for k in ("vllm.distributed","vllm.forward_context")}
  sys.modules["vllm.distributed"]=dist;sys.modules["vllm.forward_context"]=ctx
  self.module=types.SimpleNamespace(attn=types.SimpleNamespace(layer_name="model.layers.3.self_attn.attn"),num_heads=8,num_kv_heads=1,head_dim=256,scaling=1/16)
  self.q=torch.randn(5,2048,dtype=torch.bfloat16)
  self.k=torch.randn(5,256,dtype=torch.bfloat16);self.v=self.k.clone()
  self.pos=torch.tensor([0,1,2,0,0])
 def tearDown(self):
  for k,v in self.saved.items():
   if v is None:sys.modules.pop(k,None)
   else:sys.modules[k]=v
  self.temp.cleanup()
 def activate(self,n=3):
  self.control.write_text(json.dumps({"seq_len":n,"sample_id":"unit","output_dir":str(self.root),"token_ids_sha256":"test"}))
 def call(self):return self.capture.capture_inputs(self.module,self.pos,self.q,self.k,self.v)
 def test_disabled_has_no_io(self):
  self.assertIsNone(self.call());self.assertEqual(list(self.root.iterdir()),[])
 def test_padding_snapshot_and_duplicate(self):
  self.activate();before=self.q.clone();cap=self.call()
  self.assertEqual(cap[1]["q"].shape,(3,2048));self.assertTrue(torch.equal(before,self.q))
  self.q.zero_();self.assertTrue(torch.equal(cap[1]["q"],before[:3]))
  output=torch.randn(5,2048,dtype=torch.bfloat16)
  self.capture.finish_capture(cap,output)
  d=torch.load(cap[0],weights_only=True)
  self.assertTrue(torch.equal(d["attention_output_rows"],output[:3]))
  with self.assertRaises(RuntimeError):self.call()
 def test_reject_chunked_prefill(self):
  self.activate(4)
  with self.assertRaises(RuntimeError):self.call()
 def test_reject_wrong_positions(self):
  self.activate();self.pos[1]=7
  with self.assertRaises(RuntimeError):self.call()
 def test_reject_decode(self):
  self.activate();self.metadata.num_decodes=1
  with self.assertRaises(RuntimeError):self.call()

if __name__=="__main__":unittest.main()
