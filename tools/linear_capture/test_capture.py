import importlib.util,json,os,sys,tempfile,types
from pathlib import Path
import torch

with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);control=root/"control.json";os.environ["LINEAR_CAPTURE_CONTROL"]=str(control)
 class Method:
  def apply(self,layer,x,bias=None):return torch.nn.functional.linear(x,layer.weight,bias)
 fake=types.ModuleType("vllm_ascend.ops.linear");fake.AscendUnquantizedLinearMethod=Method
 sys.modules["vllm_ascend.ops.linear"]=fake
 dist=types.ModuleType("vllm.distributed")
 dist.get_tensor_model_parallel_rank=lambda:0;dist.get_tensor_model_parallel_world_size=lambda:2
 sys.modules["vllm.distributed"]=dist
 ctx=types.ModuleType("vllm.forward_context")
 meta=types.SimpleNamespace(num_actual_tokens=4,seq_lens_list=[4])
 ctx.get_forward_context=lambda:types.SimpleNamespace(attn_metadata={"attention":meta})
 sys.modules["vllm.forward_context"]=ctx
 p=Path(__file__).parents[2]/"vllm_ascend/patch/worker/patch_linear_capture.py"
 spec=importlib.util.spec_from_file_location("linear_capture_unit",p)
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 layer=types.SimpleNamespace(prefix="language_model.model.layers.3.test",weight=torch.randn(8,16),input_size=16,output_size=8,output_sizes=[8])
 x=torch.randn(6,16);original=x.clone()
 assert mod.capture_before(layer,x,None) is None and list(root.iterdir())==[]
 control.write_text(json.dumps({"seq_len":4,"sample_id":"unit","output_dir":str(root/"raw"),"modules":["layers.3.test"],"token_ids_sha256":"unit"}))
 capture=mod.capture_before(layer,x,None)
 assert torch.equal(x,original) and capture[1]["x"].shape==(4,16)
 y=Method().apply(layer,x)
 assert torch.equal(x,original)
 loaded=torch.load(capture[0],weights_only=True)
 assert torch.equal(loaded["x"],original[:4])
 assert torch.equal(loaded["y_local_rows"],(original@layer.weight.T)[:4])
 try:mod.capture_before(layer,x,None)
 except RuntimeError:pass
 else:raise AssertionError("duplicate not rejected")
 control.write_text(json.dumps({"seq_len":5,"sample_id":"other","output_dir":str(root/"raw"),"modules":["layers.3.test"],"token_ids_sha256":"unit"}))
 try:mod.capture_before(layer,x,None)
 except RuntimeError:pass
 else:raise AssertionError("incomplete prefill not rejected")
 print("LINEAR_CAPTURE_UNIT_PASSED: disabled, immutable, padding, output, duplicate, incomplete")
