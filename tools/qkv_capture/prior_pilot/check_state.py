import importlib.util,sys,torch,copy,json
from pathlib import Path
torch.set_num_threads(4)
root=Path("/tmp/qwen35_vprior_pilot")
spec=importlib.util.spec_from_file_location("prior_unit",root/"adaptive/solution.py")
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
base={"beta":0.0,"adjacent_mass":torch.tensor([[1.]+[.1]*8]*2),"lengths":torch.tensor([128,1024]),
"length_mass":torch.tensor([[[1.]+[.1]*8]*2]*2),"tail_mass":{"128":torch.tensor([1.,2.]),"1024":torch.tensor([3.,4.])}}
state={"base":base,"match":{"enabled":False},"_vfp":{"sentinel":1}}
before=copy.deepcopy(state)
samples=[{"q":[torch.zeros(t,1),None]} for t in (128,1024)]
assert m._bp_mix_state(state,samples,0.) is state
for w in (.1,.25,1.):
 changed=m._bp_mix_state(state,samples,w)
 m._validate_v_causal_feedback_state(changed["base"],2,256)
 assert changed["_vfp"] is state["_vfp"] and changed["match"] is state["match"]
 assert changed["base"]["beta"]==base["beta"]
 assert torch.equal(changed["base"]["lengths"],base["lengths"])
 assert torch.equal(changed["base"]["adjacent_mass"][:,0],torch.ones(2))
 assert torch.equal(base["adjacent_mass"],before["base"]["adjacent_mass"])
 assert torch.equal(base["length_mass"],before["base"]["length_mass"])
 assert all(torch.equal(v,before["base"]["tail_mass"][k]) for k,v in base["tail_mass"].items())
assert (root/"offline/solution.py").read_text().startswith((root/"baseline/solution.py").read_text())
assert (root/"adaptive/solution.py").read_text().startswith((root/"baseline/solution.py").read_text())
print("PRIOR_STATE_AND_PREFIX_CHECKS_PASSED")
