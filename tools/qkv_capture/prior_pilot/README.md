# Frozen V band/tail prior pilot

This folder preserves the small calibration-only experiment, not a promoted
contest solution. Parent is the online21690.13 solution at
ef65ace073703096fd4d1e1b89c6c1f46de11f90, path
wanghuanjun/20260908_0120_long_vpair2048_fast/solution.py in
pjgao/hw_competetion_2026.
Its SHA256 is9153c980e6ce318f193c73110ac61ed660acbad48f181cd1eff72598849ae98c.

Execution took place in container vllm-023-qwen35-qkv.
The scripts retain the exact task paths used in the experiment:
temporary experiment /tmp/qwen35_vprior_pilot;
persistent plan, priors, source hashes and results under
/home/w00498770/algo/algorithm_9_7_2/data/qwen35_35B/prior_pilot.

extract.py reads only development test tensors (no score labels, no validation).
It averages normalized9-lag plus alpha/T statistics over30 document/layer
samples and2 KV heads for each T. Thus70 numeric coefficients are learned.
Development is fitting data, not an independent prior evaluation.

Candidate construction: exact parent text, followed by
_BP_PRIOR = repr({int(k): v for k,v in json.load(PRIOR.json).items()})
_BP_MODE = 'offline' or 'adaptive'
then the literal calibration_extension.py.in.
No original dynamic function is changed. Existing beta, Q/K states,
predictor parameters, state shapes and iteration budgets remain fixed.
Weight0 returns the original state exactly.

Offline mode uses weight1. Adaptive mode selects0,0.1,0.25 based on real
three-role Attention output MSE on two middle-length calibration samples.
Each calibration ratio must be <=1.001 and mean ratio<0.999 to accept.
This is supervised calibration fitting, not independent leave-one-out.

Screening is fixed to layers3/19/35, all3 documents and7 lengths per bank:
9groups/63samples. LongT scores use128 sampled query rows, through512 full rows.
Baseline is reused only from the already completed frozen-source same-bank
results. Full-bank or online advancement requires same-direction pilot gain.

run_screen.py expects mode, bank and CPU list arguments, and uses the normal
test_machine CPU lease. check_state.py verifies schema, no input-state mutation,
unchanged predictor and beta, weight0 identity and intact parent source prefix.
Source variants remain temporary; only a positive candidate would be promoted.
