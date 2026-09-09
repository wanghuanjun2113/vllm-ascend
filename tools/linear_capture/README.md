# Qwen3.5-35B-A3B Linear dataset

## Scope and provenance
Dedicated container: vllm-023-qwen35-qkv.
Worktree: /home/w00498770/dev/worktrees/vllm-023-qwen35-qkv/vllm-ascend-linear-capture
Branch: feat/qwen35-linear-dataset; base 79bc337f62a955c3eceaa944b01ad0a5eed642f6.
Raw capture: /home/w00498770/dev/artifacts/vllm-023-qwen35-qkv/linear_capture_v1.
The original QKV worktree and prior experiments are not modified.

The hook wraps AscendUnquantizedLinearMethod.apply only when
LINEAR_CAPTURE_CONTROL is set. It records actual GEMM inputs, local BF16
weights, and 32 output rows before TP row-parallel reduction, output gating
or other downstream operations. The existing native implementation is called
unchanged. Warmup/profile calls and unselected modules are excluded.

Runtime: BF16 TP2 on NPU2/3, eager, complete4096-token prefills, prefix cache
and async scheduling disabled. CPU binding is explicitly disabled.
FlashComm1, matmul-allreduce fusion, shared-expert DP and shared-expert
multistream overlap are disabled to make the GEMM boundary observable.
This is a correctness/data capture run, not a performance benchmark.

## Matrix coverage
Each bank has50 distinct checkpoint matrices and50 dynamic test units:
- 15 x [8192,2048]: ten Full Attention Q+gate projections and five GDN QKV projections.
- 20 x [512,2048]: ten K projections, five V projections and five shared-expert up projections.
- 5 x [2048,4096]: three Full Attention O and two GDN output projections.
- 10 x [2048,512]: ten shared-expert down projections.

The shape ratio15:20:5:10 follows the project's recorded online probes.
The module-role mix is a coverage choice, not a recovered online frequency.
Routed-expert-specific activations are not captured in this first dataset.
SPECIFICATIONS.json lists every exact checkpoint key and fusion mapping.

## Row sampling and split
Six4096-token documents use the existing document split:
development=Alice/Python AST/Chinese systems;
validation=Holmes/Frankenstein/arithmetic.
Weights are shared between banks; all calibration/test activation documents
are disjoint across banks.

Each Linear group has five calibration tensors (64/128/256/512/1024 rows)
from its bank's other two documents; row ranges within a donor do not overlap.
One held-out document supplies the dynamic test tensor.
Test row count M covers the recorded support
1,2,7,8,10,14,15,16,18,22,25,26,31,39,42,56,58,60,128,512,1024.
Support comes from generate_simulated_v4.py LINEAR_TEST_SUPPORT; exact
shape-conditioned frequencies and official calibration lengths are unknown.
Within each shape bucket the support index is spread across its full range.
Rows are actual activations sampled from4096-token prefills, not random data,
independent short requests or decode captures. Linear is row-separable, so
these are valid X@W.T instances without inventing a new activation process.

## Reconstruction and checks
Column-parallel packed projections are split by output_sizes, gathered by
component, and mapped to checkpoint names. Row-parallel input/weight columns
are concatenated and partial outputs summed. Replicated modules are verified
and deduplicated. Every reconstructed weight must equal its checkpoint tensor.
Each matrix/document pair is checked against an FP32 GEMM on32 captured rows;
NRMSE threshold0.015 accommodates BF16 GEMM/partial-reduction rounding.
Inputs are checked for exact width, finite values and token-ID provenance.

NVFP4 uses tools/qkv_capture/export_nvfp4.py:
E2M1 RNE carriers, E4M3 per16-element block scales and FP32 global scale.
The global scale is folded into the tester's FP32 scale tensor; E2M1 numeric
carriers are stored in BF16. Files are not nibble-packed.

## Reproduce inside the container
python tools/linear_capture/test_capture.py
bash tools/linear_capture/run.sh
python tools/linear_capture/audit_and_export.py

The capture and export refuse to overwrite existing directories.
The worktree reuses this container image's untracked native libraries:
vllm_ascend/lib64, libvllm_ascend_kernels.so,
vllm_ascend_C.cpython-312-aarch64-linux-gnu.so,
and _cann_ops_custom/vendors. Preserve the image PYTHONPATH and vendor paths.

## Use with the contest tester
Data paths:
 /home/w00498770/algo/algorithm_9_7_2/data/qwen35_35B/development/linear_shards
 /home/w00498770/algo/algorithm_9_7_2/data/qwen35_35B/validation/linear_shards

Run from /home/w00498770/algo/algorithm_9_7_2:
python scripts/test_machine.py --owner qwen35-linear --cpus 0-95 \
 --solution try/0024_sota21613_perf_exact/solution.py \
 --data-dir data/qwen35_35B/development --skip-attention --timeout-seconds 0

The existing Attention shards and MANIFEST.json are unchanged.
With neither skip flag, the tester now runs50 Linear plus210 Attention
samples per bank; that260-sample score is not the official300-case total.
Detailed commands and verification receipts are in LINEAR_MANIFEST.json
and the dataset root's LINEAR_DATASET.json.

## Verified result

50 unique checkpoint weights matched exactly; all300 matrix/document replay
checks passed, maximum NRMSE=0.001956543. Each bank has50 groups
and50 tests,1195943354 bytes. Both complete standard controls were exactly zero.
A four-shape0024 smoke completed (not a full-candidate benchmark).
Capture off/on token and compared logprob matched exactly. NPU2/3 released.
Runtime source commit:a5c02ebb361834ca3e3a724e942d5a37cac22248.
