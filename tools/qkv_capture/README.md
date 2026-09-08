# Qwen3.5-35B-A3B Full Attention Q/K/V capture

This opt-in diagnostic captures the actual BF16 Full Attention operands in
vLLM-Ascend 0.23.0, immediately after fused Q/K RMSNorm and RoPE, before
attention, output gating and output projection. Gated DeltaNet layers are
not ordinary causal GQA and are deliberately not exported as GQA samples.

## Environment
Container: vllm-023-qwen35-qkv, Ascend A2 910B3.
vLLM: 0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665 (image baseline).
vLLM-Ascend parent: 5cb98caaadeff42b5b62b996e34bb2aaa29d20fd.
Python 3.12.13, torch 2.10.0+cpu, torch-npu 2.10.0.post4, CANN 9.1.0.
Artifacts: /home/w00498770/dev/artifacts/vllm-023-qwen35-qkv/.
The task worktree reuses shared libraries from this same container's image;
the library symlinks are not committed. This includes lib64 and the vendors
subdirectory under _cann_ops_custom (its tracked .gitkeep directory must stay).
The launcher preserves the image PYTHONPATH and explicitly exposes the vendor
OPP and op_api library paths before importing torch.

## Reproduce (inside the dedicated container)
Start in /home/w00498770/dev/worktrees/vllm-023-qwen35-qkv/vllm-ascend.
1. python tools/qkv_capture/download_weights.py
2. python tools/qkv_capture/prepare_corpus.py
3. python tools/qkv_capture/test_capture.py
4. bash tools/qkv_capture/run.sh --run-id YOUR_UNIQUE_RUN_ID
5. python tools/qkv_capture/audit_capture.py /home/w00498770/dev/artifacts/vllm-023-qwen35-qkv/captures/YOUR_UNIQUE_RUN_ID

The downloader uses ModelScope's Qwen publisher mirror, pins each file's
revision from its frozen manifest, validates HTTP ranges and every SHA256.
HF mirror returned HTTP 403 in the initial test. Weight data remain outside Git.

## Capture contract
- Eager TP2 on NPU2/3, one request, no prefix cache or MTP. Chunked-prefill
  scheduling remains enabled for model compatibility; the 8192-token budget
  exceeds every prompt, and the capture validates one complete prefill.
- CPU binding explicitly disabled because its default path changes host IRQ affinity.
- Lengths: 10,128,256,512,1024,2048,4096 token IDs, with no chat template or extra BOS.
- Six diagnostic document families; each longer sequence is a prefix extension
  of the same family. They are correlated samples, not independent benchmark cases.
  A future train/validation split must respect document/family identity.
- All ten Full Attention layers: 3,7,11,15,19,23,27,31,35,39 (zero-based).
- TP2 local Q/K/V shapes: [T,2048], [T,256], [T,256].
- Merged Q/K/V shapes: [T,4096], [T,512], [T,512].
- num_q_heads=16, num_kv_heads=2, head_dim=256, scaling=1/16.
- Capture excludes engine dummy/profile runs, padding, decode and cache reuse.
- An environment variable points to an atomic one-request control record.
  With it unset, the hook performs no capture I/O or tensor copies.
- Input tensors are cloned to CPU before the attention call; up to 64 output
  query rows are saved after attention and before its sigmoid output gate.
- Files include token-ID hashes, positions, layer name and TP rank.
- Data are model BF16 operands, NOT NVFP4 tuples or a claim about official contest data.
  NVFP4 conversion requires a separate validated converter.
- Sampling/replay is diagnostic; throughput measured during capture is not a serving benchmark.

## Validation
Unit tests cover disabled recording, padding trimming, immutable snapshots,
duplicate prevention, wrong positions and rejected chunk/decode captures.
The runner compares greedy token IDs and top-logprob values with capture off/on.
The audit checks every expected sample/layer/rank, dtype, shape, finiteness and
position alignment; merges head-contiguous TP shards, then recomputes causal
GQA in FP32 on the saved query rows. NRMSE must be <= 0.02 to tolerate BF16
backend rounding. It records actual NRMSE, max absolute error and all file hashes.
This sampled replay does not claim full-output bitwise equivalence.

## Outputs
captures/<run>/prompts.json: exact input token IDs.
captures/<run>/run_config.json: resolved launch arguments supplied to vLLM.
captures/<run>/raw/<sample>/*.tp{0,1}.pt: original per-rank snapshots.
captures/<run>/merged/<sample>/layer_XX.pt: reconstructed global-head tensors.
captures/<run>/generation_results.json: capture on/off control and generations.
captures/<run>/audit.json: coverage, shape, checksum and replay audit.
corpus/sources.json: source URLs / authored content tags / source hashes.

The runner explicitly shuts down its engine and removes its control file.
Verify process exit and npu-smi after use; never stop another task's processes.

## Verified run (2026-09-08)

Runtime source: b6511da55e0daa753513256484d51954a5086429.
Run: qkv_20260908_v4. All 42 requests and 420 layer records passed.
Raw shards: 840; merged files: 420; merged bytes: 5167494180.
Maximum sampled FP32 replay NRMSE: 0.00196946.
First-request capture off/on token IDs and compared logprobs matched exactly.
Five unit tests passed. NPU2/3 inference processes exited and resource release
was verified with npu-smi. See the artifact root's results.json and 数据说明.md.
