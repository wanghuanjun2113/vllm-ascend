# Qwen3.5-35B-A3B datasets v2/v3/v4

Worktree: /home/w00498770/dev/worktrees/vllm-023-qwen35-qkv/vllm-ascend-datasets-v234
Container: vllm-023-qwen35-qkv. Base: ec908f2ab3bc88e175e32c6bbc0ebea0286d72fb.
Runtime/producer commit: d2cf570416a8bd96aea3ab6081944522a1a5b87e.
Artifacts: /home/w00498770/dev/artifacts/vllm-023-qwen35-qkv/versions_v234.

Each version uses its own corpus:
- v2: processed WikiText-2 from pytorch/examples, pinned commit
  acc295dc7b90714f1bf47f06004fc19a7fe235c4; CC-BY-SA-3.0/GFDL.
- v3:20 selected Project Gutenberg public-domain books, exact source URLs and hashes.
- v4: CPython3.12.13 standard-library source from this container;
  PSF-2.0 and embedded notices. Site-packages, tests and generated build config are excluded.

Each version has development/ and test/, each with5 Linear groups and25
Attention groups, exactly5 calibration and5 test samples per group.
Thus each split has150 calibration entries and150 scored samples
(25 Linear +125 Attention), and each version has300 scored samples.
These counts are local sample counts, not a reconstruction of the judge's
group-level aggregation.

Attention:
-16 query heads /2 KV heads /256 head dimension.
-All10 Full Attention layers are covered; each appears2 or3 times per split.
-Every group contains256/1024/2048/4096.
-The fifth test sample preserves the observed25-case rare masks:
 T10 at group0; T128 at1/5/6; T512 at2/9/10/11.
-The other17 fifth samples are new distinct samples at balanced common lengths.
 Their frequency is a construction choice; it is not claimed as a probe result.
-Per-split test histogram: T10=1,T128=3,T256=30,T512=4,T1024=29,T2048=29,T4096=29.
-Calibration lengths128/256/512/1024/2048 are a local panel, not recovered official lengths.

Linear:
-Five distinct matrices per version cover all four shapes with counts1/2/1/1:
 [8192,2048], [512,2048] twice, [2048,4096], [2048,512].
-Five groups cannot exactly represent15:20:5:10 while covering all shapes.
 The manifest offers optional shape weights1.5/1/1/0.5/1; the unchanged
 test_machine still reports unweighted scores.
-Test M covers the21 recorded support values across25 samples.
 Calibration uses64/128/256/512/1024 rows.
-Weights are real checkpoint weights. X is sampled from actual captured
 GEMM inputs; this is not a claim of independent short-request/decode capture.

Source separation:
-Source articles/books/top-level Python packages are partitioned between
 development/test and between calibration/test use.
-Selected token intervals do not overlap and exact input hashes are unique.
-Short documents may be concatenated as token fragments, without repeating
 or padding text. Full source spans and token IDs are retained in CAPTURE_PLAN.json.
-Five input panels are reused across different model layers. Those produce
 different QKV tensors; a same-layer/request pair is not duplicated.
-No model-data score fitting or candidate optimization determined the split.

Capture:
-BF16 TP2 NPU2/3, eager, prefix caching and async scheduling off.
-CPU binding, FlashComm1, matmul-allreduce fusion, shared-expert DP/overlap off.
-QKV is saved after Q/K RMSNorm+RoPE, before attention/output gating.
-Linear weights/X/local output rows are saved at the actual GEMM boundary.
-Source checkpoint weights must match after TP reconstruction.
-First request of every version has capture off/on token+logprob comparison.
-Up to64 query rows and32 GEMM rows are replayed in FP32 for each used sample.
-NVFP4 uses E2M1 RNE + E4M3 per16 blocks + FP32 tensor-global scale, folded
 into the tester's effective scale tensor. Carriers are BF16 numeric values,
 not packed nibbles.

Inside the container, from the worktree:
python tools/datasets_v234/prepare.py
bash tools/datasets_v234/run.sh
python tools/datasets_v234/build.py
python tools/datasets_v234/validate.py

Capture/build refuse to overwrite an existing output. Raw corpus caches and
plans are retained; recreating a plan from a changed upstream file is not an
exact replay unless its hash matches. The original v1 data/worktrees are unchanged.

Each output root contains source provenance and exact plans; each split has
MANIFEST.json, AUDIT.json, and TEST_VALIDATION.json after validation.
Validation uses the original tester under its genuine96-CPU lease with8
diagnostic threads; T<=512 full attention output, long T128 sampled query
rows, full QKV quantization. This is a data/zero-control check, not performance.
