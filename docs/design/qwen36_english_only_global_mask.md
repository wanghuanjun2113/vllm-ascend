# Qwen3.6 target + MTP global NPU mask

## Purpose

Load a server-wide allow-list once per rank, keep it on the NPU, and set blocked target and MTP draft logits to `-inf` before sampling.

## Activation

```bash
export VLLM_ASCEND_GLOBAL_ALLOWED_TOKEN_IDS_PATH=/path/allowed_ascii_token_ids.json
vllm serve /mnt/weights/Qwen3.6-27B-w8a8 ...
```

## Main code

- `vllm_ascend/worker/model_runner_v1.py`: load the global mask and apply it to target logits.
- `vllm_ascend/spec_decode/eagle_proposer.py`: apply the same mask to MTP draft logits.

## Advantages

- Strict decoding constraint: blocked tokens have `-inf` logits.
- One configuration applies to all requests.
- Tokenizer, input embedding, LM Head shape, API token IDs, and checkpoint remain unchanged.

## Limitations

- Adds full-vocabulary `masked_fill_` work on the hot path.
- Must constrain target and draft consistently; target-only masking reduces speculative acceptance.
- Random sampling and structured output paths require workload-specific performance validation.

## Validation snapshot

Five leakage cases changed from CJK under Base to 0/5 CJK and 5/5 ASCII. The pinned ARC-Easy test remained unchanged within the measured rounds. See the main design document for the exact gates and measurements.

### 2026-09-03 TP2 validation

Environment: Qwen3.6-27B W8A8, TP2 on NPU 6/7, MTP3, prefix caching enabled,
and `FULL_DECODE_ONLY` graph mode. Base and Mask used the same runtime commit;
only `VLLM_ASCEND_GLOBAL_ALLOWED_TOKEN_IDS_PATH` changed.

- Constraint: Base emitted CJK in 5/5 cases. Global target+draft Mask emitted CJK
  in 0/5 cases; all 5 outputs were non-empty ASCII.
- English accuracy: both variants scored 280/300 on the same pinned examples:
  ARC-Challenge 99/100, HellaSwag 98/100, and MMLU 83/100. All 300 parsed
  predictions were identical.
- 8K input + 256 output, concurrency 8, 80 requests per round: across three
  rounds, Mask changed output throughput by +1.640%, mean TTFT by -9.242%, mean
  TPOT by +0.454%, mean ITL by +0.454%, and speculative acceptance from 99.987%
  to 99.989%. Mean prefix-cache token hit rate was identical at 67.805%.
- Warm rounds 2-3: output throughput changed by -0.088%, mean E2E by +0.089%,
  mean TTFT by -3.791%, mean TPOT by +1.913%, and median ITL by -0.352%.
  Prefix-cache token hit rate was 75% for both variants.
- NPU profiling: Mask added 300 full-vocabulary `MaskedFill` kernels on rank 0,
  totaling 8.940 ms in an approximately 18-second profile (about 0.05% of the
  profile window). The profiled 8-request run was 17.89 seconds for Mask versus
  18.09 seconds for Base.

The throughput and E2E capacity gates are within 1%. The warm mean-TPOT scalar
is above 1%, but it is not corroborated by throughput, E2E latency, median ITL,
or operator-level profiling; it reflects a TTFT/decode-wait redistribution in
the asynchronous concurrency test. No higher-risk sampling-path optimization
was applied on this evidence.

Raw artifacts are stored under
`/home/w00498770/dev/artifacts/vllm-018/qwen36-global-mask/20260903` in the
`vllm-018` environment.
