# Qwen3.6 compact-domain greedy sampling

## Purpose

Remove the compact-scatter bottleneck by keeping target and MTP draft logits in the 127,803-token compact domain through greedy sampling and speculative rejection. Convert token IDs only at the sampler/embedding boundary.

## Activation

```bash
export VLLM_ASCEND_COMPACT_OUTPUT_HEAD_CONFIG=/path/compact_output_head.json
export VLLM_ASCEND_COMPACT_OUTPUT_HEAD_FAST_GREEDY=1
vllm serve /mnt/weights/Qwen3.6-27B-w8a8 ...
```

Warm the intended concurrency before production admission so the rejection kernel is compiled.

## Main code

- `vllm_ascend/output_vocab/compact_head.py`: return compact logits and maintain `compact_id <-> original_id` mappings.
- `vllm_ascend/spec_decode/eagle_proposer.py`: map draft argmax IDs back before embedding lookup.
- `vllm_ascend/worker/model_runner_v1.py`: execute target greedy rejection in compact space and map accepted IDs back.

## Fast-path gate

The compact-domain path is used only for greedy requests without logprobs, penalties, bad words, per-request token masks, or active `min_tokens`. Unsupported cases use the correctness-preserving full-vocabulary fallback.

## Validation snapshot

- Full-vocabulary scatter: 416 calls / 1.651 s -> 0.
- Warm 8K+256, concurrency-8 output throughput: 148.18 -> 151.39 tok/s (+2.17%).
- Mean TPOT: 43.71 -> 43.51 ms (-0.46%).
- Mean ITL: 137.78 -> 130.88 ms (-5.00%).
- English MCQ: 279/300; Base range 276-282/300.
- Five language-induction cases: 0/5 CJK, 5/5 ASCII and non-empty.

## Recommendation

This is the preferred framework optimization for the measured greedy workload. Non-greedy compact-domain sampling remains future work.
