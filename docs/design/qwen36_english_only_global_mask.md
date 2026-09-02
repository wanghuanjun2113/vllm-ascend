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
