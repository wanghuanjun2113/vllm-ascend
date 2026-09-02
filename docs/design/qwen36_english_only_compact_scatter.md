# Qwen3.6 compact LM Head with full-vocabulary scatter

## Purpose

Keep the tokenizer and input embedding unchanged, physically remove disallowed rows only from the output LM Head, compute 127,803 compact logits, then scatter them back to the original 248,320-token ID space before using the unmodified sampler.

## Activation

```bash
export VLLM_ASCEND_COMPACT_OUTPUT_HEAD_CONFIG=/path/compact_output_head.json
vllm serve /mnt/weights/Qwen3.6-27B-w8a8 ...
```

## Main code

- `vllm_ascend/output_vocab/compact_head.py`: load the compact LM Head, mapping, and full-logit reconstruction.
- `vllm_ascend/worker/model_runner_v1.py`: replace the target output head during loading.
- `vllm_ascend/spec_decode/eagle_proposer.py`: attach the same compact output processor to MTP.

## Advantages

- Input embedding, tokenizer, special IDs, and external API IDs stay stable.
- Reduces output-head weight memory and compact GEMM dimensions.
- Preserves the original sampler interface.

## Limitations and result

This is a retained negative-result branch. Reconstructing `[N, 248320]` logits with `index_copy_` introduced 416 full-vocabulary `ScatterUpdate` kernels and about 1.651 seconds of rank-local profile time. The refill cost exceeded the compact GEMM saving and caused material TPOT/throughput regression.

Do not use this branch as the production recommendation. It exists for profiling reproduction and comparison with the compact-domain solution.
