## Why

Qwen3.6 使用 Full Attention + Linear Attention (GDN) 混合注意力架构。当前实现中，Full Attention 的 KV Cache 和 Linear Attention 的 SSM State 共享同一个 Cache Pool（`kv_cache_tensor`）。为保持共享张量的连续性，代码在 K/V cache 和 GDN state 之间插入了 padding blocks，导致显存浪费。在 910B4 32G HBM 资源紧张的部署场景下，这部分浪费直接影响可缓存 token 数量，需要消除。

## What Changes

- **修改算子支持非连续访问**：通过修改 KV cache 相关算子（如 `transpose_kv_cache_by_block`、`reshape_and_cache` 等），使其支持对非连续 block 的读写操作，从而不再需要通过 padding 保持连续性。保持 Full Attention KV Cache 和 GDN SSM State 共享同一 Cache Pool 不变。
- **移除 hybrid layout 中的 padding blocks**：消除 `conv_block_padding_size` 和 `mamba_padding`，消除 tensor layout `[kv_padding | conv]`、`[k | ssm]`、`[v | mamba_padding]` 中的对齐浪费。
- **修改 KV cache 分配和 reshape 逻辑**：`_allocate_kv_cache_tensors()` 和 `_reshape_kv_cache_tensors()` 中的 hybrid 特殊路径（`hybrid_with_attn_and_mamba`、`use_hybrid_blocks`）移除 padding 计算，按实际大小分配 KV 和 GDN state。

## Capabilities

### New Capabilities
- `non-contiguous-kv-cache-access`: 修改 KV cache 相关算子支持非连续 block 访问，消除 padding blocks，保持 Full Attention KV Cache 和 GDN SSM State 共享同一 Cache Pool
- `hybrid-cache-padding-removal`: 移除 hybrid layout 中的 padding 逻辑，简化 tensor layout 和分配路径

### Modified Capabilities

## Impact

- **核心文件**：`vllm_ascend/worker/model_runner_v1.py`（分配和 reshape 逻辑）、`vllm_ascend/worker/block_table.py`（block table 管理）
- **算子层**：`csrc/transpose_kv_cache_by_block/`（KV cache 转置算子）、`vllm_ascend/attention/attention_v1.py`（attention 后端）、`vllm_ascend/ops/gdn.py`（GDN 算子）
- **KV Transfer**：`vllm_ascend/distributed/kv_transfer/` 路径下的 Prefill Disaggregation 连接器，需适配新的 cache tensor 组织方式
- **前向兼容**：非混合注意力模型（纯 Full Attention 或纯 Linear Attention）不应受影响
