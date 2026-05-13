## Why

Qwen3.6 27B 在 310P (300IDuo) 上的基础推理已通过 eager 模式 + PyTorch fallback 实现。为进一步提升推理性能（降低 TTFT、提升吞吐），需要启用 Chunk Prefill、Prefix Caching 和 MTP 投机推理三个关键特性。这些特性在 910 上已有成熟实现，本变更重点识别可共用的框架代码，并详细描述 310P 需要单独适配的部分。

## What Changes

### Chunk Prefill
- 910 框架代码（scheduler、`enable_chunked_prefill` 配置）可共用
- 310P Attention 后端已有 `_npu_paged_attention_splitfuse` 路径，需验证与 Qwen3.6 混合注意力的兼容性
- GDN 层 chunk prefill 依赖 `chunk_gated_delta_rule_pytorch`，需验证长序列分段处理的正确性
- 310P mask 构造需 NZ 格式转换（`AttentionMaskBuilder310.get_splitfuse_mask`）

### Prefix Caching
- Full Attention 层的 block-level prefix caching 使用 vLLM V1 共享框架，310P 可直接复用
- GDN 层 SSM State Checkpoint 机制需全新实现（SSMStatePool、checkpoint/restore 逻辑）
- 310P KV cache 5D NZ 布局和 block_size 约束（`block_size * head_size ≤ 128*128`）影响 prefix cache 粒度
- 需确认 `expandable_segments` 在 310P hybrid 模型上的兼容性

### MTP 投机推理
- `AscendEagleProposer(method="mtp")` 框架可共用（draft loop、rejection sampling 编排）
- GDN 层 MTP multi-query 路径（`spec_token_indx`）已在 `gdn_310.py` 中实现
- Rejection Sampling PyTorch fallback 已在 `rejection_sampler.py` 中实现
- **关键缺口**：`AscendAttentionBackend310` 当前对 SpecDecoding 抛出 `NotImplementedError`，需新增支持
- 310P 无 ACLGraph，MTP draft/verify 走纯 eager 路径
- `npu_copy_and_expand_eagle_inputs` C++ 算子当前仅 910 注册，310P 需移植或 PyTorch 替代

## Capabilities

### New Capabilities
- `310p-chunk-prefill`: 310P 上 Qwen3.6 27B 的 Chunk Prefill 支持，含 Full Attention splitfuse + GDN chunk 并行
- `310p-prefix-cache`: 310P 上混合注意力的 Prefix Caching，含 Full Attn block-level 复用 + GDN SSM State Checkpoint
- `310p-mtp`: 310P 上 Qwen3.6 MTP 投机推理，含 draft proposer + verify + rejection sampling

### Modified Capabilities
<!-- 无现有 spec 级别的行为变更 -->

## Impact

- **Attention 后端**：`AscendAttentionBackend310` 需新增 SpecDecoding 状态支持和 splitfuse 正确性验证
- **GDN 层**：SSMStatePool 数据结构与 checkpoint/restore 逻辑，影响 `_forward_core` 执行流
- **算子依赖**：依赖 `qwen36-27b-310p-operators` 变更中的算子开发（chunk_gated_delta_rule、fused_gdn_gating 等）
- **调度器**：chunked prefill 的 chunk size 配置、prefix cache 的 block 粒度需 310P 特定调优
- **测试**：需逐特性启用验证 + 组合验收（Chunk Prefill + Prefix Cache、MTP + Chunk Prefill 等）
