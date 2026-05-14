## ADDED Requirements

### Requirement: 310P Full Attention SplitFuse 支持

`AscendAttentionBackend310` SHALL 在 `ChunkedPrefill` 和 `PrefillCacheHit` 状态下使用 `_npu_paged_attention_splitfuse` 算子执行分块预填充注意力计算。

#### Scenario: ChunkedPrefill 状态
- **WHEN** scheduler 将长 prefill 拆分为多个 chunk，attention state 为 `ChunkedPrefill`
- **THEN** 使用 `torch_npu._npu_paged_attention_splitfuse(query, key_cache, value_cache, mask, block_table, seq_len, context_lens, ...)`，其中 KV cache 为 NZ 格式 `(2, num_blocks, (kv_heads*head_dim)//16, block_size, 16)`

#### Scenario: PrefillCacheHit 状态
- **WHEN** prefix cache 命中部分 blocks，剩余 tokens 需 prefill，attention state 为 `PrefillCacheHit`
- **THEN** 复用 splitfuse 路径处理剩余 prefill tokens

### Requirement: 310P SplitFuse Mask NZ 格式构造

系统 SHALL 为 splitfuse 路径构造 ACL_FORMAT_FRACTAL_NZ 格式的 causal mask。

#### Scenario: 动态行选择 mask
- **WHEN** 执行 splitfuse attention，需要为每个 token 构造对应位置的 causal mask 行
- **THEN** 计算每个 token 的绝对位置 `range(context_len - query_len, context_len)`，从预缓存 causal mask 中 `index_select` 对应行，转换为 NZ 格式 `nd_to_nz_spec() + npu_format_cast(ACL_FORMAT_FRACTAL_NZ)`

#### Scenario: FIA sparse_mode 替代
- **WHEN** NAIE 能力中心在 310P FIA 算子中增加 `sparse_mode=3` 支持（`qwen36-27b-310p-operators` 变更）
- **THEN** 可跳过完整 mask 构造和 NZ 转换，直接使用 `sparse_mode=3` 因果 mask 压缩

### Requirement: 310P GDN 层 Chunk Prefill 支持

`AscendGatedDeltaNetAttention310` SHALL 正确处理 chunked prefill 场景下 GDN 层的 SSM state 跨 chunk 传递。

#### Scenario: 跨 chunk SSM state 传递
- **WHEN** 长序列被拆分为多个 chunk，每个 chunk 包含部分 prefill tokens
- **THEN** 每个 chunk 的 `chunk_gated_delta_rule_pytorch` 接收前一个 chunk 的最终 SSM state 作为 `initial_state`，确保递推累积正确

#### Scenario: initial_state 构造
- **WHEN** GDN 层 prefill 路径执行
- **THEN** `initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()`（无转置），`initial_state[~has_initial_state, ...] = 0`（布尔索引清零），state 写回 `ssm_state[indices] = last_recurrent_state.to(dtype)`

### Requirement: 310P Chunk Prefill 与 Scheduler 配置

系统 SHALL 支持 310P 场景下 chunked prefill 的配置和验证。

#### Scenario: enable_chunked_prefill 配置
- **WHEN** 启动参数设置 `enable_chunked_prefill=True`
- **THEN** scheduler 将长 prefill 拆分为 chunk，`max_num_batched_tokens` 控制每 chunk 大小上限，310P 与 910 使用相同的调度逻辑

#### Scenario: block_size 兼容性
- **WHEN** 310P KV cache block_size 为 64 或 128
- **THEN** splitfuse 算子使用 `get_supported_kernel_block_sizes()` 返回的兼容 block_size（`[128, 64]`）
