## ADDED Requirements

### Requirement: KV cache and SSM state 使用非连续布局共享同一 pool
系统 SHALL 在同一 `kv_cache_tensor` 内划分独立的连续子区域：KV Cache Region、SSM State Region 和 Conv State Region，各子区域使用各自的自然 page_size，不再强制统一 page_size。

#### Scenario: KV cache region 使用 block_size=128 的自然 page_size
- **WHEN** 混合注意力模型初始化 KV cache
- **THEN** KV Cache Region 的每 slot 大小为 `128 * num_kv_heads * head_size * dtype_size`，attention block_size 为 128 tokens

#### Scenario: SSM state region 使用自然 page_size
- **WHEN** 混合注意力模型初始化 GDN state
- **THEN** SSM State Region 的每 slot 大小为 `num_value_heads * value_dim * key_dim * 4` bytes（float32），独立于 KV cache 的 page_size

#### Scenario: Conv state region 使用自然 page_size
- **WHEN** 混合注意力模型初始化 GDN conv state
- **THEN** Conv State Region 的每 slot 大小为 `conv_dim * (conv_kernel_dim - 1) * dtype_size` bytes

### Requirement: 算子通过 offset 访问非连续 block
所有访问 KV cache 和 GDN state 的算子 SHALL 通过 block_table、slot_mapping 或 indices 中编码的 physical block number / slot index 定位数据，算子本身不假设 block 在内存中连续排列。

#### Scenario: CANN attention 算子通过 block_table 读取 KV cache
- **WHEN** decode 阶段调用 `npu_fused_infer_attention_score` 或 `npu_paged_attention`
- **THEN** 算子通过 `block_table` 中的 physical block number 定位 KV data，physical block number 反映 KV Cache Region 内的实际偏移

#### Scenario: reshape_and_cache 通过 slot_mapping 写入 KV cache
- **WHEN** prefill 阶段写入 KV cache
- **THEN** `slot_mapping` 中编码的 slot offset 反映 KV Cache Region 内的实际偏移，正确写入 K/V data

#### Scenario: transpose_kv_cache_by_block 通过 blockIDs 转置
- **WHEN** 执行 KV cache block 转置
- **THEN** `blockIDs` 数组中的 block ID 反映 KV Cache Region 内的实际偏移

#### Scenario: GDN 算子通过 indices 访问 SSM/conv state
- **WHEN** GDN forward 中访问 ssm_state 或 conv_state
- **THEN** `ssm_state_indices` 和 `conv_state_indices` 反映各自 Region 内的实际偏移，或直接使用已 slice 的子 tensor

### Requirement: num_kv_blocks 和 num_ssm_slots 可独立设置
系统 SHALL 允许 KV Cache Region 的 block 数量和 SSM State Region 的 slot 数量独立配置，不再强制两者相等。

#### Scenario: KV cache blocks 多于 SSM slots
- **WHEN** 长上下文场景下 KV cache 需求大于 SSM state 需求
- **THEN** KV Cache Region 可分配更多 block slots，SSM State Region 分配较少 slot，两者在同一 pool 内独立运作

#### Scenario: SSM slots 多于 KV blocks
- **WHEN** 高并发短序列场景下 SSM state 需求大于 KV cache 需求
- **THEN** SSM State Region 可分配更多 slot，KV Cache Region 分配较少 block
