## ADDED Requirements

### Requirement: 移除 conv_block_padding_size padding 逻辑
系统 SHALL 移除 `_reshape_kv_cache_tensors` 中的 `conv_block_padding_size` 计算和相关 padding 跳过逻辑。K cache 和 V cache 直接从各自子区域的起始位置提取，不再需要跳过 padding。

#### Scenario: K cache 直接从 KV Cache Region 提取
- **WHEN** 混合注意力模型的 attention 层初始化 K cache tensor
- **THEN** K cache 直接从 KV Cache Region 起始位置提取，无需跳过 conv_block_padding_size

#### Scenario: V cache 直接从 KV Cache Region 提取
- **WHEN** 混合注意力模型的 attention 层初始化 V cache tensor
- **THEN** V cache 从 KV Cache Region 中 K cache 之后的位置提取，无需 padding 对齐

### Requirement: 移除 block_size 膨胀逻辑
系统 SHALL 移除 `patch_mamba_config.py` 中强制 `attn_block_size` 对齐 SSM state page_size 的逻辑。Attention block_size 保持 kernel 原生支持的 128 tokens。

#### Scenario: attention block_size 为 128
- **WHEN** 混合注意力模型配置 block_size
- **THEN** attention 的 block_size 为 128（或用户显式指定的值），不被膨胀到 1536

#### Scenario: 移除 mamba_page_size_padded 中的 conv padding
- **WHEN** 计算 mamba page_size
- **THEN** 不再执行 `mamba_page_size_padded = attn_page_size + conv_block_page_size` 的 padding 计算

### Requirement: 移除 AttentionSpec page_size_padded 强制对齐
系统 SHALL 移除 `get_kv_cache_spec()` 中将 AttentionSpec 的 `page_size_padded` 强制设为 MambaSpec page_size 的逻辑。

#### Scenario: AttentionSpec 使用自然 page_size
- **WHEN** 生成 KV cache spec
- **THEN** AttentionSpec 的 `page_size_bytes` 基于自身的 block_size、num_kv_heads、head_size 计算，不被覆写为 MambaSpec 的 page_size

### Requirement: 移除 hybrid block splitting 逻辑
系统 SHALL 移除 `_reshape_kv_cache_tensors` 中的 `block_size_chunk` 和 physical-to-logical block 转换逻辑。block_size 恢复为 128 后，不再需要将膨胀的 block 拆分为 kernel block。

#### Scenario: 不再需要 block splitting
- **WHEN** reshape KV cache tensor
- **THEN** 直接使用 block_size=128 计算 `kv_cache_shape`，不执行 `num_blocks * block_size_chunk` 的拆分

### Requirement: 非混合注意力模型不受影响
系统 SHALL 确保纯 Full Attention 或纯 Linear Attention 模型的 cache 分配和 reshape 路径不受本次改动影响。

#### Scenario: 纯 Full Attention 模型正常工作
- **WHEN** 非 hybrid 模型（如 Qwen2.5）初始化 KV cache
- **THEN** 走原有 `else` 分支分配和 reshape 路径，行为不变

#### Scenario: 纯 Mamba 模型正常工作
- **WHEN** 纯 Mamba/Linear Attention 模型初始化 state
- **THEN** 走原有 MambaSpec 路径，行为不变
