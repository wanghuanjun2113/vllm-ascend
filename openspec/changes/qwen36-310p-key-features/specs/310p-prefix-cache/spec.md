## ADDED Requirements

### Requirement: 310P Full Attention Block-Level Prefix Caching

系统 SHALL 在 310P 上启用 Full Attention 层的 block-level prefix caching，复用 vLLM V1 共享框架。

#### Scenario: Block hash 匹配
- **WHEN** `enable_prefix_caching=True`，新请求的 prefix tokens 与已有请求匹配
- **THEN** `KVCacheManager` 通过 block hash 匹配复用已有 KV cache blocks，310P 的 NZ 格式 KV cache 不影响 block allocator 逻辑

#### Scenario: PrefillCacheHit attention state
- **WHEN** prefix cache 命中 N 个 blocks，剩余 tokens 需继续 prefill
- **THEN** attention state 设为 `PrefillCacheHit`，使用 splitfuse 路径处理剩余 prefill tokens

### Requirement: 310P GDN SSM State Checkpoint 机制

系统 SHALL 实现 SSM State Checkpoint 机制，支持 GDN 层 prefix cache 命中后恢复 SSM state。

#### Scenario: Checkpoint 创建
- **WHEN** prefill 处理到 Full Attn block 边界（每 128 tokens）
- **THEN** snapshot 所有 GDN 层的 SSM state（h 矩阵），以 `hash(prefix_tokens)` 为 key 存入 SSMStatePool

#### Scenario: Checkpoint 恢复
- **WHEN** 新请求到达，prefix hash 匹配到已有 checkpoint
- **THEN** 从 SSMStatePool 恢复 h 矩阵作为 `initial_state`，跳过 prefix tokens 的 GDN 计算，从 checkpoint token 位置开始增量计算

#### Scenario: LRU 淘汰
- **WHEN** SSMStatePool 中的 checkpoint 数量超过 `max_checkpoints` 上限
- **THEN** 按最近最少使用策略淘汰旧 checkpoint，释放显存

### Requirement: SSMStatePool 数据结构

系统 SHALL 实现 `SSMStatePool` 和 `SSMCheckpoint` 数据结构。

#### Scenario: SSMCheckpoint 存储
- **WHEN** checkpoint 创建
- **THEN** 存储 `layer_states: dict[str, Tensor]`（48 层 GDN 的 float32 h 矩阵）、`hash_key: int`（与 Full Attn prefix hash 统一）、`num_tokens: int`、`ref_count: int`

#### Scenario: 显存预算控制
- **WHEN** 310P 可用显存约 40-50G
- **THEN** 每个 checkpoint 约 144 MiB（48 layers × num_heads × head_dim² × 4 bytes），`max_checkpoints` 上限根据可用显存动态计算

### Requirement: 310P SSM Checkpoint 与 KVCacheManager 集成

系统 SHALL 扩展 `get_computed_blocks()` 同时返回 matched SSM checkpoint。

#### Scenario: 混合 prefix cache 查询
- **WHEN** 新请求到达，查询 prefix cache
- **THEN** 返回：(1) Full Attn 命中的 block 列表，(2) GDN SSM checkpoint（如有匹配）

#### Scenario: 310P model_runner 集成
- **WHEN** `NPUModelRunner310.execute_model()` 执行 GDN 层 forward
- **THEN** 在 `AscendGatedDeltaNetAttention310._forward_core()` 中使用恢复的 `initial_state`，跳过已缓存的 prefix tokens

### Requirement: 310P Block Size 对齐

系统 SHALL 确保 GDN SSM checkpoint 间隔与 Full Attn block 边界对齐。

#### Scenario: block_size=128 对齐
- **WHEN** KV cache block_size=128
- **THEN** SSM checkpoint 每 128 tokens 创建一次，与 Full Attn block hash 粒度一致

#### Scenario: block_size=64 对齐
- **WHEN** KV cache block_size=64（受 `block_size * head_size ≤ 128*128` 约束）
- **THEN** SSM checkpoint 仍每 128 tokens 创建一次（独立于 KV block_size），部分 block hash 命中可能无对应 SSM checkpoint

### Requirement: 310P GDN 算子无 APC 支持

系统 SHALL 在 GDN 算子不具备 APC（Automatic Prefix Caching）支持的情况下，通过 SSMStatePool 框架层实现等效功能。

#### Scenario: 无 APC 算子的 prefix cache
- **WHEN** 310P GDN 算子不支持 `block_idx_last_scheduled_token` / `initial_state_idx` 参数
- **THEN** prefix cache 通过 SSMStatePool 在模型层实现，不依赖算子级 APC
