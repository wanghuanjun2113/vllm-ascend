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

### Requirement: 用户可配置 Linear Attention Checkpoint 保存策略

系统 SHALL 提供外部接口配置 Linear Attention checkpoint 保存策略，用于控制 SSMStatePool 的 checkpoint 创建密度和语义锚点。

#### Scenario: 策略字段透传
- **WHEN** 外部服务或部署层传入 `linearCheckpointPolicy`
- **THEN** 系统 SHALL 支持 `mode`、`intervalTokens`、`anchors`、`maxCheckpointsPerRequest` 和 `maxCheckpointsPerPrefixProfile` 字段，并将策略传递到 SSM checkpoint 创建逻辑。

#### Scenario: 固定 tools/system prompt 只保存一个 checkpoint
- **WHEN** `mode=anchor` 且 `anchors=["tools", "system_prompt"]`，同时 `maxCheckpointsPerRequest=1`
- **THEN** 系统 SHALL 只在 chat template 渲染后的稳定 tools + system prompt token 边界保存一个 SSM checkpoint，动态 user content 不创建新的 SSM checkpoint。

#### Scenario: checkpoint key 版本绑定
- **WHEN** 创建 anchor checkpoint
- **THEN** checkpoint key SHALL 绑定 token hash、`prefix_profile_key`、模型 revision、tokenizer revision、chat template version、tool schema version、system prompt version 和 thinking mode。

#### Scenario: 显存安全上限
- **WHEN** 用户策略请求的 checkpoint 数量或间隔超过全局显存预算
- **THEN** 系统 SHALL 按全局 `max_checkpoints`、LRU 淘汰和服务端安全上限裁剪策略；用户策略只能减少 checkpoint 数量或放大保存间隔，不能突破全局上限。

#### Scenario: 策略关闭
- **WHEN** `mode=disabled`
- **THEN** 系统 SHALL 不保存 GDN SSM checkpoint，仅保留 Full Attention block-level prefix cache，并在 profiling 中记录 GDN prefix 重算开销。

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
