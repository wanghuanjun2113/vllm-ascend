## ADDED Requirements

### Requirement: FIA 算子压缩 mask 310P 支持

系统 SHALL 在 310P 的 Flash Infer Attention 算子中支持压缩 mask 模式，减少显存占用和传输带宽：

```python
# 310P FIA 接口扩展
torch_npu._npu_flash_attention(
    query, key, value,
    ...,
    sparse_mode: int = 3,  # 新增：3=causal mask 压缩
)

torch_npu._npu_paged_attention(
    query, key_cache, value_cache,
    ...,
    sparse_mode: int = 3,  # 新增：decode 路径 causal mask 压缩
)
```

当 `sparse_mode=3` 时，NPU 硬件 SHALL 内部生成 causal mask，无需外部传入完整 `[batch, seq, seq]` mask tensor。

#### Scenario: Prefill 路径 causal mask 压缩
- **WHEN** 310P 执行 prefill 注意力计算，`sparse_mode=3`
- **THEN** 硬件内部生成下三角 causal mask，不分配外部 mask tensor，节省 `batch * seq * seq * 2` 字节显存

#### Scenario: Decode 路径 causal mask 压缩
- **WHEN** 310P 执行 decode 注意力计算（`_npu_paged_attention`），`sparse_mode=3`
- **THEN** 硬件自动处理 causal mask 语义，减少 mask 传输开销

#### Scenario: 滑动窗口 mask
- **WHEN** 模型配置了 sliding_window，`sparse_mode=4`
- **THEN** 硬件生成滑动窗口 mask（上三角 + 超出窗口的下三角屏蔽）

### Requirement: FIA 压缩 mask 与 310P AttentionBackend 集成

`AscendAttentionBackend310` SHALL 在创建 attention 调用时使用 `sparse_mode` 参数替代完整 mask 物化。

#### Scenario: 替换 AttentionMaskBuilder310
- **WHEN** 310P 注意力后端初始化，检测到 FIA 算子支持 `sparse_mode`
- **THEN** 跳过 `AttentionMaskBuilder310` 的完整 mask 生成和 NZ 格式转换，直接使用 `sparse_mode` 参数

### Requirement: 拒绝采样 Greedy 路径 310P 实现

系统 SHALL 在 310P 上提供 greedy rejection sampling 算子：

```python
rejection_greedy_sample_pytorch(
    draft_token_ids: Tensor,     # [num_tokens]
    target_logits: Tensor,       # [num_tokens, vocab_size]
    bonus_token_ids: Tensor,     # [batch_size]
    num_draft_tokens: list[int], # [batch_size]
    cu_num_draft_tokens: Tensor, # [batch_size]
    max_spec_len: int,
) -> Tensor  # [batch_size, max_spec_len + 1]
```

#### Scenario: Greedy 拒绝采样
- **WHEN** 投机推理 verify 阶段使用 greedy sampling
- **THEN** 比较 `draft_token_ids == target_argmax` 逐位置匹配，第一个不匹配位置后丢弃所有 draft token，最后一个匹配的请求追加 bonus token

#### Scenario: 全部接受
- **WHEN** 所有 draft token 均与 target argmax 匹配
- **THEN** 输出包含所有 draft token + bonus token

### Requirement: 拒绝采样 Random 路径 310P 实现

系统 SHALL 在 310P 上提供 random/stochastic rejection sampling 算子：

```python
rejection_random_sample_pytorch(
    draft_token_ids: Tensor,     # [num_tokens]
    draft_probs: Tensor,         # [num_tokens, vocab_size]
    target_probs: Tensor,        # [num_tokens, vocab_size]
    bonus_token_ids: Tensor,     # [batch_size]
    recovered_token_ids: Tensor, # [num_tokens]
    num_draft_tokens: list[int],
    cu_num_draft_tokens: Tensor,
    max_spec_len: int,
    is_greedy: Tensor,           # [batch_size]
) -> Tensor  # [batch_size, max_spec_len + 1]
```

#### Scenario: Random 拒绝采样
- **WHEN** 投机推理 verify 阶段使用 stochastic sampling
- **THEN** 接受条件为 `target_prob / draft_prob >= uniform_prob`，拒绝时从残差分布 `max(0, target_probs - draft_probs) / q` 采样 recovered token

### Requirement: 拒绝采样 Block Verify 路径 310P 实现

系统 SHALL 在 310P 上提供 block verify rejection sampling（`max_spec_len >= 3` 时使用）：

```python
rejection_random_sample_block_verify_pytorch(
    # 与 random 路径相同参数
    # 增加前缀概率追踪和逐位置接受阈值
) -> Tensor
```

#### Scenario: Block Verify 拒绝采样
- **WHEN** `max_spec_len >= 3` 且 `draft_probs` 可用
- **THEN** 计算前缀概率和逐位置接受阈值 `h_block`，提高 acceptance rate

### Requirement: Recovered Token Sampling 310P 实现

系统 SHALL 在 310P 上提供 recovered token 采样算子：

```python
sample_recovered_tokens_pytorch(
    target_probs: Tensor,    # [num_tokens, vocab_size]
    draft_probs: Tensor,     # [num_tokens, vocab_size]
    q: Tensor,               # [num_tokens] — 从指数分布采样
) -> Tensor  # [num_tokens] — recovered token IDs
```

#### Scenario: 残差分布采样
- **WHEN** draft token 被拒绝，需要从残差分布中采样 recovered token
- **THEN** 计算 `residual = max(0, target_probs - draft_probs)`，从 `residual / q` 中采样 token ID

### Requirement: Eagle 输入扩展 310P 实现

系统 SHALL 在 310P 上提供 EAGLE/MTP 投机推理的输入扩展算子：

```python
# 310P 需要等效实现（PyTorch 或 AscendC）
copy_and_expand_eagle_inputs_310(
    target_token_ids: Tensor,
    target_positions: Tensor,
    next_token_ids: Tensor,
    query_start_loc: Tensor,
    query_end_loc: Tensor,
    padding_token_id: int,
    parallel_drafting_token_id: int,
    num_padding_slots_per_request: int,
    shift_input_ids: bool,
    total_draft_tokens: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
    # (out_input_ids, out_positions, out_is_rejected_token_mask,
    #  out_is_masked_token_mask, out_new_token_indices, out_hidden_state_mapping)
```

#### Scenario: EAGLE 投机推理输入准备
- **WHEN** EAGLE/MTP proposer 需要为下一轮 draft 准备输入
- **THEN** 扩展和重排 token IDs、positions、rejected masks 等信息

### Requirement: 拒绝采样算子性能要求

310P 上拒绝采样 PyTorch 实现的单次调用开销 SHALL 不超过 0.1ms。

#### Scenario: 性能基准
- **WHEN** 在 310P 上执行一次完整的 rejection sampling（含 greedy + random + block verify 路径）
- **THEN** 总耗时不超过 0.1ms（vocab_size 比较操作为主，计算量极小）

### Requirement: 注意力/投机推理算子正确性验证

每个 310P 注意力和投机推理算子 SHALL 与 910 Triton 实现的输出一致。

#### Scenario: 拒绝采样输出一致性
- **WHEN** 使用相同的 draft tokens、target logits/probs 和随机种子
- **THEN** 310P PyTorch 实现的 rejection sampling 输出与 910 Triton 实现一致
