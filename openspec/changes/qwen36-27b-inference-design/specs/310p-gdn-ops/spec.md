## ADDED Requirements

### Requirement: causal_conv1d_fwd 310P 实现

系统 SHALL 在 310P 上提供因果 1D 卷积前向算子，接口与 910 `npu_causal_conv1d_custom` 一致：

```python
causal_conv1d_310(
    x: Tensor,              # [dim, cu_seq_len]
    weight: Tensor,         # [dim, width]
    bias: Tensor | None,    # [dim]
    conv_states: Tensor,    # [..., dim, width-1]
    query_start_loc: list[int],
    cache_indices: list[int],
    initial_state_mode: list[int],
    num_accepted_tokens: list[int],
    activation_mode: int,   # 0=none, 1=silu
    pad_slot_id: int,
    run_mode: int,          # 0=prefill, 1=decode
) -> Tensor
```

算子 SHALL 支持 prefill（run_mode=0）和 decode（run_mode=1）两种运行模式。

#### Scenario: Prefill 模式 varlen 输入
- **WHEN** 输入为 varlen token 序列，`run_mode=0`，提供 `query_start_loc` 和 `cache_indices`
- **THEN** 对每个序列独立执行因果 1D 卷积，in-place 更新 `conv_states`，返回卷积输出

#### Scenario: Decode 模式单 token 更新
- **WHEN** 输入为单步 decode token，`run_mode=1`，提供 `cache_indices`
- **THEN** 使用 conv_state 执行单步卷积更新，返回更新后输出

### Requirement: causal_conv1d_update 310P 实现

系统 SHALL 在 310P 上提供因果 1D 卷积递推更新算子，支持投机推理的多 token 更新：

```python
causal_conv1d_update(
    x: Tensor,                  # [num_tokens, dim]
    conv_state: Tensor,         # [..., dim, state_len]
    weight: Tensor,             # [dim, width]
    bias: Tensor | None,
    activation: bool | str | None,
    conv_state_indices: Tensor, # [batch] int32
    num_accepted_tokens: Tensor,# [batch] int32
    query_start_loc: Tensor,    # [batch+1] int32
    pad_slot_id: int,
) -> Tensor
```

#### Scenario: 投机推理多 token 更新
- **WHEN** `num_accepted_tokens` 指定每批次接受的 token 数，`query_start_loc` 指定 varlen 边界
- **THEN** 对每个请求的前 N 个 token（N=num_accepted_tokens）执行因果卷积更新

### Requirement: fused_recurrent_gated_delta_rule_fwd 310P 实现

系统 SHALL 在 310P 上提供融合递推 gated delta rule 前向算子：

```python
fused_recurrent_gated_delta_rule_pytorch(
    q: Tensor,           # [1, T, H, K]
    k: Tensor,           # [1, T, H, K]
    v: Tensor,           # [1, T, HV, V]
    g: Tensor,           # [1, T, HV]
    beta: Tensor,        # [1, T, HV]
    initial_state: Tensor,    # SSM state tensor
    inplace_final_state: bool,
    cu_seqlens: Tensor,       # [N+1]
    ssm_state_indices: Tensor,
    num_accepted_tokens: Tensor | None,
    use_qk_l2norm_in_kernel: bool,
) -> tuple[Tensor, Tensor]  # (output, final_state)
```

#### Scenario: Decode 路径递推
- **WHEN** 输入为 decode 步骤的单 token（或投机推理的多 token），提供 `ssm_state_indices` 和 `cu_seqlens`
- **THEN** 执行递推公式 `h_t = exp(g_t)*h_{t-1} + beta_t*(v_t - h_{t-1}@k_t)⊗k_t`，输出 attention 结果并更新 SSM state

#### Scenario: 投机推理 spec decode 路径
- **WHEN** `num_accepted_tokens` 非 None，每个请求有多个 accepted token
- **THEN** 对 accepted tokens 执行递推更新，结果通过 `index_copy_` 合并回主输出

### Requirement: chunk_gated_delta_rule_fwd 310P 实现

系统 SHALL 在 310P 上提供 chunk 并行 gated delta rule 前向算子，用于 prefill 路径：

```python
chunk_gated_delta_rule_pytorch(
    q: Tensor,              # [T, H, K] (head_first=False)
    k: Tensor,              # [T, H, K]
    v: Tensor,              # [T, HV, V]
    g: Tensor,              # [T, HV]
    beta: Tensor,           # [T, HV]
    initial_state: Tensor,  # [N, H, K, V]
    output_final_state: bool,
    cu_seqlens: LongTensor | None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[Tensor, Tensor]  # (output, final_state)
```

算法 SHALL 使用 chunk 并行策略（CHUNK_SIZE=64），包含 WY 表示、下三角求解和跨 chunk 递推。

#### Scenario: Prefill 路径 chunk 并行计算
- **WHEN** 输入为 prefill 序列（长序列），`cu_seqlens` 指定 varlen 边界，`output_final_state=True`
- **THEN** 执行 chunk 分块并行计算，返回 attention 输出和最终 SSM state

#### Scenario: 初始状态恢复
- **WHEN** 提供 `initial_state`（从 SSM checkpoint 恢复）
- **THEN** 从 checkpoint 对应的 token 位置开始增量计算，跳过已缓存的 prefix tokens

### Requirement: fused_sigmoid_gating_delta_rule_310 实现

系统 SHALL 在 310P 上提供融合 sigmoid gating delta rule 算子，将 gating 计算和递推更新合并为单 kernel：

```python
fused_sigmoid_gating_delta_rule_310(
    A_log: Tensor,       # [num_heads]
    a: Tensor,           # [B, HV]
    dt_bias: Tensor,     # [num_heads]
    softplus_beta: float,
    softplus_threshold: float,
    q: Tensor,           # [B, T, H, K]
    k: Tensor,           # [B, T, H, K]
    v: Tensor,           # [B, T, HV, V]
    b: Tensor,           # [B, T, HV]
    initial_state_source: Tensor,
    initial_state_indices: Tensor,
    scale: float = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Tensor = None,
) -> Tensor  # [B, T, HV, V]
```

#### Scenario: Decode 路径单步融合
- **WHEN** decode 步骤执行 GDN attention
- **THEN** 在单 kernel 内完成：gating 计算（exp+softplus+sigmoid）+ SSM state 递推更新 + attention 输出计算

### Requirement: fused_gdn_gating 310P 实现

系统 SHALL 在 310P 上提供 GDN gating 融合算子：

```python
fused_gdn_gating_310(
    A_log: Tensor,   # [num_heads]
    a: Tensor,       # [batch, num_heads]
    b: Tensor,       # [batch, num_heads]
    dt_bias: Tensor, # [num_heads]
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[Tensor, Tensor]:
    # g: [1, batch, num_heads] float32
    # beta_output: [1, batch, num_heads]
```

计算公式：`g = -exp(A_log) * softplus(a + dt_bias)`, `beta_output = sigmoid(b)`

#### Scenario: GDN 层 gating 计算
- **WHEN** GDN 层 forward 中需要计算 decay gate (g) 和 beta gate
- **THEN** 融合执行 exp + softplus + sigmoid 计算，输出 g 和 beta_output

#### Scenario: Fallback 模式
- **WHEN** AscendC kernel 不可用（如开发阶段）
- **THEN** 退回到 `fused_gdn_gating_pytorch`（纯 Python fp32 实现），接口和输出格式一致

### Requirement: GDN 算子双路径 Fallback 机制

系统 SHALL 为每个 GDN 算子提供 AscendC 优先 + PyTorch fallback 双路径实现。

#### Scenario: AscendC 可用时使用优化路径
- **WHEN** AscendC kernel 已注册（`torch.ops._C_ascend.*` 可调用）
- **THEN** 使用 AscendC 优化实现

#### Scenario: AscendC 不可用时退回 PyTorch
- **WHEN** AscendC kernel 未注册或环境变量 `VLLM_ASCEND_USE_ASCENDC_OPS=0`
- **THEN** 退回到 PyTorch fallback 实现，保证正确性

### Requirement: GDN 算子正确性验证

每个 GDN 算子的 AscendC 实现 SHALL 与 910 参考实现（或 PyTorch fallback）的数值误差满足 `atol=1e-3, rtol=1e-3`。

#### Scenario: 数值正确性测试
- **WHEN** 使用相同输入分别调用 310P AscendC 实现和 910 Triton/PyTorch 参考实现
- **THEN** 输出张量的数值误差在 fp16 精度允许范围内（atol=1e-3, rtol=1e-3）
