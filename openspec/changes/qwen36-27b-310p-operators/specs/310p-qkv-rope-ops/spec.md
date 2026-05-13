## ADDED Requirements

### Requirement: mRope 310P 实现

系统 SHALL 在 310P 上提供多模态 Rotary Position Embedding (mRope) 算子，接口与 910 `torch_npu.npu_mrope` 一致：

```python
npu_mrope_310(
    positions: Tensor,      # [batch, num_pos_dim] — 2D 多模态位置
    query: Tensor,          # [num_tokens, num_heads, head_dim]
    key: Tensor,            # [num_tokens, num_kv_heads, head_dim]
    cos_sin_cache: Tensor,  # cos+sin 缓存
    head_size: int,
    mrope_section: list[int],  # [temporal, height, width] 分区
    rotary_mode: str = "half",
) -> tuple[Tensor, Tensor]  # (query_rotated, key_rotated)
```

mRoPE SHALL 支持三个独立旋转维度分区：temporal、height、width，每个分区使用独立的位置索引和旋转角度。

#### Scenario: Qwen3.6 多模态位置编码
- **WHEN** 输入包含 2D 位置信息（多模态场景），`mrope_section=[16, 24, 24]`
- **THEN** 对 query 和 key 分别按 temporal/height/width 三个分区应用独立的旋转位置编码

#### Scenario: 单模态 fallback
- **WHEN** mRoPE AscendC kernel 不可用
- **THEN** 退回到通用 RoPE 实现或纯 PyTorch mRoPE 计算

### Requirement: split_qkv_rmsnorm_Mrope 310P 实现

系统 SHALL 在 310P 上提供融合 QKV split + RMSNorm + mRoPE 算子：

```python
split_qkv_rmsnorm_mrope_310(
    qkv: Tensor,          # [num_tokens, q_size + gate_size + 2*kv_size]
    q_weight: Tensor,     # [head_size]
    k_weight: Tensor,     # [head_size]
    cos_sin: Tensor,      # [3*num_tokens, rope_dim]
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    eps: float,
    mrope_section: list[int],
    is_interleaved: bool,
    rope_dim: int | None = None,
    q_bias: Tensor | None = None,
    k_bias: Tensor | None = None,
    has_gate: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]
    # (q_output, k_output, v_output, gate_output)
```

#### Scenario: 融合 QKV 处理
- **WHEN** 模型前向传播中需要从融合投影输出提取 Q、K、V 并应用归一化和旋转编码
- **THEN** 在单次 kernel 调用中完成 QKV split + Q/K RMSNorm + Q/K mRoPE 三步操作

#### Scenario: 带门控输出
- **WHEN** `has_gate=True`，融合投影包含 gate 分支
- **THEN** 额外输出 gate 张量，维度与 Q 一致

### Requirement: rmsnormgated 310P 实现

系统 SHALL 在 310P 上提供融合 RMSNorm + SiLU gating 算子：

```python
rmsnorm_gated_310(
    x: Tensor,        # [num_tokens, hidden_size]
    z: Tensor,        # [num_tokens, hidden_size] — gate 分支
    weight: Tensor,   # [hidden_size]
    eps: float,
    norm_before_gate: bool = True,
) -> Tensor
    # norm_before_gate=True:  y = RMSNorm(x) * weight; y *= z * sigmoid(z)
    # norm_before_gate=False: x *= z * sigmoid(z); y = RMSNorm(x) * weight
```

#### Scenario: RMSNorm + SiLU gating 融合
- **WHEN** GDN 层输出投影需要 RMSNorm 和门控融合
- **THEN** 根据 `norm_before_gate` 参数选择先归一化再门控或先门控再归一化

#### Scenario: 无门控模式
- **WHEN** `z=None`
- **THEN** 仅执行标准 RMSNorm：`y = RMSNorm(x) * weight`

### Requirement: transposeKV 310P 实现

系统 SHALL 在 310P 上提供 KV Cache 转置算子，接口与 910 `transpose_kv_cache_by_block` 一致：

```python
transpose_kv_cache_by_block_310(
    kCache: list[Tensor],
    vCache: list[Tensor],
    blockIDs: Tensor,
    blockSize: int,
    headNum: int,
    headDim: int,
    splitNum: int,
    layerNum: int,
) -> None  # in-place
```

#### Scenario: KV Cache 布局转换
- **WHEN** KV cache 需要从一种 block 格式转换为注意力计算所需的转置格式
- **THEN** in-place 执行 KV cache 转置，适配 310P 的 5D KV cache 布局 `(2, num_blocks, (num_kv_heads*head_size)//16, block_size, 16)`

#### Scenario: Tiling 参数适配
- **WHEN** 910 的 tiling 参数（TILING_KEY 0-4）在 310P 上不兼容
- **THEN** 根据 310P AICore 架构调整 tiling 参数（block 大小、memory 对齐方式）

### Requirement: QKV/RoPE 算子正确性验证

每个 QKV/RoPE 算子的 310P 实现 SHALL 与 910 参考实现的数值误差满足 `atol=1e-3, rtol=1e-3`。

#### Scenario: 数值正确性对比
- **WHEN** 使用相同输入分别调用 310P 实现和 910 Triton 参考实现
- **THEN** Q、K、V 输出的数值误差在 fp16 精度允许范围内
