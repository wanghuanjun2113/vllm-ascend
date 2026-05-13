## Context

Qwen3.6 27B 采用 Full Attention + Linear Attention (GDN) 混合注意力架构，在 300IDuo (310P3) 上推理时，910 平台的 Triton kernel 和部分 CANN 算子无法直接使用。310P 不支持 Triton，需使用 AscendC 或纯 PyTorch 替代方案。当前 310P 的 GDN 推理已通过 PyTorch fallback 路径实现正确性，但性能远低于 910（TPOT 基线要求 ≤ 60ms）。

本设计覆盖所有算子的开发状态、接口定义与 310P 集成方案，分为四个领域：

| 领域 | 算子数量 | 性能影响 |
|------|----------|----------|
| GDN 核心（已具备） | 3 | Prefill/Decode 主路径 |
| GDN 核心（开发中） | 2 | Prefill/Decode 关键路径 |
| QKV/RoPE 处理 | 4 | 注意力前置处理 |
| 注意力/投机推理 | 1 + 8 | 注意力 mask 优化 + 投机推理采样 |

### 算子开发状态总览

| 算子名称 | 开发状态 | 优先级 | 开发方 | 备注/落地计划 |
|----------|----------|--------|--------|---------------|
| causal_conv1d_fwd_kernel | 已经具备 | - | CANN/vllm-ascend | Q2 商发 |
| causal_conv1d_update_kernel | 已经具备 | - | CANN/vllm-ascend | Q2 商发 |
| fused_recurrent_gated_delta_rule_fwd_kernel | 已经具备 | - | CANN/vllm-ascend | Q2 商发 |
| chunk_gated_delta_rule_fwd | SAIE 开发中 | - | SAIE | AscendC 实现 |
| rmsnormgated | NAIE 开发 | 低 | NAIE | 性能收益不大 |
| fused_gdn_gating | NAIE 开发 | 高 | NAIE | 每层每步调用 |
| mRope | NAIE 开发 | 高 | NAIE | 多模态位置编码 |
| split_qkv_rmsnorm_Mrope | NAIE 开发 | 低 | NAIE | 性能收益不大 |
| transposeKV | NAIE 开发 | 低 | NAIE | 性能收益不大 |
| fused_sigmoid_gating_delta_rule_310 | SAIE 开发中 | - | SAIE+NAIE | Decode 单步融合 |
| FIA 算子增加压缩 mask | NAIE 能力中心 | 高 | NAIE 能力中心 | 注意力 mask 优化 |
| rejection_greedy_sample | NAIE 能力中心 | - | NAIE 能力中心 | Greedy 拒绝采样（含 spec_len=1 快速路径） |
| rejection_random_sample | NAIE 能力中心 | - | NAIE 能力中心 | Random 拒绝采样 |
| rejection_random_sample_block_verify | NAIE 能力中心 | - | NAIE 能力中心 | Block verify 拒绝采样（max_spec_len≥3） |
| sample_recovered_tokens | NAIE 能力中心 | - | NAIE 能力中心 | 残差分布恢复采样 |
| expand_batch_to_tokens | NAIE 能力中心 | - | NAIE 能力中心 | Per-request→Per-token 扩展 |
| npu_copy_and_expand_eagle_inputs | NAIE 能力中心 | - | NAIE 能力中心 | Eagle/MTP 输入扩展（C++ 算子） |
| prepare_inputs_padded | NAIE 能力中心 | - | NAIE 能力中心 | Spec decode 输入准备 |

### 310P 平台约束

- 不支持 Triton，需 AscendC 或纯 PyTorch 替代
- float16（部分 ATB 算子不支持 BF16）
- KV cache 受限：`block_size * head_size <= 128 * 128`
- 310P KV cache 布局为 5D NZ 对齐格式：`(2, num_blocks, (num_kv_heads * head_size) // 16, block_size, 16)`
  - 例：`num_kv_heads=8, head_size=128` → `(2, num_blocks, 64, block_size, 16)`
  - 910 为标准 ND 格式：`(2, num_blocks, block_size, num_kv_heads, head_size)`
- 当前版本仅 eager 模式，未来 CANN 版本计划支持 ACLGraph
- 310P 注册算子通过 `#ifdef ASCEND_PLATFORM_310P` 编译时条件选择

### 注册机制

vllm-ascend 使用两套注册机制：

1. **C++ 层 `TORCH_LIBRARY_EXPAND`**（`csrc/torch_binding.cpp`）：
   - 310P 编译时仅注册 `npu_causal_conv1d_310` 和 `npu_recurrent_gated_delta_rule_310`
   - 910 注册完整算子集（~30 个），包括 `npu_causal_conv1d_custom`、`npu_sparse_flash_attention` 等
   - `npu_copy_and_expand_eagle_inputs` 目前仅 910 注册（`#ifndef ASCEND_PLATFORM_310P`），310P 需移植

2. **Python 层 `direct_register_custom_op`**（`ops/register_custom_ops.py`）：
   - 注册 `torch.ops.vllm.*` 命名空间的算子
   - 包含 `triton_split_qkv_rmsnorm_mrope`、`qkv_rmsnorm_rope` 等

## Goals / Non-Goals

**Goals:**
- 为每个算子提供明确的接口定义（参考 910 实现）和 310P 适配方案
- 详细描述关键输入参数的构造方式，尤其是 910 与 310P 的差异
- 高优先级算子（fused_gdn_gating、mRope、FIA 压缩 mask）优先完成
- 已具备算子确认可用性，开发中算子明确依赖与排期
- 新增算子整网耗时占比相比 910 不超过 30%

**Non-Goals:**
- 不覆盖 910 平台算子实现细节（仅作为参考接口）
- 不涉及 310P 图模式设计（独立设计文档覆盖）
- 不涉及量化精度对齐方案
- 不涉及 BF16 支持（310P 限制）

## Decisions

### 1. causal_conv1d_fwd_kernel — 已具备

**910 接口参考**（C++ 自定义算子 `npu_causal_conv1d_custom`）：

```python
torch.ops._C_ascend.npu_causal_conv1d_custom(
    x: Tensor,              # [dim, cu_seq_len] (varlen)
    weight: Tensor,         # [dim, width]
    conv_state: Tensor,     # [..., dim, width-1]
    bias: Tensor | None,    # [dim]
    query_start_loc: list[int],   # [N+1] 累积序列长度
    cache_indices: list[int],     # [N] cache 索引
    initial_state_mode: list[int],# [N] 是否有初始状态
    num_accepted_tokens: list[int],# [N] 投机推理接受数
    activation_mode: int,   # 0=none, 1=silu
    pad_slot_id: int,       # padding slot ID
    run_mode: int,          # 0=prefill, 1=decode
) -> Tensor                 # [dim, cu_seq_len]
```

**310P 实现**：`npu_causal_conv1d_310`（AscendC，`csrc/causal_conv1d_v310/`），接口签名与 910 版本一致。

**310P 参数构造（与 910 的关键差异）**：

| 参数 | 910 构造方式 | 310P 构造方式 | 差异原因 |
|------|-------------|-------------|----------|
| `conv_weights` | `self.conv1d.weight.view(...)` 不转置，Triton kernel 内部转置 | `self.conv1d.weight.view(...).transpose(0, 1)` 提前转置 | 310P AscendC 算子期望转置后的权重 |
| `conv_state` | `self_kv_cache[0].transpose(-1, -2)` 预转置 | `self_kv_cache[0]` 直接使用不转置 | 310P AscendC 算子接受原始 cache 布局 `(num_lines, dim, width-1)`；910 Triton kernel 需要 `(num_lines, width-1, dim)` |
| `query_start_loc` | 设备 Tensor（Triton）或 host tuple（C++ op） | 始终转为 Python tuple：`to_int64_tuple(spec_query_start_loc)` | 310P C++ 算子只接受 host 端参数 |
| `cache_indices` | 设备 Tensor slice | `to_int64_tuple(spec_state_indices_tensor[:, 0][:num_spec_decodes])` | 310P C++ 算子接受 host 端参数 |
| `initial_state_mode` | 不需要（Triton kernel 隐式处理）| Spec/Decode: `[1] * batch_size` 全 1；Prefill: `to_int64_tuple(has_initial_state)` | 310P 需要显式指定是否有初始状态 |
| `num_accepted_tokens` | 设备 Tensor | `to_int64_tuple(num_accepted_tokens)` 或 `[]`（非 spec 场景） | 310P C++ 算子接受 host 端参数 |
| `run_mode` | 不需要（通过选择不同函数区分） | `0`=prefill, `1`=decode | 310P 单一算子通过 run_mode 区分模式 |

**输出处理**：
- 返回卷积后的 `mixed_qkv` 张量，形状 `[dim, cu_seq_len]`
- conv_state 被 in-place 更新（写入最新 state_len 个 token 的卷积状态）
- 310P 和 910 的输出格式一致，无需额外转换

**集成位置**：`gdn_310.py` 第 103 行（spec 路径）、第 120 行（prefill 路径）、第 135 行（decode 路径）。

### 2. causal_conv1d_update_kernel — 已具备

**910 接口参考**（Triton kernel `_causal_conv1d_update_kernel_npu_tiled`）：

```python
def causal_conv1d_update_npu(
    x: Tensor,                  # [batch, dim] 或 [num_tokens, dim]
    conv_state: Tensor,         # [..., dim, state_len]
    weight: Tensor,             # [dim, width]
    bias: Tensor | None,        # [dim]
    activation: bool | str,     # True/"silu"/None
    conv_state_indices: Tensor, # [batch] int32
    num_accepted_tokens: Tensor,# [batch] int32 — 投机推理
    query_start_loc: Tensor,    # [batch+1] int32 — varlen
    max_query_len: int = -1,
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_last_scheduled_token: Tensor | None = None,  # APC
    initial_state_idx: Tensor | None = None,               # APC
) -> Tensor                    # same shape as x
```

**310P 实现**：310P decode 场景复用 `npu_causal_conv1d_310(run_mode=1)` 统一处理，不单独使用 update 函数。

**310P 参数构造差异**：
- 910 Triton kernel 接受设备端 Tensor 参数（`conv_state_indices`、`num_accepted_tokens`、`query_start_loc`）
- 310P 通过 `npu_causal_conv1d_310` 的 host 端 list 参数传入（`cache_indices`、`num_accepted_tokens`、`query_start_loc` 均为 Python tuple/list）
- 910 支持 APC（Automatic Prefix Caching）参数 `block_idx_last_scheduled_token`、`initial_state_idx`；310P 当前不支持 APC

**输出处理**：
- 返回更新后的卷积输出，形状与输入 `x` 相同
- conv_state in-place 更新

### 3. fused_recurrent_gated_delta_rule_fwd_kernel — 已具备

**910 接口参考**（两个版本）：

Triton kernel（用于 Qwen3.5 等 ssm_state 为 float32 的场景）：
```python
fused_recurrent_gated_delta_rule(
    q, k, v, g, beta,
    initial_state=ssm_state,
    inplace_final_state=True,
    cu_seqlens=spec_query_start_loc[:num_spec_decodes+1],
    ssm_state_indices=spec_state_indices_tensor,
    num_accepted_tokens=num_accepted_tokens,
    use_qk_l2norm_in_kernel=True,
) -> tuple[Tensor, Tensor]  # (output, final_state)
```

NPU 原生算子（用于 Qwen3Next 等 ssm_state 为 float16/bf16 的场景）：
```python
torch_npu.npu_recurrent_gated_delta_rule(
    query=query_spec.squeeze(0),       # 去掉 batch 维
    key=key_spec.squeeze(0),
    value=value_spec.squeeze(0),
    g=g_spec.squeeze(0),
    beta=beta_spec.squeeze(0),
    state=ssm_state,                   # 注意参数名不同
    scale=key_spec.shape[-1] ** -0.5,  # 显式传入 scale
    actual_seq_lengths=actual_seq_lengths,  # 注意：是长度而非累积长度
    ssm_state_indices=spec_state_indices_tensor.flatten(),  # 展平为 1D
    num_accepted_tokens=num_accepted_tokens.to(torch.int32), # 显式转 int32
).unsqueeze(0)  # 恢复 batch 维
```

**310P 实现**：
- AscendC：`npu_recurrent_gated_delta_rule_310`（`csrc/recurrent_gated_delta_rule_v310/`）
- PyTorch fallback：`fused_recurrent_gated_delta_rule_pytorch`（`_310p/ops/fla/fused_recurrent_gated_delta_rule.py`）

**310P 参数构造（与 910 的关键差异）**：

| 参数 | 910 NPU op | 310P PyTorch fallback |
|------|-----------|----------------------|
| `initial_state` / `state` | 参数名 `state`，batch 维 squeezed | 参数名 `initial_state`，保持原始 batch 维 `[1, T, H, K]` |
| `cu_seqlens` | 转换为 `actual_seq_lengths`（`cu[1:]-cu[:-1]`） | 直接传入累积长度 Tensor，内部逐序列遍历 |
| `ssm_state_indices` | `.flatten()` 为 1D | 保持 2D tensor，内部按 `state_idx = int(indices[i, 0].item())` 取索引 |
| `num_accepted_tokens` | `.to(torch.int32)` 类型转换 | 传入原始 Tensor，内部 `int(num_accepted_tokens[seq_idx].item())` |
| `scale` | 显式传入 `key_spec.shape[-1] ** -0.5` | 内部计算 `Kdim ** -0.5` |
| 返回值 | 单 Tensor（`.unsqueeze(0)`） | `(output, final_state)` 元组 |

**310P 内部状态更新方式**：
```python
# 310P 逐序列循环更新 ssm_state：
for seq_idx in range(n_seq):
    state_idx = int(ssm_state_indices[seq_idx, 0].item())
    h_t = ssm_state[state_idx].to(torch.float32)
    # ... 递推计算 ...
    ssm_state[state_idx] = h_t.to(ssm_state.dtype)  # in-place 写回
```

**输出处理差异**：
- 910 NPU op 返回单 Tensor，需 `.unsqueeze(0)` 恢复 batch 维；state 更新由 NPU kernel 内部完成
- 310P 返回 `(output, final_state)` 元组；310P prefill chunk 路径需显式写回：`ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(ssm_state.dtype)`
- 910 gqa_interleaved 分支需 `.transpose(-1, -2)` 转换 state 布局（`[N,H,V,K]` ↔ `[N,H,K,V]`），310P 不需要转置

**集成位置**：`gdn_310.py` 第 176 行（spec decode）、第 215 行（non-spec decode）、第 231 行（纯 decode）。

### 4. chunk_gated_delta_rule_fwd — SAIE 开发中

**910 接口参考**（Triton 实现，多个子 kernel 协作）：

```python
def chunk_gated_delta_rule(
    q: Tensor,              # [B, T, H, K] (head_first=False)
    k: Tensor,              # [B, T, H, K]
    v: Tensor,              # [B, T, HV, V]
    g: Tensor,              # [B, T, HV] — 衰减门控（log 空间）
    beta: Tensor,           # [B, T, HV]
    scale: float = None,    # 默认 1/sqrt(K)
    initial_state: Tensor = None,  # [N, H, K, V]
    output_final_state: bool = False,
    cu_seqlens: LongTensor | None = None,  # [N+1]
    prebuilt_meta = None,   # 预计算的 chunk 元数据
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[Tensor, Tensor]:  # (output, final_state)
```

**子步骤流水线**（910 Triton 实现）：
1. `chunk_local_cumsum` — 局部累积求和
2. `chunk_scaled_dot_kkt_fwd` — KKT 矩阵乘（chunk 内注意力）
3. `solve_tril` — 下三角求解（WY 表示）
4. `recompute_w_u_fwd` — 重新计算 w 和 u
5. `chunk_gated_delta_rule_fwd_h` — 跨 chunk 递推状态更新
6. `chunk_fwd_o` — 最终输出计算

**310P 参数构造（与 910 的关键差异）**：

| 参数 | 910 | 310P |
|------|-----|------|
| `initial_state` 构造 | `ssm_state[indices].transpose(-1, -2).contiguous()`（gqa_interleaved）或 `.contiguous()`（non-interleaved） | `ssm_state[indices].contiguous()`（不转置） |
| `initial_state` 清零 | `clear_ssm_states(initial_state, has_initial_state)`（专用 Triton kernel） | `initial_state[~has_initial_state, ...] = 0`（布尔索引直接置零） |
| `prebuilt_meta` | 传入预计算的 chunk 元数据（`chunk_indices`、`chunk_offsets` 等） | 不传（`prebuilt_meta` 参数不存在） |
| `cu_seqlens` 处理 | Triton kernel 直接使用设备端 Tensor | PyTorch fallback 内部逐序列遍历 |

**输出处理**：
- 两者均返回 `(output, final_state)` 元组
- 310P 写回方式：`ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(ssm_state.dtype)`（无转置）
- 910 gqa_interleaved 写回：`ssm_state[...] = last_recurrent_state.transpose(-1, -2).contiguous().to(ssm_state.dtype)`（需转置）

**310P 集成设计**：
- 目标：AscendC kernel，pipeline 化子步骤
- CHUNK_SIZE=64，WY 表示 + 下三角求解 + 跨 chunk 递推
- 需协调 `build_chunk_meta_device`（chunk 元数据预计算）的 310P 适配
- 集成入口：`gdn_310.py` 第 199 行，prefill 路径调用

### 5. rmsnormgated — NAIE 开发（低优先级）

**910 接口参考**（Triton kernel `_layer_norm_fwd_1pass_kernel_npu`）：

```python
def layer_norm_fwd_npu(
    x: Tensor,          # [M, N] 输入
    weight: Tensor,     # [N] RMSNorm 权重
    bias: Tensor | None,# [N] 偏置
    eps: float,         # 防止除零
    z: Tensor | None,   # [M, N] 门控分支
    out: Tensor | None, # [M, N] 输出
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
) -> tuple[Tensor, Tensor | None, Tensor]  # (output, mean, rstd)
```

**310P 当前 fallback**：`_310p/ops/layernorm.py` → `forward_native(x, z)`，基于 `torch_npu.npu_rms_norm` + 纯 PyTorch gating。

**310P 参数构造**：与 910 一致，无需特殊适配。输入 `x`、`z`、`weight` 均为标准形状。

**输出处理**：
- 910 返回 `(output, mean, rstd)` 三元组（mean/rstd 用于反向传播）
- 310P forward 仅返回归一化+门控后的 `output`，不需要 mean/rstd（推理场景无反向传播）

### 6. fused_gdn_gating — NAIE 开发（高优先级）

**910 接口参考**（Triton kernel `fused_gdn_gating_kernel`）：

```python
def fused_gdn_gating_patch(
    A_log: Tensor,   # [num_heads] — A 衰减参数的 log
    a: Tensor,       # [batch, num_heads] — 输入相关门控
    b: Tensor,       # [batch, num_heads] — beta 输入
    dt_bias: Tensor, # [num_heads] — dt 偏置
    beta: float = 1.0,     # softplus beta
    threshold: float = 20.0, # softplus 阈值
) -> tuple[Tensor, Tensor]:
    # g: [1, batch, num_heads] float32 — -exp(A_log) * softplus(a + dt_bias)
    # beta_output: [1, batch, num_heads] — sigmoid(b)
```

**计算公式**：`g = -exp(A_log) * softplus(a + dt_bias)`，`beta_output = sigmoid(b)`

**310P 参数构造（与 910 的差异）**：

数学计算完全一致，差异仅在执行方式：

| 方面 | 910 Triton | 310P PyTorch fallback |
|------|-----------|----------------------|
| `A_log` 展开 | kernel 内 per-element 加载 | `.unsqueeze(0).expand(batch, -1)` 预展开 |
| `dt_bias` 展开 | kernel 内 per-element 加载 | `.unsqueeze(0).expand(batch, -1)` 预展开 |
| softplus 实现 | `tl.log(1 + tl.exp(...))` | `torch.log1p(torch.exp(...))`（数值更稳定） |
| 输出分配 | `tl.store` 直接写入预分配 buffer | 返回新 Tensor |
| Grid 配置 | `(num_cores, 1)`，tiling `BLK_HEADS=8, BLK_BATCHES=64` | 无（eager PyTorch 自动并行） |

**输出处理**：
- 两者输出格式一致：`g: [1, batch, num_heads] float32`，`beta_output: [1, batch, num_heads]`
- 310P 的 `beta_output` 在 fp32 计算 `sigmoid(b)` 后转回 `b.dtype`（通常为 float16），与 910 Triton 行为一致
- 输出直接传给后续的 `fused_recurrent_gated_delta_rule` 或 `chunk_gated_delta_rule`，无需额外处理

**310P 集成设计**：
- 高优先级：每个 GDN 层每步都调用，频率极高
- AscendC 融合 kernel 将 exp + softplus + mul + sigmoid 四个操作融合为单次 kernel launch
- 接口保持与 910 版本一致，替换 `_310p/ops/fla/fused_gdn_gating.py` 中的实现
- 集成位置：`gdn_310.py` 第 153 行 `fused_gdn_gating_pytorch(self.A_log, a, b, self.dt_bias)`

### 7. mRope — NAIE 开发（高优先级）

**910 接口参考**（`AscendMRotaryEmbedding` 类）：

```python
class AscendMRotaryEmbedding(MRotaryEmbedding):
    def forward_oot(self, positions, query, key):
        # positions: [batch, num_pos_dim] — 2D 位置（多模态）
        # query: [num_tokens, num_heads, head_dim]
        # key: [num_tokens, num_kv_heads, head_dim]

        # CANN 优化路径：
        query, key = torch_npu.npu_mrope(
            positions.contiguous(),     # [batch, num_pos_dim]
            query.contiguous(),         # [num_tokens, num_heads, head_dim]
            key.contiguous(),           # [num_tokens, num_kv_heads, head_dim]
            self.cos_sin_cache.contiguous(),
            self.head_size,
            mrope_section=[t, h, w],  # 如 [16, 24, 24]
            rotary_mode="half",
        )
```

**310P 参数构造**：
- `positions`：910 为 2D `[batch, num_pos_dim]`（多模态位置），310P 需保持相同格式
- `cos_sin_cache`：需要 `.contiguous()` 保证内存连续
- `mrope_section`：三个维度分区 `[temporal, height, width]`，和与 `head_dim` 一致
- 关键差异：310P 目前无 `torch_npu.npu_mrope`，需 NAIE 提供等效 AscendC 算子

**输出处理**：
- 返回旋转编码后的 `(query, key)` 元组，形状与输入一致
- 910 `torch_npu.npu_mrope` 是 in-place 修改 query/key 并返回
- 310P 需确认输出方式（in-place 还是新 Tensor）

**310P 集成设计**：
- 集成位置：`AscendMRotaryEmbedding.forward_oot` 中根据 `is_310p()` 选择实现
- 910 fallback 路径使用 `triton_mrope`（来自上游 vLLM），310P 需独立的 fallback

### 8. split_qkv_rmsnorm_Mrope — NAIE 开发（低优先级）

**910 接口参考**（Triton kernel）：

```python
def triton_split_qkv_rmsnorm_mrope(
    qkv: Tensor,          # [num_tokens, q_size + gate_size + 2*kv_size]
    q_weight: Tensor,     # [head_size]
    k_weight: Tensor,     # [head_size]
    cos_sin: Tensor,      # [3*num_tokens, rope_dim] — T/H/W cos+sin 交错
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
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # (q_output, k_output, v_output, gate_output)
```

**310P 参数构造**：与 910 一致，输入参数均为标准 Tensor。

**输出处理**：
- 返回 4 个 Tensor：(q, k, v, gate)，其中 V 不经过 RMSNorm/RoPE 直接 pass-through
- `has_gate=False` 时 gate_output 为空 Tensor
- 310P 可拆分为独立步骤（QKV split → RMSNorm → mRope），不影响正确性

### 9. transposeKV — NAIE 开发（低优先级）

**910 接口参考**（C++ 自定义算子）：

```python
torch.ops._C_ascend.transpose_kv_cache_by_block(
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

**310P 参数构造差异**：

| 参数 | 910 KV cache 格式 | 310P KV cache 格式 |
|------|------------------|-------------------|
| `kCache` 形状 | `(num_blocks, block_size, num_kv_heads, head_size)` ND 格式 | `(num_blocks, (num_kv_heads*head_size)//16, block_size, 16)` NZ 对齐格式 |
| `vCache` 形状 | 同上 | 同上 |
| `blockSize` | 128 | 64 或 128（受 `block_size * head_size ≤ 16384` 约束） |
| `headDim` | 自然维度 | 需考虑 16 字节对齐 |

**输出处理**：in-place 修改 kCache/vCache 的内存布局，无返回值。

**310P 集成设计**：
- 310P 编译时不注册此算子（`#ifdef ASCEND_PLATFORM_310P` 条件排除），需新增 310P 条件分支
- 需调整 tiling 参数以适配 310P AICore 的 memory hierarchy 和 NZ 格式

### 10. fused_sigmoid_gating_delta_rule_310 — SAIE 开发中，NAIE 支撑

**910 接口参考**（Triton kernel `fused_sigmoid_gating_delta_rule_update_kernel`）：

```python
def fused_sigmoid_gating_delta_rule_update(
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

**310P 当前方案**：两步操作
1. `fused_gdn_gating_pytorch(A_log, a, b, dt_bias)` → `(g, beta)`
2. `fused_recurrent_gated_delta_rule_pytorch(q, k, v, g, beta, ...)` → `output`

**310P 参数构造差异**：
- 融合后不再需要中间 `g` 和 `beta` Tensor，直接在 kernel 内计算
- `initial_state_source` 和 `initial_state_indices` 的构造方式与 `fused_recurrent_gated_delta_rule` 的 `initial_state` + `ssm_state_indices` 一致
- 910 的 `cu_seqlens` 是可选的（`None` 表示非 varlen），310P 融合算子需保持相同语义

**输出处理**：
- 910 返回单 Tensor `[B, T, HV, V]`，SSM state 由 kernel 内部更新
- 310P 融合算子应返回相同格式，SSM state 同样 in-place 更新
- 集成位置：`gdn_310.py` decode 路径（第 230 行起），替换两步调用

### 11. FIA 算子增加压缩 mask — NAIE 能力中心开发（高优先级）

**910 当前实现**：
- 使用 `torch_npu.npu_fused_infer_attention_score` 的 `sparse_mode` 参数
- `sparse_mode=3`：因果 mask（硬件内部生成），无需外部 mask tensor

**310P 当前实现**：

```python
# _310p/attention/attention_mask.py — AttentionMaskBuilder310
# 1. 生成完整 causal mask [max_seq_len, max_seq_len] float16
mask = torch.zeros((max_seq_len, max_seq_len), dtype=torch.float16)
mask.masked_fill_(upper, float("-inf"))

# 2. 转换为 NZ 分形格式（310P 独有步骤）
# nd_to_nz_2d: (R, C) → (1, ceil(C/16), ceil(R/16), 16, 16)
mask_nz = torch_npu.npu_format_cast(nd_to_nz_2d(mask), ACL_FORMAT_FRACTAL_NZ)
```

**310P mask 构造（与 910 的关键差异）**：

| 方面 | 910 | 310P |
|------|-----|------|
| Mask 格式 | 标准 ND（row-major） | NZ 分形格式（16x16 块转置），需 `nd_to_nz_2d` + `npu_format_cast` |
| Mask 数据类型 | float16 用 `-inf`，bfloat16 用 `1` | 固定 float16 `-inf` |
| SplitFuse mask | 静态 2048×2048 int8 二值 mask | 动态选择行：`mask.index_select(0, position)`，position 由 `seq_lens - query_lens` 计算 |
| sparse_mode 支持 | `sparse_mode=3` 跳过 mask 生成 | 不支持，需物化完整 mask |
| 显存开销 | sparse_mode=3 时为 0 | `max_seq_len² * 2` 字节 + NZ 转换开销 |

**310P SplitFuse mask 动态行选择构造**：
```python
# 310P 独有：根据每个请求的 context_len - query_len 计算行偏移
qsl = attn_metadata.query_start_loc.to("cpu", dtype=torch.int32)
qlens = qsl[1:] - qsl[:-1]
context_lens = attn_metadata.seq_lens.to("cpu", dtype=torch.int32)
pos_list = [p for ql, cl in zip(qlens, context_lens) for p in range(cl - ql, cl)]
position = torch.tensor(pos_list, dtype=torch.int32, device=device)
splitfuse_mask = cls.chunked_prefill_attn_mask.index_select(0, position)
```

**输出处理**：
- 910 sparse_mode=3：无 mask 输出，硬件内部处理
- 310P 当前：返回 NZ 格式 mask tensor，传入 `_npu_flash_attention` 或 `_npu_paged_attention`
- 310P 优化目标：支持 sparse_mode 后跳过 mask 生成和 NZ 转换

### 12. 拒绝采样算子详细设计 — NAIE 能力中心开发

拒绝采样是投机推理 verify 阶段的核心操作，包含 8 个子算子。

#### 12a. rejection_greedy_sample — Greedy 拒绝采样

**910 接口**（两个 Triton kernel）：

```python
# 快速路径（所有请求 spec_len=1）：
rejection_greedy_sample_spec_len_1_triton(
    output_token_ids: Tensor,   # [batch_size, 2] int32，初始化为 PLACEHOLDER_TOKEN_ID
    draft_token_ids: Tensor,    # [num_tokens] == [batch_size]
    target_argmax: Tensor,      # [num_tokens]
    bonus_token_ids: Tensor,    # [batch_size]
)

# 通用路径：
rejection_greedy_sample_triton(
    output_token_ids: Tensor,   # [batch_size, max_spec_len+1]
    cu_num_draft_tokens: Tensor,# [batch_size] 累积和
    draft_token_ids: Tensor,    # [num_tokens]
    target_argmax: Tensor,      # [num_tokens]
    bonus_token_ids: Tensor,    # [batch_size]
    is_greedy: Tensor | None,   # [batch_size]
    max_spec_len: int,
)
```

**310P PyTorch fallback 构造**：

| 参数 | 构造方式 | 与 910 差异 |
|------|---------|------------|
| `output_token_ids` | `torch.full([batch_size, max_spec_len+1], PLACEHOLDER_TOKEN_ID, dtype=torch.int32)` | 一致 |
| `cu_num_draft_tokens` | 从 `num_draft_tokens` list 累积计算：`torch.tensor(np.cumsum(num_draft_tokens), dtype=torch.int32)` | 一致 |
| `target_argmax` | `torch.argmax(target_logits, dim=-1)` | 一致 |
| `bonus_token_ids` | 从 `SamplingMetadata` 的 bonus_tokens 获取，squeeze 到 `[batch_size]` | 一致 |
| `draft_token_ids` | 直接从 draft model 输出获取 `[num_tokens]` | 一致 |

**310P 实现差异**：
- 310P 使用 `torch.where(draft == target, draft, target)` 向量化比较，替代 Triton 逐元素 kernel
- 310P 通用路径构造 2D 索引矩阵 `pos_matrix [batch, max_draft]` 映射 flat token 到 (batch, position)
- 310P `bonus_renew` 逻辑用 `output_token_ids[accepted_indices] = bonus_token_ids[accepted_indices]` 替代

**输出处理**：
- 输出 `output_token_ids [batch_size, max_spec_len+1]` int32，被拒绝位置保留 `PLACEHOLDER_TOKEN_ID`
- 后续由 `rejection_sample` 主函数裁剪和格式化

#### 12b. rejection_random_sample — Random 拒绝采样

```python
rejection_random_sample_kernel(
    output_token_ids: Tensor,       # [batch_size, max_spec_len+1]
    cu_num_draft_tokens: Tensor,    # [batch_size]
    draft_token_ids: Tensor,        # [num_tokens]
    draft_probs: Tensor | None,     # [num_tokens, vocab_size]
    target_probs: Tensor,           # [num_tokens, vocab_size]
    bonus_token_ids: Tensor,        # [batch_size]
    recovered_token_ids: Tensor,    # [num_tokens] — 预计算的恢复 token
    uniform_probs: Tensor,          # [num_tokens] — 均匀随机数
    is_greedy: Tensor,              # [batch_size]
    max_spec_len: int,
    vocab_size: int,
)
```

**关键参数构造**：

| 参数 | 构造方式 | 说明 |
|------|---------|------|
| `draft_probs` | draft model 的 softmax logits `[num_tokens, vocab_size]` | ngram 方法时为 None |
| `target_probs` | `torch.softmax(target_logits, dim=-1)` | 主模型目标概率 |
| `uniform_probs` | `torch.rand(num_tokens, device=device)` | 每个位置的接受阈值 |
| `recovered_token_ids` | 由 `sample_recovered_tokens` 预计算 | 拒绝时的恢复 token |

**310P PyTorch 实现差异**：
- 接受条件：`target_prob[global_idx, draft_id] / draft_prob[global_idx, draft_id] >= uniform_prob`
- 使用 2D 索引矩阵 `global_token_indices [batch, max_draft_len]` 映射 flat token index
- `IS_NGRAM` constexpr：当 `draft_probs is None` 时跳除法，直接接受

#### 12c. rejection_random_sample_block_verify — Block Verify 拒绝采样

```python
rejection_random_sample_block_verify_kernel(
    # 与 random_sample 相同参数
    # 增加：前缀概率追踪和逐位置接受阈值 h_block
    SUB_BLOCK: int,  # = 4*1024，词汇表级分块
)
```

**Block Verify 算法（310P PyTorch 实现）**：
```python
# 1. 计算前缀概率
p_prefix[:, 0] = 1.0
for i in range(1, max_draft + 1):
    p_prefix[:, i] = p_prefix[:, i-1] * target_probs[:, i-1, draft_ids[:, i-1]] / draft_probs[:, i-1, draft_ids[:, i-1]]

# 2. 计算接受阈值 h_block
for i in range(max_draft):
    # 残差质量 = sum(max(0, target - draft) for remaining positions)
    # h_block = residual_mass * target_prob / draft_prob
    h_block = compute_acceptance_threshold(p_prefix, target_probs, draft_probs, i)

# 3. 接受条件：uniform_prob <= h_block
```

**310P 特有构造**：
- `p_prefix [batch_size, max_spec_len+1]` float32：前缀概率累积矩阵
- `h_block [batch_size, max_spec_len]` float32：每位置的接受阈值
- 激活条件：`max_spec_len >= 3` 且 `draft_probs is not None`

#### 12d. sample_recovered_tokens — 残差分布恢复采样

```python
sample_recovered_tokens_kernel(
    output_token_ids: Tensor,       # [num_tokens] 输出
    cu_num_draft_tokens: Tensor,    # [batch_size]
    draft_token_ids: Tensor,        # [num_tokens]
    draft_probs: Tensor | None,     # [num_tokens, vocab_size]
    target_probs: Tensor,           # [num_tokens, vocab_size]
    q: Tensor,                      # [batch_size, vocab_size] — 指数噪声
    vocab_size: int,
    BLOCK_VERIFY: bool,             # 是否使用 block verify 模式
)
```

**关键参数构造**：

| 参数 | 构造方式 |
|------|---------|
| `q` | `torch.zeros(batch_size, vocab_size).exponential_(generator=gen)` — 每个请求独立随机数生成器 |
| `target_probs` | `torch.softmax(target_logits, dim=-1)` |
| `draft_probs` | draft model 的 softmax 输出（可为 None） |

**310P PyTorch 实现**：
- 普通模式：`residual = max(0, target_probs - draft_probs)`，采样 `argmax(residual / q)`
- Block verify 模式：`residual = max(0, p_prefix * target_probs - draft_probs)`，采样 `argmax(residual / q)`
- 310P 使用向量化 `torch.argmax` 替代 Triton 逐元素计算

**输出处理**：
- `output_token_ids [num_tokens]`：每个 draft 位置的恢复 token ID
- 仅在拒绝发生时使用，被 `rejection_random_sample` 在拒绝位置写入

#### 12e. expand_batch_to_tokens — 批量到 token 扩展

```python
expand_kernel(
    output: Tensor,              # [num_tokens] 输出
    input: Tensor,               # [batch_size] 输入
    cu_num_tokens: Tensor,       # [batch_size] 累积 token 数
    replace_from: int,           # 替换源值
    replace_to: int,             # 替换目标值
)
```

**参数构造**：
- `cu_num_tokens`：`torch.tensor(np.cumsum(num_draft_tokens))` — 每个 batch 的 draft token 累积计数
- 用途：将 per-request 的 temperature/top_k/top_p 从 `[batch_size]` 扩展为 `[num_tokens]`

**310P PyTorch 实现**：
```python
# 使用 einsum 实现批量到 token 的广播
one_hot = torch.zeros(batch_size, num_tokens)
one_hot[cu_start, torch.arange(num_tokens)] = 1.0
output = torch.einsum("tb,b->t", one_hot, input)
```

#### 12f. npu_copy_and_expand_eagle_inputs — Eagle 输入扩展（C++ 算子）

```python
torch.ops._C_ascend.npu_copy_and_expand_eagle_inputs(
    target_token_ids: Tensor,      # [total_input_tokens] int32
    target_positions: Tensor,      # [total_input_tokens] int32
    next_token_ids: Tensor,        # [batch_size] int32
    query_start_loc: Tensor,       # [batch_size+1] int32
    query_end_loc: Tensor,         # [batch_size] int32
    padding_token_id: int,         # 0
    parallel_drafting_token_id: int,
    num_padding_slots_per_request: int,
    shift_input_ids: bool,         # EAGLE vs parallel drafting 模式
    total_draft_tokens: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
    # (out_input_ids, out_positions, out_is_rejected_token_mask,
    #  out_is_masked_token_mask, out_new_token_indices, out_hidden_state_mapping)
```

**参数构造**：
- `target_token_ids`：verify 后的目标 token IDs（包含拒绝后的 padding）
- `next_token_ids`：每个请求最后一个有效 token（bonus 或 recovered），由 `prepare_next_token_ids_padded` 计算
- `query_start_loc` / `query_end_loc`：从 `attn_metadata` 获取
- `num_padding_slots_per_request`：`self.extra_slots_per_request`（EAGLE proposer 配置）

**310P 关键问题**：
- 此算子当前仅 910 注册（`csrc/torch_binding.cpp` 中 `#ifndef ASCEND_PLATFORM_310P` 条件）
- 310P 需要：(a) 移植 AscendC kernel 到 310P AICore，或 (b) 用纯 PyTorch 实现等效逻辑
- 集成位置：`eagle_proposer.py` 第 1099 行

**输出处理**：
- 6 个输出 Tensor 用于下一轮 draft 推理的输入准备
- `out_is_rejected_token_mask` 和 `out_is_masked_token_mask` 用于标记哪些 token 需要忽略
- `out_hidden_state_mapping` 用于 hidden state 到 draft 模型的映射

#### 12g. prepare_inputs_padded — Spec Decode 输入准备

```python
prepare_inputs_padded_kernel(
    cu_num_draft_tokens: Tensor,            # [num_reqs]
    valid_sampled_tokens_count: Tensor,     # [num_reqs]
    query_start_loc: Tensor,                # [num_reqs+1]
    # 输出：
    token_indices_to_sample: Tensor,        # [num_reqs]
    num_rejected_tokens: Tensor,            # [num_reqs]
)
```

**参数构造**：
- `valid_sampled_tokens_count`：由 `prepare_next_token_ids_padded` 计算的每个请求有效采样 token 数
- `cu_num_draft_tokens`：累积 draft token 计数
- `query_start_loc`：从 scheduler 获取的查询起始位置

**310P PyTorch 实现**：
```python
# 纯 PyTorch 计算，无需 Triton
num_rejected = cu_draft[1:] - cu_draft[:-1] + 1 - valid_count
token_indices = query_start_loc[1:] - 1 - num_rejected
```

**310P 集成**：
- 已有 PyTorch fallback 路径（`eagle_proposer.py` 第 1565 行 `if not HAS_TRITON` 分支）
- 计算量极小，不需要 AscendC 优化

#### 12h. apply_sampling_constraints — 采样约束应用

```python
def apply_sampling_constraints(
    logits: Tensor,              # [num_tokens, vocab_size]
    cu_num_draft_tokens: Tensor, # [batch_size]
    sampling_metadata: SamplingMetadata,
) -> Tensor                       # [num_tokens, vocab_size]
```

**参数构造**：
- 从 `sampling_metadata` 提取 per-request 的 temperature/top_k/top_p
- 使用 `expand_batch_to_tokens` 将 `[batch_size]` 扩展为 `[num_tokens]`
- 310P 在 logits 上 in-place 应用：`logits /= temperature`，`logits[topk_mask] = -inf`

**310P 集成**：
- 已有 PyTorch 实现路径，通过 `patch_rejection_sampler.py` monkey-patch 到上游 vLLM
- 310P 分支在 `rejection_sampler.py` 中通过 `HAS_TRITON` 标志自动选择

### 13. 算子集成架构

**决策**：310P 算子采用统一注册 + 运行时分发模式。

```
算子调用链：
  patch_qwen3_6.py (模型 patch)
    ├── is_310p() → AscendGatedDeltaNetAttention310
    │                 ├── npu_causal_conv1d_310 (AscendC)
    │                 ├── fused_gdn_gating_310 (NAIE AscendC / PyTorch fallback)
    │                 ├── chunk_gated_delta_rule_310 (SAIE AscendC / PyTorch fallback)
    │                 └── fused_recurrent_gated_delta_rule_310 (AscendC)
    ├── AscendMRotaryEmbedding → npu_mrope_310 (NAIE AscendC)
    ├── split_qkv_rmsnorm_mrope → PyTorch / AscendC
    └── AscendAttentionBackend310 → FIA compressed mask

  投机推理调用链：
    AscendEagleProposer._propose()
      ├── npu_copy_and_expand_eagle_inputs (C++ / PyTorch)
      ├── rejection_sample → HAS_TRITON 分支
      │     ├── rejection_greedy_sample_pytorch (310P)
      │     ├── rejection_random_sample_pytorch (310P)
      │     ├── rejection_random_sample_block_verify_pytorch (310P)
      │     └── sample_recovered_tokens_pytorch (310P)
      ├── expand_batch_to_tokens → expand_pytorch (310P)
      └── apply_sampling_constraints → PyTorch in-place (310P)
```

**分发策略**：
- 每个 310P 算子提供两层实现：AscendC kernel（优先）+ PyTorch fallback（兜底）
- 通过 `is_310p()` + 环境变量（如 `VLLM_ASCEND_USE_ASCENDC_OPS`）控制是否启用 AscendC 版本
- 新算子注册到 `torch.ops._C_ascend` 命名空间（C++ 算子）或 `torch.ops.vllm`（Python 算子）
- 310P 编译时条件（`#ifdef ASCEND_PLATFORM_310P`）控制 C++ 算子注册范围
- 拒绝采样算子通过 `HAS_TRITON`（`importlib.util.find_spec("triton")`）自动选择 Triton 或 PyTorch 路径

**公共参数构造模式**（多个算子共享）：
- `cu_num_draft_tokens`：`torch.tensor(np.cumsum(num_draft_tokens), dtype=torch.int32)` — 从 Python list 累积求和
- `query_start_loc`：从 `attn_metadata` 或 `CommonAttentionMetadata` 直接获取
- `ssm_state_indices`：从 `GDNAttentionMetadata` 获取，310P 保持 2D，910 NPU op 需要 `.flatten()`
- `to_int64_tuple`：310P 专用，将设备 Tensor 转为 host 端 Python tuple 给 C++ 算子

## Risks / Trade-offs

- **[SAIE 算子交付节奏]** → `chunk_gated_delta_rule_fwd` 和 `fused_sigmoid_gating_delta_rule_310` 由 SAIE 开发，交付时间直接影响 310P 性能达标。缓解：PyTorch fallback 保底正确性，AscendC 版本替换后性能提升。

- **[NAIE 高优先级算子开发周期]** → `fused_gdn_gating` 和 `mRope` 为高优先级但依赖 NAIE 团队排期。缓解：明确接口规范，提前对齐参数格式，减少联调时间。

- **[310P AICore 架构差异]** → 310P 与 910 的 AICore 架构不同，AscendC kernel 移植需重新调整 tiling 参数和内存访问模式。缓解：参考已有 `causal_conv1d_v310` 和 `recurrent_gated_delta_rule_v310` 的 310P 适配经验。

- **[FIA compressed mask 兼容性]** → 310P 的 `torch_npu._npu_flash_attention` 可能不支持 `sparse_mode` 扩展。缓解：需与 NAIE 能力中心确认 API 扩展可行性，如不可行则维持当前全 mask 方案。

- **[拒绝采样算子需求]** → 现有设计认为 PyTorch 实现已足够（< 0.1ms），NAIE 能力中心的开发可能是为了统一代码路径。缓解：明确 NAIE 开发动机，避免重复投入。

- **[npu_copy_and_expand_eagle_inputs 310P 移植]** → 此算子当前仅 910 注册，310P 投机推理需要等效实现。缓解：可先实现 PyTorch 版本保底正确性，AscendC 版本后续移植。

- **[性能约束风险]** → 新增算子整网耗时占比需 ≤ 910 的 30%。缓解：逐算子 benchmark，优先优化占比最高的算子（chunk_gated_delta_rule、fused_gdn_gating）。

## Open Questions

1. `chunk_gated_delta_rule_fwd` 的 AscendC 实现预计交付时间？
2. NAIE 高优先级算子（`fused_gdn_gating`、`mRope`）的排期和 API 草案？
3. 310P `torch_npu._npu_flash_attention` 是否可扩展 `sparse_mode` 参数？如可以，NAIE 能力中心的交付时间？
4. 拒绝采样算子由 NAIE 能力中心开发的动机？是否需要 AscendC 实现，还是维持 PyTorch fallback？
5. `fused_sigmoid_gating_delta_rule_310` 是否仅用于 decode 路径？是否也需要覆盖投机推理的多 token 场景？
6. `npu_copy_and_expand_eagle_inputs` 是否计划移植到 310P？还是用 PyTorch 替代？
