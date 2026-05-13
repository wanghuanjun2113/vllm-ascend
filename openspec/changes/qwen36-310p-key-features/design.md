## Context

Qwen3.6 27B 使用 Full Attention (16 层) + Linear Attention/GDN (48 层) 混合注意力架构。910 上 Chunk Prefill、Prefix Caching、MTP 投机推理已实现；310P (300IDuo) 当前仅有 eager 基础推理能力。

**310P 关键约束**：
- KV cache 布局为 5D NZ 对齐：`(2, num_blocks, (num_kv_heads*head_size)//16, block_size, 16)`
- block_size 受限：`block_size * head_size ≤ 128*128`（head_size=128 时 block_size 上限为 128）
- 不支持 Triton，所有 Triton kernel 需 PyTorch fallback 或 AscendC 替代
- 当前仅 eager 模式，不支持 ACLGraph
- Attention 后端 `AscendAttentionBackend310` 不支持 SpecDecoding

**共用框架总览**（910/310P 共享，简要描述）：
- **Scheduler**：vLLM V1 scheduler 的 `enable_chunked_prefill`、`max_num_batched_tokens` 配置对所有硬件通用
- **Block Allocator**：vLLM V1 block-level prefix caching 通过 block hash 匹配实现，硬件无关
- **MTP Proposer**：`AscendEagleProposer(method="mtp")` 的 draft loop、verify 编排、rejection sampling 框架跨硬件复用
- **Model Patch**：`patch_qwen3_6.py` 中 `is_310p()` 条件选择 `AscendGatedDeltaNetAttention310` vs `AscendGatedDeltaNetAttention`

## Goals / Non-Goals

**Goals:**
- 310P 上启用 Chunk Prefill，控制长序列 prefill 峰值时延
- 310P 上启用混合注意力 Prefix Caching（Full Attn block-level + GDN SSM State Checkpoint）
- 310P 上启用 MTP 投机推理，减少 decode 阶段主模型推进次数
- 明确每个特性中 910/310P 可共用部分和 310P 独有适配工作

**Non-Goals:**
- 不涉及 310P 图模式设计（独立变更覆盖）
- 不涉及算子 AscendC 开发（`qwen36-27b-310p-operators` 变更覆盖）
- 不涉及量化精度对齐
- 不涉及 BF16 支持

## Decisions

### 1. Chunk Prefill — 310P 适配

#### 1.1 共用框架（简要）

Chunk Prefill 的调度逻辑完全共用：
- vLLM V1 scheduler 根据 `enable_chunked_prefill=True` 将长 prefill 请求拆分为多个 chunk
- `max_num_batched_tokens` 控制 chunk 大小上限
- `NPUModelRunner._prepare_inputs()` 和 `_prepare_attn_metadata()` 对 910/310P 使用相同的调度输出处理逻辑
- 混合 batch（prefill chunk + decode token）的调度和 metadata 构建是硬件无关的

#### 1.2 Full Attention Splitfuse — 310P 独有适配

**910 实现**：
```python
# attention_v1.py — 使用 npu_fused_infer_attention_score 的 splitfuse 模式
torch_npu.npu_fused_infer_attention_score(
    query, key, value,
    attn_mask=mask,
    sparse_mode=3,  # causal mask 压缩
    ...
)
```

**310P 实现**：
```python
# _310p/attention/attention_v1.py — 使用 _npu_paged_attention_splitfuse
torch_npu._npu_paged_attention_splitfuse(
    query=query,
    key_cache=self.key_cache,     # NZ 格式: (num_blocks, (kv_heads*head_dim)//16, block_size, 16)
    value_cache=self.value_cache, # 同上
    mask=mask,                    # NZ 格式 mask（关键差异）
    block_table=block_table,
    seq_len=qlens,
    context_lens=context_lens,
    ...
)
```

**310P 关键差异**：

| 方面 | 910 | 310P |
|------|-----|------|
| FIA 算子 | `npu_fused_infer_attention_score` | `_npu_paged_attention_splitfuse` |
| KV cache 输入 | reshape 后的独立 K/V tensor | 直接传入 NZ 格式的 key_cache/value_cache |
| Mask 格式 | `sparse_mode=3`，无需外部 mask | 完整 causal mask + NZ 分形转换 |
| Mask 构造 | 不需要 | 动态行选择 + `nd_to_nz_spec` + `npu_format_cast` |

**310P SplitFuse mask 构造**（独有，需重点适配）：

```python
# AttentionMaskBuilder310.get_splitfuse_mask()
# 1. 从预生成的全局 causal mask 中按位置选择行
qsl = attn_metadata.query_start_loc.to("cpu", dtype=torch.int32)
qlens = qsl[1:] - qsl[:-1]
context_lens = attn_metadata.seq_lens.to("cpu", dtype=torch.int32)
pos_list = [p for ql, cl in zip(qlens, context_lens) for p in range(cl - ql, cl)]
position = torch.tensor(pos_list, dtype=torch.int32, device=device)
splitfuse_mask = causal_mask.index_select(0, position)

# 2. 转换为 NZ 分形格式
splitfuse_mask_nz = torch_npu.npu_format_cast(
    nd_to_nz_spec(splitfuse_mask).contiguous(),
    ACL_FORMAT_FRACTAL_NZ  # format code 29
)
```

**适配要点**：
- SplitFuse mask 构造中 `pos_list` 的计算涉及 CPU-NPU 数据同步（`query_start_loc.to("cpu")`），在长序列高并发场景下可能成为瓶颈
- `nd_to_nz_spec` 将 `[num_tokens, max_seq_len]` 转换为 `[1, max_seq_len_pad//16, num_tokens_pad, 16]` 的 NZ 格式，需确保 padding 对齐正确
- 当前 310P `get_supported_kernel_block_sizes()` 返回 `[128, 64]`，SplitFuse 路径需使用兼容的 block_size

#### 1.3 GDN 层 Chunk Prefill — 310P 独有适配

**910 实现**：使用 Triton `chunk_gated_delta_rule` kernel 链（`chunk_local_cumsum` → `chunk_scaled_dot_kkt_fwd` → `solve_tril` → `recompute_w_u_fwd` → `chunk_fwd_h` → `chunk_fwd_o`），支持 `prebuilt_meta` 预计算 chunk 元数据加速。

**310P 实现**：
```python
# gdn_310.py — chunk_gated_delta_rule_pytorch
# 纯 PyTorch 实现，逐序列处理
core_attn_out_non_spec, last_recurrent_state = chunk_gated_delta_rule_pytorch(
    q=query_non_spec,
    k=key_non_spec,
    v=value_non_spec,
    g=g_non_spec,
    beta=beta_non_spec,
    initial_state=initial_state,  # 从 ssm_state 索引，无转置
    output_final_state=True,
    cu_seqlens=non_spec_query_start_loc,
    head_first=False,
    use_qk_l2norm_in_kernel=True,
    # 注意：无 prebuilt_meta 参数
)
```

**310P 关键差异**：

| 方面 | 910 | 310P |
|------|-----|------|
| 算法实现 | Triton kernel 链，6 步 pipeline | 纯 PyTorch 逐序列循环，CHUNK_SIZE=64 |
| initial_state 构造 | `ssm_state[indices].transpose(-1, -2).contiguous()` + `clear_ssm_states()` | `ssm_state[indices].contiguous()` + `initial_state[~has_initial_state] = 0`（无转置，布尔索引清零） |
| prebuilt_meta | 预计算 chunk 元数据加速 | 不使用（PyTorch fallback 内部手动遍历） |
| state 写回 | gqa_interleaved: `.transpose(-1, -2).contiguous()` | 无转置：`ssm_state[indices] = last_recurrent_state.to(dtype)` |
| 性能 | 高（并行化 Triton） | 低（PyTorch fallback），依赖 AscendC 升级 |

**Chunk Prefill 与 GDN 层的交互**：
- Chunk Prefill 调度器将长序列拆分为多个 chunk，每个 chunk 包含一部分 prefill tokens
- GDN 层的 `chunk_gated_delta_rule_pytorch` 接收的是一个 chunk 的 tokens，不是完整序列
- **关键问题**：GDN 层的 SSM state 是递推累积的，每个 chunk 需要前一个 chunk 的最终 state 作为 initial_state
- 当前实现中，`cu_seqlens` 参数控制序列边界，确保跨 chunk 的 SSM state 正确传递
- 310P 的 `fused_recurrent_gated_delta_rule_pytorch` 已支持 `inplace_final_state=True`，chunk 间 state 传递正确

### 2. Prefix Caching — 310P 适配

#### 2.1 共用框架（简要）

Full Attention 层的 block-level prefix caching 完全共用：
- vLLM V1 `BlockSpaceManager` 通过 block hash 匹配实现 prefix block 共享
- `KVCacheConfig` 和 block allocator 逻辑硬件无关
- `enable_prefix_caching=True` 配置对所有硬件通用
- 310P 的 `AscendAttentionBackend310.get_kv_cache_shape()` 返回 NZ 格式形状，但 block allocator 不关心内部格式

#### 2.2 GDN SSM State Checkpoint — 310P 独有实现（重点）

Full Attention 的 block-level prefix caching 对 GDN 层无效——GDN 使用 per-request 的 SSM state（递推状态矩阵 `h`），而非 block-level 可共享的 KV cache。需要新增 SSM State Checkpoint 机制。

**910 设计**（已有详细设计，310P 复用架构）：

```
SSM State Checkpoint 机制:
┌─────────────────────────────────────────────────────────────────┐
│ 每个 Full Attn block 边界 (128 tokens) 处:                        │
│   1. Snapshot 所有 GDN 层的 SSM state (h 矩阵)                    │
│   2. 以 hash(prefix_tokens) 为 key 存入 SSMStatePool              │
│                                                                  │
│ 新请求到达:                                                       │
│   1. Full Attn: get_computed_blocks() → 命中 N 个 blocks          │
│   2. GDN: hash(prefix) → 查 SSMStatePool → 恢复 h 矩阵            │
│   3. 从 checkpoint token 位置开始增量计算后续 tokens               │
└─────────────────────────────────────────────────────────────────┘
```

**数据结构**（910/310P 共用）：
```python
class SSMCheckpoint:
    layer_states: dict[str, Tensor]  # layer_name → h 矩阵
    ref_count: int
    num_tokens: int
    hash_key: int  # 与 Full Attn prefix hash 统一

class SSMStatePool:
    checkpoints: dict[int, SSMCheckpoint]  # hash → checkpoint
    max_checkpoints: int                    # LRU 淘汰上限
    per_checkpoint_size: int                # 单个 checkpoint 字节数
```

**310P 独有适配**：

1. **SSM State 精度**：
   - 310P `ssm_state` 为 float32（与 910 Qwen3.5 路径一致）
   - Checkpoint 存储 h 矩阵时保持 float32，无需额外类型转换
   - Qwen3Next gqa_interleaved 路径的 ssm_state 为 float16，需要确认 310P Qwen3.6 的实际精度

2. **Block_size 对齐**：
   - 310P block_size 受 `block_size * head_size ≤ 128*128` 约束
   - head_size=128 时 block_size=128，SSM checkpoint 间隔与 Full Attn block 边界自然对齐（每 128 tokens）
   - 如 block_size 被迫缩小到 64，checkpoint 间隔需独立配置（仍建议 128 tokens）

3. **SSMStatePool 显存预算**：
   - 310P 显存 96G（300IDuo），模型权重 + 运行时约 40-50G，可用于 KV cache 约 40-50G
   - 每个 checkpoint 包含 48 层 GDN 的 h 矩阵（`[num_heads, head_dim, head_dim]` float32）
   - 需配置 `max_checkpoints` 上限，LRU 淘汰避免显存溢出

4. **与 KVCacheManager 集成**：
   - 扩展 `get_computed_blocks()` 返回值，同时携带 matched SSM checkpoint
   - 310P 的 `model_runner_310p.py` 中 `execute_model` 需在 GDN 层 forward 前恢复 `initial_state`

5. **分池管理**：
   - 910 设计中提到 Full Attn KV Cache 和 GDN SSM State 分池管理消除碎片
   - 310P 的 NZ 格式 KV cache 与 SSM state 的内存布局差异更大，分池管理收益更高
   - 但 310P `expandable_segments` 的兼容性需确认

#### 2.3 Prefix Cache Hash 对齐 — 310P 注意事项

- Full Attn prefix hash 以 block (128 tokens) 为粒度计算
- GDN SSM checkpoint 需与 Full Attn block 边界对齐（每 128 tokens snapshot 一次）
- 310P block_size 可能小于 128（64），但 SSM checkpoint 间隔应保持 128 tokens（与 Full Attn hash 粒度一致）
- 如 block_size=64，Full Attn hash 每 64 tokens 一次，SSM checkpoint 仍每 128 tokens 一次，部分 hash 命中可能没有对应的 SSM checkpoint

### 3. MTP 投机推理 — 310P 适配

#### 3.1 共用框架（简要）

MTP 推理的框架层完全共用：
- `AscendEagleProposer(method="mtp")` 处理 draft loop、multi-step draft generation、verify 编排
- `AscendRejectionSampler` 处理 draft-target 比较、accept/reject、bonus token
- `patch_qwen3_6_mtp.py` 处理 MTP head 权重加载和 KV 绑定（跨硬件通用）
- Rejection Sampling 的 PyTorch fallback 路径已通过 `HAS_TRITON` 标志自动在 310P 上启用

GDN 层 MTP multi-query 路径已在 310P 实现：
```python
# gdn_310.py — spec_sequence_masks 分支
# 1. 分离 spec 和 non-spec tokens
mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)

# 2. Spec 部分：使用 fused_recurrent_gated_delta_rule_pytorch 的 multi-query 路径
core_attn_out_spec, _ = fused_recurrent_gated_delta_rule_pytorch(
    q=query_spec, k=key_spec, v=value_spec,
    g=g_spec, beta=beta_spec,
    initial_state=ssm_state,
    inplace_final_state=True,
    cu_seqlens=spec_query_start_loc[:num_spec_decodes+1],
    ssm_state_indices=spec_state_indices_tensor,
    num_accepted_tokens=num_accepted_tokens,  # 投机推理接受数
)

# 3. 合并输出
merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
```

#### 3.2 Attention 后端 SpecDecoding 支持 — 310P 独有适配（关键缺口）

**当前状态**：`AscendAttentionBackend310` 对 SpecDecoding 抛出 `NotImplementedError`。

**需要新增的支持**：
```python
# _310p/attention/attention_v1.py — AscendAttentionBackend310
# 当前缺失：
#   - SpecDecodeMetadata 的处理
#   - Draft token 的 slot mapping 更新
#   - Mixed batch (draft + verify) 的 attention 计算

# 需要新增的 key 方法/逻辑：
class AscendAttentionBackend310:
    def forward(self, ...):
        # 1. 检查是否有 spec_decode_metadata
        # 2. Draft 路径：为 draft tokens 计算 attention（含 slot mapping 更新）
        # 3. Verify 路径：对 draft + target tokens 批量计算 attention
        # 4. 区分 prefill/decode/spec 路径
```

**310P Spec Decode Attention 路径**：
- Draft 阶段：每个 draft token 走标准 decode 路径（`_npu_paged_attention`），更新 KV cache
- Verify 阶段：draft tokens + target token 批量计算 attention（类似 splitfuse，但有 draft-specific 的 block_table 和 slot mapping）
- 310P 不支持 ACLGraph，verify 阶段全部走 eager

#### 3.3 `npu_copy_and_expand_eagle_inputs` — 310P 独有适配

**当前状态**：C++ AscendC 算子，仅在 910 条件编译下注册。

```python
# csrc/torch_binding.cpp
# 当前: #ifndef ASCEND_PLATFORM_310P 才注册
# 310P 需要等效实现
torch.ops._C_ascend.npu_copy_and_expand_eagle_inputs(
    target_token_ids, target_positions, next_token_ids,
    query_start_loc, query_end_loc,
    padding_token_id, parallel_drafting_token_id,
    num_padding_slots_per_request, shift_input_ids, total_draft_tokens,
) -> (out_input_ids, out_positions, out_is_rejected_token_mask,
     out_is_masked_token_mask, out_new_token_indices, out_hidden_state_mapping)
```

**310P 适配方案**：

| 方案 | 优点 | 缺点 |
|------|------|------|
| A: 移植 AscendC kernel 到 310P | 性能最优，与 910 统一代码路径 | 需 NAIE/SAIE 开发，交付周期长 |
| B: 纯 PyTorch 实现 | 无需 AscendC 开发，可快速验证正确性 | 性能稍差（但计算量极小，预计 < 0.5ms） |

**推荐**：先用方案 B（PyTorch 实现）保底正确性，再由 NAIE/SAIE 移植 AscendC kernel。

PyTorch 实现核心逻辑：
```python
def copy_and_expand_eagle_inputs_pytorch(
    target_token_ids, target_positions, next_token_ids,
    query_start_loc, query_end_loc, ...
):
    # 对每个请求：
    #   1. 复制有效 input tokens + 插入 next_token_id
    #   2. 追加 num_padding_slots_per_request - 1 个 parallel_drafting_token_id
    #   3. 标记 rejected/masked token
    #   4. 构建 hidden_state_mapping
    # 逐请求循环，纯 tensor 操作
```

#### 3.4 MTP + GDN State 一致性 — 310P 关键设计

MTP 投机推理的 verify 阶段会拒绝部分 draft tokens，导致 GDN SSM state 出现不一致：
- Draft 阶段：k 个 draft tokens 递推更新了 SSM state
- Verify 后：只有前 n 个 tokens 被接受（n ≤ k），SSM state 已被 k 个 tokens 更新

**310P 处理方式**（与 910 一致）：
- `fused_recurrent_gated_delta_rule_pytorch` 的 `num_accepted_tokens` 参数限制实际更新的 token 数
- Spec 路径通过 `spec_query_start_loc` 和 `num_accepted_tokens` 只处理被接受的 tokens
- Non-spec 路径不受影响（无 draft tokens）
- GDN SSM state 通过 `inplace_final_state=True` 只在被接受的 tokens 上更新

#### 3.5 310P MTP 无 ACLGraph 的影响

910 上 MTP draft 和 verify 路径可通过 ACLGraph 加速（`ACLGraphWrapper(runtime_mode=FULL)`）。310P 不支持 ACLGraph：

| 方面 | 910 (ACLGraph) | 310P (eager) |
|------|----------------|--------------|
| Draft 阶段 | Graph capture + replay，单次 kernel launch | 逐层 eager forward，多次 kernel launch |
| Verify 阶段 | Graph capture + replay | 逐层 eager forward |
| SSM state 管理 | Graph 外部 buffer，replay 时更新指针 | 直接 in-place 更新 |
| 性能 | 高（消除 launch overhead） | 依赖算子性能，launch overhead 较大 |

**310P MTP 性能预估**：
- MTP 的收益来自减少主模型 forward 次数（k draft tokens = 1 次 verify vs k 次独立 decode）
- 即使 eager 路径有额外 launch overhead，只要 acceptance rate > 某阈值，MTP 仍有净收益
- 需实际测量 310P 上 MTP 的 acceptance rate 和端到端 TPOT 改善

### 4. 特性组合与使能顺序

**决策**：310P 三个特性逐项启用、验证、再组合。

```
Phase 1: Chunk Prefill（基础）
  ├── 验证 Full Attn splitfuse 正确性
  ├── 验证 GDN chunk prefill 正确性
  └── 性能 baseline: TTFT 改善

Phase 2: Prefix Caching（在 Chunk Prefill 基础上）
  ├── 实现 SSMStatePool
  ├── 验证 Full Attn prefix cache 命中
  ├── 验证 GDN SSM checkpoint restore 正确性
  └── 性能: TTFT 改善（有 prefix 命中时）

Phase 3: MTP（独立于 Prefix Caching）
  ├── 新增 Attention 后端 SpecDecoding 支持
  ├── 实现 npu_copy_and_expand_eagle_inputs PyTorch fallback
  ├── 验证 MTP draft + verify + rejection sampling 端到端
  └── 性能: TPOT 改善

Phase 4: 组合验收
  ├── Chunk Prefill + Prefix Caching
  ├── Chunk Prefill + MTP
  └── Chunk Prefill + Prefix Caching + MTP
```

## Risks / Trade-offs

- **[Attention 后端 SpecDecoding 改动量]** → `AscendAttentionBackend310` 当前不支持 SpecDecoding，新增支持需要修改 slot mapping、block_table 和 attention 计算路径。缓解：参考 910 `AscendAttentionBackendImpl` 的 SpecDecoding 处理逻辑。

- **[SSMStatePool 显存管理]** → 每个 checkpoint 包含 48 层 GDN 的 float32 h 矩阵，可能占用数十 MB。缓解：LRU 淘汰 + `max_checkpoints` 上限 + 分池管理。

- **[Chunk Prefill + GDN state 一致性]** → 跨 chunk 的 SSM state 传递需要正确处理，特别是在 Prefix Caching 命中后恢复 state 的场景。缓解：逐 chunk 正确性测试 + 端到端输出对比。

- **[MTP acceptance rate 在 310P 上可能较低]** → 310P eager 路径的计算精度可能与 910 不同（PyTorch fallback vs Triton），影响 draft token 质量。缓解：实测 acceptance rate，低于阈值时考虑降低 draft token 数。

- **[SplitFuse mask CPU-NPU 同步]** → mask 构造中 `query_start_loc.to("cpu")` 涉及设备到主机同步，可能成为长序列高并发场景的瓶颈。缓解：如 FIA 算子支持 `sparse_mode`（`qwen36-27b-310p-operators` 变更中规划），可消除 mask 构造。

- **[特性组合正确性]** → Chunk Prefill + Prefix Caching + MTP 三特性组合可能引入交互问题。缓解：严格按 Phase 逐项启用，每步验证正确性后再叠加。

## Open Questions

1. 310P `AscendAttentionBackend310` 新增 SpecDecoding 支持的改动范围？是否需要修改 `_npu_paged_attention` 的调用参数？
2. SSMStatePool 的 `max_checkpoints` 和显存预算如何确定？需要根据 310P 实际可用显存计算。
3. `npu_copy_and_expand_eagle_inputs` 是否计划由 NAIE/SAIE 移植到 310P？还是长期使用 PyTorch fallback？
4. 310P 上 MTP 的预期 acceptance rate 和 draft token 数建议值？
5. `expandable_segments` 在 310P hybrid 模型（Full Attn + GDN）上的兼容性如何？分池管理是否影响 `expandable_segments` 的效果？
