## Context

Qwen3.6 27B 使用混合注意力架构：16 层 Full Attention + 48 层 Linear Attention (GDN)。两种注意力层共享同一个 `kv_cache_tensor` 内存池。

### 当前 Pool 设计原理

**核心约束**：共享同一 `kv_cache_tensor` 的所有层必须拥有统一的 `page_size`（即每个 block slot 占用的字节数）。这是因为 `kv_cache_tensor.size = page_size * num_blocks`，`num_blocks = raw_tensor.numel() // page_size`。

**当前实现**（`patch_mamba_config.py`）通过三步对齐实现统一 page_size：

1. **膨胀 attention block_size**：强制 `attn_block_size` 使得 `attn_single_token_k_page_size * attn_block_size == ssm_block_page_size`。
   - Qwen3.6 27B TP=4 下：`512 bytes/token * attn_block_size = 786,432 bytes` → `attn_block_size = 1536`
   - 自然值应为 128（kernel 支持的 block_size），被膨胀到 1536（12 倍）
   - 膨胀后的 attn_page_size = `1536 * 2 * 256 * 1 * 2 = 1,572,864 bytes`

2. **添加 conv padding**：`mamba_page_size_padded = attn_page_size + conv_block_page_size`
   - conv_block_page_size = 27,648 bytes（GDN conv_state 大小）
   - 最终 padded page_size = `1,572,864 + 27,648 = 1,600,512 bytes`

3. **强制 AttentionSpec 的 page_size_padded 等于 MambaSpec 的**（`model_runner_v1.py:3460-3463`）

**内存布局**（每个 block slot 内）：

```
Offset 0:       conv_state (27 KiB)     ← Mamba 视图：conv_state
Offset 27 KiB:  k_cache (768 KiB)       ← Attn 视图：K cache；与 Mamba 视图的 ssm_state 重叠
Offset 795 KiB: v_cache (768 KiB)       ← Attn 视图：V cache
Offset 1563 KiB: (page 结束 = 1,600,512 bytes)
```

K cache 和 SSM state 物理上共享同一块 768 KiB 区域。代码在 `_reshape_kv_cache_tensors` 中通过 `conv_block_padding_size` 跳过前 27 KiB 的 conv_state 来定位 K cache。

### 显存浪费来源

**浪费 1：conv_block_page_size padding**（每 block 27 KiB）

`mamba_page_size_padded` 比 `attn_page_size` 多出 `conv_block_page_size`。这 27 KiB/block 用于存放 conv_state，但从 Attention 视角看是"padding"。对于 TP=4 四卡 69.92 GiB 总 cache 预算，num_blocks ≈ 11,708，总 padding ≈ 11,708 × 27 KiB ≈ 309 MiB。

**浪费 2：block_size 膨胀导致的碎片化**（主要浪费）

虽然通过 block splitting（1536 → 12 × 128）将物理 block 大小保持在 128，但逻辑 block_size=1536 意味着 KVCacheManager 的分配粒度为 1536 tokens/block。实际影响：
- SSM state 按 block 分配，每个 SSM state slot 对应一个 1536-token block，但 GDN 实际只需要每请求一个固定大小的 SSM state（不随 token 数增长）
- block_size 膨胀使得每个 block 同时预留了 KV cache 空间和 SSM state 空间，当某请求只需要 KV cache 或只需要 SSM state 时，另一半空间被浪费

**浪费 3：统一 page_size 限制灵活性**

KV cache 的自然 page_size（128 tokens × 2 × 256 × 1 × 2 bytes = 128 KiB）和 SSM state 的自然 page_size（786,432 + 27,648 = 795 KiB）差异很大。强制统一意味着每个 block slot 都按 1,563 KiB 分配，而实际 KV cache 只需要 128 KiB/page，利用率仅 8.2%。

如果改为非连续访问方式，KV cache 和 SSM state 可以各自使用自然 page_size，同一 pool 内不再需要 page 对齐。

## Goals / Non-Goals

**Goals:**
- 保持 Full Attention KV Cache 和 GDN SSM State 共享同一 `kv_cache_tensor` 不变
- 消除 `conv_block_padding_size` 和 `mamba_padding`
- 恢复 attention block_size 为 kernel 原生支持的 128 tokens
- 通过算子改动支持同一 pool 内的非连续访问，不再依赖 page 对齐
- 量化收益并给出典型部署场景下的回收 token 数

**Non-Goals:**
- 不做分池管理（Full Attention 和 GDN 保持共享同一 pool）
- 不修改上游 vLLM 的 KVCacheManager 或 block allocator 核心逻辑
- 不改变 GDN SSM state 的 slot-based 访问模式
- 不影响非混合注意力模型（纯 Full Attention 或纯 Linear Attention）

## Decisions

### 1. 方案选择：同一 Pool 内非连续布局

**决策**：保持共享 `kv_cache_tensor`，但将其划分为独立的连续子区域（KV cache 区域、SSM state 区域、conv_state 区域），各子区域使用各自的自然 page_size。算子通过 offset 计算访问非连续的 block。

**理由**：
- 分池管理改动量大（需修改 KVCacheConfig、block allocator、KV transfer 等），风险高
- 当前算子（CANN attention ops、reshape_and_cache、transpose_kv_cache_by_block）已通过 block_table/slot_mapping 支持非连续访问
- 只需修改 offset 计算逻辑和分配/reshape 路径

**备选方案**：
- 分池管理：改动大，风险高，用户已否决
- 保持 page 对齐但减小 padding：收益有限，不能解决 block_size 膨胀问题

### 2. 新的内存布局

**决策**：同一 `kv_cache_tensor` 内划分为三个连续子区域：

```
kv_cache_tensor 布局:
┌─────────────────────────┬────────────────────────┬───────────────────┐
│  KV Cache Region        │  SSM State Region      │  Conv State Region│
│  (num_kv_blocks slots)  │  (num_ssm_slots slots)  │  (num_ssm_slots) │
│  每slot: 128 KiB        │  每slot: 768 KiB        │  每slot: 27 KiB  │
│  block_size=128         │  固定大小               │  固定大小         │
└─────────────────────────┴────────────────────────┴───────────────────┘
```

- **KV Cache Region**：`num_kv_blocks * 131,072 bytes`，每 block 128 tokens
- **SSM State Region**：`num_ssm_slots * 786,432 bytes`，每 slot 一个请求的 GDN state
- **Conv State Region**：`num_ssm_slots * 27,648 bytes`，每 slot 一个请求的 conv state

`num_kv_blocks` 和 `num_ssm_slots` 可以独立设置，不再强制相等。

**理由**：
- KV cache 和 SSM state 的生命周期和用量不同：KV cache 随序列增长，SSM state 固定大小
- 解耦后可独立优化每种资源的容量

### 3. block_size 恢复为 128

**决策**：Attention 的 `block_size` 恢复为 kernel 原生支持的 128 tokens。

**影响**：
- `patch_mamba_config.py` 不再强制 `attn_block_size = 1536`
- `_reshape_kv_cache_tensors` 不再需要 block splitting（`block_size_chunk` 逻辑）
- `BlockTable` 不再需要 physical-to-logical block 转换
- KVCacheManager 的分配粒度变为 128 tokens，碎片化更低

### 4. 需要修改的算子和代码路径

#### 4.1 算子改动（核心）

| 算子 | 当前行为 | 需要的改动 | 改动量 |
|------|----------|-----------|--------|
| `reshape_and_cache` (写 KV) | 通过 `slot_mapping` 写入连续 KV cache 区域 | `slot_mapping` 需加上 KV region offset；或保证 slot_mapping 的 block_number 已经包含 region offset | 小 |
| `npu_fused_infer_attention_score` (读 KV) | 通过 `block_table` 读取连续 KV cache | `block_table` 中的 physical block number 需反映 KV region 内的实际偏移 | 无（block_table 值由框架计算） |
| `transpose_kv_cache_by_block` | 通过 `blockIDs` 访问 block | `blockIDs` 需反映 KV region 内的实际偏移 | 无（blockIDs 由框架传入） |
| GDN `ssm_state` 访问 | 通过 `ssm_state_indices` 访问 | `ssm_state_indices` 需反映 SSM region 内的实际偏移；或直接使用新 slice 的 tensor | 小 |
| GDN `conv_state` 访问 | 通过 `conv_state_indices` 访问 | 同上，使用新 slice 的 conv_state tensor | 小 |

**关键洞察**：CANN attention 算子（`npu_fused_infer_attention_score`、`npu_paged_attention`）通过 `block_table` 间接寻址，算子本身不假设 block 连续。只要 `block_table` 中的 physical block number 正确反映新布局中的偏移，算子无需修改。同理，`transpose_kv_cache_by_block` 使用 `blockIDs` 数组，也不假设连续。

#### 4.2 框架层改动

| 文件 | 改动 |
|------|------|
| `patch/platform/patch_mamba_config.py` | 移除 `attn_block_size` 膨胀逻辑和 `mamba_page_size_padded` padding。block_size 保持 128 或用户设定值 |
| `worker/model_runner_v1.py:3460-3463` | 移除 AttentionSpec `page_size_padded` 强制对齐逻辑 |
| `worker/model_runner_v1.py:2877-2901` | 修改 `_allocate_kv_cache_tensors`，同一 raw tensor 内按子区域分配 |
| `worker/model_runner_v1.py:3043-3084` | 修改 `_reshape_kv_cache_tensors` hybrid 路径，移除 `conv_block_padding_size`，按子区域切片 |
| `worker/model_runner_v1.py:3167-3185` | 修改 MambaSpec reshape 路径，从对应的子区域提取 conv/ssm state |
| `worker/block_table.py` | slot_mapping 计算需考虑 KV region 的起始 offset |

### 5. 收益分析

**910B4 x4 (TP=4)，69.92 GiB 总 cache 预算：**

当前方案（padded page_size = 1,600,512 bytes，block_size=1536）：
- num_blocks = 69.92 GiB / 1,600,512 bytes ≈ 46,832 blocks
- KV cache 容量 = 46,832 × 1536 = 71,940,352 tokens（但实际受限于 block 分配策略）
- 每 block 的 padding 浪费 = 27 KiB，总浪费 ≈ 46,832 × 27 KiB ≈ 1,237 MiB

新方案（自然 page_size，block_size=128）：
- KV cache 自然 page_size = 131,072 bytes (128 KiB)
- 可分配更多 KV blocks，无 padding 浪费
- 回收的 ~1.2 GiB 可额外缓存约 1,200 MiB / 64 KiB/token ≈ 19,200 tokens（逻辑唯一口径）

**block_size 恢复为 128 的间接收益：**
- KVCacheManager 分配粒度从 1536 tokens 降为 128 tokens
- 内部碎片化从最大 1535 tokens/block 降为最大 127 tokens/block
- 短序列场景（并发请求数多）下效果显著

## Risks / Trade-offs

- **[block_table 和 slot_mapping offset 正确性]** → 新布局下 physical block number 的计算逻辑变化，需要仔细验证所有 slot_mapping 和 block_table 的计算路径。缓解：逐算子单测，对比新旧 layout 下的 attention 输出。
- **[KV Transfer 兼容性]** → `mooncake_connector` 和 `mooncake_layerwise_connector` 假设 KV cache 的 0-dim 是 `num_blocks`，新布局可能影响 swap_blocks 等逻辑。缓解：确保 KV cache 子区域仍保持 `(num_blocks, block_size, num_kv_heads, head_size)` 的标准形状。
- **[Prefix Caching 交互]** → block_size 从 1536 变为 128 后，prefix cache 的 block hash 粒度变细，命中率可能提升。但 SSM state 的 checkpoint 间隔需要重新对齐。缓解：SSM checkpoint 间隔独立设置，不受 KV block_size 影响。
- **[PD Disaggregation]** → 当前 2MB alignment 逻辑假设连续 KV cache tensor。新布局下 K 和 V 的子区域可能需要独立 alignment。缓解：各子区域起始地址单独 2MB 对齐。
