# Qwen3.6 27B KV Cache 与 Linear State 容量分析

## 目录

1. [摘要](#1-摘要)
2. [基础事实与计算假设](#2-基础事实与计算假设)
   - 2.1 [模型结构事实](#21-模型结构事实)
   - 2.2 [实现与代码事实](#22-实现与代码事实)
   - 2.3 [显存与容量计算假设](#23-显存与容量计算假设)
   - 2.4 [DRAM 池化与部署假设](#24-dram-池化与部署假设)
   - 2.5 [Prefix Caching 与 Agent 请求假设](#25-prefix-caching-与-agent-请求假设)
3. [单 Token Cache 与 State 开销](#3-单-token-cache-与-state-开销)
   - 3.1 [Full Attention KV Cache 单 Token 开销](#31-full-attention-kv-cache-单-token-开销)
   - 3.2 [Linear Attention GDN State](#32-linear-attention-gdn-state)
   - 3.3 [Linear Checkpoint 摊销到每 Token 的开销](#33-linear-checkpoint-摊销到每-token-的开销)
4. [910B4 32G HBM 容量预算](#4-910b4-32g-hbm-容量预算)
   - 4.1 [容量预算口径](#41-容量预算口径)
   - 4.2 [TP=4 容量](#42-tp4-容量)
   - 4.3 [TP=8 容量](#43-tp8-容量)
   - 4.4 [Live Request State 补充](#44-live-request-state-补充)
5. [KV Cache 池化方案](#5-kv-cache-池化方案)
   - 5.1 [池化目的](#51-池化目的)
   - 5.2 [总体架构](#52-总体架构)
   - 5.3 [当前代码支持边界](#53-当前代码支持边界)
   - 5.4 [Qwen3.6 混合注意力池化简要设计](#54-qwen36-混合注意力池化简要设计)
   - 5.5 [池化容量分析](#55-池化容量分析)
   - 5.6 [TTFT 传输时延理论估算](#56-ttft-传输时延理论估算)
   - 5.7 [Layerwise KVCache 搬运方案](#57-layerwise-kvcache-搬运方案)
   - 5.8 [TTFT 收益](#58-ttft-收益)
6. [Agent 请求设计对 Prefix Caching 命中的影响](#6-agent-请求设计对-prefix-caching-命中的影响)
   - 6.1 [从 OpenAI Request 到 LLM 输入](#61-从-openai-request-到-llm-输入)
   - 6.2 [Agent 侧设计要求](#62-agent-侧设计要求)
   - 6.3 [设计结论](#63-设计结论)
7. [总结与待确认问题](#7-总结与待确认问题)
   - 7.1 [关键解读](#71-关键解读)
   - 7.2 [待确认问题](#72-待确认问题)

## 1. 摘要

本文档分析 Qwen3.6 27B 在 910B4 32G 上做长上下文 Prefix Caching 时的 cache/state 容量、DRAM 池化方案，以及 Agent 请求设计对命中率的影响。核心结论如下：

- **单 token 开销**：Qwen3.6 27B 的 Full Attention KV cache 在逻辑唯一口径下为 **64 KiB/token**。但模型只有 4 个 KV heads，TP=8 时每卡仍至少保存 1 个 KV head，因此 Full Attention KV cache 会发生复制，物理分配口径变为 **128 KiB/token**。GDN Linear Attention 的 SSM state 每个 checkpoint 为 **144 MiB**，checkpoint 间隔越短，摊销到每个 token 的开销越高。
- **HBM 容量预算**：按实测 4 卡部署单卡可用 KVCache **17.48GiB** 估算，TP=4 总预算为 **69.92GiB**。在不保存 GDN checkpoint 时可缓存约 **1.15M tokens**；checkpoint 间隔为 1,024 tokens 时约 **352K tokens**。TP=8 虽然总 HBM cache/state 预算线性外推到 **139.84GiB**，但 Full Attention 4 个 KV heads 会复制，因此不保存 GDN checkpoint 时最大 token 数仍约 **1.15M**。
- **DRAM 池化方案**：建议在单台 910B4 上部署 2 个 TP=4 的 vllm-ascend PD 混合实例，共享 1 个 Mooncake DRAM KV Cache Pool。若预留 **512GiB DRAM**，仅池化 Full KV 时可缓存约 **8.39M tokens**，约等价 **128 个 64K prefix**；若同时池化 GDN checkpoint，建议优先评估 4,096 或 8,192 token 间隔。
- **代码支持边界**：当前 AscendStore KV Pool 路径具备 block hash lookup/load/store 和 layerwise 框架，但尚未把 Qwen3.6 的 Full KV、GDN SSM checkpoint、GDN live state 建模为不同池化对象。因此本文给出 Qwen3.6 混合注意力池化的简要设计，建议先支持 Full KV 池化，再逐步加入 GDN checkpoint。
- **TTFT 收益**：实测 64K 输入场景下，DRAM cache 命中率约 90% 时，TTFT 从 **38.3s** 降至 **14.4s**，相对下降 **62.4%**，加速比约 **2.66x**。理论传输下界通常是几十到百毫秒，主要收益来自避免大部分 prefill 计算；后续 layerwise 方案预计还能进一步降低可见 TTFT。
- **Agent 请求设计**：Prefix Caching 命中的是最终渲染后的 token 前缀。Qwen3.6 chat template 会先渲染 `tools`，再追加首个 `system` message，因此 Agent 必须稳定工具集合、工具顺序、system prompt、chat template 参数和共享上下文格式，否则 DRAM 池化容量无法有效转化为命中率。

## 2. 基础事实与计算假设

本节列出全文共同依赖的事实和假设，覆盖模型结构、vLLM/vLLM Ascend 实现、HBM/DRAM 容量、池化架构和 Agent 请求渲染链路。

### 2.1 模型结构事实

来自 `Qwen/Qwen3.6-27B/config.json` 的模型 config 事实：

- `num_hidden_layers = 64`
- `layer_types` 按 3 个 `linear_attention` 层 + 1 个 `full_attention` 层重复
- `num_full_attention_layers = 16`
- `num_linear_attention_layers = 48`
- Full Attention: `num_key_value_heads = 4`, `head_dim = 256`
- Linear Attention: `linear_num_value_heads = 48`, `linear_key_head_dim = 128`, `linear_value_head_dim = 128`
- GDN SSM dtype: `mamba_ssm_dtype = float32`

### 2.2 实现与代码事实

来自本仓库的实现与设计事实：

- `openspec/changes/qwen36-27b-310p-support/design.md` 明确 Full Attention 使用 paged KV cache，Linear Attention 使用 per-request SSM state。
- `vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py` 中记录 `initial_state: [B, H, V, K]`。
- `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py` 中 linear key/value heads 会按 TP 切分。
- upstream vLLM 的 `vllm/config/model.py::get_num_kv_heads()` 在 `total_num_kv_heads < tensor_parallel_size` 时返回 `max(1, total_num_kv_heads // tensor_parallel_size)`，注释明确说明 KV heads 会复制，以保证每张卡至少有 1 个 KV head。
- upstream vLLM 的 `vllm/model_executor/models/qwen3_next.py` 中 Qwen3Next attention 初始化也明确处理 `total_num_kv_heads < tp_size` 的 KV heads 复制分支。
- upstream vLLM 的 `ChatCompletionRequest` 定义 `messages`、`tools` 与 `chat_template_kwargs`，chat serving 链路会调用 `render_chat_request()` 生成 `engine_inputs`。
- upstream vLLM 的 render 链路会把 `request.tools` 转成 `tool_dicts`，再合并到 chat template kwargs 中，最终渲染成模型输入 token。
- `Qwen/Qwen3.6-27B/chat_template.jinja` 在存在 `tools` 时，会先渲染工具说明和工具 schema；如果首条 message 是 system，再把 system 内容追加到该 system 段中。

### 2.3 显存与容量计算假设

显存计算假设：

- KV cache dtype: FP16/BF16，2 bytes。
- GDN SSM checkpoint dtype: float32，4 bytes。
- TP=4 时 Full Attention 的 4 个 KV heads 可以均匀切分，每卡 1 个 KV head。
- TP=8 时 Full Attention 的 4 个 KV heads 少于 TP size，按 vLLM 逻辑每卡至少 1 个 KV head，因此物理 KV cache 总量相对逻辑唯一口径放大 2 倍。
- Linear Attention 的 `linear_num_value_heads = 48`，TP=4/TP=8 都可均匀切分。
- 除非特别说明，容量单位使用二进制单位：KiB、MiB、GiB。
- 4 卡部署时，按实测经验单卡可用 KVCache 为 17.48GiB，四卡总 cache/state 预算为 69.92GiB。
- 8 卡部署时，暂按同样单卡可用预算线性外推，总 cache/state 预算为 139.84GiB；该值仍需生产实测确认。
- 本文档的 HBM 容量表只估算 Full Attention KV cache 与 GDN SSM checkpoint，不把模型权重、runtime、graph、通信 buffer、临时 workspace 等额外开销重复计入表格。

### 2.4 DRAM 池化与部署假设

- 目标机器为单台 910B4，8 张 32G 卡。
- 单机 DRAM 总量按 1TiB 级别考虑，可用于 KV Cache 池化的内存不小于 512GiB。
- 推荐部署形态为 2 个 TP=4 的 vllm-ascend PD 混合实例，共享 1 个 Mooncake DRAM KV Cache Pool。
- DRAM 池化主路径只考虑内存介质。SSD 因延迟和带宽限制，不作为在线 TTFT 优化主路径，只适合作为冷数据或离线预热兜底。
- 池化对象包括 Full Attention KV blocks；GDN SSM checkpoint 可选，是否池化取决于 checkpoint 间隔、热度和恢复成本。
- TTFT 传输时延估算采用保守工程带宽：PCIe 4.0 x16 按 25 GB/s/card 有效带宽估算，HCCS 跨卡分发按 100 到 200 GB/s 级别估算；生产部署需要用目标机器实测带宽替换该假设。

### 2.5 Prefix Caching 与 Agent 请求假设

- Prefix Caching 命中以最终进入模型的 token 前缀为准，而不是以 OpenAI request JSON 的语义等价为准。
- Agent 请求采用 OpenAI-compatible chat 格式，主要输入包括 `messages`、`tools`、`tool_choice` 和 `chat_template_kwargs`。
- 对 Qwen3.6，`tools` 和首个 `system` message 位于用户问题之前，因此工具集合、工具顺序、system prompt、chat template 参数和共享上下文格式都属于关键 prefix 稳定性因素。
- 生产网关可以按 `model_id`、`tokenizer_revision`、`chat_template_version`、`tool_schema_version`、`system_prompt_version`、`thinking_mode` 和 `tenant_cache_namespace` 组织 prefix profile，用于路由和缓存隔离。

## 3. 单 Token Cache 与 State 开销

### 3.1 Full Attention KV Cache 单 Token 开销

公式：

```text
full_kv_bytes_per_token_global
  = full_attention_layers * 2(K,V) * num_kv_heads * head_dim * kv_dtype_bytes
  = 16 * 2 * 4 * 256 * 2
  = 65,536 bytes
  = 64 KiB/token
```

上述 **64 KiB/token** 是逻辑唯一口径。实际物理分配要看 TP size 与 KV heads 数量：

实际物理分配口径：

| TP | 每卡 KV heads | Full KV / token / card | Full KV / token / 集群物理总量 |
|---:|---:|---:|---:|
| 4 | 1 | 16 KiB | 64 KiB |
| 8 | 1（复制） | 16 KiB | 128 KiB |

标准 128-token KV block：

```text
TP=4: 128 tokens * 64 KiB/token = 8 MiB / 集群，2 MiB/card
TP=8: 128 tokens * 128 KiB/token = 16 MiB / 集群，2 MiB/card
```

### 3.2 Linear Attention GDN State

GDN recurrent state 每个 linear layer 的形状可以按 `[num_value_heads, value_dim, key_dim]` 估算。

公式：

```text
gdn_ssm_state_bytes_global
  = linear_layers * linear_num_value_heads * linear_value_head_dim * linear_key_head_dim * fp32_bytes
  = 48 * 48 * 128 * 128 * 4
  = 150,994,944 bytes
  = 144 MiB
```

单卡口径：

| TP | GDN SSM state / checkpoint / card |
|---:|---:|
| 4 | 36 MiB |
| 8 | 18 MiB |

这部分 state 对于 live request 或已保存的 prefix checkpoint 来说是固定大小。它本身不随序列长度增长；只有当系统保存多个 checkpoint 时，总量才会随 checkpoint 数量增长。

### 3.3 Linear Checkpoint 摊销到每 Token 的开销

如果系统每 `N` tokens 保存一个 GDN checkpoint，GDN 部分的摊销开销为：

```text
linear_checkpoint_bytes_per_token_global = 144 MiB / N
```

TP=4 时，Full KV 物理总量为 64 KiB/token：

| Checkpoint 间隔 | Full KV / token | Linear checkpoint / token | Total / token | Linear / Full |
|---:|---:|---:|---:|---:|
| 128 | 64 KiB | 1.125 MiB | 1.1875 MiB | 18.00x |
| 256 | 64 KiB | 576 KiB | 640 KiB | 9.00x |
| 512 | 64 KiB | 288 KiB | 352 KiB | 4.50x |
| 1,024 | 64 KiB | 144 KiB | 208 KiB | 2.25x |
| 2,048 | 64 KiB | 72 KiB | 136 KiB | 1.125x |
| 4,096 | 64 KiB | 36 KiB | 100 KiB | 0.5625x |
| 8,192 | 64 KiB | 18 KiB | 82 KiB | 0.28125x |
| 不保存 GDN checkpoint | 64 KiB | 0 | 64 KiB | 0x |

TP=8 时，Full KV 因复制变为 128 KiB/token：

| Checkpoint 间隔 | Full KV / token | Linear checkpoint / token | Total / token |
|---:|---:|---:|---:|
| 128 | 128 KiB | 1.125 MiB | 1.25 MiB |
| 256 | 128 KiB | 576 KiB | 704 KiB |
| 512 | 128 KiB | 288 KiB | 416 KiB |
| 1,024 | 128 KiB | 144 KiB | 272 KiB |
| 2,048 | 128 KiB | 72 KiB | 200 KiB |
| 4,096 | 128 KiB | 36 KiB | 164 KiB |
| 8,192 | 128 KiB | 18 KiB | 146 KiB |
| 不保存 GDN checkpoint | 128 KiB | 0 | 128 KiB |

关键观察：TP=4 下 128-token 的 Full Attention KV block 是 **8 MiB/集群**，但同一边界上的全 GDN 层 checkpoint 是 **144 MiB/集群**。如果按 128-token 粒度保存 GDN checkpoint，Linear Attention checkpoint 会成为 cache 显存主导项。TP=8 下 Full KV block 因复制变为 **16 MiB/集群**，但仍明显小于单个 GDN checkpoint。

## 4. 910B4 32G HBM 容量预算

### 4.1 容量预算口径

本文档只保留实测工程 cache 预算，不再列裸 HBM 理论上限。根据实测经验，910B4 32G 四卡部署时单卡可用 KVCache 约 17.48G。本文档按 GiB 近似进入容量公式，即四卡总预算 `17.48 GiB * 4 = 69.92 GiB`。八卡场景暂按相同单卡可用预算线性外推，即 `17.48 GiB * 8 = 139.84 GiB`；该八卡值仍需实测确认。

TP=4 带 checkpoint 间隔 `N` 的容量公式：

```text
memory(tokens, N)
  = tokens * 64 KiB + ceil(tokens / N) * 144 MiB
```

TP=8 带 checkpoint 间隔 `N` 的容量公式：

```text
memory(tokens, N)
  = tokens * 128 KiB + ceil(tokens / N) * 144 MiB
```

### 4.2 TP=4 容量

4x 32G，69.92 GiB cache/state 预算：

| Checkpoint 间隔 | 最大可缓存 tokens |
|---:|---:|
| 128 | 60,288 |
| 256 | 114,432 |
| 512 | 207,872 |
| 1,024 | 352,256 |
| 2,048 | 538,624 |
| 4,096 | 733,153 |
| 8,192 | 892,928 |
| 不保存 GDN checkpoint | 1,145,569 |

### 4.3 TP=8 容量

8x 32G，139.84 GiB cache/state 预算：

| Checkpoint 间隔 | 最大可缓存 tokens |
|---:|---:|
| 128 | 120,576 |
| 256 | 208,128 |
| 512 | 352,256 |
| 1,024 | 538,624 |
| 2,048 | 733,153 |
| 4,096 | 893,281 |
| 8,192 | 1,003,873 |
| 不保存 GDN checkpoint | 1,145,569 |

### 4.4 Live Request State 补充

上面的容量表主要建模 prefix-cache 存储。实际 live decode 还需要为每个活跃请求保存当前 GDN SSM state：

```text
live_gdn_state = active_requests * 144 MiB / TP
```

示例：

| TP | Live GDN state / request / card |
|---:|---:|
| 4 | 36 MiB |
| 8 | 18 MiB |

长上下文场景下，Full Attention KV cache 仍然是主要开销。短上下文高并发场景下，live GDN state 会变得明显。按集群物理分配口径，它大致等价于：

```text
TP=4: 144 MiB / 64 KiB = 2,304 Full-Attention-KV tokens
TP=8: 144 MiB / 128 KiB = 1,152 Full-Attention-KV tokens
```

## 5. KV Cache 池化方案

### 5.1 池化目的

随着 Agent 应用发展，长上下文会成为推理主流。典型输入长度可达到 32K 到 64K，最大上下文长度大于 128K。长上下文下，Prefix Caching 是降低 TTFT、稳定吞吐的关键特性；但 HBM 容量有限，在大并发场景下本地 HBM prefix cache 很容易被挤出，导致命中率下降。

KV Cache 池化的目标是把“热 prefix 的 KV blocks 和必要的 Linear Attention state checkpoint”从单实例 HBM 扩展到单机 DRAM 池中：

- **提升命中率**：多个实例共享同一个 prefix cache pool，避免同一业务热 prefix 在不同实例间重复预热。
- **扩大容量**：910B4 单机通常有 1T 级 DRAM，可用于 KV Cache 池化的内存不小于 512GiB，远大于单实例 HBM 可用 cache 空间。
- **降低 TTFT**：命中 DRAM cache 后，只需从池中加载已计算 prefix 的 KV/state，再计算剩余 suffix。
- **隔离 HBM 压力**：HBM 保留 live decode 和短期热点，DRAM 承担更大的 prefix cache 工作集。

当前可选实现包括 Mooncake 和 Memcache。本文档以 Mooncake DRAM pool 为主口径；SSD 池化不作为主方案，因为 SSD 延迟和带宽对在线 TTFT 不友好，更适合离线预热或冷数据兜底。

### 5.2 总体架构

单台 910B4 机器有 8 张卡，上层推理网关负责请求路由，中间部署 2 个 TP=4 的 vllm-ascend PD 混合实例，每个实例同时具备 Prefill/Decode 能力。底层部署 1 个 Mooncake DRAM KV Cache 池，两个 vllm-ascend 实例共享该池。

<div style="border:1px solid #d0d7de;border-radius:8px;padding:16px;margin:16px 0;font-family:Arial, sans-serif;color:#24292f;background:#ffffff;">
  <div style="text-align:center;font-weight:700;font-size:16px;margin-bottom:14px;">910B4 单机 KV Cache 池化架构</div>
  <div style="border:2px solid #57606a;border-radius:8px;padding:12px;background:#f6f8fa;margin-bottom:12px;">
    <div style="font-weight:700;text-align:center;margin-bottom:8px;">推理网关</div>
    <div style="font-size:13px;text-align:center;line-height:1.5;">接收 Agent 推理请求，按负载、prefix hash、本地命中信息路由到 vllm-ascend 实例</div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:stretch;margin-bottom:8px;">
    <div style="text-align:center;font-size:13px;color:#57606a;">请求路由 ↓</div>
    <div style="text-align:center;font-size:13px;color:#57606a;">请求路由 ↓</div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:stretch;margin-bottom:8px;">
    <div style="border:1px solid #8c959f;border-radius:8px;padding:12px;background:#ffffff;">
      <div style="font-weight:700;margin-bottom:8px;">vllm-ascend 实例 A</div>
      <div style="font-size:13px;margin-bottom:8px;">PD 混合部署，TP=4，NPU 0-3</div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:10px;">
        <div style="border:1px solid #6e7781;border-radius:6px;padding:6px;text-align:center;background:#f6f8fa;">卡0</div>
        <div style="border:1px solid #6e7781;border-radius:6px;padding:6px;text-align:center;background:#f6f8fa;">卡1</div>
        <div style="border:1px solid #6e7781;border-radius:6px;padding:6px;text-align:center;background:#f6f8fa;">卡2</div>
        <div style="border:1px solid #6e7781;border-radius:6px;padding:6px;text-align:center;background:#f6f8fa;">卡3</div>
      </div>
      <div style="font-size:13px;line-height:1.5;">本地 HBM cache<br/>Prefill + Decode<br/>请求级 live state</div>
    </div>
    <div style="border:1px solid #8c959f;border-radius:8px;padding:12px;background:#ffffff;">
      <div style="font-weight:700;margin-bottom:8px;">vllm-ascend 实例 B</div>
      <div style="font-size:13px;margin-bottom:8px;">PD 混合部署，TP=4，NPU 4-7</div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:10px;">
        <div style="border:1px solid #6e7781;border-radius:6px;padding:6px;text-align:center;background:#f6f8fa;">卡4</div>
        <div style="border:1px solid #6e7781;border-radius:6px;padding:6px;text-align:center;background:#f6f8fa;">卡5</div>
        <div style="border:1px solid #6e7781;border-radius:6px;padding:6px;text-align:center;background:#f6f8fa;">卡6</div>
        <div style="border:1px solid #6e7781;border-radius:6px;padding:6px;text-align:center;background:#f6f8fa;">卡7</div>
      </div>
      <div style="font-size:13px;line-height:1.5;">本地 HBM cache<br/>Prefill + Decode<br/>请求级 live state</div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:stretch;margin-bottom:8px;">
    <div style="text-align:center;font-size:13px;color:#0969da;">lookup / load / store ↓↑</div>
    <div style="text-align:center;font-size:13px;color:#0969da;">lookup / load / store ↓↑</div>
  </div>
  <div style="border:2px solid #0969da;border-radius:8px;padding:12px;background:#ddf4ff;">
    <div style="font-weight:700;text-align:center;margin-bottom:8px;">Mooncake KV Cache Pool</div>
    <div style="font-size:13px;text-align:center;margin-bottom:10px;">底层单机共享池，使用 DRAM，建议预留 ≥512GiB；SSD 仅作为冷数据兜底，不作为在线主路径</div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;">
      <div style="border:1px solid #0969da;border-radius:6px;padding:8px;background:#ffffff;">Full Attention KV blocks<br/><span style="font-size:12px;">block hash → TP shard payload</span></div>
      <div style="border:1px solid #0969da;border-radius:6px;padding:8px;background:#ffffff;">GDN SSM checkpoints<br/><span style="font-size:12px;">prefix hash → all linear layers state</span></div>
      <div style="border:1px solid #0969da;border-radius:6px;padding:8px;background:#ffffff;">Pool 管理<br/><span style="font-size:12px;">LRU / TTL / 热点前缀</span></div>
    </div>
  </div>
</div>

关键交互：

1. 请求进入某个 PD 混合实例后，实例先在本地 HBM prefix cache 中查找已命中的 blocks。
2. 本地未命中的 prefix blocks，通过 prefix hash / block hash 查询 Mooncake DRAM Pool。
3. 命中后，实例从 DRAM pool 拉取 Full Attention KV blocks；如果启用 GDN checkpoint，也拉取对应 checkpoint 并恢复 linear state。
4. 实例只对未命中的 suffix 继续 prefill，之后进入 decode。
5. 新完成的 Full KV blocks 与 GDN checkpoint 异步写回 Mooncake，供本机另一个实例或后续请求复用。
6. 池中对象按 LRU、TTL 或业务热度淘汰；HBM 只保留 live state 与最近热点。

### 5.3 当前代码支持边界

当前代码中存在两类 KV 传输能力，需要区分：

1. **AscendStore KV Pool 路径**：`vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py` 提供 `AscendStoreConnector`，后端可选 Mooncake、Memcache、Yuanrong。`KVPoolWorker` 会根据 block hash 做 lookup/load/store，并支持 `use_layerwise`。
2. **Mooncake P2P Layerwise 路径**：`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py` 已经显式识别 `MambaSpec`、`FullAttentionSpec`、`SlidingWindowSpec`，并有混合 Attention + Mamba 的注册与传输逻辑。

代码事实：

- `KVPoolWorker.register_kv_caches()` 只从第一个 cache tensor 推导 `num_blocks` 和 `block_len`，然后把所有 cache tensor 按同一套 `block_len` 规则注册到后端。见 `pool_worker.py` 的 `register_kv_caches()`。
- `ChunkedTokenDatabase.prepare_value()` 和 `prepare_value_layer()` 基于统一的 `kv_caches_base_addr` 与 `block_len` 计算地址和长度。见 `config_data.py`。
- AscendStore 的 layerwise store/load 会把每个 block 拆成 `num_layers` 个 `LayerPoolKey`，但 key 里只有 `layer_id`，没有区分 Full Attention KV、GDN conv state、GDN SSM state 或 checkpoint interval。见 `pool_worker.py::retrieve_layer()`、`store_layer()`、`lookup_scheduler()`。
- Qwen3.6 的 GDN 路径会调用 `maybe_save_kv_layer_to_connector("", [])`，即依赖 connector 当前层计数触发 layerwise hook，而不是显式传入 GDN state 对象。见 `vllm_ascend/ops/gdn.py` 与 `vllm_ascend/_310p/ops/fla/gdn_310.py`。
- P2P 的 Mooncake layerwise connector 中，`register_kv_caches()` 会检测一个 cache tensor 是否同时被 Mamba 和 Attention layer 共享，并设置 `use_attn_mamba_hybrid`；传输侧 `get_transfer_meta()` 对 `MambaSpec` 单独处理 conv/ssm state。见 `mooncake_layerwise_connector.py`。

因此，当前代码可以确认支持的是 **Full Attention KV blocks 的外部池化基础能力**，并且具备通用 layerwise load/store 框架；但还不能直接认定已经完整支持 Qwen3.6 这种 Full Attention + GDN Linear Attention 的 **语义正确 KV Cache 池化**。主要缺口是：AscendStore pool 路径没有把 Full KV、GDN SSM checkpoint、GDN conv/live state 建模成不同对象，也没有定义 GDN checkpoint interval、checkpoint key、恢复边界和 replay 规则。

### 5.4 Qwen3.6 混合注意力池化简要设计

建议把 Qwen3.6 的池化对象拆成三类，而不是把所有 cache tensor 当成同一种 block payload：

| 对象 | Key 维度 | Payload | 读取时机 | 说明 |
|---|---|---|---|---|
| Full Attention KV block | `model + prefix_profile + block_hash + full_layer_id + tp/head_rank` | K/V block shard | prefill 前或 layerwise 到达该 full layer 前 | 必须支持 TP=8 下 KV head 复制语义 |
| GDN SSM checkpoint | `model + prefix_profile + checkpoint_token + linear_layer_id + tp_rank + checkpoint_interval` | GDN SSM state shard，FP32 | 恢复到最近 checkpoint 后 | 只保存 checkpoint，不建议每 128 token 都保存 |
| GDN conv/live state | 不作为 DRAM pool 默认对象 | live request 内部状态 | 请求执行过程中 | conv/live state 更适合留在 HBM；跨请求复用优先依赖 SSM checkpoint |

推荐恢复流程：

```text
1. 对渲染后的 token prefix 做 block hash lookup。
2. 找到 Full KV 连续命中的最大 block 边界。
3. 在该边界之前，选择最近的 GDN checkpoint。
4. 从 DRAM pool 加载：
   - 命中范围内的 Full Attention KV blocks
   - 最近 GDN SSM checkpoint
5. 从 checkpoint_token 到 hit_token 边界 replay GDN linear layers。
6. 从 hit_token 之后继续正常 prefill，然后进入 decode。
```

设计要点：

- **对象分层**：Full KV 是按 token/block 增长的 cache；GDN SSM checkpoint 是按 checkpoint interval 增长的 state。两者 key、生命周期、淘汰策略都应分开。
- **命中判定**：只有 Full KV blocks 和必要的 GDN checkpoint 同时可用，才能把 prefix 命中推进到对应 token 边界；否则只能退回到最近可恢复边界。
- **checkpoint interval**：优先评估 4,096 或 8,192 tokens。间隔过小会显著增加 DRAM 占用，间隔过大会增加 GDN replay 时间。
- **TP=8 复制语义**：Full Attention 只有 4 个 KV heads。TP=8 时 key 中的 `head_or_tp_rank` 需要表达复制后的物理 shard，不能简单按 8 份唯一 KV 计算。
- **与现有代码复用**：可以复用 AscendStore 的 block hash lookup、Mooncake/Memcache backend、layerwise load/store 线程；但需要新增 Qwen3.6 hybrid cache metadata，把 `layer_id` 扩展为 `cache_object_type + layer_id + checkpoint_id`。

### 5.5 池化容量分析

基线假设：

- 单机 DRAM 总量约 1TiB。
- 可用于 KV Cache 池化的 DRAM 不小于 512GiB。
- 两个实例均为 TP=4，因此池化对象按 TP=4 物理布局估算。
- Full Attention KV cache 为 64 KiB/token。
- GDN SSM checkpoint 为 144 MiB/checkpoint。
- 表中“仅 Full KV”表示只池化 Full Attention KV blocks，不池化 GDN checkpoint；启用 GDN checkpoint 时，按不同 checkpoint 间隔额外计入 GDN state。

512GiB DRAM pool 的容量：

| 池化内容 / GDN checkpoint 间隔 | 可缓存 tokens | 约等价 32K prefixes | 约等价 64K prefixes | 约等价 128K prefixes |
|---:|---:|---:|---:|---:|
| 仅 Full KV | 8,388,608 | 256 | 128 | 64 |
| 8,192 | 6,545,408 | 199 | 99 | 49 |
| 4,096 | 5,368,064 | 163 | 81 | 40 |
| 2,048 | 3,946,496 | 120 | 60 | 30 |
| 1,024 | 2,580,480 | 78 | 39 | 19 |
| 512 | 1,524,992 | 46 | 23 | 11 |
| 256 | 838,656 | 25 | 12 | 6 |
| 128 | 441,472 | 13 | 6 | 3 |

不同 DRAM 预算下的敏感性：

| DRAM pool | 仅 Full KV | GDN checkpoint=8,192 | GDN checkpoint=4,096 | GDN checkpoint=1,024 |
|---:|---:|---:|---:|---:|
| 512GiB | 8,388,608 tokens | 6,545,408 tokens | 5,368,064 tokens | 2,580,480 tokens |
| 768GiB | 12,582,912 tokens | 9,820,416 tokens | 8,052,736 tokens | 3,871,488 tokens |
| 1,024GiB | 16,777,216 tokens | 13,093,120 tokens | 10,736,128 tokens | 5,161,984 tokens |

解读：

- 如果只池化 Full KV，512GiB DRAM 可以缓存约 **128 个 64K prefix**，或 **64 个 128K prefix**。
- 如果同时池化 GDN checkpoint，checkpoint 间隔不要过小。128-token checkpoint 在 DRAM 中也非常昂贵，512GiB 只能容纳约 **6 个 64K prefix**。
- 4,096 或 8,192 token checkpoint 更适合作为 DRAM 池化的默认候选：512GiB 下分别可容纳约 **81 个** 或 **99 个 64K prefix**。
- 若业务可以接受命中 Full KV 后重算部分 GDN state，则可以只池化 Full KV，把 GDN checkpoint 作为二期优化或只对超热点 prefix 开启。

### 5.6 TTFT 传输时延理论估算

以 64K 输入、90% DRAM cache 命中为例，按 128-token block 对齐后约命中 58,880 tokens。TP=4 时 Full Attention KV 物理总量为 64 KiB/token，因此需要从 DRAM 读取：

```text
Full KV 数据量 = 58,880 * 64 KiB = 3.594 GiB / TP=4 实例
单卡读取量 = 3.594 GiB / 4 = 0.898 GiB/card
若同时加载 1 个 GDN SSM checkpoint = 144 MiB / 实例 = 36 MiB/card
合计约 3.734 GiB / 实例，0.934 GiB/card
```

不同上下文长度下，90% 命中需要搬运的数据量如下：

| 输入长度 | 90% 命中 tokens（128 对齐） | Full KV 数据量 / TP=4 实例 | Full KV 数据量 / card | 加 1 个 GDN checkpoint |
|---:|---:|---:|---:|---:|
| 32K | 29,440 | 1.797 GiB | 0.449 GiB | 1.938 GiB |
| 64K | 58,880 | 3.594 GiB | 0.898 GiB | 3.734 GiB |
| 128K | 117,888 | 7.195 GiB | 1.799 GiB | 7.336 GiB |

带宽口径采用保守工程估算，实际值需要用目标 910B4 机器实测固化：

- PCIe 4.0 x16 单向理论带宽约 32 GB/s；考虑 DMA、协议、NUMA、host DRAM、并发 copy 影响，按 **25 GB/s/card 有效带宽** 做保守估算。
- HCCS 是 NPU-NPU 互联，不是 host DRAM 到 NPU HBM 的主路径。如果需要先落到某张卡再跨卡分发，按 **100 到 200 GB/s** 级别估算 HCCS 分发成本。
- DRAM pool 到 NPU HBM 的主耗时通常由 host DRAM 读取、PCIe DMA、后端调度、同步点共同决定；纯链路带宽只给出下界。

64K、90% 命中场景的理论传输下界：

| 传输路径假设 | 数据量 | 带宽假设 | 理论耗时 |
|---|---:|---:|---:|
| 4 卡并行 PCIe 读取，仅 Full KV | 0.898 GiB/card | 25 GB/s/card | 约 39 ms |
| 4 卡并行 PCIe 读取，Full KV + 1 个 GDN checkpoint | 0.934 GiB/card | 25 GB/s/card | 约 40 ms |
| 单链路串行读取，仅 Full KV | 3.594 GiB | 25 GB/s | 约 154 ms |
| 单链路串行读取，Full KV + 1 个 GDN checkpoint | 3.734 GiB | 25 GB/s | 约 160 ms |
| HCCS 跨卡分发，仅 Full KV | 3.594 GiB | 100 GB/s | 约 39 ms |
| HCCS 跨卡分发，仅 Full KV | 3.594 GiB | 200 GB/s | 约 19 ms |

因此，64K 输入下 DRAM 命中后从 38.3s 降到 14.4s 的主要收益不是来自“传输很快”本身，而是避免了大部分 prefill 计算。纯数据搬运的理论量级通常是几十到百毫秒；剩余 TTFT 主要由未命中 suffix prefill、GDN checkpoint replay、后端 lookup、DMA 调度、同步等待、batch 排队和 layerwise overlap 效果决定。

### 5.7 Layerwise KVCache 搬运方案

非 layerwise 方案会在 prefill 开始前尽量把命中的 KV 全部搬入 HBM。优点是实现简单，缺点是首 token 必须等待完整 KV load 完成，且一次性 HBM/PCIe 压力较大。

Layerwise 方案把 KV load/store 从“请求级”拆到“层级”：

```text
普通 load：
  lookup all blocks
  load all matched KV/state
  wait
  run prefill

layerwise load：
  lookup all blocks
  load layer 0 KV/state
  run layer 0
  while run layer i:
      async load layer i+1
      async store layer i-1
  run final layers
```

在当前代码中，AscendStore 路径已有 `use_layerwise` 开关：`start_load_kv()` 会创建 `retrieve_layer()` 生成器并先发起第一层 load；`wait_for_layer_load()` 在模型执行到下一层前推进下一次 load；`save_kv_layer()` 则按层触发 store。底层 `KVCacheStoreLayerSendingThread` 和 `KVCacheStoreLayerRecvingThread` 按 `layer_id` 调用 `prepare_value_layer()` 做 put/get。

对 Qwen3.6，layerwise 设计需要额外处理混合层：

- Full Attention 层：按 layer_id 搬运 K/V block shard。
- GDN Linear Attention 层：按 checkpoint boundary 搬运 SSM checkpoint；若 checkpoint 不在当前命中边界，需要先恢复 checkpoint，再 replay 到目标 token。
- 层顺序：Qwen3.6 是 3 个 linear_attention + 1 个 full_attention 重复。layerwise load 不应只按 `num_layers` 平铺同质 KV，而应根据 `layer_types` 决定当前层需要 Full KV、GDN state，还是只需要 live state。
- 同步点：每层计算前只等待本层必要对象；下一层对象异步预取，上一层对象异步写回。
- 命中判断：layerwise lookup 应能区分“Full KV 全层命中”与“GDN checkpoint 缺失”。否则可能出现 Full KV 命中但 GDN state 不可恢复，导致语义上不能跳过 prefix 计算。

Layerwise 的预期收益是把几十到百毫秒级的 DRAM/PCIe 搬运隐藏到逐层计算中，并降低请求开始阶段的一次性等待。它不能减少需要搬运的总字节数，但可以减少可见 TTFT；最终收益取决于每层计算时间、每层 KV/state 大小、DMA 并发度和同步开销。

### 5.8 TTFT 收益

实测 64K 输入场景，在 DRAM cache 命中率约 90% 时：

| 场景 | TTFT |
|---|---:|
| 未命中，需要完整 prefill | 38.3s |
| 90% 命中 DRAM cache | 14.4s |

收益：

```text
TTFT 降低 = 38.3s - 14.4s = 23.9s
相对下降 = 23.9 / 38.3 = 62.4%
加速比 = 38.3 / 14.4 = 2.66x
```

这说明在 32K 到 64K 平均输入长度的 Agent 场景中，DRAM 池化能显著改善 TTFT。进一步使用 layerwise 方案后，KV 读取可以与逐层计算更细粒度重叠，TTFT 预计会继续下降；具体收益需要单独 profiling 固化。

## 6. Agent 请求设计对 Prefix Caching 命中的影响

Prefix Caching 的命中不是只由缓存池容量决定，也由 Agent 组织 OpenAI-compatible request 的方式决定。对 Qwen3.6 27B 这类模型，`tools` 和 `system` 都位于用户问题之前，任何工具列表、工具 JSON 序列化、system prompt 或 chat template 参数的变化，都会改变后续全部 token 的前缀，从而降低命中率。

### 6.1 从 OpenAI Request 到 LLM 输入

vLLM 的 OpenAI-compatible chat 请求大致经过以下链路：

<div style="border:1px solid #d0d7de;border-radius:8px;padding:14px;margin:14px 0;font-family:Arial, sans-serif;color:#24292f;background:#ffffff;">
  <div style="display:grid;grid-template-columns:1fr 28px 1fr 28px 1fr;gap:8px;align-items:center;margin-bottom:10px;">
    <div style="border:1px solid #8c959f;border-radius:8px;padding:10px;background:#f6f8fa;text-align:center;">
      <div style="font-weight:700;">Agent / SDK</div>
      <div style="font-size:12px;margin-top:4px;">messages + tools</div>
    </div>
    <div style="text-align:center;font-weight:700;color:#57606a;">→</div>
    <div style="border:1px solid #8c959f;border-radius:8px;padding:10px;background:#ffffff;text-align:center;">
      <div style="font-weight:700;">ChatCompletionRequest</div>
      <div style="font-size:12px;margin-top:4px;">OpenAI 请求对象</div>
    </div>
    <div style="text-align:center;font-weight:700;color:#57606a;">→</div>
    <div style="border:1px solid #8c959f;border-radius:8px;padding:10px;background:#ffffff;text-align:center;">
      <div style="font-weight:700;">Render Chat</div>
      <div style="font-size:12px;margin-top:4px;">tools 注入模板参数</div>
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 28px 1fr 28px 1fr;gap:8px;align-items:center;">
    <div style="border:1px solid #0969da;border-radius:8px;padding:10px;background:#ddf4ff;text-align:center;">
      <div style="font-weight:700;">Chat Template</div>
      <div style="font-size:12px;margin-top:4px;">渲染 prompt / token ids</div>
    </div>
    <div style="text-align:center;font-weight:700;color:#57606a;">→</div>
    <div style="border:1px solid #0969da;border-radius:8px;padding:10px;background:#ddf4ff;text-align:center;">
      <div style="font-weight:700;">Engine Generate</div>
      <div style="font-size:12px;margin-top:4px;">提交 engine_input</div>
    </div>
    <div style="text-align:center;font-weight:700;color:#57606a;">→</div>
    <div style="border:1px solid #0969da;border-radius:8px;padding:10px;background:#ddf4ff;text-align:center;">
      <div style="font-weight:700;">LLM 执行</div>
      <div style="font-size:12px;margin-top:4px;">prefix lookup → prefill → decode</div>
    </div>
  </div>
</div>

代码锚点：`ChatCompletionRequest` 定义 `messages`、`tools` 和 `chat_template_kwargs`；`create_chat_completion()` 调用 `render_chat_request()`；`preprocess_chat()` 将 `request.tools` 转成 `tool_dicts` 并合并到 chat template kwargs；最终由 `engine_client.generate()` 提交给推理引擎。

这里的关键点是：Prefix Caching 命中的是最终进入模型的 token 前缀，而不是 OpenAI request 的 JSON 语义。两个请求即使业务含义相同，只要工具数组顺序、JSON 字段顺序、system prompt 空白字符、chat template 参数或消息角色顺序导致渲染后的 token 不一致，就会从变化位置开始失去 prefix cache 复用。

Qwen3.6 的 chat template 进一步放大了这个影响。根据 `Qwen/Qwen3.6-27B/chat_template.jinja`，当请求携带 `tools` 时，模板会先生成一个 system 段写入工具说明和序列化后的 tools；如果第一条 message 是 system，再把这条 system 内容追加到同一个 system 段中；之后才渲染后续对话消息。因此，对 Agent 来说，最靠前、最值得稳定的 prefix 结构是：

```text
┌─────────────────────────────────────────────────────────────┐
│ Qwen3.6 rendered prompt prefix                              │
├─────────────────────────────────────────────────────────────┤
│ 1. tools block                                               │
│    - tool schema JSON                                        │
│    - function-call format instruction                        │
│                                                             │
│ 2. system prompt                                             │
│    - Agent role / policy / safety / output format            │
│                                                             │
│ 3. stable shared context                                     │
│    - product docs / common task context / reusable examples  │
│                                                             │
│ 4. request-specific user input                               │
│    - 当前问题、检索结果、用户态动态信息                       │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Agent 侧设计要求

#### 6.2.1 固定工具集合与工具序列化

工具列表是 Qwen3.6 模板中的最前缀内容。Agent 应该按“工具剖面”管理 tools，而不是按单次请求临时拼装 tools。

- 对同一类 Agent，保持工具集合、工具顺序、tool name、description、parameters schema 稳定。
- 工具按固定规则排序，例如按 tool name 和 tool version 排序，避免 Python dict / SDK 构造顺序差异造成 token 前缀抖动。
- tool description 中不要放时间、用户 ID、租户名、实验开关、trace id、请求级权限说明等动态内容。
- 工具 schema 变更应视为 cache-busting 版本发布，使用显式 `tool_schema_version` 做灰度和路由。
- 如果不同租户或不同产品线确实需要不同工具集合，应拆成不同 cache profile，而不是混在同一个 prefix cache 热集里。

一个稳定的工具剖面可以这样定义：

```text
tool_profile_key =
  model_id
  + chat_template_version
  + tool_schema_version
  + tool_order_policy
  + thinking_mode
```

推理网关可以用该 key 辅助路由，把相同工具剖面的请求尽量路由到相同 vllm-ascend 实例或相同 Mooncake DRAM pool 热区。

#### 6.2.2 固定 System Prompt，把动态信息后移

System prompt 位于 tools 之后、用户消息之前。它适合放稳定的 Agent 身份、长期策略和输出规范，不适合放请求级动态信息。

- 不要在 system prompt 中拼接当前时间、用户名、会话 ID、AB 实验标记、临时权限、检索摘要。
- 对同一 Agent profile，system prompt 文本、段落顺序、分隔符和空白字符都应保持稳定。
- system prompt 变更应显式版本化，例如 `system_prompt_version = agent-research-v3`。
- 请求级动态指令放到后续 user message 中，尽量不要污染 tools + system 这段最昂贵的共享前缀。

推荐结构：

```text
messages = [
  {"role": "system", "content": stable_agent_policy},
  {"role": "user", "content": stable_shared_context + "\n\n" + request_specific_input}
]
```

不推荐结构：

```text
messages = [
  {"role": "system", "content": stable_agent_policy + current_time + user_profile + request_flags},
  {"role": "user", "content": request_specific_input}
]
```

#### 6.2.3 稳定 Chat Template 参数

vLLM 会将 request 里的 `chat_template_kwargs` 与 server 侧默认值合并后传给渲染链路。对 Qwen3.6，这类参数会影响最终 prompt/token ids，因此也应纳入 prefix profile。

- `enable_thinking`、`add_generation_prompt`、多模态相关模板参数应保持同一 profile 内稳定。
- 不建议 Agent 每次请求传自定义 `chat_template`；生产中应使用服务端固定模板。
- 如果使用 `cache_salt` 做缓存隔离，不要按请求随机生成。随机 salt 会主动切断跨请求复用。它更适合用于租户隔离、安全隔离或灰度隔离。

#### 6.2.4 稳定共享上下文的位置、顺序和格式

长上下文 Agent 往往会携带产品文档、工具使用示例、历史任务摘要或 RAG 检索内容。Prefix Caching 只能复用“完全相同的前缀”，因此共享上下文应尽量前置且规范化，请求级内容后置。

- 对公共长文档，保持文档排序、标题、分隔符、空白字符稳定。
- 对 RAG 结果，区分“公共热文档”和“请求级检索结果”。公共热文档可以进入稳定前缀，请求级检索结果放在更靠后的 user 内容中。
- JSON、YAML、Markdown 表格等结构化内容应做 canonical serialization，避免字段顺序或缩进变化破坏命中。
- Agent 多轮对话中，如果需要压缩历史，压缩摘要的格式要稳定；摘要内容本身变化不可避免，但不要让摘要前面的系统策略和工具定义一起变化。

可以把一个请求拆成两段思考：

```text
稳定 prefix：
  tools
  system prompt
  common docs
  common examples

动态 suffix：
  current user question
  per-request retrieved chunks
  current memory / scratchpad
```

#### 6.2.5 网关按 Prefix Profile 路由

KV Cache 池化解决的是容量问题，Agent 和网关还要共同解决“同类前缀能否相遇”的问题。推理网关应在请求进入 vllm-ascend 前计算 prefix profile，并将同类请求尽量路由到相同实例或相同 pool 热区。

建议路由 key 至少包含：

```text
prefix_profile_key =
  model_id
  + tokenizer_revision
  + chat_template_version
  + tool_schema_version
  + system_prompt_version
  + thinking_mode
  + tenant_cache_namespace
```

其中 `tenant_cache_namespace` 用于安全隔离。跨租户共享 prefix cache 之前，需要确认稳定 prefix 中没有租户私有信息；否则应使用独立 namespace 或 `cache_salt` 隔离。

#### 6.2.6 可观测性与验收指标

Agent 设计是否利于 Prefix Caching，应通过 token 级指标验证，而不是只看请求字段。

- 记录渲染后 token prefix 的 hash、公共前缀 token 长度、命中 blocks 数、DRAM/HBM 命中来源。
- 统计 `tool_schema_hash`、`system_prompt_hash`、`chat_template_kwargs_hash` 的基数。如果这些 hash 在同一 Agent 内高度发散，说明 Agent 请求设计正在破坏缓存。
- 对典型 32K、64K、128K 输入，分别观察 TTFT、prefill tokens saved、DRAM load latency、layerwise overlap 收益。
- 在发布新工具或新 system prompt 前，用离线 canary 渲染比较新旧请求的 token ids，确认预期的 cache-busting 范围。

### 6.3 设计结论

对 Qwen3.6 Agent 场景，Prefix Caching 命中率的第一优先级是稳定 tools，第二优先级是稳定 system prompt，第三优先级是稳定共享上下文格式。KV Cache 池化提供更大的 DRAM 热集，但如果 Agent 每次请求都改变最前面的 tools 或 system，池化只能缓存大量彼此不共享的前缀，无法充分转化为 TTFT 收益。

## 7. 总结与待确认问题

### 7.1 关键解读

- 128-token GDN checkpoint 的显存成本很高。它对 block-aligned 恢复很友好，但会把四卡实测工程预算下的容量从不保存 GDN checkpoint 时的约 **1.15M tokens** 降到约 **60K tokens**。
- 1,024-token checkpoint 是更均衡的点：linear checkpoint 摊销成本为 144 KiB/token，四卡实测工程预算下容量约 **352K tokens**。
- TP=8 不能简单按 TP=4 容量翻倍。由于 Qwen3.6 27B Full Attention 只有 4 个 KV heads，TP=8 时 Full KV cache 在物理上复制，Full KV 物理总量从 64 KiB/token 变为 128 KiB/token。
- 不保存 GDN checkpoint 时，TP=8 虽然总 cache/state 预算翻倍，但 Full KV 物理开销也翻倍，所以最大可缓存 tokens 仍约 **1.15M**，与 TP=4 基本相同。
- 保存 GDN checkpoint 时，TP=8 仍有一定收益，因为 GDN SSM state 的 48 个 value heads 可以按 8 卡切分；例如 1,024-token checkpoint 下，TP=8 容量约 **539K tokens**，高于 TP=4 的 **352K tokens**，但不是 2 倍。
- 4,096-token 或 8,192-token checkpoint 会让 GDN checkpoint 开销低于 Full KV，但恢复时可能需要从最近 checkpoint 之后重算更多 GDN prefix tokens。

### 7.2 待确认问题

- GDN checkpoint 应该放在 HBM、host DRAM，还是分层缓存池？
- checkpoint 间隔应该全局固定，还是根据 prefix 热度和长度自适应？
- 生产八卡部署的单卡可用 KVCache 是否同样约为 17.48G，还是会因通信、runtime 或 graph 开销发生变化？
- live GDN state 与 prefix-checkpoint GDN state 应该共享淘汰池，还是保持独立管理？
