# Qwen3.6-27B 推理设计文档

## Context

Qwen3.6-27B 的推理交付同时受模型能力、硬件规格、版本打包、安全红线和业务开关约束。本设计从用户给定指标和当前仓库代码事实独立推导，整合 310P 算子开发、310P 关键特性使能、混合注意力缓存优化和 KV Cache 容量分析四个子领域。本设计只覆盖 vllm-ascend 侧和外部微服务的接口契约，不描述当前仓库中不存在的微服务内部实现。

### 1. 设计目标与验收指标

#### 通用验收边界

| 维度 | 910B4 标准路径 | 300IDuo 标准路径 |
|---|---:|---:|
| 硬件 | 910B4*4 | 300IDuo*2 |
| 显存 | 32G/卡 | 96G/卡 |
| CPU | 单实例 16 核，峰值 32 核 | 单实例 16 核 |
| 内存 | 150G，峰值 200G | 150G |
| 量化 | W8 动态量化 | W8 动态量化 |
| 启动就绪 | <= 8 min | <= 13.5 min |
| 最大输出 | 64K | 64K |
| 最大上下文 | 128K | 128K |

#### 26.3 性能基线

| 部署环境 | 模型 | 硬件 | 输入 | 输出 | 并发 | 首字 TTFT | 增量 TPOT |
|---|---|---|---:|---:|---:|---:|---:|
| 裸机容器 | Qwen3 32B | 910B4*4 | 8K | 8K | 4 | 3000 ms | 40 ms/字符 |
| 裸机容器 | Qwen3 32B | 300IDuo*2 | 4K | 4K | 2 | 5000 ms | 100 ms/字符 |

#### 630 达成目标

| 部署环境 | 模型 | 硬件 | 输入 | 输出 | 并发 | 首字 TTFT | 增量 TPOT |
|---|---|---|---:|---:|---:|---:|---:|
| 裸机容器 | Qwen3.6 27B | 910B4*4 | 8K | 8K | 8 | 2500 ms | 20 ms/字符 |
| 裸机容器 | Qwen3.6 27B | 910B4*4 | 32K | 10K | 4 | 8000 ms | 20 ms/字符 |
| 裸机容器 | Qwen3.6 27B | 300IDuo*2 | 4K | 4K | 2 | 5000 ms | 80 ms/字符 |

#### 930 达成目标

| 部署环境 | 模型 | 硬件 | 输入 | 输出 | 并发 | 首字 TTFT | 增量 TPOT |
|---|---|---|---:|---:|---:|---:|---:|
| 裸机容器 | Qwen3.6 27B | 910B4*4 | 8K | 8K | 8 | 2500 ms | 20 ms/字符 |
| 裸机容器 | Qwen3.6 27B | 910B4*4 | 32K | 10K | 4 | 5000 ms | 20 ms/字符 |
| 裸机容器 | Qwen3.6 27B | 300IDuo*2 | 4K | 4K | 4 | 4000 ms | 60 ms/字符 |

2 张 300VPro 等价于 1 张 300IDuo；4 卡 300VPro 同样适用 300IDuo*2 的性能基线。Prefix Cache 的收益不得用于替代关闭 Prefix Cache 的阶段验收基线。开启 Prefix Cache 后单独验收命中率、TTFT 改善和输出一致性。

### 当前代码事实

- Patch 加载入口在 `vllm_ascend/patch/worker/__init__.py`。其中 310P 与非 310P 路径存在差异，任何 Qwen3.6 patch 都必须显式处理平台分支。
- 当前可参考的相邻实现包括 `vllm_ascend/patch/worker/patch_qwen3_5.py`，但它只能作为代码锚点，不能替代 Qwen3.6 的 `config.json`、模型类和权重结构确认。
- NPU runner 的 scheduler 到 attention metadata 链路在 `vllm_ascend/worker/model_runner_v1.py`。该文件处理 chunked prefill、spec decode metadata、GDN metadata、KV cache group 和 hybrid block 场景。
- ACLGraph 抽象在 `vllm_ascend/compilation/acl_graph.py`。`ACLGraphWrapper` 按 forward context 中的 `cudagraph_runtime_mode` 和 `batch_descriptor` 进行 capture/replay，不负责持久化 runtime input buffer。
- MTP/spec decode 分发在 `vllm_ascend/spec_decode/__init__.py`。`method in ("eagle", "eagle3", "mtp")` 会路由到 `AscendEagleProposer`。
- MTP draft 模型 embedding/head 共享和 full graph draft capture 逻辑在 `vllm_ascend/spec_decode/eagle_proposer.py`。
- 310P GDN 当前 eager 路径在 `vllm_ascend/_310p/ops/fla/gdn_310.py`，调用 `fused_gdn_gating_pytorch`、`chunk_gated_delta_rule_pytorch` 和 `fused_recurrent_gated_delta_rule_pytorch`。
- 310P Attention 后端 `AscendAttentionBackend310` 不支持 SpecDecoding（抛出 `NotImplementedError`）。
- 310P KV cache 为 5D NZ 对齐格式：`(2, num_blocks, (num_kv_heads*head_size)//16, block_size, 16)`，block_size 受 `block_size * head_size <= 128*128` 约束。
- 混合注意力共享同一 `kv_cache_tensor`，当前通过膨胀 block_size（128→1536）和添加 conv padding 实现统一 page_size，导致显存浪费。
- W8A8 动态量化相关融合 pass 在 `vllm_ascend/compilation/passes/norm_quant_fusion_pass.py`。
- Profiling 符号配置在 `vllm_ascend/profiling_config.py`。
- CoT 调用级透传已有测试证据：`tests/e2e/singlecard/test_qwen3_multi_loras.py`。

## Goals / Non-Goals

**Goals:**

- 给出 Qwen3.6-27B 在 910B4*4 与 300IDuo*2 上的可实施推理设计。
- 明确 0.13.0 与 Qwen3.6 候选版本线的路由和 RTSP 包选择规则。
- 明确 310P 算子开发状态、接口定义和 310P 集成方案，涵盖 GDN 核心、QKV/RoPE、注意力/投机推理四领域。
- 明确 310P Chunk Prefill、Prefix Caching、MTP 三个关键特性的 910/310P 可共用部分和 310P 独有适配工作。
- 明确混合注意力缓存优化方案：非连续布局、消除 padding、恢复 block_size=128。
- 建立 KV Cache/SSM state 容量分析模型、DRAM 池化方案和 Agent 请求设计指导。
- 建立 FP16 baseline、W8 动态量化精度对齐、Profiling 和性能回归闭环。

**Non-Goals:**

- 不实现 `NetrsnQwenLargeService` 或 `NetrsnQwenMoeMediumService` 内部逻辑。
- 不在设计中假定 Qwen3.6 的最终 `config.json`、MTP head 结构或 GDN 参数已经确定。
- 不修改上游 vLLM 模型注册机制。
- 不把 310P graph 作为首个可交付底座；310P 先以 eager 正确性和关键算子性能为交付底线。
- 不涉及 BF16 支持（310P 限制）。

## Decisions

### 2. 总体方案

请求链路固定为：

<div style="width:760px;max-width:100%;aspect-ratio:16/9;border:1px solid #c9d1d9;border-radius:10px;padding:18px;box-sizing:border-box;font-size:15px;line-height:1.3;margin:10px 0;background:#f6f8fa;color:#24292f;">
  <div style="display:grid;grid-template-columns:1fr 28px 1.35fr 28px 1.45fr 28px 1.35fr;align-items:center;gap:4px;height:42%;">
    <div style="height:100%;border:1px solid #0969da;border-radius:8px;background:#ddf4ff;display:flex;align-items:center;justify-content:center;text-align:center;padding:8px;font-weight:600;color:#0a3069;">产品 CR</div>
    <div style="text-align:center;color:#0969da;font-weight:800;font-size:20px;">→</div>
    <div style="height:100%;border:1px solid #8250df;border-radius:8px;background:#fbefff;display:flex;align-items:center;justify-content:center;text-align:center;padding:8px;font-weight:600;color:#512a97;">服务路由<br><code>NetrsnQwen*</code></div>
    <div style="text-align:center;color:#0969da;font-weight:800;font-size:20px;">→</div>
    <div style="height:100%;border:1px solid #1a7f37;border-radius:8px;background:#dafbe1;display:flex;align-items:center;justify-content:center;text-align:center;padding:8px;font-weight:600;color:#116329;">运行包选择<br>vLLM / vllm-ascend / RTSP / 镜像</div>
    <div style="text-align:center;color:#0969da;font-weight:800;font-size:20px;">→</div>
    <div style="height:100%;border:1px solid #bf8700;border-radius:8px;background:#fff8c5;display:flex;align-items:center;justify-content:center;text-align:center;padding:8px;font-weight:600;color:#7d4e00;">chat template<br><code>chat_template_kwargs</code></div>
  </div>
  <div style="height:16%;display:flex;align-items:center;justify-content:flex-end;padding-right:10%;color:#0969da;font-weight:800;font-size:20px;">↓</div>
  <div style="display:grid;grid-template-columns:1fr 28px 1.25fr 28px 1.7fr 28px 1.25fr 28px 1fr;align-items:center;gap:4px;height:42%;">
    <div style="height:100%;border:1px solid #0969da;border-radius:8px;background:#ddf4ff;display:flex;align-items:center;justify-content:center;text-align:center;padding:8px;font-weight:600;color:#0a3069;">vLLM scheduler</div>
    <div style="text-align:center;color:#0969da;font-weight:800;font-size:20px;">→</div>
    <div style="height:100%;border:1px solid #8250df;border-radius:8px;background:#fbefff;display:flex;align-items:center;justify-content:center;text-align:center;padding:8px;font-weight:600;color:#512a97;"><code>NPUModelRunner</code></div>
    <div style="text-align:center;color:#0969da;font-weight:800;font-size:20px;">→</div>
    <div style="height:100%;border:1px solid #1a7f37;border-radius:8px;background:#dafbe1;display:flex;align-items:center;justify-content:center;text-align:center;padding:8px;font-weight:600;color:#116329;">执行后端<br>attention / GDN / quant / ACLGraph / MTP</div>
    <div style="text-align:center;color:#0969da;font-weight:800;font-size:20px;">→</div>
    <div style="height:100%;border:1px solid #bf8700;border-radius:8px;background:#fff8c5;display:flex;align-items:center;justify-content:center;text-align:center;padding:8px;font-weight:600;color:#7d4e00;">sampler / rejection sampler</div>
    <div style="text-align:center;color:#0969da;font-weight:800;font-size:20px;">→</div>
    <div style="height:100%;border:1px solid #0969da;border-radius:8px;background:#ddf4ff;display:flex;align-items:center;justify-content:center;text-align:center;padding:8px;font-weight:600;color:#0a3069;">response</div>
  </div>
</div>

#### 性能闭环拆分

总体实现思路是把性能目标拆到 910 和 310 两条硬件路径上分别闭环。

#### 910 路径：关键特性使能

910 侧以关键特性使能为主，优先组合 W8 动态量化、ACLGraph Full decode、Chunk Prefill、Prefix Caching 和 MTP，目标是在硬件能力充足的场景下压缩 prefill、decode 和采样链路的端到端时延。

#### 310 路径：四阶段推进

310 侧分四步推进：
1. **算子补齐**：GDN 核心、QKV/RoPE、注意力/投机推理算子开发或验证
2. **eager 正确性**：FP16/W8 eager baseline 端到端推理正确性
3. **关键特性使能**：Chunk Prefill、Prefix Caching、MTP 逐项启用
4. **性能达标**：组合验收，630 阶段 TPOT <= 80 ms/字符，930 阶段 TPOT <= 60 ms/字符

#### 关键风险：think token 膨胀

**Qwen3.6 的核心时延风险不是单 token 执行，而是 think 过程显著拉长输出链路。** 与 Qwen32B 相比，Qwen3.6 的 think 过程预计需要约 4 倍 token，这会直接放大端到端时延压力。即使使能 MTP、Prefix Caching、ACLGraph 等加速特性，端到端时延相对 Qwen32B 仍可能存在 **30%～50% 差距**。

因此，性能验收必须同时记录生成 token 数、think token 占比、TTFT、TPOT 和端到端耗时，不能只看单项算子或单 token 指标。

### 3. 版本与打包设计

版本选择规则：

| CR 字段 | vLLM/vllm-ascend | RTSP 包 | 模型范围 |
|---|---|---|---|
| `spec.vllmVersion=0.13.0` | 0.13.0 版本组合 | `netrsnpython3rd` | Qwen3.6 以外既有模型 |
| `spec.vllmVersion=0.18.0` | 0.18.0 版本组合 | `netrsnpython3rdadvance` | Qwen3.6-27B |
| `spec.vllmVersion=0.19.x.rcx` | 0.19.x.rcx 候选版本组合 | `netrsnpython3rdadvance` | Qwen3.6-27B 备选路径 |

Qwen3.6 当前按 0.18.0 包线设计；后续若产品选择 0.19.x.rcx，则 0.18.0 路径整体替换为 0.19.x.rcx 路径。除 vLLM/vllm-ascend 版本组合与镜像标签外，RTSP 包、安全扫描、GCC 红线和性能验收要求保持不变。

0.18.0 包线以及后续可能替换的 0.19.x.rcx 包线必须满足：

- 运行态不安装、不调用 GCC。
- vllm、vllm-ascend、torch-npu、CANN 依赖以 wheel 或系统基础镜像形式提前固化。
- AscendC 自定义算子以 `.so` 交付，构建期完成编译。
- Triton 路径若参与 910 推理，必须在构建期预热 cache 或在安全认可的非运行态阶段生成。
- 启动脚本在运行态只允许加载包、选择配置、启动服务，不允许触发源码编译。

### 4. 910 推理方案

910 标准交付配置：

- 硬件：910B4 32G x4。
- 量化：W8 动态量化，优先复用 `--quantization ascend` 和模型量化配置。
- Decode：ACLGraph Full decode，prefill 或 mixed batch 可退回 eager。
- Prefill：启用 Chunk Prefill，长输入按 scheduler 切分，避免单次 prefill 峰值过高。
- Prefix Cache：启用后单独验证命中收益；关闭时用于 TTFT 基线。
- MTP：入口使用现有 `mtp` speculative dispatch 到 `AscendEagleProposer`；Qwen3.6 需要独立确认模型结构、权重加载、KV/SSM state 绑定和 GDN 状态一致性。
- CoT：调用级通过 `chat_template_kwargs.enable_thinking` 控制；非调用级由服务启动默认 chat template kwargs 控制。Ascend kernel、runner、sampler 不新增 CoT 语义。

ACLGraph 约束来自 `ACLGraphWrapper`：capture/replay 依赖稳定的 batch descriptor 和 runtime input 地址。Qwen3.6 的 Full decode 只能在 uniform decode batch 下进入 full graph；prefill、chunked prefill 或 shape 不稳定路径不得强行 full graph。

MTP 约束来自 `model_runner_v1.py` 和 `eagle_proposer.py`：spec decode 会改变 draft token、logits indices、GDN metadata、accepted token 和 rejection sampling 链路。Qwen3.6 接入时必须验证 GDN `spec_token_indx`、SSM state indices、accepted token 数量与 full graph capture size 一致。

MTP 增量优化不以直接训练 Eagle3 draft 模型作为首选路径。原因是 Qwen3.6 MTP 接受率预期较高，单独训练 Eagle3 模型的边际收益有限。可评估的增量方向包括：

1. 在主模型微调已带 MTP 的基础上，对 MTP 分支单独使用领域化数据微调。
2. 使能 DFlash，提高投机步长和 draft/verify 执行速度，并对 DFlash 模型进行领域化数据微调。

#### 极低时延方案

极低时延方案面向单请求长上下文交互场景，使用 8 卡 910B4 配置承载 Qwen3.6-27B。该方案以 batchSize=1、32K 输入为目标，通过 Prefix Caching 降低重复前缀 prefill 开销，通过 MTP 降低 decode 单 token 推进次数，并叠加 ACLGraph Full decode、W8 动态量化和通信优化以压缩端到端时延。

| 指标 | 目标值 | 条件 |
|---|---:|---|
| 硬件 | 910B4 x8 | 单实例极低时延配置 |
| 输入长度 | 32K | 长上下文单请求 |
| batchSize | 1 | 低并发、低时延优先 |
| Prefix Caching | 开启 | 依赖业务前缀复用 |
| MTP | 开启 | 需验证 Qwen3.6 MTP head 与 GDN state 一致性 |
| TTFT | <= 1.5 s | Prefix Cache 命中场景 |
| TPOT | <= 10 ms/token | MTP、Full decode graph 与量化优化叠加 |

### 5. 310 推理方案

310 标准交付配置：

- 硬件：300IDuo*2，单卡显存 96G；4 卡 300VPro 可按同一基线验收。
- dtype：W8 动态量化作为性能目标。
- 首个交付底座：eager correctness。
- Graph：在 CANN/torch_npu 对 310P graph 能力确认后分阶段启用。

#### 5.1 算子补齐

310P 第一阶段目标是补齐 Qwen3.6-27B 推理所需的关键算子，先保证 GDN、MRoPE、KV 转置等路径具备可验证的正确性和基础性能。

缺失算子中，`chunk_gated_delta_rule_fwd` 对 310P TTFT 影响最大。该算子覆盖 GDN prefill 主路径，替换当前 PyTorch 路径后预计可将当前 TTFT 降低约 40%；该收益属于工程预测，必须在 4K/4K 阶段验收和 profiling 报告中单独验证。

**算子开发状态总览**：

| 算子名称 | 开发状态 | 优先级 | 开发方 |
|----------|----------|--------|--------|
| causal_conv1d_fwd_kernel | 已具备 | - | CANN/vllm-ascend |
| causal_conv1d_update_kernel | 已具备 | - | CANN/vllm-ascend |
| fused_recurrent_gated_delta_rule_fwd_kernel | 已具备 | - | CANN/vllm-ascend |
| chunk_gated_delta_rule_fwd | SAIE 开发中 | 最高 | SAIE |
| fused_sigmoid_gating_delta_rule_310 | SAIE 开发中 | - | SAIE+NAIE |
| fused_gdn_gating | NAIE 开发 | **高** | NAIE |
| mRope | NAIE 开发 | **高** | NAIE |
| FIA 算子增加压缩 mask | NAIE 能力中心 | **高** | NAIE 能力中心 |
| rmsnormgated | NAIE 开发 | 低 | NAIE |
| split_qkv_rmsnorm_Mrope | NAIE 开发 | 低 | NAIE |
| transposeKV | NAIE 开发 | 低 | NAIE |

**投机推理算子**：

| 算子名称 | 开发方 | 说明 |
|----------|--------|------|
| rejection_greedy_sample | NAIE 能力中心 | Greedy 拒绝采样（含 spec_len=1 快速路径） |
| rejection_random_sample | NAIE 能力中心 | Random 拒绝采样 |
| rejection_random_sample_block_verify | NAIE 能力中心 | Block verify 拒绝采样 |
| sample_recovered_tokens | NAIE 能力中心 | 残差分布恢复采样 |
| npu_copy_and_expand_eagle_inputs | NAIE 能力中心 | Eagle/MTP 输入扩展（当前仅 910 注册） |
| prepare_inputs_padded | PyTorch fallback | 已有 fallback 路径 |
| apply_sampling_constraints | PyTorch fallback | 已有 fallback 路径 |

**310P 平台约束**：

- 不支持 Triton，需 AscendC 或纯 PyTorch 替代
- KV cache 为 5D NZ 对齐格式：`(2, num_blocks, (num_kv_heads*head_size)//16, block_size, 16)`
- block_size 受 `block_size * head_size <= 128*128` 约束
- 当前仅 eager 模式
- 算子注册通过 `#ifdef ASCEND_PLATFORM_310P` 编译时条件选择

**算子集成架构**：

```
算子调用链：
  patch_qwen3_6.py (模型 patch)
    ├── is_310p() → AscendGatedDeltaNetAttention310
    │                 ├── npu_causal_conv1d_310 (AscendC)
    │                 ├── fused_gdn_gating_310 (NAIE AscendC / PyTorch fallback)
    │                 ├── chunk_gated_delta_rule_310 (SAIE AscendC / PyTorch fallback)
    │                 └── fused_recurrent_gated_delta_rule_310 (AscendC)
    ├── AscendMRotaryEmbedding → npu_mrope_310 (NAIE AscendC)
    └── AscendAttentionBackend310 → FIA compressed mask

  投机推理调用链：
    AscendEagleProposer._propose()
      ├── npu_copy_and_expand_eagle_inputs (C++ / PyTorch)
      ├── rejection_sample → HAS_TRITON 分支
      │     ├── rejection_greedy_sample_pytorch
      │     ├── rejection_random_sample_pytorch
      │     └── sample_recovered_tokens_pytorch
      └── apply_sampling_constraints → PyTorch in-place
```

**分发策略**：
- 每个 310P 算子提供两层实现：AscendC kernel（优先）+ PyTorch fallback（兜底）
- 通过 `is_310p()` + 环境变量 `VLLM_ASCEND_USE_ASCENDC_OPS` 控制
- 拒绝采样算子通过 `HAS_TRITON` 标志自动选择 Triton 或 PyTorch 路径

**910/310P 算子关键差异**：

GDN 算子的参数构造存在系统性差异：
- **host vs 设备参数**：910 Triton kernel 接受设备端 Tensor，310P C++ 算子接受 host 端 Python tuple/list（通过 `to_int64_tuple` 转换）
- **state 布局**：910 gqa_interleaved 路径需 `.transpose(-1, -2)` 转换 ssm_state 布局，310P 不需要转置
- **state 清零**：910 使用 `clear_ssm_states()` Triton kernel，310P 使用布尔索引 `initial_state[~has_initial_state] = 0`
- **mask 格式**：910 使用 `sparse_mode=3` 跳过 mask 生成，310P 需物化完整 NZ 格式 causal mask
- **run_mode**：910 通过选择不同函数区分 prefill/decode，310P 单一算子通过 `run_mode` 参数区分

#### 5.2 关键特性适配

310P 第二/三阶段目标是在算子路径稳定后启用关键推理特性。三个特性逐项启用、验证、再组合。

**Chunk Prefill — 310P 适配**：

- **共用框架**：vLLM V1 scheduler 的 `enable_chunked_prefill`、`max_num_batched_tokens` 对所有硬件通用
- **310P Full Attention SplitFuse**：使用 `_npu_paged_attention_splitfuse`，需构造 NZ 格式 mask（`AttentionMaskBuilder310.get_splitfuse_mask`），涉及 CPU-NPU 数据同步（`query_start_loc.to("cpu")`）
- **310P GDN 层**：使用 `chunk_gated_delta_rule_pytorch`，跨 chunk SSM state 传递通过 `initial_state` + `inplace_final_state=True` 实现

**Prefix Caching — 310P 适配**：

- **共用框架**：Full Attention 层 block-level prefix caching 使用 vLLM V1 共享框架
- **GDN SSM State Checkpoint（全新实现）**：需要新增 `SSMStatePool` 数据结构、checkpoint/restore 逻辑、LRU 淘汰策略。每个 checkpoint 包含 48 层 GDN 的 float32 h 矩阵（~144 MiB/checkpoint）
- **Block_size 对齐**：310P block_size=128 时 SSM checkpoint 间隔与 Full Attn block 边界自然对齐
- **用户可配置保存策略**：Linear Attention checkpoint 显存占用较大，不应只提供固定间隔保存。外部服务需开放 checkpoint 保存策略接口，允许用户按语义锚点保存少量 checkpoint。例如固定 tools 与 system prompt 的 Agent 场景，只在该稳定前缀末尾保存 1 个 checkpoint，后续动态 user content 从该 checkpoint 继续计算。

建议接口契约：

```yaml
linearCheckpointPolicy:
  mode: auto | disabled | interval | anchor
  intervalTokens: 128 | 1024 | 4096 | 8192
  anchors:
    - tools
    - system_prompt
    - shared_context
  maxCheckpointsPerRequest: 1
  maxCheckpointsPerPrefixProfile: 1
```

`anchor` 模式下，系统只允许在 chat template 渲染后的稳定 token 边界创建 checkpoint；checkpoint key 必须绑定 `prefix_profile_key`、token hash、模型 revision、tokenizer revision、chat template version、tool schema version、system prompt version 与 thinking mode。用户策略只能减少 checkpoint 数量或放大间隔，不能绕过全局显存上限、LRU 淘汰和正确性校验。

**MTP 投机推理 — 310P 适配**：

- **共用框架**：`AscendEagleProposer(method="mtp")` 框架跨硬件复用
- **关键缺口**：`AscendAttentionBackend310` 当前不支持 SpecDecoding，需新增 Draft/Verify 路径
- **`npu_copy_and_expand_eagle_inputs`**：当前仅 910 注册，310P 需 PyTorch fallback（推荐先用 PyTorch 保底正确性）
- **GDN multi-query 路径**：已在 `gdn_310.py` 中通过 `spec_sequence_masks` 分支实现
- **Rejection Sampling**：PyTorch fallback 已在 `rejection_sampler.py` 中实现，通过 `HAS_TRITON` 自动选择
- **增量优化方向**：Qwen3.6 MTP 接受率预期较高，直接训练 Eagle3 draft 模型收益有限；后续优先评估 MTP 分支领域化微调，或使能 DFlash 并对 DFlash 模型做领域化数据微调

**特性使能顺序**：

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

Graph 推进顺序：

1. 验证 310P `torch.npu.graph`/TorchAir 支持范围。
2. 仅 Full Attention decode graph，GDN 仍走 eager。
3. GDN state 双缓冲或 input/output state 拆分，解决 inplace state 更新和地址稳定性。
4. 全模型 graph + spec decode graph。

### 6. 混合注意力缓存优化

Qwen3.6 27B 使用 Full Attention (16 层) + Linear Attention/GDN (48 层) 混合注意力架构。当前实现中，Full Attention KV Cache 和 GDN SSM State 共享同一个 `kv_cache_tensor`。为保持共享张量的连续性，代码通过膨胀 block_size（128→1536）和添加 padding 实现统一 page_size，导致显著显存浪费。

#### 6.1 显存浪费来源

- **conv_block_page_size padding**：每 block 27 KiB，总 padding 约 309 MiB
- **block_size 膨胀碎片化**：block_size 从 128 膨胀到 1536，KVCacheManager 分配粒度为 1536 tokens/block
- **统一 page_size 限制灵活性**：KV cache 自然 page_size 128 KiB vs SSM state 自然 page_size 795 KiB，强制统一后 KV cache 利用率仅 8.2%

#### 6.2 优化方案：同一 Pool 内非连续布局

保持共享 `kv_cache_tensor`，划分为三个独立连续子区域：

```
kv_cache_tensor 布局:
┌─────────────────────────┬────────────────────────┬───────────────────┐
│  KV Cache Region        │  SSM State Region      │  Conv State Region│
│  (num_kv_blocks slots)  │  (num_ssm_slots slots)  │  (num_ssm_slots) │
│  每slot: 128 KiB        │  每slot: 768 KiB        │  每slot: 27 KiB  │
│  block_size=128         │  固定大小               │  固定大小         │
└─────────────────────────┴────────────────────────┴───────────────────┘
```

关键洞察：CANN attention 算子（`npu_fused_infer_attention_score`、`npu_paged_attention`）通过 `block_table` 间接寻址，算子本身不假设 block 连续。只要 `block_table` 中的 physical block number 正确反映新布局中的偏移，算子无需修改。

#### 6.3 需要修改的代码路径

| 文件 | 改动 |
|------|------|
| `patch/platform/patch_mamba_config.py` | 移除 `attn_block_size` 膨胀逻辑和 `mamba_page_size_padded` padding |
| `worker/model_runner_v1.py:3460-3463` | 移除 AttentionSpec `page_size_padded` 强制对齐 |
| `worker/model_runner_v1.py:2877-2901` | `_allocate_kv_cache_tensors` 按子区域分配 |
| `worker/model_runner_v1.py:3043-3084` | `_reshape_kv_cache_tensors` 按子区域切片 |
| `worker/block_table.py` | slot_mapping 计算需考虑 KV region 起始 offset |

### 7. KV Cache 容量分析与池化

#### 7.1 单 Token Cache/State 开销

- **Full Attention KV cache**：64 KiB/token（TP=4，16 层 × 2 × 4 KV heads × 256 head_dim × 2 bytes）
- **TP=8 复制**：Full Attention 4 个 KV heads < TP size，物理分配 128 KiB/token
- **GDN SSM state**：144 MiB/checkpoint（48 层 × 48 value heads × 128 × 128 × 4 bytes float32）
- **Checkpoint 间隔影响**：128-token 间隔时 linear checkpoint 为 1.125 MiB/token（是 Full KV 的 18 倍），1024-token 间隔时降为 144 KiB/token

#### 7.2 HBM 容量预算（910B4 32G）

实测四卡部署单卡可用 KVCache 17.48 GiB，总预算 69.92 GiB。

| Checkpoint 间隔 | TP=4 最大 tokens | TP=8 最大 tokens |
|---:|---:|---:|
| 不保存 GDN checkpoint | ~1.15M | ~1.15M |
| 1,024 | ~352K | ~539K |
| 4,096 | ~733K | ~893K |

关键结论：TP=8 不能简单按 TP=4 容量翻倍。由于 Full Attention 4 个 KV heads 复制，不保存 GDN checkpoint 时 TP=4 和 TP=8 最大 tokens 基本相同。

#### 7.3 DRAM 池化方案

单台 910B4 上部署 2 个 TP=4 PD 混合实例，共享 1 个 Mooncake DRAM KV Cache Pool。

512GiB DRAM pool 容量：

| 池化内容 | 可缓存 tokens | 约等价 64K prefixes |
|---|---:|---:|
| 仅 Full KV | ~8.39M | 128 |
| + GDN checkpoint=8,192 | ~6.55M | 99 |
| + GDN checkpoint=4,096 | ~5.37M | 81 |
| + GDN checkpoint=1,024 | ~2.58M | 39 |

建议先支持 Full KV 池化，再逐步加入 GDN checkpoint。checkpoint 间隔优先评估 4,096 或 8,192 tokens。

#### 7.4 Linear Attention Checkpoint 保存策略

Linear Attention checkpoint 单点开销高，固定每 128 tokens 保存会显著压缩可缓存 token 数。保存策略应从“按固定 block 全量保存”扩展为“按用户可声明的稳定前缀保存”。

推荐策略：

| 策略 | 保存行为 | 适用场景 |
|---|---|---|
| `disabled` | 不保存 GDN checkpoint，仅复用 Full KV | 显存优先、可接受 GDN prefix 重算 |
| `interval` | 按 `intervalTokens` 保存 | 长文档共享前缀，前缀长度大且复用稳定 |
| `anchor` | 仅在声明的稳定语义段末尾保存 | 固定 tools、固定 system prompt、固定共享上下文 |
| `auto` | 系统按显存预算和命中统计选择 | 默认策略，受全局上限控制 |

Agent 默认建议使用 `anchor` 策略：固定 tools 与 system prompt 只保存一个 checkpoint，动态用户输入不创建 checkpoint。该策略减少 linear checkpoint 常驻显存，同时保留稳定前缀复用收益。

#### 7.5 TTFT 收益

实测 64K 输入 90% DRAM cache 命中：TTFT 从 38.3s 降至 14.4s，相对下降 62.4%，加速比 2.66x。纯数据搬运理论量级为几十到百毫秒，主要收益来自避免大部分 prefill 计算。

#### 7.6 Agent 请求设计指导

Prefix Caching 命中的是最终渲染后的 token 前缀，不是 OpenAI request JSON 的语义。Agent 设计优先级：

1. **稳定 tools**：固定工具集合、工具顺序、tool schema，变更视为 cache-busting 版本发布
2. **稳定 system prompt**：不放动态内容，变更显式版本化
3. **稳定共享上下文**：公共文档前置且规范化，请求级内容后置

网关按 `prefix_profile_key = model_id + tokenizer_revision + chat_template_version + tool_schema_version + system_prompt_version + thinking_mode + tenant_cache_namespace` 路由。

### 8. 精度、Profiling 与性能闭环

验证顺序：

1. FP16 eager baseline：确认模型结构、权重加载、tokenizer/CoT 语义和输出正确性。
2. W8 动态量化：验证权重映射、scale/offset、quant fusion 和输出偏差。
3. 性能开关逐项启用：Chunk Prefill、ACLGraph、Prefix Cache、MTP。
4. 组合验收：910 和 310 按性能门禁运行。

Profiling 分段：

- 启动：模型加载、量化权重处理、graph capture、cache 初始化。
- Prefill：chunked prefill、full attention、GDN chunk recurrent、KV/SSM state 写入。
- Decode：ACLGraph replay、eager 路径、GDN recurrent update、通信。
- MTP：draft、verify、accepted token、rejection sampling。
- Sampling：logits、sampler、CoT token 计量。
- 通信：TP allreduce/allgather、FlashComm 或其他通信优化。
- 缓存：KV block、Prefix Cache hit、SSM state 更新。

精度对齐矩阵：

| 维度 | 组合 |
|---|---|
| dtype/quant | FP16 baseline、W8 动态量化 |
| CoT | enable_thinking=true、enable_thinking=false |
| Prefix Cache | 关闭、开启未命中、开启命中 |
| MTP | 关闭、开启 |
| Graph | eager、Full decode graph |

## Risks / Trade-offs

- [Qwen3.6 最终结构未确认] -> 前置调研必须确认 `config.json`、模型类、GDN 参数和 MTP head。
- [外部微服务不在仓库内] -> 文档只定义 CR 路由契约和验收输出。
- [Qwen3.6 候选版本线运行态 GCC 红线] -> 所有编译动作前移到构建期。
- [SAIE 算子交付节奏] -> `chunk_gated_delta_rule_fwd` 和 `fused_sigmoid_gating_delta_rule_310` 交付时间直接影响 310P 性能。缓解：PyTorch fallback 保底正确性。
- [NAIE 高优先级算子开发周期] -> `fused_gdn_gating` 和 `mRope` 依赖 NAIE 团队排期。缓解：明确接口规范，提前对齐。
- [310P AICore 架构差异] -> AscendC kernel 移植需重新调整 tiling 参数。缓解：参考已有 310P 适配经验。
- [Attention 后端 SpecDecoding 改动量] -> `AscendAttentionBackend310` 需新增 Draft/Verify 路径。缓解：参考 910 实现。
- [MTP + GDN state 交互复杂] -> 先验证非 MTP，再验证 MTP eager，最后叠加 full graph。
- [SSMStatePool 显存管理] -> 每个 checkpoint 144 MiB。缓解：LRU 淘汰 + max_checkpoints 上限。
- [Chunk Prefill + GDN state 一致性] -> 跨 chunk SSM state 传递需正确处理。缓解：逐 chunk 正确性测试。
- [FIA compressed mask 兼容性] -> 310P `_npu_flash_attention` 可能不支持 sparse_mode。缓解：维持全 mask 方案。
- [block_table 和 slot_mapping offset 正确性] -> 新布局下 offset 计算逻辑变化。缓解：逐算子单测。
- [KV Transfer 兼容性] -> mooncake_connector 假设连续 KV cache tensor。缓解：确保子区域保持标准形状。
- [KV Cache 容量 TP=8 非线性] -> Full KV heads 复制导致 TP=8 容量不是 TP=4 的 2 倍。

## Migration Plan

1. 新建 Qwen3.6 候选版本线专用镜像和 `netrsnpython3rdadvance` 包，不影响 0.13.0 既有模型路径。
2. 在灰度环境按 CR `spec.vllmVersion=0.18.0` 或 `0.19.x.rcx` 路由 Qwen3.6 实例。
3. 先启用 FP16 eager，随后逐项启用 W8 动态量化、Chunk Prefill、ACLGraph、Prefix Cache、MTP。
4. 310P 按 Phase 1-4 顺序推进：算子补齐 → eager 正确性 → 关键特性使能 → 组合验收。
5. 缓存优化在 310P 算子稳定后执行，按子区域布局迁移，对比新旧 layout 的 attention 输出一致性。
6. 若 Qwen3.6 候选版本线运行态依赖或性能不达标，通过 CR 切换到 0.13.0 路径或禁用 Qwen3.6 实例。

## Open Questions

- Qwen3.6-27B 上游 vLLM 模型类名、模块路径和 `model_type`。
- Qwen3.6-27B `config.json` 中 Full Attention/Linear Attention 层分布、head_dim、num_heads、GDN state 维度。
- Qwen3.6 MTP head 结构、权重命名和是否复用主模型 embedding/lm_head。
- 300IDuo 目标 CANN/torch-npu 版本是否支持可用的 graph capture/replay。
- W8 动态量化权重格式。
- SAIE `chunk_gated_delta_rule_fwd` AscendC 实现预计交付时间。
- NAIE 高优先级算子（`fused_gdn_gating`、`mRope`）排期和 API 草案。
- 310P `torch_npu._npu_flash_attention` 是否可扩展 `sparse_mode`。
- GDN checkpoint 存储层级（HBM/DRAM/分层）和间隔策略。
- 生产八卡部署单卡可用 KVCache 是否与四卡一致。
- `expandable_segments` 在 310P hybrid 模型上的兼容性。
- `npu_copy_and_expand_eagle_inputs` 是否计划移植到 310P。
