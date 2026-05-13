# Qwen3.6-27B 推理设计文档

## Context

Qwen3.6-27B 的推理交付同时受模型能力、硬件规格、版本打包、安全红线和业务开关约束。本设计从用户给定指标和当前仓库代码事实独立推导。本设计只覆盖 vllm-ascend 侧和外部微服务的接口契约，不描述当前仓库中不存在的微服务内部实现。

### 1. 设计目标与验收指标

| 维度 | Ascend 910 | Ascend 310 |
|---|---:|---:|
| 硬件 | 910B4 x4 | 300IDuo x2 |
| 显存 | 32G/卡 | 96G |
| CPU | 单实例 16 核，峰值 32 核 | 单实例 16 核 |
| 内存 | 150G，峰值 200G | 150G |
| 并发 | 4 | 4 |
| 量化 | W8 动态量化 | W8 动态量化 |
| 启动就绪 | <= 8 min | <= 13.5 min |
| 基线输入/输出 | 输入 32K，输出 10K | 输入 4K，输出 4K |
| TTFT 基线 | <= 5000 ms，关闭 Prefix Cache | <= 4000 ms，关闭 Prefix Cache |
| TPOT 基线 | <= 20 ms/字符 | <= 60 ms/字符 |
| 最大输出 | 64K | 64K |
| 最大上下文 | 128K | 128K |

Prefix Cache 的收益不得用于替代上述关闭 Prefix Cache 的 TTFT 基线。开启 Prefix Cache 后单独验收命中率、TTFT 改善和输出一致性。

### 当前代码事实

- Patch 加载入口在 `vllm_ascend/patch/worker/__init__.py`。其中 310P 与非 310P 路径存在差异，任何 Qwen3.6 patch 都必须显式处理平台分支。
- 当前可参考的相邻实现包括 `vllm_ascend/patch/worker/patch_qwen3_5.py`，但它只能作为代码锚点，不能替代 Qwen3.6 的 `config.json`、模型类和权重结构确认。
- NPU runner 的 scheduler 到 attention metadata 链路在 `vllm_ascend/worker/model_runner_v1.py`。该文件处理 chunked prefill、spec decode metadata、GDN metadata、KV cache group 和 hybrid block 场景。
- ACLGraph 抽象在 `vllm_ascend/compilation/acl_graph.py`。`ACLGraphWrapper` 按 forward context 中的 `cudagraph_runtime_mode` 和 `batch_descriptor` 进行 capture/replay，不负责持久化 runtime input buffer。
- MTP/spec decode 分发在 `vllm_ascend/spec_decode/__init__.py`。`method in ("eagle", "eagle3", "mtp")` 会路由到 `AscendEagleProposer`。
- MTP draft 模型 embedding/head 共享和 full graph draft capture 逻辑在 `vllm_ascend/spec_decode/eagle_proposer.py`。
- 310P GDN 当前 eager 路径在 `vllm_ascend/_310p/ops/fla/gdn_310.py`，调用 `fused_gdn_gating_pytorch`、`chunk_gated_delta_rule_pytorch` 和 `fused_recurrent_gated_delta_rule_pytorch`。
- `chunk_gated_delta_rule_pytorch` 位于 `vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py`，当前为 PyTorch 实现，内部包含 chunk size 64、变长 TND 输入归一化、float32 recurrent state 计算。
- W8A8 动态量化相关融合 pass 在 `vllm_ascend/compilation/passes/norm_quant_fusion_pass.py`，`AddRMSNormQuantFusionPass` 注册 dynamic quant pattern。
- Profiling 符号配置在 `vllm_ascend/profiling_config.py`，已覆盖 scheduler、KV cache、model execute、NPUModelRunner、MTP 和 rejection sampler。
- CoT 调用级透传已有测试证据：`tests/e2e/singlecard/test_qwen3_multi_loras.py` 调用 `llm.chat(..., chat_template_kwargs={"enable_thinking": False})`。
- `NetrsnQwenLargeService`、`NetrsnQwenMoeMediumService`、`netrsnpython3rdadvance`、`spec.vllmVersion` 未在当前仓库代码中找到实现，只能作为外部微服务和打包系统契约处理。

## Goals / Non-Goals

**Goals:**

- 给出 Qwen3.6-27B 在 910B4 x4 与 300IDuo x2 上的可实施推理设计。
- 明确 0.13.0 与 Qwen3.6 候选版本线的路由和 RTSP 包选择规则；候选版本线当前按 0.18.0 设计，但可切换为 0.19.x.rcx。
- 明确 Qwen3.6 候选版本线 `netrsnpython3rdadvance` 的运行态无 GCC 依赖验收方式。
- 明确 W8 动态量化、Prefix Cache、MTP、ACLGraph Full decode、Chunk Prefill、CoT 控制的配置边界。
- 明确 310P 必补算子、接口落点和 Profiling 验收方式。
- 建立 FP16 baseline、W8 动态量化精度对齐、Profiling 和性能回归闭环。

**Non-Goals:**

- 不实现 `NetrsnQwenLargeService` 或 `NetrsnQwenMoeMediumService` 内部逻辑。
- 不在设计中假定 Qwen3.6 的最终 `config.json`、MTP head 结构或 GDN 参数已经确定；这些信息必须在前置调研中验证。
- 不修改上游 vLLM 模型注册机制；若上游未注册 Qwen3.6，按任务先确认模型类和 patch 入口。
- 不用 Prefix Cache 命中结果替代关闭 Prefix Cache 的性能基线。
- 不把 310P graph 作为首个可交付底座；310P 先以 eager 正确性和关键算子性能为交付底线。

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

#### 310 路径：两阶段推进

310 侧分两步推进：第一步补齐 GDN 和 MRoPE 等关键算子，先实现 Qwen3.6-27B 的基本推理性能；第二步在算子路径稳定后，再使能 Chunk Prefill、Prefix Caching、MTP 等关键特性，进一步提升长上下文和 decode 性能。

#### 关键风险：think token 膨胀

**Qwen3.6 的核心时延风险不是单 token 执行，而是 think 过程显著拉长输出链路。** 与 Qwen32B 相比，Qwen3.6 的 think 过程预计需要约 4 倍 token，这会直接放大端到端时延压力。即使使能 MTP、Prefix Caching、ACLGraph 等加速特性，端到端时延相对 Qwen32B 仍可能存在 **30%～50% 差距**。

因此，性能验收必须同时记录生成 token 数、think token 占比、TTFT、TPOT 和端到端耗时，不能只看单项算子或单 token 指标。

设计选择：微服务只负责产品 CR 到运行包的选择，不把模型执行策略散落到微服务。执行策略必须通过 vLLM 参数、`additional_config`、环境变量和模型配置进入 vllm-ascend。

替代方案：在微服务中为 Qwen3.6 写专用分支。该方案会把模型运行语义从 vllm-ascend 扩散到服务层，版本升级时难以验证，不采用。

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

替代方案：在启动时按机器环境即时编译自定义算子。该方案违反 Qwen3.6 候选版本线 GCC 安全红线，不采用。

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

- 硬件：300IDuo 96G。
- dtype：W8 动态量化作为性能目标。
- 首个交付底座：eager correctness。
- Graph：在 CANN/torch_npu 对 310P graph 能力确认后分阶段启用。

#### 算子补齐

310P 第一阶段目标是补齐 Qwen3.6-27B 推理所需的关键算子，先保证 GDN、MRoPE、KV 转置等路径具备可验证的正确性和基础性能。

| 算子 | 当前锚点或参考 | 目标 |
|---|---|---|
| `chunk_gated_delta_rule_fwd` | `_310p/ops/fla/chunk_gated_delta_rule.py` PyTorch 实现 | AscendC prefill 主路径 |
| `rmsnormgated` | `_310p/ops/layernorm.py` 和通用 RMSNorm 路径 | RMSNorm + gate fused |
| `fused_gdn_gating` | `_310p/ops/fla/fused_gdn_gating.py` PyTorch 实现 | exp/softplus/sigmoid fused |
| `split_qkv_rmsnorm_Mrope` | `ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py` 910 Triton 参考 | 310P AscendC 版本 |
| `transposeKV` | `VLLM_ASCEND_FUSION_OP_TRANSPOSE_KV_CACHE_BY_BLOCK` 相关路径 | 校验 310P tiling 与端到端收益 |

#### 特性适配

310P 第二阶段目标是在算子路径稳定后启用关键推理特性。Chunk Prefill 用于控制长上下文 prefill 峰值和调度粒度；Prefix Caching 用于降低重复前缀输入的 TTFT；MTP 用于减少 decode 阶段主模型推进次数。上述特性必须逐项启用和验证，再进入组合验收。

Graph 推进顺序：

1. 验证 310P `torch.npu.graph`/TorchAir 支持范围。
2. 仅 Full Attention decode graph，GDN 仍走 eager。
3. GDN state 双缓冲或 input/output state 拆分，解决 inplace state 更新和地址稳定性。
4. 全模型 graph + spec decode graph。

替代方案：310P 直接对全模型做 graph capture。当前 GDN recurrent state、conv state、变长 metadata 和 PyTorch 路径均可能破坏 graph 稳定性，不采用。

### 6. 精度、Profiling 与性能闭环

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

每个组合必须记录 prompt、采样参数、模型 revision、vLLM/vllm-ascend 版本、CANN/torch-npu 版本、硬件型号和关键环境变量。

## Risks / Trade-offs

- [Qwen3.6 最终结构未确认] -> 前置调研必须确认 `config.json`、模型类、GDN 参数和 MTP head；未确认前不得把任何相邻模型 patch 作为 Qwen3.6 事实。
- [外部微服务不在仓库内] -> 文档只定义 CR 路由契约和验收输出，不描述服务内部实现。
- [Qwen3.6 候选版本线运行态 GCC 红线] -> 所有 C++/AscendC/Triton 编译动作前移到构建期；运行态扫描作为发布门禁。
- [310P 算子周期长] -> 先交付 eager 正确性和 profiling，再按端到端瓶颈优先级开发 `chunk_gated_delta_rule_fwd`、`split_qkv_rmsnorm_Mrope` 等高优先级算子。
- [MTP 与 GDN state 交互复杂] -> 先验证非 MTP，再验证 MTP eager，最后叠加 full graph；每一步固定 accepted token、SSM state 和输出一致性检查。
- [Prefix Cache 与 Chunk Prefill 组合可能影响正确性] -> Prefix Cache 性能收益独立验收；关闭 Prefix Cache 的 TTFT 仍作为主基线。

## Migration Plan

1. 新建 Qwen3.6 候选版本线专用镜像和 `netrsnpython3rdadvance` 包，不影响 0.13.0 既有模型路径；候选版本可为 0.18.0 或 0.19.x.rcx。
2. 在灰度环境按 CR `spec.vllmVersion=0.18.0` 或 `0.19.x.rcx` 路由 Qwen3.6 实例，同时保留 `0.13.0` 既有模型路径。
3. 先启用 FP16 eager，随后逐项启用 W8 动态量化、Chunk Prefill、ACLGraph、Prefix Cache、MTP；每次只打开一个新增变量。
4. 若 Qwen3.6 候选版本线运行态依赖或性能不达标，通过 CR 切换到不承载 Qwen3.6 的 0.13.0 路径或禁用 Qwen3.6 实例。

## Open Questions

- Qwen3.6-27B 上游 vLLM 模型类名、模块路径和 `model_type`。
- Qwen3.6-27B `config.json` 中 Full Attention/Linear Attention 层分布、head_dim、num_heads、GDN state 维度。
- Qwen3.6 MTP head 结构、权重命名和是否复用主模型 embedding/lm_head。
- 300IDuo 目标 CANN/torch-npu 版本是否支持可用的 graph capture/replay。
- W8 动态量化权重格式是 W8A8、W8A16 还是产品自定义格式；最终以实际量化描述文件为准。
