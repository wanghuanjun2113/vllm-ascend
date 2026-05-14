## Why

Qwen3.6-27B 需要在 Ascend 910B4 与 300IDuo 两类硬件上形成可验收的推理方案，覆盖动态量化、流水线、Prefix Cache、MTP、ACLGraph、Chunk Prefill、CoT 控制、精度对齐与 Profiling 闭环。本 change 是总括性设计文档，整合 310P 算子开发、310P 关键特性使能、混合注意力缓存优化和 KV Cache 容量分析四个子领域的完整设计。

## What Changes

### Runtime Routing 与打包
- 定义微服务 CR 参数到 vLLM/vllm-ascend/RTSP 包/镜像的动态路由契约；`NetrsnQwenLargeService` 与 `NetrsnQwenMoeMediumService` 未在当前仓库中找到实现，设计仅覆盖外部接口边界。
- 规定 0.13.0 与 Qwen3.6 候选版本线双路径选型：0.13.0 支持 Qwen3.6 以外模型；Qwen3.6 候选版本线当前按 0.18.0 设计，但允许切换为 0.19.x.rcx，并新增 `netrsnpython3rdadvance` RTSP 包。
- 规定 Qwen3.6 候选版本线运行态安全红线：不得依赖 GCC；wheel、AscendC `.so`、Triton cache 均需在构建期完成编译或预热。

### 910 推理链路
- 明确 910 推理链路：W8 动态量化、ACLGraph Full decode、Chunk Prefill、MTP、Prefix Cache 与 CoT 控制。
- 极低时延方案：910B4 x8、batchSize=1、Prefix Caching + MTP 叠加，目标 TTFT <= 1.5s（Prefix Cache 命中）、TPOT <= 10ms/token。

### 310P 算子开发（qwen36-27b-310p-operators）
- GDN 核心算子（已具备）：`causal_conv1d_fwd_kernel`、`causal_conv1d_update_kernel`、`fused_recurrent_gated_delta_rule_fwd_kernel`
- GDN 核心算子（开发中）：`chunk_gated_delta_rule_fwd`（SAIE）、`fused_sigmoid_gating_delta_rule_310`（SAIE+NAIE）
- QKV/RoPE 算子：`mRope`（高优先级）、`split_qkv_rmsnorm_Mrope`、`rmsnormgated`、`transposeKV`（NAIE 开发）
- 注意力/投机推理算子：`fused_gdn_gating`（高优先级）、FIA 压缩 mask、rejection sampling 系列（greedy/random/block_verify/recovered_tokens）、`npu_copy_and_expand_eagle_inputs`、`prepare_inputs_padded`、`apply_sampling_constraints`

### 310P 关键特性使能（qwen36-310p-key-features）
- Chunk Prefill：Full Attention SplitFuse（310P NZ 格式 mask 适配）+ GDN 层 chunk prefill（跨 chunk SSM state 传递）
- Prefix Caching：Full Attention block-level（复用 vLLM 框架）+ GDN SSM State Checkpoint（SSMStatePool 全新实现）
- MTP 投机推理：Attention 后端 SpecDecoding 支持修复、`npu_copy_and_expand_eagle_inputs` PyTorch fallback、GDN multi-query 路径、rejection sampling PyTorch fallback

### 混合注意力缓存优化（qwen36-hybrid-attn-cache-optimization）
- 修改算子支持非连续 block 访问，消除 KV cache 和 GDN SSM state 之间的 padding blocks
- 恢复 attention block_size 为 kernel 原生 128 tokens（当前被膨胀到 1536）
- 同一 `kv_cache_tensor` 内划分为 KV Cache Region、SSM State Region、Conv State Region 三个独立子区域

### KV Cache 容量分析（qwen36-kvcache-capacity-analysis）
- Full Attention KV cache 单 token 开销 64 KiB/token（TP=4），GDN SSM checkpoint 144 MiB/checkpoint
- 910B4 x4 (TP=4) 实测工程预算 69.92 GiB：不保存 GDN checkpoint 时可缓存 ~1.15M tokens
- KV Cache 池化方案：单机 2 个 TP=4 PD 混合实例共享 1 个 Mooncake DRAM pool（>=512GiB），仅 Full KV 可缓存 ~128 个 64K prefix
- 实测 TTFT 收益：64K 输入 90% DRAM 命中时 TTFT 从 38.3s 降至 14.4s（-62.4%，2.66x）
- Agent 请求设计指导：稳定 tools > 稳定 system prompt > 稳定共享上下文格式

### 精度与性能闭环
- 建立精度和性能闭环：FP16 baseline、W8 动态量化精度对齐、CoT/MTP/Prefix Cache 组合验证
- 按启动、prefill、decode、draft、verify、sampling、通信、KV/SSM state 分段 Profiling

## Capabilities

### New Capabilities

- `qwen36-runtime-routing`: 产品 CR 参数驱动 vLLM/vllm-ascend/RTSP 包/镜像选择，覆盖 0.13.0 与 Qwen3.6 候选版本线
- `qwen36-910-inference`: Qwen3.6-27B 在 910B4 x4 上的推理配置、优化特性和性能验收
- `qwen36-310p-inference`: Qwen3.6-27B 在 300IDuo x2 上的 eager/graph 推理路径、算子要求和性能验收
- `qwen36-quantization-accuracy`: FP16 baseline、W8 动态量化、CoT/MTP/Prefix Cache 组合下的精度对齐和 Profiling 闭环
- `qwen36-packaging`: RTSP 包、运行态依赖、GCC 去依赖和版本包验收要求
- `310p-gdn-ops`: GDN 混合注意力核心算子集（causal_conv1d、gated_delta_rule、sigmoid_gating），含 prefill/decode 双路径
- `310p-qkv-rope-ops`: QKV 处理与 RoPE 编码算子集（mRope、split_qkv_rmsnorm_Mrope、rmsnormgated、transposeKV）
- `310p-attention-spec-ops`: 注意力优化与投机推理算子集（FIA compressed mask、rejection sampling 系列）
- `310p-chunk-prefill`: 310P 上 Qwen3.6 27B 的 Chunk Prefill 支持，含 Full Attention splitfuse + GDN chunk 并行
- `310p-prefix-cache`: 310P 上混合注意力的 Prefix Caching，含 Full Attn block-level 复用 + GDN SSM State Checkpoint
- `310p-mtp`: 310P 上 Qwen3.6 MTP 投机推理，含 draft proposer + verify + rejection sampling
- `non-contiguous-kv-cache-access`: 修改 KV cache 相关算子支持非连续 block 访问，消除 padding blocks
- `hybrid-cache-padding-removal`: 移除 hybrid layout 中的 padding 逻辑，恢复 block_size=128
- `qwen36-cache-capacity`: KV Cache/SSM state 容量分析、DRAM 池化方案和 Agent 请求设计指导

### Modified Capabilities

<!-- 无现有 OpenSpec 基线能力需要修改；本 change 新增独立能力规格。 -->

## Impact

- 文档产物：新增 OpenSpec proposal、design、tasks 和 14 个 capability specs
- 代码锚点：
  - 推理链路：`vllm_ascend/patch/worker/__init__.py`、`vllm_ascend/patch/worker/patch_qwen3_5.py`、`vllm_ascend/worker/model_runner_v1.py`、`vllm_ascend/compilation/acl_graph.py`、`vllm_ascend/spec_decode/__init__.py`
  - 310P 算子：`vllm_ascend/_310p/ops/fla/gdn_310.py`、`vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py`、`csrc/torch_binding.cpp`
  - Attention 后端：`vllm_ascend/_310p/attention/attention_v1.py`、`vllm_ascend/_310p/attention/attention_mask.py`
  - 缓存优化：`vllm_ascend/worker/model_runner_v1.py`、`vllm_ascend/worker/block_table.py`、`vllm_ascend/patch/platform/patch_mamba_config.py`
  - KV Transfer：`vllm_ascend/distributed/kv_transfer/`
- 外部系统：`NetrsnQwenLargeService`、`NetrsnQwenMoeMediumService`、产品 CRD、RTSP 打包流水线和镜像构建系统
- 验收依赖：910B4 x4、300IDuo x2、Qwen3.6-27B FP16/W8 权重、Profiling 工具链和安全扫描工具
- 依赖团队：SAIE（chunk_gated_delta_rule_fwd、fused_sigmoid_gating_delta_rule_310）、NAIE（fused_gdn_gating、mRope 等）、NAIE 能力中心（FIA compressed mask、rejection sampling）
