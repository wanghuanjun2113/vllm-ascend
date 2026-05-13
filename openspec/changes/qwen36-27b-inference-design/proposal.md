## Why

Qwen3.6-27B 需要在 Ascend 910B4 与 300IDuo 两类硬件上形成可验收的推理方案，覆盖动态量化、流水线、Prefix Cache、MTP、ACLGraph、Chunk Prefill、CoT 控制、精度对齐与 Profiling 闭环。本 change 从需求指标和当前仓库代码事实独立定义设计。

## What Changes

- 新增独立的《Qwen3.6-27B 推理设计文档》，明确 910B4 x4 与 300IDuo x2 的部署拓扑、启动参数、性能门禁和验收方式。
- 定义微服务 CR 参数到 vLLM/vllm-ascend/RTSP 包/镜像的动态路由契约；`NetrsnQwenLargeService` 与 `NetrsnQwenMoeMediumService` 未在当前仓库中找到实现，设计仅覆盖外部接口边界。
- 规定 0.13.0 与 Qwen3.6 候选版本线双路径选型：0.13.0 支持 Qwen3.6 以外模型；Qwen3.6 候选版本线当前按 0.18.0 设计，但允许切换为 0.19.x.rcx，并新增 `netrsnpython3rdadvance` RTSP 包。
- 规定 Qwen3.6 候选版本线运行态安全红线：不得依赖 GCC；wheel、AscendC `.so`、Triton cache 均需在构建期完成编译或预热。
- 明确 910 推理链路：W8 动态量化、ACLGraph Full decode、Chunk Prefill、MTP、Prefix Cache 与 CoT 控制。
- 明确 310 推理链路：先交付 eager 正确性，再推进 graph；补齐 `chunk_gated_delta_rule_fwd`、`rmsnormgated`、`fused_gdn_gating`、`split_qkv_rmsnorm_Mrope`、`transposeKV` 五个算子。
- 建立精度和性能闭环：FP16 baseline、W8 动态量化精度对齐、CoT/MTP/Prefix Cache 组合验证，以及按启动、prefill、decode、draft、verify、sampling、通信、KV/SSM state 分段 Profiling。

## Capabilities

### New Capabilities

- `qwen36-runtime-routing`: 产品 CR 参数驱动 vLLM/vllm-ascend/RTSP 包/镜像选择，覆盖 0.13.0 与 Qwen3.6 候选版本线。
- `qwen36-910-inference`: Qwen3.6-27B 在 910B4 x4 上的推理配置、优化特性和性能验收。
- `qwen36-310p-inference`: Qwen3.6-27B 在 300IDuo x2 上的 eager/graph 推理路径、算子要求和性能验收。
- `qwen36-quantization-accuracy`: FP16 baseline、W8 动态量化、CoT/MTP/Prefix Cache 组合下的精度对齐和 Profiling 闭环。
- `qwen36-packaging`: RTSP 包、运行态依赖、GCC 去依赖和版本包验收要求。

### Modified Capabilities

<!-- 无现有 OpenSpec 基线能力需要修改；本 change 新增独立能力规格。 -->

## Impact

- 文档产物：新增 OpenSpec proposal、design、tasks 和 capability specs。
- 代码锚点：设计引用 `vllm_ascend/patch/worker/__init__.py`、`vllm_ascend/patch/worker/patch_qwen3_5.py`、`vllm_ascend/worker/model_runner_v1.py`、`vllm_ascend/compilation/acl_graph.py`、`vllm_ascend/spec_decode/__init__.py`、`vllm_ascend/_310p/ops/fla/gdn_310.py`、`vllm_ascend/_310p/ops/fla/chunk_gated_delta_rule.py`、`vllm_ascend/profiling_config.py`。
- 外部系统：`NetrsnQwenLargeService`、`NetrsnQwenMoeMediumService`、产品 CRD、RTSP 打包流水线和镜像构建系统。
- 验收依赖：910B4 x4、300IDuo x2、Qwen3.6-27B FP16/W8 权重、Profiling 工具链和安全扫描工具。
