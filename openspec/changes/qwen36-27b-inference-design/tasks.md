## 1. 前置调研确认

- [ ] 1.1 确认上游 vLLM 是否已注册 Qwen3.6-27B 模型类，记录类名、模块路径和 `model_type`
- [ ] 1.2 获取 Qwen3.6-27B `config.json`，确认层数、Full Attention/Linear Attention 分布、head_dim、num_heads、KV heads 和 GDN state 维度
- [ ] 1.3 确认 Qwen3.6-27B MTP head 结构、权重命名、embedding/lm_head 共享方式和 speculative method 名称
- [ ] 1.4 确认 W8 动态量化权重格式、量化描述文件字段和 vllm-ascend quantization 解析路径
- [ ] 1.5 确认 300IDuo 目标 CANN/torch-npu 版本对 `torch.npu.graph` 或 TorchAir graph 的支持范围
- [ ] 1.6 在当前仓库外确认 `NetrsnQwenLargeService`、`NetrsnQwenMoeMediumService` 和产品 CRD 的实际字段名

## 2. Runtime Routing 与打包

- [ ] 2.1 定义 `spec.vllmVersion=0.13.0` 到 0.13.0 vLLM/vllm-ascend、`netrsnpython3rd` 和既有模型镜像的路由规则
- [ ] 2.2 定义 `spec.vllmVersion=0.18.0` 或 `0.19.x.rcx` 到 Qwen3.6 候选 vLLM/vllm-ascend、`netrsnpython3rdadvance` 和 Qwen3.6 镜像的路由规则
- [ ] 2.3 在外部微服务或部署层输出最终 vLLM 版本、vllm-ascend 版本、RTSP 包名和镜像标签
- [ ] 2.4 为 Qwen3.6 候选版本线构建预编译 wheel、AscendC `.so` 和必要 Triton cache，确保运行态不触发源码编译
- [ ] 2.5 对 Qwen3.6 候选版本线镜像或 RTSP 包执行运行态依赖扫描，证明无 GCC 依赖

## 3. 910B4 推理实现与验证

- [ ] 3.1 基于 Qwen3.6 模型类实现或调整 worker patch；相邻模型实现只能作为代码锚点，不能作为 Qwen3.6 事实依据
- [ ] 3.2 验证 910B4 x4、TP=4、FP16 eager baseline 正确性
- [ ] 3.3 验证 W8 动态量化加载、quant fusion 和输出精度
- [ ] 3.4 启用 Chunk Prefill，验证 32K 输入下 prefill 正确性、显存峰值和调度行为
- [ ] 3.5 启用 ACLGraph Full decode，验证 uniform decode batch 的 capture/replay 和 eager fallback 边界
- [ ] 3.6 接入 Qwen3.6 MTP，验证 draft、verify、accepted tokens、rejection sampling 和 GDN state 一致性
- [ ] 3.7 启用 Prefix Cache，分别验证未命中、命中和关闭 Prefix Cache 的输出一致性
- [ ] 3.8 验收 910 性能基线：启动 <= 8 min，关闭 Prefix Cache 时 32K 输入、10K 输出、TTFT <= 5000 ms、TPOT <= 20 ms/字符

## 4. 310P 推理实现与验证

- [ ] 4.1 验证 300IDuo x2、TP=2、FP16 eager baseline 正确性
- [ ] 4.2 验证 310P W8 动态量化权重加载和输出精度
- [ ] 4.3 将 `chunk_gated_delta_rule_fwd` 从 PyTorch fallback 替换或新增为 310P AscendC kernel，并保留 fallback 对照测试
- [ ] 4.4 实现或适配 310P `rmsnormgated` fused kernel，覆盖 RMSNorm + gate 路径
- [ ] 4.5 实现或适配 310P `fused_gdn_gating` fused kernel，覆盖 exp、softplus、mul、sigmoid 路径
- [ ] 4.6 实现 310P `split_qkv_rmsnorm_Mrope`，替代 910 Triton 参考路径
- [ ] 4.7 校验 `transposeKV` 在 310P 上的 tiling、正确性和端到端收益
- [ ] 4.8 对五个新增算子做逐算子正确性和 micro benchmark，并输出与 PyTorch fallback 的误差
- [ ] 4.9 输出 310P 端到端 Profiling，证明新增算子整网耗时占比相对 910 不超过 30%
- [ ] 4.10 验收 310 性能基线：启动 <= 13.5 min，关闭 Prefix Cache 时 4K 输入、4K 输出、TTFT <= 4000 ms、TPOT <= 60 ms/字符

## 5. 精度对齐与 Profiling 闭环

- [ ] 5.1 建立统一测试集，覆盖普通问答、长上下文、CoT 开/关、MTP 开/关、Prefix Cache 命中/未命中
- [ ] 5.2 记录 FP16 baseline 的输出 token、关键 logits 或业务指定精度指标
- [ ] 5.3 对 W8 动态量化输出做白盒或黑盒精度对齐，输出误差报告
- [ ] 5.4 生成 910 分段 Profiling 报告，覆盖启动、prefill、decode、MTP draft、verify、sampling、通信、KV/SSM state 更新
- [ ] 5.5 生成 310 分段 Profiling 报告，覆盖五个新增算子和 fallback 路径耗时
- [ ] 5.6 固化测试元数据记录格式，包含模型 revision、vLLM/vllm-ascend 版本、CANN/torch-npu 版本、硬件型号、量化格式和所有关键开关

## 6. 文档与验收交付

- [ ] 6.1 补充 Qwen3.6-27B 部署教程，列出 910 和 310 推荐启动参数
- [ ] 6.2 补充故障定位章节，覆盖启动超时、GCC 依赖、graph capture 失败、MTP 输出不一致、Prefix Cache 命中异常和 310P fallback 性能不达标
- [ ] 6.3 将 910、310、打包、安全扫描和精度报告归档到发布验收记录
- [ ] 6.4 执行 OpenSpec 状态检查，确认 `proposal`、`design`、`specs`、`tasks` 均为 done
