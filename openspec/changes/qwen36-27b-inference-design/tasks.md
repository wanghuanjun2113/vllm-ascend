## 1. 前置调研确认

- [ ] 1.1 确认上游 vLLM 是否已注册 Qwen3.6-27B 模型类，记录类名、模块路径和 `model_type`
- [ ] 1.2 获取 Qwen3.6-27B `config.json`，确认层数、Full Attention/Linear Attention 分布、head_dim、num_heads、KV heads 和 GDN state 维度
- [ ] 1.3 确认 Qwen3.6-27B MTP head 结构、权重命名、embedding/lm_head 共享方式和 speculative method 名称
- [ ] 1.4 确认 W8 动态量化权重格式、量化描述文件字段和 vllm-ascend quantization 解析路径
- [ ] 1.5 确认 300IDuo 目标 CANN/torch-npu 版本对 `torch.npu.graph` 或 TorchAir graph 的支持范围
- [ ] 1.6 在当前仓库外确认 `NetrsnQwenLargeService`、`NetrsnQwenMoeMediumService` 和产品 CRD 的实际字段名
- [ ] 1.7 与 SAIE 对齐 `chunk_gated_delta_rule_fwd` AscendC 实现的交付时间、接口草案和联调计划
- [ ] 1.8 与 SAIE/NAIE 对齐 `fused_sigmoid_gating_delta_rule_310` AscendC 实现的交付时间和接口
- [ ] 1.9 与 NAIE 对齐 `fused_gdn_gating` 高优先级算子的排期、API 草案和精度要求
- [ ] 1.10 与 NAIE 对齐 `mRope` 高优先级算子的排期和 API
- [ ] 1.11 与 NAIE 能力中心确认 310P FIA 算子是否可扩展 `sparse_mode` 参数
- [ ] 1.12 与 NAIE 能力中心对齐拒绝采样相关算子的开发动机（AscendC vs PyTorch fallback）
- [ ] 1.13 确认 NAIE 低优先级算子（`rmsnormgated`、`split_qkv_rmsnorm_Mrope`、`transposeKV`）预计交付时间

## 2. Runtime Routing 与打包

- [ ] 2.1 定义 `spec.vllmVersion=0.13.0` 到 0.13.0 vLLM/vllm-ascend、`netrsnpython3rd` 和既有模型镜像的路由规则
- [ ] 2.2 定义 `spec.vllmVersion=0.18.0` 或 `0.19.x.rcx` 到 Qwen3.6 候选 vLLM/vllm-ascend、`netrsnpython3rdadvance` 和 Qwen3.6 镜像的路由规则
- [ ] 2.3 在外部微服务或部署层输出最终 vLLM 版本、vllm-ascend 版本、RTSP 包名和镜像标签
- [ ] 2.4 为 Qwen3.6 候选版本线构建预编译 wheel、AscendC `.so` 和必要 Triton cache，确保运行态不触发源码编译
- [ ] 2.5 对 Qwen3.6 候选版本线镜像或 RTSP 包执行运行态依赖扫描，证明无 GCC 依赖

## 3. 910B4 推理实现与验证

- [ ] 3.1 基于 Qwen3.6 模型类实现或调整 worker patch
- [ ] 3.2 验证 910B4*4、TP=4、FP16 eager baseline 正确性
- [ ] 3.3 验证 W8 动态量化加载、quant fusion 和输出精度
- [ ] 3.4 启用 Chunk Prefill，验证 32K 输入下 prefill 正确性、显存峰值和调度行为
- [ ] 3.5 启用 ACLGraph Full decode，验证 uniform decode batch 的 capture/replay 和 eager 路径切换
- [ ] 3.6 接入 Qwen3.6 MTP，验证 draft、verify、accepted tokens、rejection sampling 和 GDN state 一致性
- [ ] 3.7 启用 Prefix Cache，分别验证未命中、命中和关闭 Prefix Cache 的输出一致性
- [ ] 3.8 验收 910 阶段性能：记录 26.3 Qwen3 32B 对照基线；630 目标覆盖 8K/8K 并发 8、TTFT <= 2500 ms、TPOT <= 20 ms/字符，以及 32K/10K 并发 4、TTFT <= 8000 ms、TPOT <= 20 ms/字符
- [ ] 3.9 验收 930 910 目标：关闭 Prefix Cache 时 8K/8K 并发 8、TTFT <= 2500 ms、TPOT <= 20 ms/字符；32K/10K 并发 4、TTFT <= 5000 ms、TPOT <= 20 ms/字符

## 4. 310P 已具备算子验证

- [ ] 4.1 验证 `npu_causal_conv1d_310` AscendC kernel 在 Qwen3.6 27B prefill 场景下的正确性（run_mode=0）
- [ ] 4.2 验证 `npu_causal_conv1d_310` AscendC kernel 在 decode 场景下的正确性（run_mode=1）
- [ ] 4.3 验证 `npu_causal_conv1d_310` 在投机推理多 token 更新场景下的正确性
- [ ] 4.4 验证 `npu_recurrent_gated_delta_rule_310` AscendC kernel 在 decode 路径的正确性
- [ ] 4.5 验证 `npu_recurrent_gated_delta_rule_310` 在投机推理 spec decode 路径的正确性
- [ ] 4.6 性能 benchmark：已具备算子的 latency profiling

## 5. 310P 高优先级算子开发

- [ ] 5.1 `fused_gdn_gating_310`：确认 NAIE AscendC kernel 接口，在 `csrc/torch_binding.cpp` 添加 310P 分支，正确性测试 vs 910 Triton 版本
- [ ] 5.2 `mRope 310P`：确认 NAIE AscendC kernel 接口，在 `ops/rotary_embedding.py` 添加 310P 路径，正确性测试 vs 910 `torch_npu.npu_mrope`
- [ ] 5.3 `FIA 压缩 mask 310P`：确认 NAIE 能力中心 `sparse_mode` 支持，更新 `AscendAttentionBackend310` 跳过完整 mask 生成

## 6. 310P GDN 核心算子开发（SAIE）

- [ ] 6.1 `chunk_gated_delta_rule_fwd 310P`：接收 SAIE AscendC kernel，注册 310P 算子，正确性测试 vs 910 Triton 实现，并单独验证替换当前路径后 TTFT 是否降低约 40%
- [ ] 6.2 `fused_sigmoid_gating_delta_rule_310`：接收 SAIE/NAIE AscendC kernel，注册 310P 融合算子，正确性测试 vs 分步实现
- [ ] 6.3 在 `gdn_310.py` 中替换 PyTorch fallback 为双路径分发调用

## 7. 310P 低优先级算子适配

- [ ] 7.1 `rmsnormgated`：确认 NAIE AscendC kernel，替换 `_310p/ops/layernorm.py` 中 `forward_native`
- [ ] 7.2 `split_qkv_rmsnorm_Mrope`：确认 NAIE AscendC 融合算子，替换 Triton 版本
- [ ] 7.3 `transposeKV`：确认 310P AICore 兼容性，调整 tiling 参数

## 8. 310P 双路径 Fallback 机制

- [ ] 8.1 建立统一双路径分发机制：`is_310p()` + 环境变量 `VLLM_ASCEND_USE_ASCENDC_OPS`
- [ ] 8.2 每个算子封装为 `try AscendC → catch fallback PyTorch` 调用模式
- [ ] 8.3 在 `csrc/torch_binding.cpp` 中扩展 310P 条件编译分支
- [ ] 8.4 更新 `meta_registration.py` 为新增算子注册 Meta dispatch key

## 9. 310P 推理验证与性能基线

- [ ] 9.1 验证 300IDuo*2、TP=2、FP16 eager baseline 正确性
- [ ] 9.2 验证 310P W8 动态量化权重加载和输出精度
- [ ] 9.3 性能 profiling：输出逐算子 latency 占比、瓶颈排序和优化建议
- [ ] 9.4 验收 310P 阶段性能：记录 26.3 Qwen3 32B 对照基线；630 目标为 4K/4K 并发 2、TTFT <= 5000 ms、TPOT <= 80 ms/字符；930 目标为 4K/4K 并发 4、TTFT <= 4000 ms、TPOT <= 60 ms/字符
- [ ] 9.5 投机推理端到端测试：310P rejection sampling + draft proposer 联合验证

## 10. 310P Chunk Prefill

- [ ] 10.1 验证 `AscendAttentionBackend310` SplitFuse 路径正确性（NZ 格式 KV cache 参数）
- [ ] 10.2 实现/验证 `AttentionMaskBuilder310.get_splitfuse_mask()` NZ 格式 mask 构造
- [ ] 10.3 验证 GDN 层跨 chunk SSM state 传递正确性
- [ ] 10.4 验证 310P `initial_state` 构造和 state 写回（无转置）
- [ ] 10.5 配置验证：`enable_chunked_prefill=True` + `max_num_batched_tokens` 在 310P 生效
- [ ] 10.6 Chunk Prefill 端到端正确性测试：长序列分 chunk prefill vs 完整 prefill 输出对齐
- [ ] 10.7 Chunk Prefill TTFT 性能 baseline 测量

## 11. 310P Prefix Caching

- [ ] 11.1 启用 `enable_prefix_caching=True`，验证 310P block allocator 和 hash 匹配
- [ ] 11.2 实现 `SSMCheckpoint` 和 `SSMStatePool` 数据结构（LRU 淘汰）
- [ ] 11.3 实现 SSM checkpoint 创建逻辑：支持固定间隔和用户可配置 `linearCheckpointPolicy`，默认不得突破全局显存上限
- [ ] 11.4 实现 SSM checkpoint 恢复逻辑：以 hash(prefix_tokens) 查找并恢复 h 矩阵
- [ ] 11.5 扩展 `get_computed_blocks()` 同时携带 matched SSM checkpoint
- [ ] 11.6 在 `NPUModelRunner310.execute_model()` 中集成 SSM checkpoint 恢复
- [ ] 11.7 验证 SSM checkpoint 间隔与 Full Attn block 边界对齐
- [ ] 11.8 Prefix Cache 端到端正确性测试：KV cache block 复用 + GDN state 恢复
- [ ] 11.9 Prefix Cache TTFT 性能测量：有命中 vs 无命中
- [ ] 11.10 验证 anchor checkpoint 策略：固定 tools + system prompt 只保存一个 Linear Attention checkpoint，动态 user content 不创建 checkpoint，并对比显存占用和 TTFT

## 12. 310P MTP 投机推理

- [ ] 12.1 在 `AscendAttentionBackend310` 中新增 `SpecDecoding` 状态路由
- [ ] 12.2 实现 Draft 阶段 attention：每个 draft token 走标准 decode 路径
- [ ] 12.3 实现 Verify 阶段 attention：draft tokens + target token 批量计算
- [ ] 12.4 实现 Spec mask 构造：NZ 格式因果 mask，支持 draft token 间因果约束
- [ ] 12.5 在 `NPUModelRunner310` 中支持初始化 `AscendEagleProposer(method="mtp")`
- [ ] 12.6 实现 `copy_and_expand_eagle_inputs_pytorch` PyTorch fallback
- [ ] 12.7 验证 GDN 层 MTP multi-query 路径：`spec_sequence_masks` + `num_accepted_tokens`
- [ ] 12.8 验证 rejection sampling PyTorch fallback 系列算子正确性
- [ ] 12.9 MTP 端到端正确性测试：draft + verify + rejection sampling
- [ ] 12.10 MTP TPOT 性能测量 vs eager decode
- [ ] 12.11 评估 MTP 增量优化方向：基于 acceptance rate 判断 Eagle3 直接训练收益，并分别评估 MTP 分支领域化微调、DFlash 使能和 DFlash 模型领域化微调

## 13. 310P 组合验收

- [ ] 13.1 Chunk Prefill + Prefix Caching 组合测试
- [ ] 13.2 Chunk Prefill + MTP 组合测试
- [ ] 13.3 Chunk Prefill + Prefix Caching + MTP 三特性组合测试
- [ ] 13.4 组合性能测试：全特性 vs 基础 eager 推理
- [ ] 13.5 回归测试：确认特性启用后不影响已有 eager 基础推理

## 14. 混合注意力缓存优化

- [ ] 14.1 配置层：移除 `patch_mamba_config.py` 中 `attn_block_size` 膨胀和 `mamba_page_size_padded` padding
- [ ] 14.2 配置层：移除 `model_runner_v1.py` 中 AttentionSpec `page_size_padded` 强制对齐
- [ ] 14.3 分配层：在 `_allocate_kv_cache_tensors()` 中实现 KV Cache/SSM State/Conv State 三子区域布局
- [ ] 14.4 Reshape 层：修改 `_reshape_kv_cache_tensors()` hybrid 路径，移除 `conv_block_padding_size`
- [ ] 14.5 Block Table：修改 `block_table.py` slot_mapping 计算考虑 KV region offset
- [ ] 14.6 KV Transfer：验证 `mooncake_connector` 和 `mooncake_layerwise_connector` 在新布局下正确
- [ ] 14.7 验证非混合注意力模型（纯 Full Attention）不受影响
- [ ] 14.8 对比测试：新布局 vs 旧 layout attention 输出数值一致性
- [ ] 14.9 性能测试：对比新旧布局 KV cache 容量和显存使用

## 15. KV Cache 容量与池化

- [ ] 15.1 通过实际部署 memory profiling 校验八卡 runtime/cache 显存预算
- [ ] 15.2 决定目标 GDN checkpoint 间隔与存储层级
- [ ] 15.3 验证 DRAM 池化方案：Mooncake 连接器在 Qwen3.6 混合注意力下的正确性
- [ ] 15.4 验证 layerwise KVCache 搬运：Full KV + GDN checkpoint 按混合层类型分发

## 16. 精度对齐与 Profiling 闭环

- [ ] 16.1 建立统一测试集，覆盖普通问答、长上下文、CoT 开/关、MTP 开/关、Prefix Cache 命中/未命中
- [ ] 16.2 记录 FP16 baseline 的输出 token、关键 logits 或业务指定精度指标
- [ ] 16.3 对 W8 动态量化输出做精度对齐，输出误差报告
- [ ] 16.4 生成 910 分段 Profiling 报告
- [ ] 16.5 生成 310 分段 Profiling 报告
- [ ] 16.6 固化测试元数据记录格式

## 17. 文档与验收交付

- [ ] 17.1 补充 Qwen3.6-27B 部署教程，列出 910 和 310 推荐启动参数
- [ ] 17.2 补充故障定位章节
- [ ] 17.3 将 910、310、打包、安全扫描和精度报告归档到发布验收记录
