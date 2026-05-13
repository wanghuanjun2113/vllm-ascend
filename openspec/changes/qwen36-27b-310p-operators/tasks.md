## 0. 前置对齐与调研

- [ ] 0.1 与 SAIE 对齐 `chunk_gated_delta_rule_fwd` AscendC 实现的交付时间、接口草案和联调计划
- [ ] 0.2 与 SAIE/NAIE 对齐 `fused_sigmoid_gating_delta_rule_310` AscendC 实现的交付时间和接口
- [ ] 0.3 与 NAIE 对齐 `fused_gdn_gating` 高优先级算子的排期、API 草案和精度要求
- [ ] 0.4 与 NAIE 对齐 `mRope` 高优先级算子的排期和 API（确认与 `torch_npu.npu_mrope` 接口一致性）
- [ ] 0.5 与 NAIE 能力中心确认 310P `torch_npu._npu_flash_attention` / `_npu_paged_attention` 是否可扩展 `sparse_mode` 参数
- [ ] 0.6 与 NAIE 能力中心对齐拒绝采样相关算子的开发动机（是否需要 AscendC，还是维持 PyTorch fallback）
- [ ] 0.7 确认 NAIE 低优先级算子（`rmsnormgated`、`split_qkv_rmsnorm_Mrope`、`transposeKV`）的预计交付时间

## 1. 已具备算子验证与集成确认

- [ ] 1.1 验证 `npu_causal_conv1d_310` AscendC kernel 在 Qwen3.6 27B prefill 场景下的正确性（run_mode=0）
- [ ] 1.2 验证 `npu_causal_conv1d_310` AscendC kernel 在 decode 场景下的正确性（run_mode=1）
- [ ] 1.3 验证 `npu_causal_conv1d_310` 在投机推理多 token 更新场景下的正确性（`num_accepted_tokens` 参数）
- [ ] 1.4 验证 `npu_recurrent_gated_delta_rule_310` AscendC kernel 在 decode 路径的正确性
- [ ] 1.5 验证 `npu_recurrent_gated_delta_rule_310` 在投机推理 spec decode 路径的正确性
- [ ] 1.6 性能 benchmark：已具备算子的 latency profiling，确认 baseline 数据

## 2. fused_gdn_gating 310P 算子开发（高优先级）

- [ ] 2.1 确认 NAIE 提供的 `fused_gdn_gating_310` AscendC kernel 接口和编译产物
- [ ] 2.2 在 `csrc/torch_binding.cpp` 中添加 310P 条件编译分支，注册 `npu_fused_gdn_gating_310` 算子
- [ ] 2.3 在 `_310p/ops/fla/fused_gdn_gating.py` 中添加 AscendC 调用路径（优先使用 AscendC，fallback 到 PyTorch）
- [ ] 2.4 编写正确性测试：`fused_gdn_gating_310` 输出与 910 Triton 版本对比（atol=1e-3, rtol=1e-3）
- [ ] 2.5 编写性能 benchmark：对比 AscendC vs PyTorch fallback 的 latency
- [ ] 2.6 在 `gdn_310.py` 中替换 `fused_gdn_gating_pytorch` 调用为双路径分发

## 3. mRope 310P 算子开发（高优先级）

- [ ] 3.1 确认 NAIE 提供的 `npu_mrope_310` AscendC kernel 接口和编译产物
- [ ] 3.2 在 `csrc/torch_binding.cpp` 中注册 `npu_mrope_310` 算子（310P 条件编译）
- [ ] 3.3 在 `ops/rotary_embedding.py` `AscendMRotaryEmbedding.forward_oot` 中添加 310P 路径
- [ ] 3.4 编写正确性测试：`npu_mrope_310` 输出与 910 `torch_npu.npu_mrope` 对比
- [ ] 3.5 编写性能 benchmark：对比 mRope 在 310P 上的 latency

## 4. chunk_gated_delta_rule_fwd 310P 算子开发（SAIE）

- [ ] 4.1 接收 SAIE 提供的 `chunk_gated_delta_rule_fwd_310` AscendC kernel 编译产物
- [ ] 4.2 在 `csrc/torch_binding.cpp` 中注册 310P chunk 算子
- [ ] 4.3 在 `_310p/ops/fla/chunk_gated_delta_rule.py` 中添加 AscendC 调用路径（双路径分发）
- [ ] 4.4 编写正确性测试：对比 310P AscendC 和 910 Triton 实现的输出（atol=1e-3, rtol=1e-3）
- [ ] 4.5 编写性能 benchmark：chunk_gated_delta_rule 在 prefill 路径的 latency 占比
- [ ] 4.6 在 `gdn_310.py` 中替换 `chunk_gated_delta_rule_pytorch` 为双路径分发调用

## 5. fused_sigmoid_gating_delta_rule_310 算子开发（SAIE+NAIE）

- [ ] 5.1 接收 SAIE/NAIE 提供的 `fused_sigmoid_gating_delta_rule_310` AscendC kernel
- [ ] 5.2 在 `csrc/torch_binding.cpp` 中注册 310P 融合算子
- [ ] 5.3 在 `_310p/ops/fla/` 中新建 `sigmoid_gating_delta_rule_310.py`，封装 AscendC 调用
- [ ] 5.4 编写正确性测试：对比融合算子和分步（gating + recurrent）的输出
- [ ] 5.5 编写性能 benchmark：对比融合 vs 分步的 decode latency
- [ ] 5.6 在 `gdn_310.py` decode 路径中集成融合算子（替换两步调用）

## 6. FIA 压缩 mask 310P 适配（高优先级）

- [ ] 6.1 确认 NAIE 能力中心在 310P FIA 算子中增加 `sparse_mode` 参数支持的交付时间
- [ ] 6.2 更新 `_310p/attention/attention_v1.py` 中 `AscendAttentionBackend310.forward()` 使用 `sparse_mode` 参数
- [ ] 6.3 条件化 `AttentionMaskBuilder310`：当 FIA 支持 sparse_mode 时跳过完整 mask 生成
- [ ] 6.4 编写正确性测试：对比使用 compressed mask 和完整 mask 的注意力输出
- [ ] 6.5 编写性能测试：测量 compressed mask 对显存占用和推理 latency 的影响

## 7. 拒绝采样算子 310P 集成

- [ ] 7.1 确认 NAIE 能力中心开发的拒绝采样算子范围和接口
- [ ] 7.2 如果提供 AscendC 算子：在 `csrc/torch_binding.cpp` 注册，并在 `rejection_sampler.py` 中添加 310P 路径
- [ ] 7.3 如果维持 PyTorch fallback：验证 `rejection_greedy_sample_pytorch`、`rejection_random_sample_pytorch`、`sample_recovered_tokens_pytorch` 在 310P 上的正确性
- [ ] 7.4 实现 `copy_and_expand_eagle_inputs` 的 310P 版本（AscendC 或 PyTorch）
- [ ] 7.5 编写端到端测试：310P 投机推理 rejection sampling 正确性
- [ ] 7.6 性能验证：确认 rejection sampling 开销 < 0.1ms

## 8. 低优先级算子适配

- [ ] 8.1 `rmsnormgated`：确认 NAIE 提供的 AscendC kernel，替换 `_310p/ops/layernorm.py` 中 `forward_native`
- [ ] 8.2 `split_qkv_rmsnorm_Mrope`：确认 NAIE 提供的 AscendC 融合算子，替换 Triton 版本
- [ ] 8.3 `transposeKV`：确认 310P AICore 兼容性，调整 tiling 参数，在 310P 条件编译中注册

## 9. 双路径 Fallback 机制与注册

- [ ] 9.1 建立统一的双路径分发机制：`is_310p()` + 环境变量 `VLLM_ASCEND_USE_ASCENDC_OPS`
- [ ] 9.2 每个算子封装为：`try AscendC → catch fallback PyTorch` 的调用模式
- [ ] 9.3 在 `csrc/torch_binding.cpp` 中扩展 310P 条件编译分支，注册所有新增算子
- [ ] 9.4 更新 `meta_registration.py`：为新增算子注册 Meta dispatch key（`torch.compile` 兼容）

## 10. 整网验证与性能达标

- [ ] 10.1 Qwen3.6 27B 310P 单卡 FP16 正确推理验证
- [ ] 10.2 Qwen3.6 27B 310P 双卡 TP=2 FP16 正确推理验证
- [ ] 10.3 性能 profiling：逐算子 latency 占比分析，确认整网耗时占比 ≤ 910 的 30%
- [ ] 10.4 TPOT 基线测试：确认 310P 上 TPOT ≤ 60ms 目标
- [ ] 10.5 投机推理端到端测试：310P rejection sampling + draft proposer 联合验证
