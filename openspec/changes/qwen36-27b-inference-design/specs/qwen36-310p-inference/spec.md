## ADDED Requirements

### Requirement: 310P 标准推理配置
系统 SHALL 支持 Qwen3.6-27B 在 300IDuo*2、单卡显存 96G、TP=2 条件下运行，并覆盖 630 阶段并发 2 与 930 阶段并发 4 两类验收负载。

#### Scenario: 310 标准实例启动
- **WHEN** 使用 300IDuo*2 启动 Qwen3.6-27B 标准实例
- **THEN** 系统 SHALL 使用 TP=2，并在动态量化配置下于 13.5 分钟内完成启动就绪。

#### Scenario: 300VPro 等价验收
- **WHEN** 使用 4 卡 300VPro 部署 Qwen3.6-27B
- **THEN** 系统 SHALL 按 300IDuo*2 的同一性能基线验收。

### Requirement: 310 关闭 Prefix Cache 性能基线
系统 SHALL 在关闭 Prefix Cache 的条件下满足 310 阶段性能验收目标，并在报告中保留 26.3 Qwen3 32B 对照基线。

#### Scenario: 26.3 Qwen3 32B 对照基线
- **WHEN** 在裸机容器、300IDuo*2、Qwen3 32B、输入 4K、输出 4K、并发 2 条件下执行基线测试
- **THEN** 验收报告 SHALL 记录 TTFT 5000 ms、TPOT 100 ms/字符。

#### Scenario: 630 310 目标
- **WHEN** Prefix Cache 关闭、Qwen3.6 27B、输入长度为 4K、输出长度为 4K、并发为 2
- **THEN** 系统 SHALL 达到 TTFT <= 5000 ms 且 TPOT <= 80 ms/字符。

#### Scenario: 930 310 目标
- **WHEN** Prefix Cache 关闭、Qwen3.6 27B、输入长度为 4K、输出长度为 4K、并发为 4
- **THEN** 系统 SHALL 达到 TTFT <= 4000 ms 且 TPOT <= 60 ms/字符。

### Requirement: 310 先交付 eager 正确性
系统 MUST 先完成 310P eager 推理正确性，再启用 graph 推理。

#### Scenario: eager baseline 通过后推进 graph
- **WHEN** 310P FP16 eager 和 W8 动态量化 eager 正确性未通过
- **THEN** 系统 MUST NOT 将 310P graph 作为性能验收路径。

### Requirement: 310P 必补算子
系统 SHALL 为 310P 补齐 `chunk_gated_delta_rule_fwd`、`rmsnormgated`、`fused_gdn_gating`、`split_qkv_rmsnorm_Mrope` 和 `transposeKV` 五个算子能力。

#### Scenario: GDN prefill 主路径算子替换
- **WHEN** Qwen3.6-27B 在 310P 上执行 GDN prefill
- **THEN** 系统 SHALL 使用 AscendC 优化后的 `chunk_gated_delta_rule_fwd`，并保留 PyTorch 参考实现用于正确性对照；验收报告 SHALL 单独记录该算子替换对 TTFT 的影响，目标预测为相对当前路径降低约 40%。

#### Scenario: QKV 与 MRoPE 融合算子可用
- **WHEN** Qwen3.6-27B 在 310P 上执行 QKV split、RMSNorm 和 MRoPE
- **THEN** 系统 SHALL 使用 310P 可执行的 `split_qkv_rmsnorm_Mrope` 实现，避免依赖 910 Triton kernel。

### Requirement: 310P graph 分阶段启用
系统 SHALL 在确认 CANN/torch-npu 支持后分阶段启用 310P graph。

#### Scenario: graph 能力未确认
- **WHEN** 目标 CANN/torch-npu 版本未确认 310P graph capture/replay 可用
- **THEN** 系统 SHALL 仅使用 eager 路径交付，不 SHALL 将 graph 作为必选路径。

#### Scenario: graph 能力确认
- **WHEN** 310P graph capture/replay 能力确认可用
- **THEN** 系统 SHALL 先启用 Full Attention decode graph，再评估 GDN state 双缓冲或 input/output state 拆分方案。
