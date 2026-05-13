## ADDED Requirements

### Requirement: 310P 标准推理配置
系统 SHALL 支持 Qwen3.6-27B 在 300IDuo 96G、TP=2、并发 4 条件下运行。

#### Scenario: 310 标准实例启动
- **WHEN** 使用 300IDuo x2 启动 Qwen3.6-27B 标准实例
- **THEN** 系统 SHALL 使用 TP=2，并在动态量化配置下于 13.5 分钟内完成启动就绪。

### Requirement: 310 关闭 Prefix Cache 性能基线
系统 SHALL 在关闭 Prefix Cache 的条件下满足 310 性能基线。

#### Scenario: 310 长上下文基线
- **WHEN** Prefix Cache 关闭、并发为 4、输入长度为 4K、输出长度为 4K
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
- **THEN** 系统 SHALL 使用 AscendC 优化后的 `chunk_gated_delta_rule_fwd`，并保留 PyTorch 参考实现用于正确性对照。

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
