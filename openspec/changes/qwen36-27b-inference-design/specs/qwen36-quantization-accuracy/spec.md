## ADDED Requirements

### Requirement: FP16 baseline 先行
系统 MUST 在启用 W8 动态量化和性能优化前建立 FP16 baseline。

#### Scenario: FP16 baseline 验收
- **WHEN** Qwen3.6-27B 首次接入 910 或 310
- **THEN** 系统 MUST 先记录 FP16 eager 输出、关键 logits 或业务指定精度指标，作为后续量化和优化对齐基线。

### Requirement: W8 动态量化精度对齐
系统 SHALL 对 W8 动态量化输出进行白盒或黑盒精度对齐。

#### Scenario: W8 动态量化对比 FP16
- **WHEN** 使用 W8 动态量化运行同一组 prompt 和采样参数
- **THEN** 系统 SHALL 将输出 token、关键 logits 或业务指定指标与 FP16 baseline 对齐，并记录可接受误差范围。

### Requirement: 优化特性组合精度矩阵
系统 SHALL 覆盖 CoT、MTP、Prefix Cache、ACLGraph 和 W8 动态量化的组合精度验证。

#### Scenario: CoT 开关精度验证
- **WHEN** 分别使用 `enable_thinking=true` 和 `enable_thinking=false` 运行同一测试集
- **THEN** 系统 SHALL 验证 prompt 构造符合预期，并分别记录生成 token 数和性能指标。

#### Scenario: MTP 与非 MTP 输出对齐
- **WHEN** 开启 MTP 后运行确定性采样用例
- **THEN** 系统 SHALL 验证最终 accepted tokens 与非 MTP 路径的输出一致或满足定义的业务误差阈值。

#### Scenario: Prefix Cache 命中输出对齐
- **WHEN** Prefix Cache 命中复用前缀状态
- **THEN** 系统 SHALL 验证输出与未命中或关闭 Prefix Cache 的结果一致。

### Requirement: Profiling 分段归因
系统 SHALL 按启动、prefill、decode、MTP draft、verify、sampling、通信、KV/SSM state 更新分段归因性能瓶颈。

#### Scenario: 生成 Profiling 报告
- **WHEN** 执行 910 或 310 性能验收
- **THEN** 系统 SHALL 输出分段耗时、关键算子耗时、通信耗时、graph capture/replay 耗时和缓存命中统计。

### Requirement: 测试元数据完整记录
系统 SHALL 记录每次精度和性能测试的模型、版本、硬件和配置元数据。

#### Scenario: 保存测试上下文
- **WHEN** 生成精度或性能报告
- **THEN** 报告 SHALL 包含模型 revision、vLLM/vllm-ascend 版本、CANN/torch-npu 版本、硬件型号、量化格式、prompt、采样参数、Prefix Cache 状态、MTP 状态和 CoT 状态。
