## ADDED Requirements

### Requirement: 910B4 标准推理配置
系统 SHALL 支持 Qwen3.6-27B 在 910B4*4、32G/卡、TP=4 条件下运行，并覆盖 8K/8K 并发 8 与 32K/10K 并发 4 两类验收负载。

#### Scenario: 910 标准实例启动
- **WHEN** 使用 910B4*4 启动 Qwen3.6-27B 标准实例
- **THEN** 系统 SHALL 使用 TP=4，并在动态量化配置下于 8 分钟内完成启动就绪。

### Requirement: 910 阶段性能验收
系统 SHALL 在关闭 Prefix Cache 的条件下满足 910 阶段性能验收目标，并在报告中保留 26.3 Qwen3 32B 对照基线。

#### Scenario: 26.3 Qwen3 32B 对照基线
- **WHEN** 在裸机容器、910B4*4、Qwen3 32B、输入 8K、输出 8K、并发 4 条件下执行基线测试
- **THEN** 验收报告 SHALL 记录 TTFT 3000 ms、TPOT 40 ms/字符。

#### Scenario: 630 910 短上下文目标
- **WHEN** Prefix Cache 关闭、Qwen3.6 27B、输入长度为 8K、输出长度为 8K、并发为 8
- **THEN** 系统 SHALL 达到 TTFT <= 2500 ms 且 TPOT <= 20 ms/字符。

#### Scenario: 630 910 长上下文目标
- **WHEN** Prefix Cache 关闭、Qwen3.6 27B、输入长度为 32K、输出长度为 10K、并发为 4
- **THEN** 系统 SHALL 达到 TTFT <= 8000 ms 且 TPOT <= 20 ms/字符。

#### Scenario: 930 910 短上下文目标
- **WHEN** Prefix Cache 关闭、Qwen3.6 27B、输入长度为 8K、输出长度为 8K、并发为 8
- **THEN** 系统 SHALL 达到 TTFT <= 2500 ms 且 TPOT <= 20 ms/字符。

#### Scenario: 930 910 长上下文目标
- **WHEN** Prefix Cache 关闭、Qwen3.6 27B、输入长度为 32K、输出长度为 10K、并发为 4
- **THEN** 系统 SHALL 达到 TTFT <= 5000 ms 且 TPOT <= 20 ms/字符。

### Requirement: 910 优化特性配置
系统 SHALL 支持 W8 动态量化、ACLGraph Full decode、Chunk Prefill、MTP 和 Prefix Cache 的组合配置。

#### Scenario: 启用 910 优化组合
- **WHEN** 用户启用 W8 动态量化、Full decode graph、Chunk Prefill、MTP 和 Prefix Cache
- **THEN** 系统 SHALL 保持输出正确，且 SHALL 分别记录每个特性对 TTFT、TPOT、显存和启动耗时的影响。

### Requirement: MTP 使用通用 speculative 分发
系统 SHALL 通过 vllm-ascend 中通用 `mtp` speculative 分发路径承载 Qwen3.6 MTP。

#### Scenario: MTP 方法分发
- **WHEN** speculative method 解析为 `mtp`
- **THEN** 系统 SHALL 通过 `vllm_ascend/spec_decode/__init__.py` 中的 MTP 分发进入 `AscendEagleProposer`，Qwen3.6 专项逻辑只处理模型结构、权重加载、KV/SSM state 绑定和 GDN 状态一致性。

#### Scenario: MTP 增量优化方向
- **WHEN** Qwen3.6 MTP 接受率已经较高
- **THEN** 系统 SHOULD 优先评估 MTP 分支领域化数据微调，或使能 DFlash 并对 DFlash 模型进行领域化数据微调；直接训练 Eagle3 draft 模型不作为首选优化路径。

### Requirement: CoT 控制不进入 Ascend 执行语义
系统 SHALL 通过 tokenizer/chat template 层控制 CoT 思考开关，不在 Ascend kernel、runner 或 sampler 中新增 CoT 专用语义。

#### Scenario: 调用级关闭 CoT
- **WHEN** 请求携带 `chat_template_kwargs={"enable_thinking": false}`
- **THEN** 系统 SHALL 将该参数透传给 chat template，并保持 NPU 执行链路不感知 CoT 专用分支。

#### Scenario: 非调用级默认 CoT 配置
- **WHEN** 服务启动参数设置默认 `chat_template_kwargs.enable_thinking`
- **THEN** 未显式覆盖的请求 SHALL 使用该默认值生成 prompt。
