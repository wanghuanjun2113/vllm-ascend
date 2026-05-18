## ADDED Requirements

### Requirement: CR 驱动运行版本选择
系统 SHALL 根据产品 CR 参数选择 vLLM、vllm-ascend、RTSP 包和镜像版本。

#### Scenario: 选择 0.13.0 路径
- **WHEN** 产品 CR 声明 `spec.vllmVersion=0.13.0`
- **THEN** 系统 SHALL 选择 0.13.0 对应的 vLLM/vllm-ascend 版本组合和 `netrsnpython3rd` RTSP 包，并仅承载 Qwen3.6 以外的既有模型。

#### Scenario: 选择 Qwen3.6 候选版本路径
- **WHEN** 产品 CR 声明 `spec.vllmVersion=0.18.0` 或 `spec.vllmVersion=0.19.x.rcx`
- **THEN** 系统 SHALL 选择对应的 Qwen3.6 候选 vLLM/vllm-ascend 版本组合和 `netrsnpython3rdadvance` RTSP 包，并允许承载 Qwen3.6-27B；0.19.x.rcx 路径 SHALL 作为 0.18.0 路径的整体替换选项处理。

### Requirement: 微服务边界清晰
系统 MUST 将 `NetrsnQwenLargeService` 和 `NetrsnQwenMoeMediumService` 视为外部路由层，不在 vllm-ascend 内实现其内部逻辑。

#### Scenario: 当前仓库缺少微服务实现
- **WHEN** 在当前仓库内未找到 `NetrsnQwenLargeService` 或 `NetrsnQwenMoeMediumService` 的实现
- **THEN** 设计和任务 SHALL 仅定义 CR 输入、运行包选择和验收输出，不 SHALL 描述微服务内部代码修改。

### Requirement: 路由结果可观测
系统 SHALL 在实例启动日志或部署状态中暴露最终选择的 vLLM 版本、vllm-ascend 版本、RTSP 包名和镜像标签。

#### Scenario: 启动后检查路由结果
- **WHEN** Qwen3.6-27B 实例完成启动
- **THEN** 运维或测试 SHALL 能从日志或状态中确认该实例使用 Qwen3.6 候选版本组合和 `netrsnpython3rdadvance` 包。

### Requirement: 模型元数据包承载投机推理启动配置
系统 SHALL 通过模型元数据包的 `basic_configs.json` 承载投机推理启动级配置。

#### Scenario: 读取 basic_configs.json 投机配置
- **WHEN** Qwen3.6-27B 实例启动
- **THEN** 外部服务或部署层 SHALL 从模型元数据包 `basic_configs.json` 读取 `speculative_config.method`、`speculative_config.model`、`speculative_config.draft_tensor_parallel_size`、`speculative_config.enforce_eager` 和 `speculative_config.num_speculative_tokens`，并转换为 vLLM/vllm-ascend 启动参数。

#### Scenario: 禁止请求级覆盖投机配置
- **WHEN** 请求携带与投机推理相关的运行时字段
- **THEN** 系统 SHALL 不允许请求级参数覆盖 `basic_configs.json` 中的投机推理启动级配置。

#### Scenario: 投机配置可观测
- **WHEN** Qwen3.6-27B 实例完成启动
- **THEN** 启动日志或部署状态 SHALL 输出最终生效的 `method`、`model`、`draft_tensor_parallel_size`、`enforce_eager` 和 `num_speculative_tokens`。

#### Scenario: 投机配置校验失败
- **WHEN** `model`、`draft_tensor_parallel_size` 与主模型 TP、草稿模型权重或模型 revision 不一致
- **THEN** 实例启动 SHALL 失败，并输出明确错误，不 SHALL 静默回退到默认投机配置。
