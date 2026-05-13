## ADDED Requirements

### Requirement: CR 驱动运行版本选择
系统 SHALL 根据产品 CR 参数选择 vLLM、vllm-ascend、RTSP 包和镜像版本。

#### Scenario: 选择 0.13.0 路径
- **WHEN** 产品 CR 声明 `spec.vllmVersion=0.13.0`
- **THEN** 系统 SHALL 选择 0.13.0 对应的 vLLM/vllm-ascend 版本组合和 `netrsnpython3rd` RTSP 包，并仅承载 Qwen3.6 以外的既有模型。

#### Scenario: 选择 0.18.0 路径
- **WHEN** 产品 CR 声明 `spec.vllmVersion=0.18.0`
- **THEN** 系统 SHALL 选择 0.18.0 对应的 vLLM/vllm-ascend 版本组合和 `netrsnpython3rdadvance` RTSP 包，并允许承载 Qwen3.6-27B。

### Requirement: 微服务边界清晰
系统 MUST 将 `NetrsnQwenLargeService` 和 `NetrsnQwenMoeMediumService` 视为外部路由层，不在 vllm-ascend 内实现其内部逻辑。

#### Scenario: 当前仓库缺少微服务实现
- **WHEN** 在当前仓库内未找到 `NetrsnQwenLargeService` 或 `NetrsnQwenMoeMediumService` 的实现
- **THEN** 设计和任务 SHALL 仅定义 CR 输入、运行包选择和验收输出，不 SHALL 描述微服务内部代码修改。

### Requirement: 路由结果可观测
系统 SHALL 在实例启动日志或部署状态中暴露最终选择的 vLLM 版本、vllm-ascend 版本、RTSP 包名和镜像标签。

#### Scenario: 启动后检查路由结果
- **WHEN** Qwen3.6-27B 实例完成启动
- **THEN** 运维或测试 SHALL 能从日志或状态中确认该实例使用 0.18.0 版本组合和 `netrsnpython3rdadvance` 包。
