## ADDED Requirements

### Requirement: Qwen3.6 候选版本线新增 RTSP 包
系统 SHALL 为 Qwen3.6 候选版本线提供 `netrsnpython3rdadvance` RTSP 包；候选版本线当前按 0.18.0 设计，但允许切换为 0.19.x.rcx。

#### Scenario: 候选版本包选择
- **WHEN** 产品 CR 选择 0.18.0 或 0.19.x.rcx 版本路径
- **THEN** 系统 SHALL 加载 `netrsnpython3rdadvance`，并使用该包内固定的 vLLM/vllm-ascend 依赖运行 Qwen3.6-27B。

### Requirement: Qwen3.6 候选版本线运行态无 GCC 依赖
系统 MUST 消除 Qwen3.6 候选版本线运行态 GCC 依赖。

#### Scenario: 运行态安全扫描
- **WHEN** 对 Qwen3.6 候选版本线镜像或 RTSP 包执行运行态依赖扫描
- **THEN** 扫描结果 MUST 证明运行态不包含 GCC 编译依赖，且启动过程不触发源码编译。

### Requirement: 自定义算子预编译
系统 SHALL 在构建期完成 AscendC 自定义算子编译并以 `.so` 形式交付。

#### Scenario: 启动加载自定义算子
- **WHEN** Qwen3.6-27B 实例启动并加载自定义算子
- **THEN** 系统 SHALL 直接加载预编译 `.so`，不 SHALL 调用 GCC 或构建脚本。

### Requirement: Triton cache 运行态不编译
若 910 推理路径依赖 Triton kernel，系统 SHALL 在构建期或安全认可阶段完成 Triton cache 预热。

#### Scenario: 910 启动使用 Triton 路径
- **WHEN** 910 Qwen3.6-27B 实例启动并使用 Triton kernel
- **THEN** 系统 SHALL 使用已准备的 cache 或预编译产物，不 SHALL 在运行态触发 GCC 相关编译。

### Requirement: 包版本可回滚
系统 SHALL 保留 0.13.0 与 Qwen3.6 候选版本线两条可独立选择的包路径。

#### Scenario: Qwen3.6 候选版本不达标回滚
- **WHEN** Qwen3.6 候选版本线路径性能、精度或安全扫描不达标
- **THEN** 系统 SHALL 支持通过 CR 参数停止 Qwen3.6 实例或切换到 0.13.0 既有模型路径，不影响 0.13.0 模型服务。
