## ADDED Requirements

### Requirement: Qwen3.6 缓存容量分析

本 change MUST 提供一份独立的 Qwen3.6 27B cache 与 linear-attention state 显存容量分析。

#### Scenario: 推导 Full Attention KV cache 单 token 开销

- **当** 读者需要了解 Qwen3.6 27B 的 Full Attention KV cache 开销
- **则** 分析必须根据模型的 full-attention 层数、KV head 数量、head dimension 与 KV dtype 字节数，推导全局单 token 字节数
- **并且** 分析必须给出 TP=4 与 TP=8 下的单卡数值

#### Scenario: 推导 Linear Attention checkpoint 开销

- **当** 读者评估 GDN prefix checkpoint 显存占用
- **则** 分析必须根据模型的 linear-attention 层数、value-head 数量、value dimension、key dimension 与 SSM dtype 字节数，推导单个 GDN SSM checkpoint 的大小
- **并且** 分析必须给出多个 checkpoint 间隔下摊销到每个 token 的开销
- **并且** 分析必须说明固定间隔保存与用户可配置 anchor 保存策略对显存占用的影响

#### Scenario: 估算 910B4 32G 容量

- **当** 读者比较四卡与八卡 910B4 32G 部署
- **则** 分析必须在明确显存预算假设下估算最大可缓存 token 数量
- **并且** 分析必须使用实测工程预算，不列裸 HBM 理论上限
- **并且** 分析必须考虑 TP=8 下 Full Attention KV heads 数量少于 TP size 导致的 KV cache 复制

#### Scenario: 说明 KV Cache 池化方案

- **当** 读者评估长上下文 Agent 应用的 Prefix Caching 命中率问题
- **则** 分析必须说明 KV Cache 池化的目的，以及为什么 DRAM 池化可以缓解 HBM 容量限制
- **并且** 分析必须描述单机 8 卡、2 个 TP=4 PD 混合实例共享 1 个 Mooncake 池的总体架构与关键交互
- **并且** 分析必须给出 DRAM 池容量与可缓存上下文长度的估算
- **并且** 分析必须记录 64K 输入在 90% DRAM cache 命中时 TTFT 从 38.3s 降到 14.4s 的实测收益

#### Scenario: 给出 Agent 请求设计指导

- **当** 读者设计长上下文 Agent 请求格式
- **则** 分析必须说明 OpenAI-compatible request 如何经过 chat template 形成最终 LLM 输入
- **并且** 分析必须说明 Qwen3.6 chat template 中 tools 与 system prompt 对 prefix 稳定性的影响
- **并且** 分析必须给出提高 Prefix Caching 命中率的 Agent 侧设计要求
- **并且** 分析必须建议固定 tools 与 system prompt 场景使用 anchor checkpoint 策略，只在稳定前缀末尾保存一个 Linear Attention checkpoint
