## 背景与动机

Qwen3.6 27B 使用 Full Attention + Linear Attention (GDN) 混合注意力架构。做容量规划时不能只计算传统 KV cache，因为 GDN 还会保存 recurrent SSM state，并且 prefix checkpoint 会按 checkpoint 数量额外占用显存。checkpoint 间隔过小时，GDN checkpoint 的显存开销会反过来主导整体缓存容量。

本 change 用独立文档沉淀以下分析：

- Full Attention KV cache 的单 token 字节数。
- Linear Attention GDN state 的单 checkpoint 字节数，以及不同 checkpoint 间隔下摊销到每个 token 的字节数。
- Ascend 910B4 32G 四卡与八卡配置下，在明确显存预算假设下可缓存的 token 数量。
- 长上下文 Agent 场景下，使用 Mooncake 或 Memcache 做 DRAM KV Cache 池化的架构、容量与收益。
- 面向 Prefix Caching 命中率的 Agent 请求设计指导。

## 范围

- 只分析 Qwen3.6 27B text model 的 cache/state 显存占用。
- 覆盖 FP16/BF16 KV cache 与 float32 GDN SSM state。
- 覆盖 910B4 32G 上 TP=4 与 TP=8 两种部署形态。
- 给出 GDN prefix checkpoint 间隔对容量的敏感性分析。
- 工程预算采用实测经验值：四卡部署时单卡可用 KVCache 约 17.48G。
- 覆盖单机 8 卡、2 个 TP=4 PD 混合实例共享 1 个 Mooncake DRAM 池的池化方案。
- 覆盖 OpenAI-compatible request 到最终 LLM 输入 token ids 的形成机制，以及 Agent 侧稳定 prefix 的设计要求。

## 变更内容

- 新增一份独立设计说明，包含 cache/state 公式与容量表。
- 新增 KV Cache 池化章节，说明目的、总体架构、关键交互、DRAM 容量分析和 TTFT 实测收益。
- 新增 Agent 请求设计章节，说明 tools/system prompt 稳定性对 Qwen3.6 Prefix Caching 命中的影响。
- 新增一个 OpenSpec requirement delta，用于描述容量分析文档应覆盖的内容。
- 保持文档分析范围，不修改 serving 代码、allocator 代码或 scheduler 代码。

## 非目标

- 不修改应用代码。
- 不修改调度器、显存分配器或 prefix-cache 实现。
- 不提供实测 runtime profiling；本文档是基于模型 config 与当前设计假设的理论容量模型。
