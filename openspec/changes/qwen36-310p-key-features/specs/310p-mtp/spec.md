## ADDED Requirements

### Requirement: 310P Attention 后端 SpecDecoding 支持

`AscendAttentionBackend310` SHALL 支持 `SpecDecoding` attention state，不再抛出 `NotImplementedError`。

#### Scenario: SpecDecoding 状态路由
- **WHEN** MTP 投机推理执行，attention state 为 `SpecDecoding`
- **THEN** 310P attention 后端使用 `_npu_paged_attention_splitfuse` 或等效算子处理 draft+verify tokens 的 attention 计算

#### Scenario: Spec mask 构造
- **WHEN** spec decode attention 需要处理多 token per request 的因果 mask
- **THEN** 构造 spec-decode-appropriate 的 NZ 格式 mask，支持 draft token 间的因果约束

### Requirement: 310P GDN 层 MTP Multi-Query 路径

`AscendGatedDeltaNetAttention310` SHALL 正确处理 MTP 投机推理的 multi-query tokens。

#### Scenario: Spec/Non-spec token 分离
- **WHEN** MTP draft 阶段产生 k 个 draft tokens，verify 后部分 tokens 被拒绝
- **THEN** 通过 `spec_sequence_masks`、`spec_token_indx`、`non_spec_token_indx` 分离 spec 和 non-spec tokens，分别调用 `fused_recurrent_gated_delta_rule_pytorch` 的 multi-query 路径

#### Scenario: num_accepted_tokens 限制
- **WHEN** verify 后只有前 n 个 draft tokens 被接受（n ≤ k）
- **THEN** `fused_recurrent_gated_delta_rule_pytorch` 的 `num_accepted_tokens` 参数限制 SSM state 只在被接受的 tokens 上更新

#### Scenario: 输出合并
- **WHEN** spec 和 non-spec tokens 分别计算完 attention
- **THEN** 使用 `index_copy_` 将两部分输出合并回主输出 tensor：`merged_out.index_copy_(1, spec_token_indx, spec_output)` + `merged_out.index_copy_(1, non_spec_token_indx, non_spec_output)`

### Requirement: 310P npu_copy_and_expand_eagle_inputs 替代实现

系统 SHALL 在 310P 上提供 `npu_copy_and_expand_eagle_inputs` 的等效实现。

#### Scenario: PyTorch fallback 实现
- **WHEN** 310P 不具备 `npu_copy_and_expand_eagle_inputs` C++ 算子
- **THEN** 使用纯 PyTorch 实现等效逻辑：对每个请求复制有效 input tokens + 插入 next_token_id + 追加 parallel drafting placeholder tokens + 标记 rejected/masked tokens

#### Scenario: AscendC 移植后替换
- **WHEN** NAIE/SAIE 提供 310P 版本的 AscendC kernel
- **THEN** 替换 PyTorch fallback，注册到 `torch.ops._C_ascend.npu_copy_and_expand_eagle_inputs`

### Requirement: 310P MTP Draft Model 初始化

`NPUModelRunner310` SHALL 支持初始化 `AscendEagleProposer(method="mtp")`。

#### Scenario: Draft model加载
- **WHEN** 启动参数配置了 `speculative_config.method="mtp"`
- **THEN** 310P model runner 初始化 `AscendEagleProposer`，加载 MTP head 权重，共享主模型 embedding 和 lm_head

#### Scenario: KV binding
- **WHEN** MTP head 的 attention 层需要绑定 KV cache
- **THEN** 使用 `bind_kv_cache` 将 draft model 的 attention 层关联到正确的 KV cache slots

### Requirement: 310P Rejection Sampling PyTorch 路径

系统 SHALL 在 310P 上通过 `HAS_TRITON=False` 自动选择 PyTorch rejection sampling fallback。

#### Scenario: Greedy rejection sampling
- **WHEN** MTP verify 阶段使用 greedy sampling
- **THEN** 使用 `rejection_greedy_sample_pytorch`，通过 `torch.where` + 2D 索引矩阵进行向量化比较

#### Scenario: Random/Block verify rejection sampling
- **WHEN** MTP verify 阶段使用 stochastic 或 block verify sampling
- **THEN** 使用 `rejection_random_sample_pytorch` 或 `rejection_random_sample_block_verify_pytorch` PyTorch fallback

#### Scenario: Recovered token sampling
- **WHEN** draft token 被拒绝，需要从残差分布采样恢复 token
- **THEN** 使用 `sample_recovered_tokens_pytorch`，通过 `torch.argmax(residual / q)` 向量化采样

### Requirement: 310P MTP Eager 模式执行

系统 SHALL 在 310P 上以纯 eager 模式执行 MTP draft/verify 路径。

#### Scenario: 无 ACLGraph 的 MTP 执行
- **WHEN** 310P 不支持 ACLGraph
- **THEN** draft 和 verify 阶段均走 eager forward，SSM state 通过 `inplace_final_state=True` 直接 in-place 更新

#### Scenario: 性能可接受性
- **WHEN** MTP eager 路径的 acceptance rate 足够高
- **THEN** 即使无 ACLGraph 加速，减少主模型 forward 次数的收益超过 eager 路径的额外 launch overhead

### Requirement: 310P Sampler MTP 兼容性

`AscendSampler310` SHALL 正确处理 MTP 场景下的采样请求。

#### Scenario: CPU exponential 随机采样
- **WHEN** MTP verify 阶段需要 random sampling
- **THEN** `AscendTopKTopPSampler310` 使用 CPU exponential 生成随机数（`q.cpu().exponential_().npu()`），避免 310P NPU exponential 同步问题

#### Scenario: Top-K/Top-P PyTorch 路径
- **WHEN** MTP 场景下 logits 需要 top-k/top-p 过滤
- **THEN** 使用 `_apply_top_k_top_p_pytorch`（非 AscendC kernel），与 910 的 `_apply_top_k_top_p_ascendc` 路径分离
