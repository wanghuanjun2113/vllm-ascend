## 1. Chunk Prefill — Full Attention SplitFuse

- [ ] 1.1 验证 `AscendAttentionBackend310` 在 `ChunkedPrefill` 状态下调用 `_npu_paged_attention_splitfuse` 的正确性，确认 NZ 格式 KV cache 传入参数无误
- [ ] 1.2 验证 `AscendAttentionBackend310` 在 `PrefillCacheHit` 状态下复用 splitfuse 路径，确认 prefix 命中后剩余 tokens 的 attention 计算正确
- [ ] 1.3 实现/验证 `AttentionMaskBuilder310.get_splitfuse_mask()` 的 NZ 格式 mask 构造：动态行选择 + `nd_to_nz_spec` + `npu_format_cast(ACL_FORMAT_FRACTAL_NZ)`
- [ ] 1.4 验证 splitfuse mask 中 `pos_list` 计算的正确性（`range(context_len - query_len, context_len)`），确保 CPU-NPU 同步无误
- [ ] 1.5 验证 `nd_to_nz_spec` padding 对齐：`[num_tokens, max_seq_len]` → `[1, max_seq_len_pad//16, num_tokens_pad, 16]` 对齐逻辑
- [ ] 1.6 验证 splitfuse 路径与 `get_supported_kernel_block_sizes()` 返回值 `[128, 64]` 的兼容性

## 2. Chunk Prefill — GDN 层

- [ ] 2.1 验证 `chunk_gated_delta_rule_pytorch` 在 310P chunk prefill 场景下的正确性，确认逐序列处理逻辑无误差
- [ ] 2.2 验证 GDN 层跨 chunk SSM state 传递：每个 chunk 接收前一个 chunk 的 `last_recurrent_state` 作为 `initial_state`，确保递推累积正确
- [ ] 2.3 验证 310P `initial_state` 构造：`ssm_state[indices].contiguous()`（无转置）+ `initial_state[~has_initial_state] = 0`（布尔索引清零）
- [ ] 2.4 验证 310P state 写回：`ssm_state[indices] = last_recurrent_state.to(dtype)`（无转置），与 910 的 `.transpose(-1, -2)` 差异确认
- [ ] 2.5 验证 `fused_recurrent_gated_delta_rule_pytorch` 的 `inplace_final_state=True` 在 chunk 间传递场景的正确性
- [ ] 2.6 验证 `cu_seqlens` 参数在 chunk 边界处的正确构造，确保序列边界不串扰

## 3. Chunk Prefill — 调度与集成

- [ ] 3.1 配置验证：`enable_chunked_prefill=True` + `max_num_batched_tokens` 在 310P 启动参数中生效
- [ ] 3.2 验证混合 batch（prefill chunk + decode token）的调度和 metadata 构建在 310P 上的正确性
- [ ] 3.3 Chunk Prefill 端到端正确性测试：长序列（>4096 tokens）分 chunk prefill，输出与完整 prefill 对齐
- [ ] 3.4 Chunk Prefill TTFT 性能 baseline 测量：对比 eager 模式，确认 TTFT 改善

## 4. Prefix Caching — Full Attention Block-Level

- [ ] 4.1 启用 `enable_prefix_caching=True` 配置，验证 310P block allocator 和 hash 匹配逻辑正常工作
- [ ] 4.2 验证 NZ 格式 KV cache 不影响 block allocator 逻辑（block allocator 不关心内部格式）
- [ ] 4.3 验证 prefix cache 命中后 `PrefillCacheHit` 状态的 splitfuse 路径正确性（与 1.2 联动）
- [ ] 4.4 验证 `expandable_segments` 在 310P hybrid 模型上的兼容性

## 5. Prefix Caching — SSMStatePool 数据结构

- [ ] 5.1 实现 `SSMCheckpoint` 数据结构：`layer_states: dict[str, Tensor]`、`hash_key: int`、`num_tokens: int`、`ref_count: int`
- [ ] 5.2 实现 `SSMStatePool` 数据结构：`checkpoints: dict[int, SSMCheckpoint]`、`max_checkpoints: int`、`per_checkpoint_size: int`
- [ ] 5.3 实现 SSMStatePool 的 checkpoint 创建逻辑：在 Full Attn block 边界（每 128 tokens）snapshot 所有 48 层 GDN 的 SSM state
- [ ] 5.4 实现 SSMStatePool 的 checkpoint 恢复逻辑：以 `hash(prefix_tokens)` 为 key 查找并恢复 h 矩阵
- [ ] 5.5 实现 LRU 淘汰策略：checkpoint 数量超过 `max_checkpoints` 时淘汰最近最少使用的 checkpoint
- [ ] 5.6 计算并配置 `max_checkpoints`：基于 310P 可用显存（~40-50G）和单 checkpoint 大小（~144 MiB for 48 layers float32）动态计算上限

## 6. Prefix Caching — SSM Checkpoint 集成

- [ ] 6.1 扩展 `get_computed_blocks()` 返回值，同时携带 matched SSM checkpoint（Full Attn block 列表 + GDN checkpoint）
- [ ] 6.2 在 `NPUModelRunner310.execute_model()` 中集成 SSM checkpoint 恢复：GDN 层 forward 前从 SSMStatePool 恢复 `initial_state`
- [ ] 6.3 在 `AscendGatedDeltaNetAttention310._forward_core()` 中使用恢复的 `initial_state`，跳过已缓存的 prefix tokens 的 GDN 计算
- [ ] 6.4 验证 SSM checkpoint 间隔与 Full Attn block 边界对齐：block_size=128 时每 128 tokens 创建一次
- [ ] 6.5 验证 block_size=64 场景：SSM checkpoint 仍每 128 tokens 创建（独立于 KV block_size），处理部分 block hash 命中无对应 SSM checkpoint 的情况

## 7. Prefix Caching — 正确性与性能

- [ ] 7.1 Prefix Cache 端到端正确性测试：相同 prefix 的多个请求，验证 KV cache block 复用和 GDN state 恢复
- [ ] 7.2 GDN SSM checkpoint restore 后增量计算的输出与完整计算输出对齐验证
- [ ] 7.3 Prefix Cache TTFT 性能测量：有 prefix 命中 vs 无命中的 TTFT 对比
- [ ] 7.4 SSMStatePool 显存使用监控：确认 checkpoint 存储不导致 OOM

## 8. MTP — Attention 后端 SpecDecoding 支持

- [ ] 8.1 在 `AscendAttentionBackend310` 中新增 `SpecDecoding` 状态路由，不再抛出 `NotImplementedError`
- [ ] 8.2 实现 Draft 阶段 attention：每个 draft token 走标准 decode 路径（`_npu_paged_attention`），更新 KV cache
- [ ] 8.3 实现 Verify 阶段 attention：draft tokens + target token 批量计算 attention，使用 `_npu_paged_attention_splitfuse` 或等效算子
- [ ] 8.4 实现 Spec mask 构造：为 multi-token per request 的 spec decode 构造 NZ 格式因果 mask，支持 draft token 间的因果约束
- [ ] 8.5 验证 SpecDecodeMetadata 的处理：slot mapping 更新、block_table 构造、mixed batch attention 计算
- [ ] 8.6 参考 910 `AscendAttentionBackendImpl` 的 SpecDecoding 处理逻辑，确认 310P 路径完整覆盖

## 9. MTP — Draft Model 初始化与 Eager 执行

- [ ] 9.1 在 `NPUModelRunner310` 中支持初始化 `AscendEagleProposer(method="mtp")`：加载 MTP head 权重，共享主模型 embedding 和 lm_head
- [ ] 9.2 验证 MTP head 的 KV cache 绑定：`bind_kv_cache` 将 draft model 的 attention 层关联到正确的 KV cache slots
- [ ] 9.3 验证 310P 纯 eager 模式下 MTP draft/verify 路径执行正确性（无 ACLGraph）
- [ ] 9.4 验证 `inplace_final_state=True` 在 MTP eager 路径中 SSM state 直接 in-place 更新的正确性

## 10. MTP — GDN 层 Multi-Query 路径

- [ ] 10.1 验证 `AscendGatedDeltaNetAttention310` 的 `spec_sequence_masks` 分支：分离 spec 和 non-spec tokens
- [ ] 10.2 验证 spec tokens 的 `fused_recurrent_gated_delta_rule_pytorch` multi-query 路径：`spec_query_start_loc` 和 `num_accepted_tokens` 参数正确
- [ ] 10.3 验证 `num_accepted_tokens` 限制 SSM state 只在被接受的 tokens 上更新
- [ ] 10.4 验证输出合并：`merged_out.index_copy_(1, spec_token_indx, spec_output)` + `merged_out.index_copy_(1, non_spec_token_indx, non_spec_output)`

## 11. MTP — `npu_copy_and_expand_eagle_inputs` PyTorch Fallback

- [ ] 11.1 实现纯 PyTorch 版本的 `copy_and_expand_eagle_inputs_pytorch`：逐请求循环，复制有效 input tokens + 插入 next_token_id + 追加 parallel drafting placeholder + 标记 rejected/masked tokens
- [ ] 11.2 验证 PyTorch fallback 输出与 910 C++ 算子输出一致性
- [ ] 11.3 在 310P 条件编译中注册 PyTorch fallback 到 `torch.ops._C_ascend.npu_copy_and_expand_eagle_inputs`（或通过 dispatch 机制）
- [ ] 11.4 测量 PyTorch fallback 性能开销（预计 < 0.5ms），确认不成为瓶颈

## 12. MTP — Rejection Sampling 与 Sampler

- [ ] 12.1 验证 `HAS_TRITON=False` 时 310P 自动选择 PyTorch rejection sampling fallback 路径
- [ ] 12.2 验证 `rejection_greedy_sample_pytorch`：`torch.where` + 2D 索引矩阵向量化比较正确性
- [ ] 12.3 验证 `rejection_random_sample_pytorch` 和 `rejection_random_sample_block_verify_pytorch` PyTorch fallback
- [ ] 12.4 验证 `sample_recovered_tokens_pytorch`：`torch.argmax(residual / q)` 向量化采样正确性
- [ ] 12.5 验证 `AscendSampler310` 在 MTP 场景下的 CPU exponential 随机采样：`q.cpu().exponential_().npu()` 避免 310P NPU exponential 同步问题
- [ ] 12.6 验证 `_apply_top_k_top_p_pytorch` 在 MTP 场景下的正确性（非 AscendC kernel 路径）

## 13. MTP — 端到端与性能

- [ ] 13.1 MTP draft + verify + rejection sampling 端到端正确性测试：验证 draft token 生成、verify 比较、accept/reject 流程
- [ ] 13.2 MTP acceptance rate 测量：在不同输入上统计 310P 的 acceptance rate
- [ ] 13.3 MTP TPOT 性能测量：对比 eager decode vs MTP decode 的 TPOT 改善
- [ ] 13.4 确认 MTP eager 路径的收益条件：acceptance rate 足够高时，减少主模型 forward 次数的收益 > eager launch overhead

## 14. 组合验收

- [ ] 14.1 Chunk Prefill + Prefix Caching 组合测试：长序列 prefill + prefix cache 命中场景正确性
- [ ] 14.2 Chunk Prefill + MTP 组合测试：chunk prefill + MTP draft/verify 正确性
- [ ] 14.3 Chunk Prefill + Prefix Caching + MTP 三特性组合测试：全特性启用端到端正确性
- [ ] 14.4 组合性能测试：全特性启用 vs 基础 eager 推理的 TTFT/TPOT/吞吐对比
- [ ] 14.5 回归测试：确认三个特性启用后不影响已有 eager 基础推理的正确性
