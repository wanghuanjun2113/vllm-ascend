## 1. 配置层：移除 padding 和 block_size 膨胀

- [ ] 1.1 修改 `patch/platform/patch_mamba_config.py`：移除 `attn_block_size = kernel_block_size * cdiv(...)` 膨胀逻辑，block_size 保持 128 或用户设定值
- [ ] 1.2 修改 `patch/platform/patch_mamba_config.py`：移除 `mamba_page_size_padded = attn_page_size + conv_block_page_size` 的 padding 计算
- [ ] 1.3 修改 `worker/model_runner_v1.py` 的 `get_kv_cache_spec()`：移除 `page_size_padded` 强制对齐逻辑（L3460-3463），AttentionSpec 使用自然 page_size

## 2. 分配层：实现非连续子区域布局

- [ ] 2.1 设计子区域布局结构：在 `_allocate_kv_cache_tensors()` 中，将单一 raw tensor 划分为 KV Cache Region、SSM State Region、Conv State Region 三个连续子区域
- [ ] 2.2 实现子区域 offset 计算：定义各子区域的起始 offset 和大小，支持 `num_kv_blocks` 和 `num_ssm_slots` 独立设置
- [ ] 2.3 修改 `_allocate_kv_cache_tensors()` hybrid 路径：按子区域布局分配，确保各子区域起始地址满足 2MB alignment（PD disaggregation 场景）

## 3. Reshape 层：适配新布局

- [ ] 3.1 修改 `_reshape_kv_cache_tensors()` hybrid 路径（L3043-3084）：移除 `conv_block_padding_size` 计算，K/V cache 从 KV Cache Region 直接提取
- [ ] 3.2 移除 `block_size_chunk` 和 block splitting 逻辑（L3067-3076）：block_size=128 后不再需要拆分
- [ ] 3.3 修改 MambaSpec reshape 路径（L3167-3185）：conv_state 和 ssm_state 从各自的子区域提取
- [ ] 3.4 修改 `may_reinitialize_input_batch()`：移除 `kernel_block_sizes` 中与 hybrid block splitting 相关的逻辑

## 4. Block Table 和 Slot Mapping 适配

- [ ] 4.1 修改 `worker/block_table.py` 的 `compute_slot_mapping()`：`slot_id = kv_region_offset + block_number * block_size + offset_in_block`
- [ ] 4.2 修改 GDN 算子的 `ssm_state_indices` 和 `conv_state_indices` 计算：使用各自 region 内的偏移
- [ ] 4.3 确保 `BlockTable` 的 `_convert_physical_to_logical_blocks()` 在 block_size=128 下正确工作（移除不再需要的 split ratio）

## 5. KV Transfer 适配

- [ ] 5.1 验证 `mooncake_connector.py` 和 `mooncake_layerwise_connector.py` 的 KV cache 读写路径在新布局下正确
- [ ] 5.2 确保 `swap_blocks` / `copy_blocks` 的 index-based copy 在 KV Cache Region 内正确执行
- [ ] 5.3 验证 `npu_paged_cache_load` 和 `npu_scatter_pa_kv_cache` 的 block_table/slot_mapping 反映新 offset

## 6. 验证和测试

- [ ] 6.1 编写单元测试：验证子区域 offset 计算正确性
- [ ] 6.2 编写集成测试：Qwen3.6 27B 模型在新布局下 prefill + decode 正确性
- [ ] 6.3 对比测试：新布局 vs 旧 layout 的 attention 输出数值一致性
- [ ] 6.4 验证非混合注意力模型（Qwen2.5 等纯 Full Attention）不受影响
- [ ] 6.5 性能测试：对比新旧布局下的 KV cache 容量和显存使用
