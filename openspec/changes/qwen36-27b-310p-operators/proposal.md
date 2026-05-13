## Why

Qwen3.6 27B 采用 Full Attention + Linear Attention (GDN) 混合注意力架构，在 300IDuo (310P3) 上推理时，910 平台的 Triton kernel 和部分 AscendC 算子无法直接使用。为实现 310P 上的推理性能基线（TPOT ≤ 60ms），需要系统性地规划 12+ 个算子的开发、适配与集成工作，涵盖 GDN 核心、QKV 处理、注意力优化和投机推理四个领域。

## What Changes

### GDN 核心算子（已具备，Q2 商发）
- `causal_conv1d_fwd_kernel`: 因果 1D 卷积前向 kernel，GDN prefill 路径
- `causal_conv1d_update_kernel`: 因果 1D 卷积递推更新 kernel，GDN decode 路径
- `fused_recurrent_gated_delta_rule_fwd_kernel`: 融合递推 gated delta rule 前向 kernel，GDN decode 路径

### GDN 核心算子（开发中）
- `chunk_gated_delta_rule_fwd`: Chunk 并行 gated delta rule 前向，GDN prefill 主路径（SAIE 开发中）
- `fused_sigmoid_gating_delta_rule_310`: 融合 sigmoid gating delta rule，GDN decode 单步更新（SAIE 开发中，NAIE 支撑）

### QKV/RoPE 算子
- `mRope`: 多模态 Rotary Position Embedding 算子，高优先级（NAIE 开发）
- `split_qkv_rmsnorm_Mrope`: 融合 QKV split + RMSNorm + MRoPE，低优先级（NAIE 开发）
- `rmsnormgated`: 融合 RMSNorm + SiLU gating，低优先级（NAIE 开发）
- `transposeKV`: KV Cache 转置算子，低优先级（NAIE 开发）

### 注意力/投机推理算子
- `fused_gdn_gating`: GDN gating 融合算子（exp + softplus + sigmoid），高优先级（NAIE 开发）
- `FIA 算子增加压缩 mask`: Flash Infer Attention 压缩 mask 支持，高优先级（NAIE 能力中心开发）
- `rejection_greedy_sample`: Greedy 拒绝采样（含 spec_len=1 快速路径），NAIE 能力中心开发
- `rejection_random_sample`: Random/stochastic 拒绝采样，NAIE 能力中心开发
- `rejection_random_sample_block_verify`: Block verify 拒绝采样（max_spec_len≥3 时启用），NAIE 能力中心开发
- `sample_recovered_tokens`: 残差分布恢复采样（Gumbel-like），NAIE 能力中心开发
- `expand_batch_to_tokens`: Per-request 到 Per-token 参数扩展，NAIE 能力中心开发
- `npu_copy_and_expand_eagle_inputs`: Eagle/MTP 投机推理输入扩展（C++ 算子），NAIE 能力中心开发
- `prepare_inputs_padded`: Spec decode 输入准备（计算 token_indices_to_sample、num_rejected），NAIE 能力中心开发
- `apply_sampling_constraints`: 采样约束应用（temperature/top_k/top_p），NAIE 能力中心开发

## Capabilities

### New Capabilities
- `310p-gdn-ops`: GDN 混合注意力核心算子集（causal_conv1d、gated_delta_rule、sigmoid_gating），含 prefill/decode 双路径
- `310p-qkv-rope-ops`: QKV 处理与 RoPE 编码算子集（mRope、split_qkv_rmsnorm_Mrope、rmsnormgated、transposeKV）
- `310p-attention-spec-ops`: 注意力优化与投机推理算子集（FIA compressed mask、rejection sampling）

### Modified Capabilities
<!-- 无现有 spec 级别的行为变更，本次为全新算子能力建设 -->

## Impact

- **算子层**: 新增 12+ 个 AscendC/PyTorch 算子的 310P 适配层，分布在 `vllm_ascend/_310p/ops/` 和 `csrc/` 目录
- **模型 Patch**: `patch_qwen3_6.py` 中条件选择 310P 算子实现
- **注册机制**: `torch.library` / ATB 算子注册，需适配 310P AICore 架构
- **依赖**: SAIE 和 NAIE 团队算子开发交付节奏，影响 310P 性能达标时间线
- **测试**: 每个算子需正确性测试（与 910 参考实现对比）和性能 benchmark
