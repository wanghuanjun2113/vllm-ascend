# 配图生成记录

- 工具：Codex built-in `image_gen`
- 样式库模板：`Infographic Engine`
- 示例案例：case 334、case 1、case 8
- 统一风格：扁平化工业技术信息图 / Technical Whitepaper Infographic；白色或极浅灰背景、深海军蓝结构线、青色正常路径、橙色屏蔽/风险路径；禁止渐变、霓虹、玻璃拟态、人物、伪 3D 和水印。

## 01-overview.png

16:9 总体架构图。按 `Allow-list 文件 → 启动时加载 → NPU Bool Mask` 展开，分为 `Target Logits` 和 `MTP Draft Logits` 两路，均经过 `Blocked → -Inf`，汇入 `Sampler / Verify → English Output`；底部精确标注 `Embedding / LM Head 维度不变`。

## 02-mask-generation.png

16:9 Mask 生成流程图。模块为 `Tokenizer 枚举 → 单 Token 解码 → ASCII / EOS / 必需协议 Token? → Allowed IDs / Blocked IDs → Bool Mask: False / True → 上 NPU + SHA 校验`；侧栏标注 `每个模型单独生成`、`Qwen3.6` 和 `DS V4 Flash`。

## 03-runtime-loop.png

16:9 投机解码时序图。主链为 `Request Batch → Target Forward → Target Logits → Global Mask → Sample + Verify → Accept / Reject → Output Tokens`；下方三个模块分别为 `MTP Step 1/2/3: Draft Logits → Global Mask → Argmax`，虚线标注 `异步调度`。Bool 图例必须为 `允许 (False)`、`屏蔽 (True)`；MTP3 最大单轮输出标注 `B × (≤ 4)`。

## 04-model-adaptation.png

16:9 双栏模型适配图。左栏 `Qwen3.6 27B / Vocab 248320 / Target LM Head / MTP Draft Head / 同一 Bool Mask`；右栏 `DS V4 Flash / Vocab 129280 / Main Head / MTP / DSpark Head / 模型专属 Mask`；中央原则 `Mask Logits, 不改权重`；底部边界 `Qwen: 已穿刺`、`DS V4: 仅方案分析`。

## 05-validation-method.png

16:9 验证方法图。8 张 NPU 划分为 `TP4 Base: NPU 0-3` 和 `TP4 Mask: NPU 4-7`，显示 `AB / BA 卡组互换`；负载为 `Natural English / HellaSwag / 8K Input / 1K Output / Concurrency 8 / 80 Requests`；证据链为 `中文输出用例`、`ARC-Challenge + HellaSwag`、`TTFT / TPOT / Throughput / Accept Rate → Raw JSON → RAW_RESULTS.md → 设计说明书`。
