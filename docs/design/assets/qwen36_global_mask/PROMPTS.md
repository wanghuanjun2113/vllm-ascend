# 配图生成记录

所有配图遵循 Infographic Engine 风格：白色或极浅灰背景、深海军蓝结构线、青色表示正常路径、橙色表示屏蔽和风险；不使用渐变、霓虹、玻璃拟态、人物或伪 3D。

## 01 overview

使用 Codex built-in image generation 生成。总体链路为 `Allow-list 文件 -> 启动时加载 -> NPU Bool Mask`，随后分成 `Target Logits` 和 `MTP Draft Logits` 两路，均执行 `Blocked -> -Inf`，最后进入 `Sampler / Verify -> English Output`。底部标注 `Embedding / LM Head 维度不变`。

## 02 mask generation

使用 Codex built-in image generation 生成。流程为 `Tokenizer 枚举 -> 单 Token 解码 -> ASCII / EOS / 必需协议 Token -> Allowed IDs / Blocked IDs -> Bool Mask -> 上 NPU + SHA 校验`，并强调每个模型单独生成 Mask。

## 03 runtime sequence

使用确定性 Pillow 绘制，确保时序和文字准确。参与者为服务进程、Target Model、Global Mask、Sampler / Verify、MTP Draft 和 Output Stream。图区分启动阶段和解码迭代，包含三个 MTP Draft step 的 repeat 区域，并标明 MTP3 单轮最多输出四个 Token。

## 05 validation method

使用确定性 Pillow 绘制，明确区分三条证据链：效果验证、精度验证、性能验证。三条链分别输出 `language_*.json`、`accuracy_*.json`、`perf_*.json`，再汇入 `RESULT_SUMMARY.json`、`RAW_RESULTS.md` 和设计说明书。
