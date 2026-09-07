# Logits Flight Recorder 交付入口

[统一下载与代码改动入口](https://github.com/wanghuanjun2113/vllm-ascend/releases/tag/logits-flight-recorder-20260907)

本交付用于 Qwen3.6-27B W8A8、vLLM-Ascend 0.23.0、TP=2、MTP=3、图模式下的逐生成位置 logits 记录与离线分析。主路径使用有界快照池，后台统计 Top-K 并持久化；离线页面支持非 Top-1、低概率、Top-1 与 Top-2 概率差筛选。

## 全部代码改动

| 仓库 | 内容 | 完整差异 |
| --- | --- | --- |
| vLLM | 后台记录进程、IPC、分布统计、持久化与采样/输出事件接入 | [从 v0.23.0 到任务分支](https://github.com/wanghuanjun2113/vllm/compare/0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665...feat/logits-flight-recorder) |
| vLLM-Ascend | NPU/CPU 槽池、MTP 映射、Ascend 接入、分析工具、HTML、测试与 Word 生成代码 | [从 v0.23.0 到任务分支](https://github.com/wanghuanjun2113/vllm-ascend/compare/5cb98caaadeff42b5b62b996e34bb2aaa29d20fd...feat/logits-flight-recorder) |

两个仓库分支均为 feat/logits-flight-recorder。Release 保存此次交付的精确 commit SHA，以及相对上述基线的完整 patch；本页面的分支链接会随后续提交更新。

## Release 附件

- flight-recorder-delivery.tar.gz：68.4 MB 原始离线包，包含实际日志、SQLite 索引、HTML 示例、分析工具、tokenizer、公开数据与许可证、测试结果。不是邮件分卷，可直接解压。
- logits-design-windows.docx：仅描述最终方案的设计说明书，含 4 张技术图、2 张真实页面截图及存储容量估算；使用微软雅黑、Arial、Consolas。
- flight-recorder-report-assets.zip：Word 生成所需图片、提示词与图片校验清单；生成脚本见 [report/README.md](../../tools/flight_recorder/report/README.md)。
- vllm-flight-recorder.patch 和 vllm-ascend-flight-recorder.patch：两个仓库的完整代码改动。
- delivery-manifest.json 和 SHA256SUMS：源码、环境及附件完整性信息。

直接查看已有 HTML 不需要模型或 Python。查询日志需 Python 与 NumPy；导出新请求页面另需 Transformers。包内不含模型权重和依赖安装包。

## 验证范围

指定负载下共 95 个含预热请求，22,392 个采样位置、22,004 个最终输出 token 审计通过；15 项回归测试通过。Top128 的 TPOT 中位数增加 0.84%～3.15%，Top1024 增加 4.47%～5.54%。这些是固定公开子集与本次环境的实测结果，不是其他负载的性能保证。

发布阶段核对远端提交、生成完整 patch、复建 Word 生成链路并校验附件，不重新运行模型基准。Word 已做容器内替代字体渲染检查，尚未在 Windows 实机验证。运行参数及原始指标以各 run 的 manifest 和结果文件为准。
