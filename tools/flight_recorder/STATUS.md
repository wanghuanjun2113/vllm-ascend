# 完成状态（2026-09-07）

环境：41_7 / vllm-023-logits-save。
模型：Qwen3.6-27B-W8A8，TP2，MTP3，target FULL_DECODE_ONLY / draft FULL。

实现：默认cpu_pool有界快照与后台CPU统计；inline参考后端保留。
HTML：排名>1、所选token概率阈值、同一步Top1-Top2差值阈值、raw/final选择、AND组合；真实日志浏览器测试通过。
正式池化测试：pooled128、pooled1024；共95个含预热请求，22004个最终输出token，22392个采样位置，审计全部通过。
C1 TPOT：关闭9.602ms；池化Top128 9.905ms（+3.15%）；池化Top1024 10.032ms（+4.47%）。
C4 TPOT：关闭14.474ms；池化Top128 14.595ms（+0.84%）；池化Top1024 15.277ms（+5.54%）。
贪心所有对照均16/16输出逐token一致。随机C1 4/4一致，随机C4不保证一致。
最终回归15项通过。NPU及临时快照槽已核对释放，回环HTML预览服务保留用于交付。

结果、实际命令、manifest、审计、环境、测试日志位于 /home/w00498770/dev/artifacts/vllm-023-logits-save。
说明书：docs/design/logits-flight-recorder.md。
默认CPU绑定已关闭，避免宿主IRQ写入；早期默认路径触发过该写入，未擅自恢复原值。
未来启动配置使用 --shutdown-timeout 30；本次各run保留原始启动参数。
