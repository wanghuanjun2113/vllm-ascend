# Logits flight recorder

本工具用于 Qwen3.6-27B-W8A8 / vLLM-Ascend 0.23.0 / TP2 / MTP3 的事后回溯。
输入 token 与所有生成位置均可追溯；每个位置保存两个阶段的 Top-K 和实际输出/EOS。

## 运行边界

本任务所有命令在 41_7 的 vllm-023-logits-save 容器内运行。两个仓库需要同时使用
feat/logits-flight-recorder 分支。不能只安装 Ascend 修改而漏掉 vLLM 的采样和输出事件钩子。

设置 PYTHONPATH 时要保留 CANN 原有路径。编译依赖使用同一个容器镜像里的产物。
服务建议设置 --shutdown-timeout 30，给记录池排空及清理留出时间。
本任务的 manage.py 同时核对整个任务进程组退出并清理对应的临时槽。
服务启动必须设置 --additional-config '{"enable_cpu_binding":false}'：
框架默认 CPU 绑定会写宿主机 IRQ 亲和性，不适用于本项目的宿主机只读边界。

本次任务目录：
- /home/w00498770/dev/worktrees/vllm-023-logits-save/vllm
- /home/w00498770/dev/worktrees/vllm-023-logits-save/vllm-ascend
- /home/w00498770/dev/artifacts/vllm-023-logits-save

## 开启记录

在服务进程启动前设置以下变量，输出目录每次使用新目录：

```bash
export VLLM_FLIGHT_RECORDER_DIR=/home/w00498770/dev/artifacts/vllm-023-logits-save/my-run/trace
export VLLM_FLIGHT_RECORDER_BACKEND=cpu_pool
export VLLM_FLIGHT_RECORDER_TOPK=128
export VLLM_FLIGHT_RECORDER_PROBES=248044,248046
export VLLM_FLIGHT_RECORDER_QUEUE=8
```

默认 cpu_pool：NPU与固定页CPU/共享槽池，独立CPU进程4线程统计；inline为参考后端。
不设置 VLLM_FLIGHT_RECORDER_DIR 即关闭。Top-K 记录配置不改变 sampling top_k。
EOS probe 必须与实际模型及停止 token 对应。完整启动命令及环境见每个 run 的 manifest.json
和任务 artifacts 根目录 launch.json。

复用已核对的本任务启动配置：

```bash
cd /home/w00498770/dev/worktrees/vllm-023-logits-save/vllm-ascend/tools/flight_recorder
python manage.py start --name my-run --top-k 128
# 测试结束后，停止本工具记录的服务 PID；不会停止 Docker。
python manage.py stop
```

## 离线查询

```bash
python analyze.py /path/to/run/trace list
python analyze.py /path/to/run/trace audit
python analyze.py /path/to/run/trace query --request REQUEST_ID --position 20
python analyze.py /path/to/run/trace html --request REQUEST_ID \
  --tokenizer /mnt/weights/Qwen3.6-27B-w8a8 --out /path/to/result.html
```

查询/审计仅需 Python + NumPy，不需要 NPU/vLLM。HTML导出还需要本地对应 tokenizer 和
Transformers。页面支持原始/处理后分布、采样排名>1、所选token概率阈值、同一步Top1-Top2概率差阈值，条件按AND组合；默认排除停止后的丢弃尾部。
生成的 HTML 不访问 CDN、不需要服务器；同目录 JSON 可用页面的文件选择器载入。
默认只导出一个请求，避免把整个测试集的压缩二进制膨胀成大量文本。

## 测试

```bash
python -m pytest -q test_recorder.py
RUN_NPU_TESTS=1 ASCEND_RT_VISIBLE_DEVICES=2 python -m pytest -q test_recorder.py
```

第二条只能在确认 NPU2 为本任务可用资源后运行。端到端公开用例：

```bash
python benchmark.py --datasets /path/to/datasets --out /path/to/results \
  --name example --n 8 --max-tokens 512 --concurrency 1 4
```

数据文件是 MGSM mgsm_zh.tsv 和 GSM8K test.jsonl（本任务命名 gsm8k_test.jsonl）。
必须保留源文件、许可证及校验和。固定示例子集用于采集机制穿刺，不是官方完整基准分数。

## 已知边界

- 只保存 target 的两个阶段与实际输出/EOS，没有完整长尾和完整 draft 分布/RNG。
- MTP 的 rejected draft 概率不能从 target Top-K 精确重建。
- enable_reduce_sample 明确拒绝，避免局部词表被当成全局词表。
- PP/DP、多模态和可续接流式输入未验证。
- 队列满会背压。写入失败不会静默降采样；检查 FAILED 标记和 audit，API成功不代表trace完整。
- 突然杀进程可能丢失队列尾部；只有持久化并通过审计的记录才可视为完整。
- 性能报告使用实际输出长度；length 停止不当成模型自然EOS。
