# 推理 Logits 飞行记录器设计说明书

## 1. 背景介绍

低概率的语言切换、提前结束和内容错误往往无法在事后稳定复现。仅保存最终文本无法判断：模型原本偏好什么、采样约束改变了什么、实际提交了哪个 token，以及结束来自 EOS、停止字符串还是长度上限。

本任务对每个生成位置保存有界 Top-K 证据，而不是全词表。记录服务于事后诊断，不承诺随机过程逐位重放，不把高概率解释为语义正确，也不将人工可见的第一个错误视为根因起点。

指定环境：41_7 / vllm-023-logits-save；Qwen3.6-27B-w8a8；vLLM-Ascend 0.23.0；TP=2、MTP=3、target FULL_DECODE_ONLY，draft FULL。输入为纯文本，使用模型原始 chat template。模型目录的架构字段为 Qwen3_5ForConditionalGeneration，不据此改称其他模型。词表 248,320，量化配置来自现存 W8A8 权重目录。

## 2. 整体方案

控制流为：模型计算 logits → pre-grammar 原始快照 → grammar/penalties/temperature/top-k/top-p → target 候选快照 → 普通采样或 MTP 验证 → 采样输出定位 → 小摘要异步 D2H → 有界传输队列 → 独立 CPU 写入进程 → SQLite 索引和压缩二进制数据块。

前端独立记录有效输入 token IDs、完整生效采样参数、输出 token 流、external/internal request ID 对应、最终 finish_reason/stop_reason 和取消事件。离线工具将 worker 的采样输出与 scheduler/frontend 最终提交序列对齐。

配置 VLLM_FLIGHT_RECORDER_TOPK 与生成参数 top_k 独立。默认捕获128，可切换1024；不能在历史128记录上恢复1024。两阶段的Top-K独立保存，另行补录实际输出及EOS。greedy的softmax仅表示偏好；MTP目标分布不是接受/拒绝过程的完整抽样概率。

存储为每进程shard：PID.sqlite、PID.bin、PID.closed；run manifest单独保存。二进制块采用NPZ的无损DEFLATE压缩；这是对初步方案Zstd的简化，无新增压缩依赖。SQLite存每个请求/token位置到块偏移的索引，每块有SHA256。ID为int32，logits为FP32，实际正词表ID不超过int32范围。没有文本化的逐候选重复字符串。

每token两阶段候选数据为16K字节：Top128 2KiB，Top1024 16KiB；词表248320的两阶段完整FP32为1,986,560字节。这些数字不含实际输出/EOS、索引、块头和归一化信息，也没有假设压缩比。

## 3. 详细设计

### 3.1 主要修改文件

vLLM：
- vllm/v1/sample/flight_recorder.py：有界队列、异步传输线程、独立写入子进程、数据块和SQLite索引、失败标记、输入/输出事件接口。
- vllm/v1/sample/sampler.py：在采样处理器之后、greedy/随机分支之前记录候选，保留greedy行的实际语义。
- vllm/v1/sample/ops/topk_topp_sampler.py：native fallback的截断后采集点。
- vllm/v1/engine/output_processor.py：请求元信息、最终提交token、停止与取消事件。

vLLM-Ascend：
- vllm_ascend/sample/flight_recorder.py：NPU快照、Top-K、logsumexp、特殊值计数、实际输出/EOS值与全局排名、MTP行映射和D2H。
- vllm_ascend/worker/model_runner_v1.py：grammar前建立快照，_sample之后提交记录。
- vllm_ascend/sample/sampler.py：Ascend随机采样截断后记录。
- vllm_ascend/sample/rejection_sampler.py：bonus与target行映射、constraints之后的目标分布。
- tools/flight_recorder/analyze.py：纯CPU离线列表、查询、审计、HTML导出。
- tools/flight_recorder/viewer.html：无CDN和服务依赖的HTML浏览器。
- tools/flight_recorder/benchmark.py：公开用例、自然EOS、流式客户端测量。
- tools/flight_recorder/manage.py：本任务服务启动与进程清理，不操作Docker生命周期。
- tools/flight_recorder/test_recorder.py：传输、损坏、失败及NPU完整词表对照测试。

### 3.2 数据模型与记录位置

每个采样输出具有request_id、position、token_id、chunk、row。batch元信息保存当时req_ids顺序、MTP草稿数量/IDs、槽宽、丢弃请求行、Top-K及验证策略参数。最终采样槽映射到target logits行，padding槽不写入索引。

每个输出位置包含：
- raw_ids/raw_values与final_ids/final_values；
- raw_lse/final_lse：完整捕获域上的logsumexp；
- raw_counts/final_counts：finite、NaN、+Inf、-Inf；
- raw_probe_values/final_probe_values：实际输出+指定EOS；
- raw_probe_ranks/final_probe_ranks：全词表严格大于该值的元素数+1；并列同名次，非有限分布需结合计数解释；
- final_seen：验证实际到达了处理后采集点；
- selected、flat_slots。

输入只记录token IDs，不额外计算每个输入位置的prompt logits。prefill后的第一个生成token纳入采集。模型快照位于grammar之前，避免把经过语法屏蔽的sampler raw误称为原始模型输出。原始FP16/BF16数值转换为FP32不会恢复模型未计算的精度。

MTP数据只保留最终采样输出对应的两阶段target分布和完整草稿token IDs/数量。它能识别accepted_draft、replacement、bonus，以及被scheduler停止逻辑截断的尾部；不保存全部draft概率、验证随机数或所有被拒绝位置的完整候选分布，不支持精确重放拒绝采样。

### 3.3 异步与完整性

模型侧只复制小摘要到pinned CPU buffer，用NPU Event通知传输线程。线程在事件完成后通过有界IPC管道发送数据；独立写入进程负责压缩、fsync和SQLite事务，并返回确认。默认队列8个batch，满时背压，不丢token、不自动降低K。

队列容量以batch数计：单batch受服务max_num_seqs、MTP深度及K共同约束。运行中还有当前步骤的原始和处理后完整设备tensor，用于补录Top-K外实际输出；这些只在当前步骤暂存。复制所在流必须遵循tensor生命周期，不能在D2H完成前回收CPU buffer。

写入失败生成FAILED标记并在后续提交/flush时抛错；已经向客户端返回的成功响应不代表trace完整。读取时检查SHA256、位置连续、输入事件、final_seen以及最终输出逐token匹配。正常采样但被停止逻辑丢弃的尾部单独报告unused_after_stop，不将其混入最终文本。进程崩溃可能丢失尚在队列内的记录；此时只能报告不完整，不能宣称零丢失。

### 3.4 查询和HTML

analyze.py只需CPU及NumPy；HTML导出额外需要与运行一致的本地tokenizer和Transformers。list列出请求；query定位一个生成位置；audit扫描完整性；html导出单请求独立HTML和JSON。

HTML显示输入/输出、token位置导航、原始Top-K覆盖率曲线、两个阶段的候选表、实际输出高亮、EOS名次、采样参数和停止原因。支持候选文本/ID搜索与载入其他导出JSON。候选文本使用textContent写入；嵌入JSON转义小于号，防止数据中的HTML/script字符串成为代码。

### 3.5 配置与边界

采集环境变量：
- VLLM_FLIGHT_RECORDER_DIR：本次运行独立目录，未设置则关闭。
- VLLM_FLIGHT_RECORDER_TOPK：128或1024等固定K。
- VLLM_FLIGHT_RECORDER_PROBES：本任务使用248044,248046，须与运行时EOS/停止token对应。
- VLLM_FLIGHT_RECORDER_QUEUE：默认8。

当前支持本任务的TP2全词表采样和MTP路径。enable_reduce_sample启用时明确报不支持，不把单卡局部词表当全词表记录。未验证PP/DP、多模态输入、可续接流式输入等路径，不扩大兼容性结论。raw/final不包含每一种处理器的中间状态，无法单凭两端快照归因到某一个处理器。分布不是语义正确性的标签。

runtime worktree复用同一容器镜像中的.so与完整custom OPP包；不重新编译算子，不使用其他容器的产物。启动时追加CANN Python路径并设置custom OPP及op_api库路径。

## 5. 穿刺结果

待本次端到端测试完成后填写实测结果，当前不能以单元测试替代模型端到端证据。

已运行：
- CPU传输、MTP索引、写入失败和损坏检测4项通过。
- 实际NPU上Top128/1024与完整2048词表的值、ID、归一化量、Top-K之外实际输出、EOS屏蔽计数对照通过；合计6项通过。小词表测试只验证采集正确性，不是目标模型质量或性能证明。

计划对照：固定MGSM中文与GSM8K英文子集，温度0、自然EOS、相同chat template与输出上限、相同TP2/MTP3/图模式；关闭记录/128/1024分别测C1和C4。TPOT为流式客户端首末token块时间除以输出间隔，MTP会一次返回多个token，不能将它称为每个内部token的独立ITL。报告真实输出长度、正确性、错误、MTP接受统计和磁盘用量。

公开数据来源：
- https://github.com/google-research/url-nlp/tree/main/mgsm ：使用中文TSV，项目说明为从GSM8K选择250题的人类翻译，不能把两个数据集当独立题库。
- https://github.com/openai/grade-school-math ：GSM8K test.jsonl。
数据及许可证保存在artifacts/datasets，附来源URL、Git blob SHA和SHA256；不提交原始测试日志到代码仓库。
