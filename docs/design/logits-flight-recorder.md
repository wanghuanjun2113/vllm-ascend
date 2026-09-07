# 推理 Logits 记录与离线分析设计说明书

验证日期：2026-09-07。环境：41_7 / vllm-023-logits-save。

## 1. 背景介绍

低概率的中英文切换、提前结束和内容错误往往难以复现。只保留最终文本，无法判断异常位置的模型偏好、采样处理影响、实际选择和终止原因。本工具覆盖全部请求的全部生成位置，保存有界 Top-K、实际输出和 EOS 的证据，供事后回溯。

“全部生成位置”包括 prefill 后的第一个输出、MTP 接受/替换/bonus token、EOS 等特殊 token。输入 token IDs 完整记录一次，不额外计算每个输入位置的 prompt logits。异步执行中停止后的推测尾部仍可记录，但与最终输出分开。

本工具用于诊断，不将低概率、非最高概率采样直接判断为错误；高概率也不证明语义正确。保存 Top-K 无法恢复全部长尾，更不能仅凭 seed 保证随机采样逐位重放。

## 2. 整体方案

最终默认使用 cpu_pool 后端。主采样路径只保存快照和元信息，统计与落盘在后台执行。

```mermaid
flowchart LR
    A["模型输出"] --> B["NPU 槽池：raw 快照"]
    B --> C["原有约束与采样"]
    C --> D["同槽保存 final 与实际 token"]
    D --> E["事件通知复制线程"]
    E --> F["固定页 CPU 缓冲"]
    F --> G["共享内存槽"]
    G --> H["独立进程：4 个 CPU 线程统计"]
    H --> I["Top-K 二进制块及 SQLite 索引"]
    I --> J["持久化 ACK"]
    J --> K["归还槽"]
```

raw 位于 grammar mask 之前；final 是目标模型经过采样处理后的分布。普通 greedy、随机采样、MTP bonus 和 target 验证分别接入对应采集点。记录不改变温度、top-k/top-p、约束和采样算法。

主要取舍是用有界的短期内存与异步传输，换取采样关键路径上更少的工作。完整快照只短期存在于槽池；磁盘仍只保存 Top-K 和少量诊断字段。池满时施加背压，不丢弃记录、不自动降低 K。

与初版相比，优化同时引入缓冲复用、后台 CPU 统计和共享内存传递，不能把收益单独归因于“池化”一项。

### 配置下的缓冲容量

本次 max_num_seqs=4、MTP=3，因此最多按 16 行 logits 配置槽；词表 248,320，默认 8 槽，两阶段 FP32：

```text
单类缓冲容量 = 8 × 2 × 16 × 248320 × 4
             = 254,279,680 字节 ≈ 242.5 MiB
```

NPU 快照、固定页 CPU 缓冲、CPU 共享区各一份，合计约 727.5 MiB 的主要载荷容量；其中额外 NPU 容量约 242.5 MiB。这是按分配形状计算的容量，不是进程 RSS/PSS 或分配器峰值测量。仅负责记录的 TP rank 0 创建这些池。

CPU 共享区的实际文件大小也核对为 254,279,680 字节。提高并发、MTP 深度或槽数时必须按公式重新计算。

## 3. 详细设计

### 3.1 环境和目录

- 镜像：quay.io/ascend/vllm-ascend:v0.23.0-openeuler；ID cb25ae391cad9549f884b9ba57877e67b5a1da6f9200b870dde15b38df6696ba。
- Python 3.12.13，PyTorch 2.10.0+cpu，torch-npu 2.10.0.post4，vLLM 0.23.0+empty，vLLM-Ascend 0.23.0，CANN 9.1.0，npu-smi/驱动报告 26.0.rc1。
- 模型：/mnt/weights/Qwen3.6-27B-w8a8；量化描述及 best-practice YAML 确认 W8A8。配置架构字段为 Qwen3_5ForConditionalGeneration，此字段不用于改称用户指定模型。
- vLLM worktree：/home/w00498770/dev/worktrees/vllm-023-logits-save/vllm。
- Ascend worktree：/home/w00498770/dev/worktrees/vllm-023-logits-save/vllm-ascend。
- artifacts：/home/w00498770/dev/artifacts/vllm-023-logits-save。
- 两仓库分支：feat/logits-flight-recorder。源码和结果均在授权目录内，通过目标容器写入。

官方 v0.23.0 标签已核对：vLLM 0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665，Ascend 5cb98caaadeff42b5b62b996e34bb2aaa29d20fd。池化核心验证对应 vLLM a20a56b、Ascend d83abb560；各 run 的 manifest.json 保存启动参数、初始 HEAD/diff 指纹及首次请求前的来源快照。最终工具/文档提交见交付目录 delivery.json。

运行时复用同一容器镜像内的编译产物，通过 worktree 的运行时链接及 CANN/custom OPP 环境定位；没有重新编译模型算子，没有使用其他容器的源码或测试结果。

### 3.2 主要修改文件

| 仓库 | 文件 | 主要改动 |
|---|---|---|
| vLLM | vllm/v1/sample/flight_recorder.py | 有界 IPC、独立写入进程、共享快照统计、索引/校验/ACK、完整采样参数事件 |
| vLLM | vllm/v1/sample/sampler.py | 采样处理之后的回调；保留 greedy 分支语义 |
| vLLM | vllm/v1/sample/ops/topk_topp_sampler.py | native 随机路径截断后的采集回调 |
| vLLM | vllm/v1/engine/output_processor.py | 输入、最终提交 token、停止和取消事件 |
| Ascend | vllm_ascend/sample/flight_recorder.py | 采集入口、backend 选择、inline 参考实现 |
| Ascend | vllm_ascend/sample/flight_recorder_pool.py | NPU/固定页 CPU/共享槽池、事件和后台复制、槽生命周期 |
| Ascend | vllm_ascend/sample/sampler.py | Ascend 随机采样处理后的快照点 |
| Ascend | vllm_ascend/sample/rejection_sampler.py | bonus/target 行映射及处理后 target 快照 |
| Ascend | vllm_ascend/worker/model_runner_v1.py | grammar 前开始记录、采样后提交、实际图模式证据 |
| Ascend | vllm_ascend/worker/worker.py | 退出时尝试排空并关闭记录池 |
| Ascend | tools/flight_recorder/analyze.py | 纯 CPU 列表、查询、完整性审计、tokenizer 校验和 HTML 导出 |
| Ascend | tools/flight_recorder/viewer.html | 独立离线 HTML、概率筛选、位置导航及候选查看 |
| Ascend | tools/flight_recorder/benchmark.py | 固定公开用例、自然 EOS、并发预热及流式计时 |
| Ascend | tools/flight_recorder/manage.py | 本任务进程组管理、退出检查和临时槽清理 |
| Ascend | tools/flight_recorder/test_recorder.py | 数值、MTP 映射、异步复用、失败与进程生命周期回归 |
| Ascend | tools/flight_recorder/profile_components.py | 独立组件耗时穿刺，不替代 E2E |
| Ascend | tools/flight_recorder/run_pool_matrix.py、summarize.py | 测试编排和结果汇总 |

### 3.3 关键代码改动

模型 runner 在 grammar 修改前调用 begin，在 _sample 返回后提交；因此 raw 不是已经被语法屏蔽的 sampler 输入：

```python
flight_capture = begin(self, logits, spec_decode_metadata)
# 原有 grammar 和 _sample 流程继续执行
if flight_capture is not None:
    flight_capture.finish(sampler_output)
```

池内快照使用独立存储，后续采样器原地修改 logits 不会覆盖 raw：

```python
slot.raw[:self.nrows].copy_(logits)
slot.filled[:self.nrows].zero_()
# processed() 中：
self.slot.final.index_copy_(0, rows, logits.to(torch.float32))
```

后台复制线程必须独立启用 inference_mode。PyTorch 的该状态是线程局部的，不能假定由主推理线程继承：

```python
@torch.inference_mode()
def _copy_loop(self):
    # 等待快照事件，异步复制到固定页 CPU 缓冲，再写共享槽
    ...
```

消费者完成统计、二进制数据 fsync 和 SQLite 提交后发送 ACK。槽归还发生在 ACK 之后：

```python
self.writer.submit(meta, arrays,
                   on_done=lambda s=slot: self.release(s))
```

CPU 统计读取当前槽中的完整词表，随后仅保存 Top-K：

```python
ids = np.argpartition(row, vocab-k)[-k:]
vals = row[ids]
order = np.lexsort((ids, -vals))
maximum = float(np.max(row))
# 有限分布用 FP64 稳定计算 logsumexp；NaN/Inf 保留为证据。
```

raw/final 数值相同时复用统计结果；MTP 实際输出槽先映射到正确 target 行，再计算实际输出/EOS 的值和全词表排名。不会用 rejected draft 的占位槽冒充输出位置。

### 3.4 记录格式

每次运行使用独立目录。每个生产进程一个 shard：

```text
run/manifest.json
run/trace/PID.sqlite
run/trace/PID.bin
run/trace/PID.closed             # 正常写入进程退出标记
run/trace/FAILED-*               # 写入或复制失败
tokenizer/                      # 共用的本次 tokenizer 快照
sampling-defaults.json          # 兼容早期稀疏参数记录
```

SQLite 索引保存 request_id、position、token_id、chunk、row。数据块记录偏移、字节数和 SHA256。PID.bin 内连续追加 NPZ 压缩块，使用 DEFLATE 无损压缩，无额外 Zstd 依赖。

每个生成位置记录：
- 两阶段 Top-K ID/int32 和 logits/FP32；
- 实际输出与配置 EOS 的 logits、严格大于计数加一得到的全词表排名；
- 完整捕获域的 logsumexp；池化版 CPU 使用 FP64 统计；
- finite、NaN、+Inf、-Inf 数量及 final_seen；
- MTP 草稿 IDs/数量、输出槽位置、普通/接受/替换/bonus 来源；
- batch 对应请求、时间、实际 NONE/FULL 图模式及采样/验证策略标记。

请求事件保存实际输入 token IDs 和生效 SamplingParams 的全部字段，包括默认值；最终输出事件保存提交的 token 和 finish_reason/stop_reason。早期 inline Top128 记录使用了 msgspec 的省略默认值编码，工具通过固定版本的 sampling-defaults.json 恢复并明确标注来源，不修改原始事件。新版直接记录完整字段。

tokenizer 快照约 22.9 MB，一次保存供各 run 共用。导出前校验 tokenizer.json、tokenizer_config.json、vocab.json 指纹，避免错误解码历史 ID。

### 3.5 概率和 MTP 解释边界

```text
p(token) = exp(logit(token) - logsumexp(full_logits))
Top-K mass = sum(p(topk_ids))
gap = p(Top-1) - p(Top-2)
```

不能只对 Top-K 做 softmax。CPU FP64 归一化提供目标分布的数值统计，不承诺逐位复现设备 softmax 的舍入。

greedy 的 softmax 用于分析模型偏好，不是其确定性 argmax 的抽样概率。MTP 记录的是 target 分布，replacement 的条件抽样概率、完整 draft 分布和验证 RNG 没有全部保存；不能据此精确重放接受/拒绝过程。实际输出、来源和停止原因必须一起看。

### 3.6 HTML 筛选和查询

三类筛选均可选择 raw 或 final，勾选的条件按 AND 组合：
1. 采样排名 > 1；并列最高概率的候选排名都是 1。
2. 所选 token 在目标分布中的概率小于指定阈值。
3. 同一步 Top-1 与 Top-2 的概率差小于指定阈值。

默认只筛选最终输出，可勾选包含停止后丢弃的位置。未知概率或差值不匹配概率条件。箭头在匹配位置之间移动，输入框可直接定位任意位置，并标识不符合当前筛选的直接查询。

HTML 同时显示完整输入/输出、候选序号、token ID/text/logit/probability、EOS 名次、Top-K 覆盖率、图模式、来源和参数。候选序号与并列情况下的采样排名不是同一概念。页面不依赖 CDN 或服务器；导出后直接打开即可。

```bash
python analyze.py /path/to/run/trace list
python analyze.py /path/to/run/trace audit
python analyze.py /path/to/run/trace query --request REQUEST_ID --position 75
python analyze.py /path/to/run/trace html --request REQUEST_ID \
  --tokenizer /path/to/tokenizer --out result.html
```

查询只需要 Python 和 NumPy；HTML 导出额外需要本地匹配 tokenizer 与 Transformers。只按需导出单请求，避免把整个压缩测试集展开成大量文本 JSON。

### 3.7 配置、故障和退出

```bash
export VLLM_FLIGHT_RECORDER_DIR=/path/to/new-run/trace
export VLLM_FLIGHT_RECORDER_BACKEND=cpu_pool
export VLLM_FLIGHT_RECORDER_TOPK=128
export VLLM_FLIGHT_RECORDER_PROBES=248044,248046
export VLLM_FLIGHT_RECORDER_QUEUE=8
```

未设置 DIR 即关闭。inline 保留为参考后端。记录的 TOPK 与生成参数 top_k 完全独立。

队列满会背压。写入/复制失败会被上报，不能把 HTTP 成功等同于日志完整。审计检查 SHA256、索引/载荷一致、位置连续、final_seen 和最终输出逐 token 对齐。异常杀进程可能丢失未持久化尾部，此时只能标记不完整。

本镜像默认 shutdown_timeout=0 会强制终止 engine，不能依赖所有退出回调都执行。交付启动配置增加 --shutdown-timeout 30；manage.py 还检查整个已验证任务进程组退出，并仅清理对应已退出创建者的临时槽。性能表来自各 manifest 中保存的实际启动参数；退出宽限不改变推理参数。本次测试结束后另外核对并清理了已退出任务的临时槽。

enable_reduce_sample 明确报不支持，避免把局部词表误记为全词表。PP/DP、多模态输入、可续接流式输入等未验证，不扩大兼容性结论。

正式启动必须 --additional-config '{"enable_cpu_binding":false}'。早期试启动触发了框架默认的宿主 IRQ 亲和性写入路径；发现后关闭该功能，正式对照全部采用关闭设置。原始 IRQ 值未记录，因此未擅自写宿主机进行恢复。

## 5. 穿刺结果

### 5.1 工作负载与验收边界

模型为指定的 Qwen3.6-27B-W8A8，TP=2、MTP=3，target FULL_DECODE_ONLY、draft FULL；max_model_len=4096、max_num_seqs=4、GPU memory utilization=0.75。测试使用物理 NPU0/1，单元测试另使用任务空闲的 NPU2。

固定 MGSM 中文前8题和 GSM8K 英文前8题；每种配置分别运行 C1/C4。温度0、seed42、生成 top_p=1/top_k=-1（框架 greedy 会记录归一化后的参数）、输出上限512，enable_thinking=false，使用自然 EOS，不设置 ignore_eos。按对应并发度预热。另用两数据集各2题、温度0.7、top_p=0.9/top_k=20，验证随机 C1/C4。

数据下载来源、Git blob SHA、SHA256及许可证均保留。MGSM 来自 GSM8K 的翻译子集，两个来源不能作为独立题库计数。此处是固定子集的机制穿刺，不是官方全量评测或模型能力分数。

TPOT 是客户端首末输出 token 块间隔除以输出间隔数的有效平均值。MTP 可能一次返回多个 token，因此不将其表述为每个内部 token 的独立 ITL。主表给出每请求该指标的中位数。未取得独占 CPU 租约，结果仅代表本次负载与环境。

### 5.2 端到端性能

| 配置 | 并发 | TPOT中位数 ms | 相对基线 | 输出token数 |
|---|---:|---:|---:|---:|
| 关闭记录 | 1 | 9.602 | +0.00% | 4039 |
| 关闭记录 | 4 | 14.474 | +0.00% | 4115 |
| 初版 inline Top128 | 1 | 13.584 | +41.47% | 4039 |
| 初版 inline Top128 | 4 | 18.170 | +25.53% | 4212 |
| 初版 inline Top1024 | 1 | 13.927 | +45.04% | 4039 |
| 初版 inline Top1024 | 4 | 19.285 | +33.24% | 4115 |
| 池化 Top128 | 1 | 9.905 | +3.15% | 4039 |
| 池化 Top128 | 4 | 14.595 | +0.84% | 4115 |
| 池化 Top1024 | 1 | 10.032 | +4.47% | 4039 |
| 池化 Top1024 | 4 | 15.277 | +5.54% | 4115 |

池化 Top128 从初版的 C1 +41.5% 降至 +3.2%。池化 Top1024 的 C1/C4 额外开销约4.5%/5.5%。所有池化贪心配置均为16/16请求与对应关闭记录基线逐token相同，输出长度及MTP接受计数也一致。

C4总吞吐还受到首批延迟、批次组合与调度波动影响；不能根据某次总吞吐较高就宣称记录器提升模型性能。因此使用TPOT及逐token/接受计数一致性作为主要对照。

### 5.3 完整性、存储与覆盖率

| 配置 | 请求数 | 采样位置 | 压缩后记录大小 MB | 字节/位置 | 审计 |
|---|---:|---:|---:|---:|---|
| 池化 Top128 | 41 | 10150 | 27.95 | 2754 | 通过 |
| 池化 Top1024 | 54 | 12242 | 90.16 | 7365 | 通过 |

池化版合计95个请求（含预热），最终提交22,004个token，保留22,392个实际采样位置；差额为停止后的推测尾部。全部请求审计通过。NONE/FULL模式以及普通、accepted_draft、replacement、bonus来源均实际出现。

Top128每百万采样位置约2.75GB；Top1024约7.36GB，是本批次包括索引/事件/块开销的压缩后平均值。共同tokenizer快照另计约22.9MB。两阶段全词表FP32仅数值载荷就为1,986,560字节/位置。存储节约很大，但K=128与1024的样本构成不同，不能把两个压缩率当作纯K缩放规律。

Top128的raw覆盖率最小98.5742%，5分位99.9866%；Top1024最小99.5382%，5分位99.9959%。这些统计包含预热与采样尾部，不是任意未来请求的保证。实际输出与EOS始终单独保留。

### 5.4 池化与背压观察

独立的4×248320 FP32组件测试中：NPU快照复制中位数0.046ms，单份完整D2H到固定页CPU约0.285ms；4线程CPU摘要约10.1ms。这些是同步测量的组件结果，不能相加替代E2E。

实际非预热批次中，主线程记录钩子CPU耗时中位数约0.59～0.60ms/批，CPU后台统计中位数约10ms/批。获取槽的等待中位数约0.008～0.009ms。Top1024观察到一次最大52.44ms的槽等待；这是批次级背压，不是单token TPOT。不能把优化版描述为零影响或没有尾延迟。

### 5.5 正确性、随机采样和页面验证

- 15项最终回归测试通过：CPU持久化/损坏/失败、NPU完整词表对照、MTP变长映射、池化即时覆盖与背压、inference_mode跨线程场景、默认参数保留及任务进程组清理。
- 完整248320词表的NPU合成对照验证Top128/1024和MTP行映射；此测试验证采集正确性，不代替真实模型E2E。
- 池化与inline的随机C1为4/4输出逐token相同；随机C4为0/4相同。相同seed不能作为并发场景完全复现的保证，工具记录各次实际执行，不把C4差异隐藏为“不变”。
- 贪心子集每组均15/16数值答案匹配，另1条达到长度上限；不把length停止作为模型自然EOS，也不据此推导全量准确率。
- 浏览器验证了三种筛选、AND组合、raw/final切换、空结果、清除和位置导航。示例中非最高概率有3个位置；概率<0.2定位位置68；排名>1、概率<0.5、Top1−Top2差<0.1定位位置75。
- 示例是公开题目的正常生成，用于证明可回溯性，没有声称复现用户原先的低概率故障。

### 5.6 验证命令和交付

```bash
# 必须在目标容器中
cd /home/w00498770/dev/worktrees/vllm-023-logits-save/vllm-ascend/tools/flight_recorder
RUN_NPU_TESTS=1 ASCEND_RT_VISIBLE_DEVICES=2 python -m pytest -q test_recorder.py
python analyze.py /home/w00498770/dev/artifacts/vllm-023-logits-save/pooled128/trace audit
python analyze.py /home/w00498770/dev/artifacts/vllm-023-logits-save/pooled1024/trace audit
python summarize.py
```

原始命令、请求、响应、metrics、manifest、审计和软件版本位于artifacts。结果摘要为results.json，最终回归为unit-tests-delivery.log，资源释放证据为npu-final-release.txt。失败的pool128和failed-pooled128-orphan-occupancy仅作为调试记录，排除在正式结果外。

推理进程已停止，目标NPU资源及本任务临时共享槽已核对释放。保留的回环HTTP服务仅用于查看和下载离线交付物，不占NPU。

### 数据来源

- [MGSM官方数据与说明](https://github.com/google-research/url-nlp/tree/main/mgsm)
- [GSM8K官方数据](https://github.com/openai/grade-school-math/tree/master/grade_school_math/data)
