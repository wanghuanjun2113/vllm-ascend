# 310P Qwen3.6 ITL 补丁

下载单文件：[310p-qwen36-itl.patch](310p-qwen36-itl.patch)。
本分支用于发布补丁，仓库 main 的运行代码没有被修改。

本目录随单一 workspace patch 安装。仅支持以下已验证源码组合：

- vLLM-Ascend：`f2f74a16c3c50a76f4349d807918e83edec1e35c`
- vLLM：`568afb3a13806beb53bb2e6bd518269357b237c0`
- aarch64、Ascend 310P3、CANN 9.1.0、torch 2.10.0、torch_npu 2.10.0.post4

补丁合并了原工作树的必要前置修改和本轮保留优化，适用于上述 commit 的干净源码。
不是针对仓库最新 main 的补丁，也不要重复应用到已经有相同修改的目录。
保留原模型权重，不新增量化；已有 lm_head 量化支持代码不会自动修改或量化权重。

## 应用和构建

停止目标 Docker 中使用这些源码的服务后执行。先确认两个源码目录都在
`/vllm-workspace` 下；不同位置可在对应共同父目录建立同名的源码副本后应用。

```bash
cd /vllm-workspace
git -C vllm-ascend rev-parse HEAD
git -C vllm rev-parse HEAD
git apply --check /path/to/310p-qwen36-itl.patch
git apply /path/to/310p-qwen36-itl.patch
python3 vllm-ascend/tools/itl20/verify.py --ascend vllm-ascend --vllm vllm
```

校验应显示两个仓库均为 `optimized`。不符合时不要强制应用。
修改前请保留目标源码备份。`git apply` 不安装 Python 包；当前 Python 必须实际
从这两个源码目录导入 vLLM 与 vLLM-Ascend，而不是另一份 site-packages 副本。

加载目标容器的 CANN 环境，使 `ASCEND_HOME_PATH` 指向其 CANN 安装目录。
需要 cmake、C++ 编译工具、ninja，以及匹配的 torch/torch_npu 开发头文件。
在目标容器内重新编译，不复制其他镜像的二进制：

```bash
bash vllm-ascend/tools/itl20/install_kernel.sh /vllm-workspace/vllm-ascend
# 选择目标机器上可用的设备；下面的 0 仅为示例。
ASCEND_RT_VISIBLE_DEVICES=0 TASK_QUEUE_ENABLE=0 \
  python3 vllm-ascend/tools/itl20/wy_solve_310p/test_kernel.py
```

安装脚本会构建并安装 `libwy_solve_kernel.so` 和 `wy_solve_bindings.so`，
位置为 `vllm_ascend/_wy_solve/`。两者缺一不可；完成构建前不要启动打过补丁的服务。
测试成功后用原启动命令重启，并设置 `export TASK_QUEUE_ENABLE=0`。
本次保留的是正常请求接纳与异步调度，没有批次解码优先策略。

## 修改范围与验证

- 复现前置工作树：拒绝采样快速路径、计数镜像、pinned 拷贝等已有修改；
  vLLM 的一处前置补丁将 draft token 的非连续视图转为连续布局后再复制。
- 本轮优化：减少无用 D2H/同步；GDN 状态索引在设备端构造；WY 重复衰减修复；
  FP32、64×64 WY 递推 AscendC 算子。
- 附带三个原有单元测试文件的增量和独立算子数值对拍脚本。
- 不包含 WY v3、已回退的回放同步删除、批次等待策略或机器专属运行库。

既有 WY 路径把衰减乘了两次，本补丁改为一次。因此这是包含数值纠错的优化，
不能保证生成文本与错误基线逐字一致。此前定向 CPU 检查共 22 个用例通过；
NPU 算子最大绝对误差 1.49e-8，生产形状整段 WY 的 W/U 相对误差约
0.0341%/0.0355%；未做通用任务能力评测。

原权重、TP4、MTP=3、图模式、8192 入/1024 出、4 并发、24 条请求的原环境实测：

| 平均指标 | 原工作树基线 | 保留配置 |
|---|---:|---:|
| TTFT | 15.09 s | 13.36 s |
| TPOT | 70.67 ms | 65.11 ms |
| ITL | 180.13 ms | 163.70 ms |

24/24 成功；ITL 降低 9.12%，尚未达到 20% 目标。这里的基线是前置修改后的
工作树，不是干净 commit；结果不能直接推定为其他 Docker/硬件的收益。

## 回退

先停止目标服务。在共同父目录执行 `git apply --reverse --check`，确认没有
后续冲突修改，再执行 `git apply --reverse` 回退同一个 patch。两个编译产生的
运行库不在 patch 中；回退后原路径不再加载它们，可单独保留或清理。
恢复启动环境中原来的 `TASK_QUEUE_ENABLE` 设置，再重启服务。
