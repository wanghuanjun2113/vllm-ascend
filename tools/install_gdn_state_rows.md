# 一键安装 GDN speculative state rows 修复

**需要编译算子。** 此补丁同时修改 Python 调用、AscendC host tiling 和设备 kernel。
仅更新 `gdn.py`、仅安装 Python 包或仅重编 host 库，都不能完整生效。

脚本固定应用提交
[`7b909dbdf4789b4c25eb2c9f00e447334c056a72`](https://github.com/wanghuanjun2113/vllm-ascend/commit/7b909dbdf4789b4c25eb2c9f00e447334c056a72)，
保留目标源码仓库的当前分支，不会把 main 切换到 v0.25.1rc1。
已经包含这个补丁的源码也可以运行，脚本会重新编译并验证。

## 已编译产物

[main d76a8c62 的 910B / CANN 9.1 / ARM64 / Python 3.12 产物](../artifacts/gdn-state-rows/main-d76a8c62-910b-cann9.1-aarch64-py312/README.md)
已提供 `.so`、配套设备文件和完整 `.run` 包。
这些二进制不适用于任意 v0.25.1rc1 环境；本脚本仍按下面的流程编译源码。

## 使用方法

在运行 vLLM 的 **Linux Ascend 容器内** 执行，使用服务所用的 Python/venv。
准备与目标源码兼容的 vLLM、CANN、torch-npu、编译工具，以及 `pytest`、
`setuptools-scm`、`pybind11`；脚本不会自动升级这些依赖。
当前安装脚本面向已验证的 **910B**，编译目标为 `ascend910b1`。

先停止使用目标源码的所有服务，包括其他容器中的服务，并选择一个空闲 NPU。
脚本不会停止、杀死或自动重启服务，也不要与该源码的其他构建同时执行。
它会检查当前 PID 命名空间中可读取的已加载原生库，以及 `npu-smi` 的设备空闲状态。

以下示例在 `/task` 任务目录内下载安装工具，给**已有的源码目录**打补丁：

```bash
# 下载工具；这个目录不替代现有的服务源码。
git clone --branch codex/fix-gdn-state-rows-v0.25.1rc1 \
  https://github.com/wanghuanjun2113/vllm-ascend.git /task/gdn-fix-tools

# 一条命令完成打补丁、清缓存、编译、安装及 NPU 回归。
/task/.venv/bin/python /task/gdn-fix-tools/tools/install_gdn_state_rows.py \
  --repo /task/src/vllm-ascend \
  --work-dir /task/gdn-state-rows-install \
  --device 4
```

- 将 Python 和 `--repo` 替换成实际服务环境及源码路径。
- `--device` 是空闲的**物理 NPU ID**，验证进程只看这一张卡。
- 所有日志、缓存及安装记录写入 `--work-dir/run-<时间>/`。
- 编译并发默认 32，可在命令前设置 `MAX_JOBS=16`。
- 如果直接使用本分支作为服务源码，`--repo` 可以指定本分支的 checkout；
  其 v0.25.1rc1 基线需要匹配的 vLLM/CANN 环境。

## 脚本具体做什么

1. 检查环境、目标源码是否正在使用，以及验证设备是否空闲。
2. 优先从目标仓库或工具仓库读取固定修复提交，必要时从 GitHub 获取该提交及其父提交。
   `git apply --check` 通过后应用；反向检查通过则识别为已应用。
   冲突或部分应用会报错，不会 reset 或覆盖解决冲突。
3. 初始化项目子模块，删除 GDN 的 generated source、设备二进制目录和 `.done` 标记。
   其他算子的缓存保留。
4. 使用当前 Python 执行 `pip install --no-deps --no-build-isolation -e <repo>`，
   设置 `COMPILE_CUSTOM_KERNELS=1`，编译安装项目原生库与自定义算子包。
5. 记录设备 `.o` 的 SHA-256。首次应用补丁时，如果仍有旧 GDN 二进制未改变则失败。
6. 在源码目录之外启动新 Python 进程，核对实际导入路径，运行补丁自带的
   BF16/FP32 共 **10 项 NPU 回归**。全部运行并通过后才写入 `receipt.json`。

验证沿用当前环境的 `PYTHONPATH`，不会靠临时指向目标源码掩盖安装路径错误。
如果导入路径仍指向另一个 checkout，请修正服务环境的 `PYTHONPATH` 后重试。

## 为什么必须清理 GDN 缓存

在 Ascend 910B4 的实测中，增量编译曾仅更新 host tiling 库，
两份 GDN 设备 `.o` 仍与未修复版本相同，回归依然失败。
清除 generated source、binary 和 `.done` 后，设备二进制真正更新，回归通过。
脚本每次都会执行这一步；已应用补丁的重复安装允许正确二进制的哈希保持不变。

## 成功、失败与重启

成功时终端显示 `VERIFIED: 10/10 NPU regressions passed`，安装目录包含：

- `install.log`：完整命令及输出。
- `fix.patch`、`cache-cleanup.json`：实际补丁及清理记录。
- `regressions.xml`、`verification.json`：NPU 回归及导入路径。
- `receipt.json`：源码 HEAD、修复提交、前后设备二进制哈希及通过数量。

随后用原服务启动命令重启，确保仍使用上述 Python 和源码。
新进程才能加载新原生库。脚本不修改服务参数，不做整网压测。

失败会返回非零状态，保存 `failure.txt`，不会生成成功凭据。
安装不是事务式回滚：源码补丁或部分构建产物可能已经写入。
解决日志中的问题后，保持服务停止，重复同一命令即可。
已有的服务镜像或源码备份应保留用于回退。

硬件验证范围是 910B；此脚本不声称验证 arch35 或其他设备。
