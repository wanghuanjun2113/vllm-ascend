# GDN state rows：已验证的预编译产物

**这套二进制来自 main `d76a8c62f84c` 加 GDN 修复 `7b909db`，不是 v0.25.1rc1 的通用二进制。**
虽然文件存放在 `codex/fix-gdn-state-rows-v0.25.1rc1` 分支，
也不能据此把它覆盖到该分支的 v0.25.1rc1 源码或 CANN 9.0.1 环境。

## 构建和验证环境

| 项目 | 版本 |
|---|---|
| vLLM-Ascend 源码 | `d76a8c62f84c49d40d5cd94ed13f685d3f439d03` + `7b909dbdf4789b4c25eb2c9f00e447334c056a72` 补丁 |
| 配套 vLLM | `84030bbe3d74d99bad477a3d2e37a973ccd8865c` |
| 系统 / 架构 | openEuler 24.03 LTS-SP3 / Linux AArch64 |
| Python ABI | CPython 3.12（`cp312`） |
| CANN / torch-npu | 9.1.0 / 2.10.0.post4 |
| PyTorch / Triton Ascend | 2.10.0+cpu / 3.2.2 |
| 构建 glibc / GCC | 2.38 / 12.3.1（不是最低兼容版本承诺） |
| NPU | Ascend 910B，已在 910B4 验证；arch35 未硬件验证 |

对应构建已通过 29/29 项独立 NPU 检查、补丁自带 10/10 项回归、
受控 TP4 状态读写对照，以及 24/24 条 8192+1024、4 并发服务请求。
本次发布复核字节与哈希，复用上述测试证据，没有重跑 NPU 测试或修改服务。

## 下载内容

`payload/vllm_ascend/` 保留原包内路径，包含：

- `vllm_ascend_C.cpython-312-aarch64-linux-gnu.so`：PyTorch 扩展。
- `libvllm_ascend_kernels.so` 及 `lib64/` 中的同名库。
- 自定义算子接口、host tiling、proto 和 AICPU `.so`。
- 两份 GDN 设备 `.o`、对应 JSON 和 kernel 配置。

`liboptiling.so` 保留原相对符号链接，实际文件是
`op_tiling/lib/linux/aarch64/libcust_opmaster_rt2.0.so`。
单文件下载请取这个实体文件；完整 Git checkout 会保留链接结构。

完整的 CANN 自定义算子安装包也已提供：

[`payload/cann-ops-transformer-custom_linux-aarch64.run`](payload/cann-ops-transformer-custom_linux-aarch64.run)

这个 `.run` 包含完整自定义算子 vendor 目录，包括配套设备代码、加载元数据和生成源码。
单独列出的 `.so`/GDN 文件便于查看和下载，不是一个独立完整的 Python 安装包。

## 校验与使用边界

在本目录、Linux 匹配环境下校验：

```bash
sha256sum -c SHA256SUMS
```

`manifest.json` 记录完整源码 SHA、构建环境和逐文件 SHA-256。
符号链接的 SHA-256 按其指向的实体内容计算；请保留链接目标。
`.run` 的内嵌归档校验和也已检查，其中原生文件与已验证构建的清单比对一致。

使用前停止相关服务，确认源码、CANN、PyTorch、Python 和系统 ABI 匹配，并保留原安装用于回退。
需要已有上述版本的完整 vLLM/vLLM-Ascend 环境，且 Python `gdn.py` 修复必须同时存在。
不能仅替换一个 `.so`，也不能混用旧 GDN 设备 `.o`。

若需要安装 CANN 包，使用其原生安装入口，并把临时文件放入任务目录：

```bash
# ASCEND_SRC 是匹配上述 main + 修复的源码目录；WORK_DIR 是任务工作目录。
mkdir -p "$WORK_DIR/tmp"
TMPDIR="$WORK_DIR/tmp" bash payload/cann-ops-transformer-custom_linux-aarch64.run \
  --install-path="$ASCEND_SRC/vllm_ascend/_cann_ops_custom"
```

PyTorch 扩展需来自匹配环境；随附的 `.so` 不表示跨版本 ABI 兼容。
安装后重启服务，并运行补丁的 10 项 NPU 回归。

已有的[一键源码安装脚本](../../../tools/install_gdn_state_rows.md)仍会编译源码，
不会自动选择这里的预编译产物。环境不同或版本无法确认时，应使用源码安装流程。
