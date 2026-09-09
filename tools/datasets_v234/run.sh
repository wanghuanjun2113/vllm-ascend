#!/usr/bin/env bash
set -euo pipefail
cd /home/w00498770/dev/worktrees/vllm-023-qwen35-qkv/vllm-ascend-datasets-v234
export PYTHONPATH="$PWD:/vllm-workspace/vllm${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=2,3
export ASCEND_CUSTOM_OPP_PATH="$PWD/vllm_ascend/_cann_ops_custom/vendors/custom_transformer${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="$PWD/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=false
export HCCL_BUFFSIZE=200
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
exec python tools/datasets_v234/run_capture.py
