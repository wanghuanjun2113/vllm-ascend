#!/usr/bin/env bash
set -euo pipefail
cd /home/w00498770/dev/worktrees/vllm-023-qwen35-qkv/vllm-ascend
test -f /home/w00498770/dev/artifacts/vllm-023-qwen35-qkv/weights/Qwen3.5-35B-A3B/VERIFIED.json
export PYTHONPATH="$PWD:/vllm-workspace/vllm"
export ASCEND_RT_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8
export OMP_PROC_BIND=false
export HCCL_BUFFSIZE=200
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ASCEND_ENABLE_FLASHCOMM1=0
exec python tools/qkv_capture/run_capture.py "$@"
