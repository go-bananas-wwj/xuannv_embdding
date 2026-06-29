#!/usr/bin/env bash
set -e

CONFIG=${1:-configs/harbin_128.yaml}
GPUS=${2:-4}

cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# 根据 GPUS 数量动态导出可见 NPU 设备，例如 GPUS=2 时导出 0,1。
VISIBLE_DEVICES=$(seq -s, 0 $((GPUS - 1)))
export ASCEND_RT_VISIBLE_DEVICES="${VISIBLE_DEVICES}"

torchrun --nproc_per_node="${GPUS}" scripts/train/train.py --config "${CONFIG}"
