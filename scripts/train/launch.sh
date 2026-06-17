#!/usr/bin/env bash
set -e

CONFIG=${1:-configs/harbin_monthly.yaml}
GPUS=${2:-4}

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node="${GPUS}" scripts/train/train.py --config "${CONFIG}"
