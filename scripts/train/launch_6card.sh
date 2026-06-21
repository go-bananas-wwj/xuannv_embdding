#!/usr/bin/env bash
# 6× Ascend NPU DDP 训练启动脚本。
set -euo pipefail

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "WARNING: 未设置 WANDB_API_KEY，WANDB 将被禁用或初始化失败。" >&2
fi

CONFIG="${1:-configs/v1.1_distill_long_stable_50ep.yaml}"
shift || true
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-6}"

cd "$(dirname "$0")/../.."

echo "Launching training on ${NNODES} node(s) x ${NPROC_PER_NODE} NPUs"
echo "Config: ${CONFIG}"
echo "Extra args: $*"

exec torchrun \
    --nnodes "${NNODES}" \
    --nproc-per-node "${NPROC_PER_NODE}" \
    --standalone \
    scripts/train/train.py \
    --config "${CONFIG}" \
    "$@"
