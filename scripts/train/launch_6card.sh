#!/usr/bin/env bash
# 6× Ascend NPU DDP 训练启动脚本（Harbin 128）。
set -euo pipefail

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: 请设置环境变量 WANDB_API_KEY 以启用 WANDB 实时监控。" >&2
    echo "例如: export WANDB_API_KEY=<your_key>" >&2
    exit 1
fi

CONFIG="${1:-configs/harbin_128.yaml}"
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
