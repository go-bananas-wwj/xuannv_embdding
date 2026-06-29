#!/usr/bin/env bash
# 6× Ascend NPU DDP 训练启动脚本（Harbin 128）。
set -euo pipefail

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "WARN: WANDB_API_KEY 未设置；若配置 use_wandb=false 可忽略。" >&2
fi

CONFIG="${1:-configs/harbin_128.yaml}"
shift || true
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-6}"

cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

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
