#!/usr/bin/env bash
# Wait for the P5H dev run to finish, then launch the all-320-patch run.
set -euo pipefail

DEV_SESSION="${DEV_SESSION:-p5h_haidian_masked_mim_dev_20260702}"
FULL_SESSION="${FULL_SESSION:-p5h_haidian_masked_mim_full_20260702}"
DEV_OUTPUT="${DEV_OUTPUT:-/data/xuannv_embedding/outputs/v2_p5h_haidian_olmo_masked_mim_dev_20260702}"
FULL_OUTPUT="${FULL_OUTPUT:-/data/xuannv_embedding/outputs/v2_p5h_haidian_olmo_masked_mim_full_20260702}"
DEV_FINAL_CKPT="${DEV_FINAL_CKPT:-${DEV_OUTPUT}/epoch_300.pt}"
DEV_BEST_CKPT="${DEV_BEST_CKPT:-${DEV_OUTPUT}/best.pt}"
FULL_CONFIG="${FULL_CONFIG:-configs/v2_p5h_haidian_olmo_masked_mim_full_20260702.yaml}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-300}"

cd "$(dirname "$0")/../.."
mkdir -p "${FULL_OUTPUT}"

log() {
    printf '%s %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*"
}

if tmux has-session -t "${FULL_SESSION}" 2>/dev/null; then
    log "Full session already exists: ${FULL_SESSION}"
    exit 0
fi

log "Watching dev session ${DEV_SESSION}; full run will start after ${DEV_FINAL_CKPT} exists."
while tmux has-session -t "${DEV_SESSION}" 2>/dev/null; do
    if [[ -f "${DEV_FINAL_CKPT}" ]]; then
        log "Detected final dev checkpoint while session is still closing."
        break
    fi
    sleep "${CHECK_INTERVAL_SECONDS}"
done

if [[ ! -f "${DEV_FINAL_CKPT}" ]]; then
    log "Dev session ended but final checkpoint is missing: ${DEV_FINAL_CKPT}"
    log "Not launching full run. Inspect ${DEV_OUTPUT}/train.log"
    exit 1
fi

if [[ ! -f "${DEV_BEST_CKPT}" ]]; then
    log "Dev best checkpoint is missing: ${DEV_BEST_CKPT}"
    exit 1
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    log "WANDB_API_KEY is not set in watcher environment; launching anyway."
fi

log "Launching full run ${FULL_SESSION} from ${DEV_BEST_CKPT}"
tmux new-session -d -s "${FULL_SESSION}" \
    "cd /root/workspace/xuannv && \
     export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5 && \
     export WANDB_API_KEY=\"\${WANDB_API_KEY:-}\" && \
     export PYTHONPATH=/root/workspace/xuannv/src:\${PYTHONPATH:-} && \
     bash scripts/train/launch_6card.sh ${FULL_CONFIG} --init-from ${DEV_BEST_CKPT} \
     2>&1 | tee ${FULL_OUTPUT}/train.log"

log "Full run launched: ${FULL_SESSION}"
