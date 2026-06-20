#!/usr/bin/env bash
# Harbin 128 阶段二训练监控脚本。
# 由 cron 定期调用：检查进程是否存活，若因非 OOM/共享内存原因中断则自动恢复。
set -euo pipefail

WORKDIR="/root/workspace/xuannv"
CONFIG="configs/harbin_128_stage2.yaml"
LOG="/data/xuannv_embedding/outputs/stage2_harbin_128.log"
OUTDIR="/data/xuannv_embedding/outputs/harbin_128_stage2_v1"
INIT_CKPT="/data/xuannv_embedding/outputs/harbin_128_v1/best.pt"
RETRY_FILE="/data/xuannv_embedding/outputs/stage2_monitor_retries"
MAX_RETRIES=3

# 1) 训练进程存活则无需处理。
if pgrep -f "train.py.*${CONFIG}" > /dev/null; then
    echo "$(date -Iseconds) stage2 training is running"
    exit 0
fi

# 2) 已正常结束（日志中出现最后一个 epoch 的验证指标）。
if [ -f "$LOG" ] && tail -30 "$LOG" | grep -qE "epoch 99:.*val_loss="; then
    echo "$(date -Iseconds) stage2 training completed"
    exit 0
fi

# 3) 超过最大重试次数则停止自动恢复。
retries=$(cat "$RETRY_FILE" 2>/dev/null || echo 0)
if [ "$retries" -ge "$MAX_RETRIES" ]; then
    echo "$(date -Iseconds) stage2 auto-recovery stopped: max retries reached"
    exit 1
fi

# 4) OOM / 共享内存类错误不自动重启，需要人工干预。
if [ -f "$LOG" ] && tail -80 "$LOG" | grep -qE \
    "out of shared memory|CUDA out of memory|NPU out of memory|No space left on device|Bus error"; then
    echo "$(date -Iseconds) stage2 stopped due to OOM/shared-memory issue; manual intervention needed"
    exit 1
fi

# 5) 选择恢复点：优先从阶段二已保存的最新 epoch 恢复；否则从阶段一 best.pt 初始化。
latest_ckpt=$(ls -1 "$OUTDIR"/epoch_*.pt 2>/dev/null | sort -V | tail -1 || true)
if [ -n "$latest_ckpt" ]; then
    resume_arg="--resume $latest_ckpt"
else
    resume_arg="--init-from $INIT_CKPT"
fi

# 6) 重启训练。
echo "$(date -Iseconds) stage2 not running; restarting from $resume_arg (retry $((retries + 1))/$MAX_RETRIES)" | tee -a "$LOG"
echo $((retries + 1)) > "$RETRY_FILE"
cd "$WORKDIR"
export WANDB_API_KEY="$(cat /data/xuannv_embedding/.secrets/wandb_key)"
nohup bash scripts/train/launch_6card.sh "$CONFIG" $resume_arg >> "$LOG" 2>&1 &
