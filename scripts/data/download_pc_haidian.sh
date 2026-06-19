#!/usr/bin/env bash
set -e

trap 'kill 0' EXIT

LOG_DIR=/data/xuannv_embedding/logs
mkdir -p "$LOG_DIR"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
MAX_BG_JOBS=6

download() {
    local source=$1 start=$2 end=$3 workers=$4 log=$5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START haidian $source $start..$end (workers=$workers)"
    python "$SCRIPT_DIR/download_pc.py" \
        --region haidian \
        --source "$source" \
        --start "$start" \
        --end "$end" \
        --region-file "$SCRIPT_DIR/../../configs/regions/haidian.geojson" \
        --output-root /data/xuannv_embedding/raw \
        --workers "$workers" \
        > "$log" 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE haidian $source $start..$end"
}

# S2: 2025 by quarter, 2026 Jan-May as one block. Use 4 workers.
download s2 2025-01-01 2025-03-31 4 "$LOG_DIR/download_pc_haidian_s2_2025q1.log" &
if (( $(jobs -r | wc -l) >= MAX_BG_JOBS )); then wait -n; fi
download s2 2025-04-01 2025-06-30 4 "$LOG_DIR/download_pc_haidian_s2_2025q2.log" &
if (( $(jobs -r | wc -l) >= MAX_BG_JOBS )); then wait -n; fi
download s2 2025-07-01 2025-09-30 4 "$LOG_DIR/download_pc_haidian_s2_2025q3.log" &
if (( $(jobs -r | wc -l) >= MAX_BG_JOBS )); then wait -n; fi
download s2 2025-10-01 2025-12-31 4 "$LOG_DIR/download_pc_haidian_s2_2025q4.log" &
if (( $(jobs -r | wc -l) >= MAX_BG_JOBS )); then wait -n; fi
download s2 2026-01-01 2026-05-31 4 "$LOG_DIR/download_pc_haidian_s2_2026.log" &
wait
echo "[$(date '+%Y-%m-%d %H:%M:%S')] S2 block complete"

# S1: monthly for 2025 + 2026-01..2026-05. Use 4 workers.
for y in 2025; do
    for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
        last=$(date -d "$y-$m-01 +1 month -1 day" +%d)
        download s1 "$y-$m-01" "$y-$m-$last" 4 "$LOG_DIR/download_pc_haidian_s1_${y}${m}.log" &
        if (( $(jobs -r | wc -l) >= MAX_BG_JOBS )); then wait -n; fi
    done
done
for m in 01 02 03 04 05; do
    last=$(date -d "2026-$m-01 +1 month -1 day" +%d)
    download s1 "2026-$m-01" "2026-$m-$last" 4 "$LOG_DIR/download_pc_haidian_s1_2026${m}.log" &
    if (( $(jobs -r | wc -l) >= MAX_BG_JOBS )); then wait -n; fi
done
wait
echo "[$(date '+%Y-%m-%d %H:%M:%S')] S1 block complete"

# Landsat: monthly for 2025 + 2026-01..2026-05. Use 4 workers.
for y in 2025; do
    for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
        last=$(date -d "$y-$m-01 +1 month -1 day" +%d)
        download landsat "$y-$m-01" "$y-$m-$last" 4 "$LOG_DIR/download_pc_haidian_landsat_${y}${m}.log" &
        if (( $(jobs -r | wc -l) >= MAX_BG_JOBS )); then wait -n; fi
    done
done
for m in 01 02 03 04 05; do
    last=$(date -d "2026-$m-01 +1 month -1 day" +%d)
    download landsat "2026-$m-01" "2026-$m-$last" 4 "$LOG_DIR/download_pc_haidian_landsat_2026${m}.log" &
    if (( $(jobs -r | wc -l) >= MAX_BG_JOBS )); then wait -n; fi
done
wait
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Landsat block complete"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Haidian PC download complete"
