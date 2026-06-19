#!/usr/bin/env bash
set -e

LOG_DIR=/data/xuannv_embedding/logs
mkdir -p "$LOG_DIR"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

download() {
    local source=$1 start=$2 end=$3 workers=$4 log=$5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] START harbin $source $start..$end (workers=$workers)"
    python "$SCRIPT_DIR/download_pc.py" \
        --region harbin \
        --source "$source" \
        --start "$start" \
        --end "$end" \
        --region-file "$SCRIPT_DIR/../../configs/regions/harbin.geojson" \
        --output-root /data/xuannv_embedding/raw \
        --workers "$workers" \
        > "$log" 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE harbin $source $start..$end"
}

# S1: monthly for 2025 + 2026-01..2026-05. Use 8 workers.
for y in 2025; do
    for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
        last=$(date -d "$y-$m-01 +1 month -1 day" +%d)
        download s1 "$y-$m-01" "$y-$m-$last" 8 "$LOG_DIR/download_pc_harbin_s1_${y}${m}.log" &
    done
done
for m in 01 02 03 04 05; do
    last=$(date -d "2026-$m-01 +1 month -1 day" +%d)
    download s1 "2026-$m-01" "2026-$m-$last" 8 "$LOG_DIR/download_pc_harbin_s1_2026${m}.log" &
done
wait
echo "[$(date '+%Y-%m-%d %H:%M:%S')] S1 block complete"

# Landsat: monthly for 2025 only. Use 8 workers.
for y in 2025; do
    for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
        last=$(date -d "$y-$m-01 +1 month -1 day" +%d)
        download landsat "$y-$m-01" "$y-$m-$last" 8 "$LOG_DIR/download_pc_harbin_landsat_${y}${m}.log" &
    done
done
wait
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Landsat block complete"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Harbin S1/Landsat PC download complete"
