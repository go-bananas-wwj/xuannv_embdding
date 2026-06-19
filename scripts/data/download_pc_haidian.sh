#!/usr/bin/env bash

LOG_DIR=/data/xuannv_embedding/logs
mkdir -p "$LOG_DIR"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
MAX_BG_JOBS=6

PIDS=()
SOURCES=()
STARTS=()
ENDS=()
LOGS=()
FAILURES=()

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
        --overwrite \
        > "$log" 2>&1
    local rc=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE haidian $source $start..$end (rc=$rc)"
    return $rc
}

wait_slot() {
    while (( $(jobs -r | wc -l) >= MAX_BG_JOBS )); do
        sleep 1
    done
}

collect_block_results() {
    local block_name=$1
    local failed=0
    for i in "${!PIDS[@]}"; do
        wait "${PIDS[$i]}"
        local rc=$?
        if [ $rc -ne 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED haidian ${SOURCES[$i]} ${STARTS[$i]}..${ENDS[$i]} (rc=$rc) log=${LOGS[$i]}"
            FAILURES+=("haidian ${SOURCES[$i]} ${STARTS[$i]}..${ENDS[$i]} -> ${LOGS[$i]} (rc=$rc)")
            failed=$((failed + 1))
        fi
    done
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $block_name block complete ($failed failures, ${#PIDS[@]} total)"
    PIDS=()
    SOURCES=()
    STARTS=()
    ENDS=()
    LOGS=()
}

# S2: 2025 by quarter, 2026 Jan-May as one block. Use 4 workers.
download s2 2025-01-01 2025-03-31 4 "$LOG_DIR/download_pc_haidian_s2_2025q1.log" &
PIDS+=($!); SOURCES+=(s2); STARTS+=(2025-01-01); ENDS+=(2025-03-31); LOGS+=("$LOG_DIR/download_pc_haidian_s2_2025q1.log")
wait_slot

download s2 2025-04-01 2025-06-30 4 "$LOG_DIR/download_pc_haidian_s2_2025q2.log" &
PIDS+=($!); SOURCES+=(s2); STARTS+=(2025-04-01); ENDS+=(2025-06-30); LOGS+=("$LOG_DIR/download_pc_haidian_s2_2025q2.log")
wait_slot

download s2 2025-07-01 2025-09-30 4 "$LOG_DIR/download_pc_haidian_s2_2025q3.log" &
PIDS+=($!); SOURCES+=(s2); STARTS+=(2025-07-01); ENDS+=(2025-09-30); LOGS+=("$LOG_DIR/download_pc_haidian_s2_2025q3.log")
wait_slot

download s2 2025-10-01 2025-12-31 4 "$LOG_DIR/download_pc_haidian_s2_2025q4.log" &
PIDS+=($!); SOURCES+=(s2); STARTS+=(2025-10-01); ENDS+=(2025-12-31); LOGS+=("$LOG_DIR/download_pc_haidian_s2_2025q4.log")
wait_slot

download s2 2026-01-01 2026-05-31 4 "$LOG_DIR/download_pc_haidian_s2_2026.log" &
PIDS+=($!); SOURCES+=(s2); STARTS+=(2026-01-01); ENDS+=(2026-05-31); LOGS+=("$LOG_DIR/download_pc_haidian_s2_2026.log")

collect_block_results "S2"

# S1: monthly for 2025 + 2026-01..2026-05. Use 4 workers.
for y in 2025; do
    for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
        last=$(date -d "$y-$m-01 +1 month -1 day" +%d)
        download s1 "$y-$m-01" "$y-$m-$last" 4 "$LOG_DIR/download_pc_haidian_s1_${y}${m}.log" &
        PIDS+=($!); SOURCES+=(s1); STARTS+=("$y-$m-01"); ENDS+=("$y-$m-$last"); LOGS+=("$LOG_DIR/download_pc_haidian_s1_${y}${m}.log")
        wait_slot
    done
done
for m in 01 02 03 04 05; do
    last=$(date -d "2026-$m-01 +1 month -1 day" +%d)
    download s1 "2026-$m-01" "2026-$m-$last" 4 "$LOG_DIR/download_pc_haidian_s1_2026${m}.log" &
    PIDS+=($!); SOURCES+=(s1); STARTS+=("2026-$m-01"); ENDS+=("2026-$m-$last"); LOGS+=("$LOG_DIR/download_pc_haidian_s1_2026${m}.log")
    wait_slot
done

collect_block_results "S1"

# Landsat: monthly for 2025 + 2026-01..2026-05. Use 4 workers.
for y in 2025; do
    for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
        last=$(date -d "$y-$m-01 +1 month -1 day" +%d)
        download landsat "$y-$m-01" "$y-$m-$last" 4 "$LOG_DIR/download_pc_haidian_landsat_${y}${m}.log" &
        PIDS+=($!); SOURCES+=(landsat); STARTS+=("$y-$m-01"); ENDS+=("$y-$m-$last"); LOGS+=("$LOG_DIR/download_pc_haidian_landsat_${y}${m}.log")
        wait_slot
    done
done
for m in 01 02 03 04 05; do
    last=$(date -d "2026-$m-01 +1 month -1 day" +%d)
    download landsat "2026-$m-01" "2026-$m-$last" 4 "$LOG_DIR/download_pc_haidian_landsat_2026${m}.log" &
    PIDS+=($!); SOURCES+=(landsat); STARTS+=("2026-$m-01"); ENDS+=("2026-$m-$last"); LOGS+=("$LOG_DIR/download_pc_haidian_landsat_2026${m}.log")
    wait_slot
done

collect_block_results "Landsat"

if [ ${#FAILURES[@]} -gt 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Haidian PC download completed with failures:"
    for f in "${FAILURES[@]}"; do
        echo "  - $f"
    done
    exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Haidian PC download complete"
exit 0
