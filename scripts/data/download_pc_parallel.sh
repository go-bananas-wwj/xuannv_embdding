#!/usr/bin/env bash
# 并行下载 Planetary Computer 上剩余的 PC 数据。
# 哈尔滨 S2 2025 按季度切分为 4 个文件，避免单文件过大导致写入卡住。
set -e

LOG_DIR=/data/xuannv_embedding/logs
mkdir -p "$LOG_DIR"

# 清理之前残留的 0 字节文件，避免脚本误认为是已完成文件而跳过下载
zero_files=(
  /data/xuannv_embedding/raw/harbin/s2/s2_20250101_20251231.nc
  /data/xuannv_embedding/raw/harbin/s1/s1_20260101_20260531.nc
  /data/xuannv_embedding/raw/harbin/landsat/landsat_20260101_20260531.nc
  /data/xuannv_embedding/raw/haidian/landsat/landsat_20260101_20260531.nc
)
for f in "${zero_files[@]}"; do
    if [ -f "$f" ] && [ ! -s "$f" ]; then
        echo "[$(date)] 删除 0 字节残留文件: $f"
        rm -f "$f"
    fi
done

# 下载函数：失败时不退出整个脚本，仅记录 FAIL
download() {
    local region=$1 source=$2 start=$3 end=$4 log=$5
    echo "[$(date)] START $region $source $start..$end -> $log"
    if python "$(dirname "$0")/../data/download_pc.py" \
        --region "$region" \
        --source "$source" \
        --start "$start" \
        --end "$end" \
        --region-file "$(dirname "$0")/../../configs/regions/${region}.geojson" \
        --output-root /data/xuannv_embedding/raw \
        --workers 12 \
        > "$log" 2>&1; then
        echo "[$(date)] DONE $region $source $start..$end"
    else
        echo "[$(date)] FAIL $region $source $start..$end (see $log)"
    fi
}

# 并行启动所有下载任务
# 哈尔滨 S2 2025 按季度拆分
download harbin s2 2025-01-01 2025-03-31 "$LOG_DIR/download_pc_harbin_s2_2025q1.log" &
download harbin s2 2025-04-01 2025-06-30 "$LOG_DIR/download_pc_harbin_s2_2025q2.log" &
download harbin s2 2025-07-01 2025-09-30 "$LOG_DIR/download_pc_harbin_s2_2025q3.log" &
download harbin s2 2025-10-01 2025-12-31 "$LOG_DIR/download_pc_harbin_s2_2025q4.log" &

# 哈尔滨 S1 / Landsat 2026，海淀 Landsat 2026
download harbin s1 2026-01-01 2026-05-31 "$LOG_DIR/download_pc_harbin_s1.log" &
download harbin landsat 2026-01-01 2026-05-31 "$LOG_DIR/download_pc_harbin_landsat.log" &
download haidian landsat 2026-01-01 2026-05-31 "$LOG_DIR/download_pc_haidian_landsat.log" &

wait

echo "[$(date)] 所有 PC 并行下载任务结束，开始完整性校验..."

python3 - <<'PY'
import xarray as xr
from pathlib import Path

files = [
    Path('/data/xuannv_embedding/raw/harbin/s2/s2_20250101_20250331.nc'),
    Path('/data/xuannv_embedding/raw/harbin/s2/s2_20250401_20250630.nc'),
    Path('/data/xuannv_embedding/raw/harbin/s2/s2_20250701_20250930.nc'),
    Path('/data/xuannv_embedding/raw/harbin/s2/s2_20251001_20251231.nc'),
    Path('/data/xuannv_embedding/raw/harbin/s1/s1_20260101_20260531.nc'),
    Path('/data/xuannv_embedding/raw/harbin/landsat/landsat_20260101_20260531.nc'),
    Path('/data/xuannv_embedding/raw/haidian/landsat/landsat_20260101_20260531.nc'),
]

all_ok = True
for f in files:
    if not f.exists():
        print('MISSING', f)
        all_ok = False
        continue
    try:
        ds = xr.open_dataset(f, decode_times=False)
        print('OK', f.name, dict(ds.sizes))
    except Exception as e:
        print('FAIL', f, e)
        all_ok = False

if all_ok:
    print('[verify] 所有 PC 文件校验通过')
else:
    print('[verify] 部分 PC 文件缺失或损坏，需要重新下载')
    exit(1)
PY

echo "[$(date)] PC 下载流程完成"
