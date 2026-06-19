#!/usr/bin/env bash
# 下载 Planetary Computer 上剩余的哈尔滨 S2 2025（按季度）以及校验所有 PC 文件。
# S2 按季度顺序跑，避免多个大文件同时写入导致资源争抢。
set -e

LOG_DIR=/data/xuannv_embedding/logs
mkdir -p "$LOG_DIR"

# 清理所有 0 字节 NetCDF 残留文件，避免被误判为已完成
find /data/xuannv_embedding/raw -name '*.nc' -size 0 -delete

download() {
    local region=$1 source=$2 start=$3 end=$4 log=$5
    echo "[$(date)] START $region $source $start..$end"
    python "$(dirname "$0")/../data/download_pc.py" \
        --region "$region" \
        --source "$source" \
        --start "$start" \
        --end "$end" \
        --region-file "$(dirname "$0")/../../configs/regions/${region}.geojson" \
        --output-root /data/xuannv_embedding/raw \
        > "$log" 2>&1
    echo "[$(date)] DONE $region $source $start..$end"
}

# 哈尔滨 S2 2025 按季度顺序下载
download harbin s2 2025-01-01 2025-03-31 "$LOG_DIR/download_pc_harbin_s2_2025q1.log"
download harbin s2 2025-04-01 2025-06-30 "$LOG_DIR/download_pc_harbin_s2_2025q2.log"
download harbin s2 2025-07-01 2025-09-30 "$LOG_DIR/download_pc_harbin_s2_2025q3.log"
download harbin s2 2025-10-01 2025-12-31 "$LOG_DIR/download_pc_harbin_s2_2025q4.log"

echo "[$(date)] 哈尔滨 S2 2025 下载完成，开始完整性校验..."

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
