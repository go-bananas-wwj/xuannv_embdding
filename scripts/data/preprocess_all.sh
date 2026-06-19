#!/usr/bin/env bash
# 预处理哈尔滨与海淀的低分辨率时序数据（S1/S2/Landsat）与 ESRI 2023 LULC 标签。
# 高分辨率数据（DOM / SAR / Planet）需要单独的预处理脚本，待下载完成后补充。
set -e

# 1) 低分辨率时序数据
python scripts/data/preprocess.py --config scripts/data/preprocess_harbin.json --overwrite
python scripts/data/preprocess.py --config scripts/data/preprocess_haidian.json --overwrite

# 2) ESRI 2023 LULC 分类标签
python scripts/data/preprocess_lulc.py --config scripts/data/preprocess_lulc_harbin.json --overwrite
python scripts/data/preprocess_lulc.py --config scripts/data/preprocess_lulc_haidian.json --overwrite

# 3) 生成 manifest（worldcover 作为分类 target 来源）
python scripts/data/generate_manifest.py \
    --processed-dir /data/xuannv_embedding/processed/harbin \
    --output /data/xuannv_embedding/processed/harbin/manifest.json \
    --source-dir s2=patches/s2 \
    --source-dir s1=patches/s1 \
    --source-dir landsat=patches/landsat \
    --source-dir worldcover=labels/worldcover \
    --sources s2 s1 landsat worldcover

python scripts/data/generate_manifest.py \
    --processed-dir /data/xuannv_embedding/processed/haidian \
    --output /data/xuannv_embedding/processed/haidian/manifest.json \
    --source-dir s2=patches/s2 \
    --source-dir s1=patches/s1 \
    --source-dir landsat=patches/landsat \
    --source-dir worldcover=labels/worldcover \
    --sources s2 s1 landsat worldcover

# 4) 计算统计量（仅连续 source；worldcover 为类别标签，不需要归一化统计量）
python scripts/data/compute_statistics.py \
    --processed-dir /data/xuannv_embedding/processed/harbin/patches \
    --output-dir /data/xuannv_embedding/statistics/harbin \
    --sources s2 s1 landsat

python scripts/data/compute_statistics.py \
    --processed-dir /data/xuannv_embedding/processed/haidian/patches \
    --output-dir /data/xuannv_embedding/statistics/haidian \
    --sources s2 s1 landsat

echo "[$(date)] 预处理完成"
