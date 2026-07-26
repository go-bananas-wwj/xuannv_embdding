# 中国季度嵌入采样空间文件说明

## 1. 交付位置

本次物化结果保存在：

`/data/xuannv_embedding/outputs/china_quarterly_sampling_62000_candidate_20260726/`

完整压缩包：

`/data/xuannv_embedding/outputs/china_quarterly_sampling_62000_candidate_20260726.zip`

## 2. 文件用途

1. `china_quarterly_62000_candidate_centers.shp`
   62,000 个采样中心点，适合空间查询、数据下载任务分片和覆盖统计。
2. `china_quarterly_62000_candidate_footprints.shp`
   62,000 个 1,280 m × 1,280 m 采样范围，适合遥感影像裁切。
3. `china_quarterly_62000_candidate.gpkg`
   同时包含 `centers` 和 `footprints`，字段限制少，日常使用优先选择该文件。
4. `china_quarterly_62000_candidate_registry.jsonl`
   完整审计字段和哈希，不受 Shapefile 十字符字段名限制。
5. `china_quarterly_62000_candidate_registry.csv`
   用于批量调度和人工筛选。
6. `china_quarterly_62000_candidate_audit.json`
   记录旧点修复、重复消解、跨 UTM 重叠消解和最终检查。

所有空间图层统一采用 WGS84（EPSG:4326）。采样范围是在各点所属 UTM
分区内按 1,280 m 方格构造，再转换到 WGS84；它不是用经纬度近似画出的方框。

## 3. 当前数量

| 采样层 | 数量 |
|---|---:|
| 1% 基础空间样本 | 57,405 |
| UTM 与空间平衡补样 | 2,595 |
| 海岸线补样 | 500 |
| 语义与困难样本复核池 | 1,500 |
| 合计 | 62,000 |

## 4. 已完成检查

1. 点图层、面图层和 GPKG 图层均为 62,000 条。
2. `patch_id`、网格身份哈希和 footprint 哈希均无重复。
3. 所有采样中心均位于冻结的中国边界内。
4. 所有点均属于由中心经度唯一确定的 UTM 分区。
5. 跨 UTM 分区的 footprint 重叠比例均不超过 1%。
6. 500 个海岸样本的 footprint 均同时与冻结陆地面和海洋面相交。
7. 基础层每个宏网格最多一个点，三个补样层合计每个宏网格最多一个点。
8. 点为有效 Point，范围为有效 Polygon。
9. ZIP 可完整解压，目录内提供 SHA-256 校验值。

## 5. 使用限制

这批文件是**数据获取和质量检查用的设计候选版**，不能直接改名为最终训练集。

1. 全国语义数据尚未落盘，1,500 个点目前是确定性复核池；后续要用
   WorldCover、历史 OSM、DEM 和季度影像质量结果重新排序。
2. 57,405 个基础点是对历史候选骨架进行 UTM 修复、宏网格去重和确定性补齐，
   尚未基于完整整数候选 atlas 重新执行获批的逐宏网格概率抽样。
3. 少量 ADM0 内点位不在冻结 ADM1 面覆盖范围内，属性标记为
   `ADM1_NOT_COVERED_BY_FROZEN_SOURCE`，不能把它当作一个省级类别。
4. 2020Q1 至 2021Q4 的 Sentinel-2、Sentinel-1 和 Landsat 覆盖率、云掩膜、
   有效观测数和配准误差尚未逐点计算。
5. 只有完成上述检查，才能冻结最终 registry，并生成空间隔离的
   train/validation/test 划分。

## 6. 重建命令

```bash
python scripts/data/materialize_china_sampling_shapes.py \
  --existing-points /data2/xuannv_embedding/china_v1/atlas/china_v1_provisional_60500_sample_points.jsonl \
  --inventory /data/xuannv_embedding/raw/china_v1/national_macrocell_inventory.jsonl \
  --country /data/xuannv_embedding/raw/china_v1/geoboundaries/geoBoundaries-CHN-ADM0.geojson \
  --adm1 /data/xuannv_embedding/raw/china_v1/geoboundaries/geoBoundaries-CHN-ADM1.geojson \
  --ocean /data2/xuannv_embedding/china_v1/source_snapshots/natural_earth/ne_10m/ne_10m_ocean.shp \
  --policy configs/national/china_quarterly_2020_2021_sampling_policy_20260726.json \
  --output-dir /data/xuannv_embedding/outputs/china_quarterly_sampling_62000_candidate_20260726
```
