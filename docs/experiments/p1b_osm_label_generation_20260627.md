# P1B OSM 建筑/道路标签生成记录 - 2026-06-27

本记录对应 P1B 扩展下游任务的第一步：使用 OSM 自动生成建筑物和道路标签，减少人工标注依赖。

## 生成脚本

- 脚本：`scripts/data/build_osm_labels.py`
- 标签类型：`osm_gt`
- 输出训练 mask：`masks/`，128x128，10m 分辨率，与 P1B embedding 对齐。
- 可选高分 mask：`masks_highres/`，用于人工抽检和汇报图叠加。

## 已生成标签

| Region | Task | OSM features | Patches | Positive patches | Positive pixels | Mean positive ratio | Max positive ratio | Output |
|---|---|---:|---:|---:|---:|---:|---:|---|
| haidian | building_osm | 44,484 | 320 | 290 | 503,142 | 0.0960 | 0.4288 | `/data/xuannv_embedding/processed/haidian/labels/building_osm` |
| haidian | road_osm | 77,675 | 320 | 317 | 819,511 | 0.1563 | 0.3399 | `/data/xuannv_embedding/processed/haidian/labels/road_osm` |
| harbin | building_osm | 13,093 | 424 | 166 | 209,990 | 0.0302 | 0.6677 | `/data/xuannv_embedding/processed/harbin/labels/building_osm` |
| harbin | road_osm | 10,523 | 424 | 375 | 502,262 | 0.0723 | 0.3738 | `/data/xuannv_embedding/processed/harbin/labels/road_osm` |

## 复现命令

```bash
python scripts/data/build_osm_labels.py --region haidian --task building_osm \
  --output-root /data/xuannv_embedding/processed/haidian/labels/building_osm \
  --overwrite-cache

python scripts/data/build_osm_labels.py --region haidian --task road_osm \
  --output-root /data/xuannv_embedding/processed/haidian/labels/road_osm \
  --overwrite-cache

python scripts/data/build_osm_labels.py --region harbin --task building_osm \
  --output-root /data/xuannv_embedding/processed/harbin/labels/building_osm \
  --overwrite-cache

python scripts/data/build_osm_labels.py --region harbin --task road_osm \
  --output-root /data/xuannv_embedding/processed/harbin/labels/road_osm \
  --overwrite-cache
```

## 初步判断

- 海淀建筑和道路覆盖率较好，适合优先做 quick downstream。
- 哈尔滨建筑 OSM 稀疏，可能存在 OSM 漏标或区域本身稀疏，需要在可视化 QA 后决定是否启用 LocateAnything/SAM2 伪标签补召回。
- 哈尔滨道路覆盖率可用，可以进入 quick downstream。
- 所有 OSM 标签都应在汇报中标注为 `osm_gt`，不等同于人工精标。

## 下一步

1. 为四个 OSM 任务生成 fixed 5-fold split。
2. 先用 P1B embedding + downstream head 跑 quick fold-0。
3. 生成每个任务的预测/GT/embedding 可视化。
4. 如果建筑召回不足，接入 LocateAnything bbox/point + SAM2 mask 伪标签。
