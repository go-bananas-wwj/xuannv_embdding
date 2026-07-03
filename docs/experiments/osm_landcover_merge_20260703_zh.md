# OSM 合并土地分类说明

日期：2026-07-03

## 目的

用清洗后的 OSM 弱标签生成一张类似 WorldCover 的单通道土地/地物分类图，替换原来的 WorldCover 训练目标。下游任务仍继续使用更细的 OSM 二值标签，例如 `osm_building`、`osm_major_road`、`osm_minor_road`。

## 合并规则

| ID | 合并类 | 来源 OSM 标签 |
| --- | --- | --- |
| 0 | background | 未被 OSM 覆盖或弱标签空白 |
| 1 | residential | `osm_residential` |
| 2 | commercial | `osm_commercial` |
| 3 | industrial | `osm_industrial` |
| 4 | agriculture | `osm_agriculture` |
| 5 | green | `osm_green` |
| 6 | recreation | `osm_playground` |
| 7 | construction | `osm_construction` |
| 8 | water | `osm_water` |
| 9 | building | `osm_building` |
| 10 | transport | `osm_path_walk`、`osm_rail`、`osm_minor_road`、`osm_major_road` |

已排除噪声标签：`osm_activity_poi`。该标签来自 POI/活动点缓冲区，边界不可靠，会大面积遮挡背景。

## 覆盖统计

| ID | 类别 | 像素占比 | 有效 patch |
| --- | --- | ---: | ---: |
| 0 | background | 15.88% | 297 / 320 |
| 1 | residential | 7.93% | 255 / 320 |
| 2 | commercial | 2.18% | 208 / 320 |
| 3 | industrial | 1.13% | 106 / 320 |
| 4 | agriculture | 5.60% | 150 / 320 |
| 5 | green | 26.92% | 317 / 320 |
| 6 | recreation | 0.91% | 210 / 320 |
| 7 | construction | 4.31% | 214 / 320 |
| 8 | water | 3.03% | 252 / 320 |
| 9 | building | 8.03% | 286 / 320 |
| 10 | transport | 24.09% | 320 / 320 |

## 训练接入

- 新标签目录：`/data/xuannv_embedding/processed/haidian/labels/osm_landcover/`
- 新 manifest：`/data/xuannv_embedding/processed/haidian/manifest_p6a_202512_202605_pixelmask_clean_osm_landcover.json`
- 当前 P6A 配置已切换到新 manifest。
- 配置中仍复用 `worldcover` target head，但实际路径已经指向 `osm_landcover`。
- 分类头通道数从 9 改为 11，对应 0 背景 + 10 个 OSM 合并类。

## 下游任务

下游道路提取不直接依赖合并类里的 `transport`，仍可使用：

- `osm_major_road`
- `osm_minor_road`
- `osm_path_walk`
- 需要交通线网时再加入 `osm_rail`

这样主训练获得更稳定的土地分类语义，下游任务仍保留细粒度标签灵活性。
