# xuannv_embedding 数据清单

> 本文档由 `scripts/data/update_inventory.py` 自动生成，最后更新：2026-06-19 08:49 UTC
> 如需查看最新状态，请直接运行该脚本。

## Harbin

| 类别 | Source / Target | 路径 | 大小 | 文件数 | 状态 |
|------|-----------------|------|------|--------|------|
| Raw 时序 | Sentinel-1 | `raw/harbin/s1` | 6.1G | 2 | ✅ 已存在 |
| Raw 时序 | Sentinel-2 | `raw/harbin/s2` | 451G | 13 | ✅ 已存在 |
| Raw 时序 | Landsat | `raw/harbin/landsat` | 3.5G | 2 | ✅ 已存在 |
| Raw 标签 | ESRI 2023 LULC | `raw/harbin/esri_lulc_2023` | 133M | 4 | ✅ 已存在 |
| Raw 高分辨率光学 | 百度 DOM | `raw/harbin/highres_optical` | 45G | 6 | 🔄 百度 DOM 202512/202605 已下完；待预处理 |
| Raw 地理辅助 | DEM | `raw/harbin/dem` | 4.0K | 0 | ⏳ 预留 |
| Raw 地理辅助 | JRC Water | `raw/harbin/jrc_water` | 4.0K | 0 | ⏳ 预留 |
| Raw 地理辅助 | Dynamic World | `raw/harbin/dynamic_world` | 4.0K | 0 | ⏳ 预留 |
| Processed Patch | S2/S1/Landsat | `processed/harbin/patches` | 32G | 43403 | ✅ 已存在 |
| Processed 标签 | worldcover | `processed/harbin/labels/worldcover` | 14M | 675 | ✅ 已存在 |
| Processed 地理辅助 | - | `processed/harbin/geography` | 0 | 0 | ⏳ 预留 |
| 元数据 | manifest | `processed/harbin/manifest.json` | 2.0M | 1 | ✅ 已存在 |
| 统计量 | mean/std | `statistics/harbin` | 8.0K | 1 | ✅ 已存在 |

## Haidian

| 类别 | Source / Target | 路径 | 大小 | 文件数 | 状态 |
|------|-----------------|------|------|--------|------|
| Raw 时序 | Sentinel-1 | `raw/haidian/s1` | 11G | 2 | ✅ 已存在 |
| Raw 时序 | Sentinel-2 | `raw/haidian/s2` | 104G | 2 | ✅ 已存在 |
| Raw 时序 | Landsat | `raw/haidian/landsat` | 3.0G | 2 | ✅ 已存在 |
| Raw 标签 | ESRI 2023 LULC | `raw/haidian/esri_lulc_2023` | 336M | 4 | ✅ 已存在 |
| Raw 高分辨率光学 | PlanetScene | `raw/haidian/highres_optical` | 11G | 283 | 🔄 PlanetScene 已下完并解压；待预处理 |
| Raw 高分辨率 SAR | 天仪 SAR | `raw/haidian/highres_sar` | 288G | 778 | 🔄 天仪 SAR 已下完；待解压/地理编码/配准 |
| Raw 地理辅助 | DEM | `raw/haidian/dem` | 4.0K | 0 | ⏳ 预留 |
| Raw 地理辅助 | JRC Water | `raw/haidian/jrc_water` | 4.0K | 0 | ⏳ 预留 |
| Raw 地理辅助 | Dynamic World | `raw/haidian/dynamic_world` | 4.0K | 0 | ⏳ 预留 |
| Processed Patch | S2/S1/Landsat | `processed/haidian/patches` | 38G | 51486 | ✅ 已存在 |
| Processed 标签 | worldcover | `processed/haidian/labels/worldcover` | 13M | 625 | ✅ 已存在 |
| Processed 地理辅助 | - | `processed/haidian/geography` | 0 | 0 | ⏳ 预留 |
| 元数据 | manifest | `processed/haidian/manifest.json` | 2.3M | 1 | ✅ 已存在 |
| 统计量 | mean/std | `statistics/haidian` | 8.0K | 1 | ✅ 已存在 |

## 图例

- ✅ 已存在/已完成
- 🔄 进行中或部分完成
- ⏳ 等待前置条件/预留
- ❌ 失败/缺失
