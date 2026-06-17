# 哈尔滨样本数据状态（2026-06-17）

- 区域：harbin
- 时间范围：2025-01-01 至 2025-01-31
- 已下载 source：s2、s1、landsat
- 原始 NetCDF：
  - `/data/xuannv_embedding/raw/harbin/s2/s2_20250101_20250131.nc`
  - `/data/xuannv_embedding/raw/harbin/s1/s1_20250101_20250131.nc`
  - `/data/xuannv_embedding/raw/harbin/landsat/landsat_20250101_20250131.nc`
- 预处理 patches：
  - s2：20 patches（1 个时点：2025-01-29）
  - s1：40 patches（2 个时点：2025-01-11、2025-01-23）
  - landsat：80 patches（4 个时点：2025-01-02、2025-01-10、2025-01-18、2025-01-26）
- manifest：`/data/xuannv_embedding/processed/harbin/scenes/manifest.json`
  - 共 20 条记录，每条记录均以 s2 的 `YYYYMMDD_pXXX_rXXX` 为 `patch_id`
  - s1、landsat 按空间格网 `pXXX_rXXX` 归到同一条目，支持多日期
- statistics：`/data/xuannv_embedding/statistics/harbin/`
  - `s2_stats.json`
  - `s1_stats.json`
  - `landsat_stats.json`

## 过程记录

1. S1 / Landsat 下载过程中 Landsat 首次因 DNS 解析失败；重试后成功。
2. S1 首次保存 NetCDF 时触发 `int64` 时间编码错误，已修复 `scripts/data/download_pc.py`：
   - 在写入前将 `time` 坐标规范化为 `datetime64[s]`，并显式指定 `int32` + `seconds since 1970-01-01` 编码。
3. 多源 manifest 匹配时，不同 sensor 的观测日期不一致导致无法关联；已增强 `src/xuannv_embedding/utils/manifest.py`：
   - 在精确匹配 `{source}_{patch_id}` 之外，增加按空间格网后缀 `p\d{3}_r\d{3}$` 的退阶匹配。
   - 现有 `tests/test_manifest.py` 全部通过。
