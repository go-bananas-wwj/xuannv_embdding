# 先读我：中国 1,280 m 全国完整父网格包

版本：2026-08-05

## 一、这个包解决什么问题

本包交付中国陆地区域的**全国完整父网格**：在 WGS84 / UTM 43N 至 53N 的各自米制
坐标系中，以 `(0 m, 0 m)` 为原点建立边长严格为 1,280 m 的规则 patch。只有中心点被
冻结中国 ADM0 边界覆盖、且中心经度属于该 UTM 半开归属带的 patch 才进入父网格。

父网格是长期、可复现的空间索引，不包含 Sentinel-1、Sentinel-2、Landsat 影像，也不表示
已经完成季度质量验收或已经生成 embedding。

## 二、与既有 62,000 个采样候选的关系

此前交付的 62,000 个候选位置是用于全国季度嵌入数据获取、覆盖统计和质量复核的采样骨架。
它们不是另一套坐标网格：每个候选必须按 `grid_epsg + grid_col + grid_row` 精确对应父网格中的
一个 patch。

- **已采样 patch**：父网格中与既有 62,000 候选注册表精确匹配的成员。它们保留原有的采样
  层、质量复核和训练准入限制。
- **未采样 patch**：同一完整父网格内其余成员。它们是可查询的空间索引，不是负样本、也不是
  已下载或已验收的训练数据。
- 三个图层满足：`all = sampled + unsampled`，且 `sampled` 与 `unsampled` 没有共享的
  `parent_key`。

## 三、坐标系和轴顺序

网格几何、面积、邻接和重叠在所属的 WGS84 / UTM 北半球坐标系中计算。中国使用
EPSG:32643 至 EPSG:32653；例如 UTM 43N 是 EPSG:32643。交付图层统一转换为 WGS84
地理坐标系 EPSG:4326，坐标单位为十进制度，**X=经度、Y=纬度**。

因此，EPSG:4326 适合全国汇总和地图显示，不能直接用经纬度差计算 1,280 m 的长度或面积。
需要计算尺度、面积、缓冲、邻接或 overlap 时，应按每条记录的 `grid_epsg` 转回相应 UTM
坐标系。

## 四、文件清单

- `all/`：全国完整父网格的分区 GeoParquet 与分片 Shapefile。
- `sampled/`：与既有采样注册表精确匹配的父网格成员。
- `unsampled/`：完整父网格中不在既有采样注册表的成员。
- `china_full_1280m_macrocell_index.gpkg`：`macrocells` 和 `utm_zones` 索引图层，含精确的
  all/sample/unsampled 计数。
- `china_full_grid_membership_audit.json`：成员关系、几何、哈希和重叠审计。
- `china_full_1280m_grid_package_manifest.json`：本次交付的计数、索引、预览和审计元数据。
- `national_patch_density_by_utm_zone.png`：按 UTM 分区检查 patch 密度。
- `local_1280m_grid_sampled_unsampled.png`：局部 1,280 m 网格中已采样和未采样成员的共同视图。
- `utm_owner_zone_seams.png`：UTM owner-zone 接缝及跨区 overlap 规则检查图。
- `先读我.docx`：本说明的 Word 版。

## 五、审计结论如何使用

只有当成员审计明确报告既有采样成员全部精确匹配一次、宏网格计数之和等于完整 GeoParquet
计数、`all = sampled + unsampled` 且无交集时，才能将本包作为空间索引使用。边界与 UTM
接缝附近的 footprint 可以跨越 owner-zone 边界。验收要求重叠只发生在相邻 UTM owner zone
的边界窄带、两侧中心归属正确、非相邻重叠为零，并且全国累计重复覆盖面积不超过父网格
总面积的 0.1%。任何重新采样、下载或训练准入都仍须遵循既有采样说明和相应质量审计。
