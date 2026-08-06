# 玄女中国全国 1280 米父网格（V1）

本数据集提供中国全国统一的 1280 米 patch 划分，用于全国遥感数据下载、模型训练、embedding
推理与结果拼接。网格覆盖 WGS84 / UTM North 的 43 至 53 分区，交付几何统一保存为 EPSG:4326。

## 数据概览

| 集合 | patch 数量 | 用途 |
|---|---:|---|
| 全国完整父网格 all | 5,785,781 | 全国推理和统一空间索引 |
| 已采样 sampled | 62,000 | 1% 基础采样与重点区域补采样 |
| 未采样 unsampled | 5,723,781 | 全国父网格中尚未进入训练采样的部分 |

三个集合满足 `all = sampled + unsampled`，sampled 与 unsampled 没有交集。既有 62,000 个
采样 patch 全部属于全国父网格，缺失和重复均为 0。

## 下载文件

主要交付文件位于 `releases/china-full-1280m-v1-20260805/`：

1. `china_full_1280m_grid_package_v1_20260805.zip`：完整交付包，约 2.07 GB。
2. `china_full_1280m_grid_package_v1_20260805.zip.sha256.json`：ZIP 大小、SHA256 与完整性结果。
3. `先读我.docx`：采样方法、父网格关系、坐标系与文件说明。
4. `先读我_原1%采样版.docx`：此前 1% 采样方案的原始说明。
5. `china_full_grid_final_qa_20260805.md`：最终检查记录。
6. 三张 PNG：全国分区密度、局部 sampled/unsampled 网格和 UTM 接缝检查图。

ZIP SHA256：

```text
7d0e70436acc5e20afe0ffbaaf141c4173fcf716def57276cc30bead36db9a9c
```

## 压缩包结构

```text
china_full_1280m_grid_package_v1_20260805/
├── all/                         # 全国完整父网格，按 UTM 分区保存
├── sampled/                     # 62,000 个已采样 patch
├── unsampled/                   # 5,723,781 个未采样 patch
├── provenance/                  # 边界、修正版宏网格、采样 registry 与配置
├── previous_sampling/           # 原 1% 采样说明和审计
├── china_full_1280m_macrocell_index.gpkg
├── china_full_grid_membership_audit.json
├── china_full_grid_utm_seam_audit.json
├── SHA256SUMS.json
├── README.md
└── 先读我.docx
```

GeoParquet 是完整信息的标准版本。Shapefile 便于在常见 GIS 软件中直接查看，字段名已缩短，并按
UTM 分区与行块拆分，避免单文件超过 2 GB。

## 坐标系和网格规则

每个 patch 在自身 owner UTM 分区中为严格的 1280 米 × 1280 米正方形，整数网格从该 UTM
坐标系的 `(0 m, 0 m)` 起算。唯一键为：

```text
{grid_epsg}:{grid_col}:{grid_row}
```

交付多边形使用 EPSG:4326，X 为经度、Y 为纬度。计算距离、面积、缓冲或邻接关系时，应根据记录
中的 `grid_epsg` 转回对应 UTM 坐标系。

相邻 UTM 分区的米制正方形在经线接缝处允许出现窄带重叠。专项审计确认：非相邻分区重叠为 0，
离开接缝的重叠为 0，中心归属错误为 0；累计重复覆盖面积约占全国父网格总面积的 0.030%。

## 检查结果

1. 全国父网格、已采样和未采样集合计数一致。
2. 62,000 个采样 patch 全部精确匹配一次。
3. 无同一 UTM 分区内正面积重叠。
4. 无无效几何、尺寸错误、owner zone 错配或哈希不一致。
5. 已读取全部 189 个 GeoParquet 分片和 33 个代表 Shapefile，CRS 均为 EPSG:4326。
6. ZIP 通过 `zipfile.testzip()` 完整性检查。

代码与生成流程见 GitHub 仓库：
[go-bananas-wwj/xuannv_embdding](https://github.com/go-bananas-wwj/xuannv_embdding/tree/feature/china-full-grid-20260805)
