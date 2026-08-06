# 中国全国 1280 米父网格交付包最终检查

## 一、结论

中国全国 1280 米父网格已完成生成、集合划分、成员审计、UTM 接缝审计、独立读取和压缩包检查。
最终父网格共 5,785,781 个 patch，其中既有 1% 采样及补采样集合 62,000 个，未采样集合
5,723,781 个。三者满足 `all = sampled + unsampled`，且 sampled 与 unsampled 无交集。

本地交付目录：

`/data/xuannv_embedding/outputs/china_full_1280m_grid_package_v1_20260805/`

本地压缩包：

`/data/xuannv_embedding/outputs/china_full_1280m_grid_package_v1_20260805.zip`

## 二、网格与集合检查

1. 全国父网格数量：5,785,781。
2. 已采样 patch：62,000，全部精确匹配一次，缺失 0，重复 0。
3. 未采样 patch：5,723,781。
4. sampled 与 unsampled 交集：0。
5. 同一 UTM 分区内正面积重叠：0。
6. 无效几何、owner zone 错配、1280 米尺寸错误、坐标哈希错误：均为 0。
7. 输出覆盖 EPSG:32643 至 EPSG:32653；交付几何统一为 EPSG:4326，X 为经度、Y 为纬度。

## 三、UTM 接缝说明

旧验收条件将“任意一对跨 UTM patch 的重叠比例超过 1%”视为错误。全国实测表明，该条件不适合
由 11 套独立 UTM 米制网格组成的全国图集：相邻分区的 1280 米正方形在经线接缝处允许出现窄带
重叠，而每个 patch 的中心仍只属于一个 owner zone。

因此保留原始审计结果，并增加 UTM 接缝专项审计。专项检查结果如下：

1. 跨区正面积重叠 pair：10,411。
2. 非相邻 UTM 分区重叠：0。
3. 离开 UTM 接缝窄带的重叠：0。
4. 两侧 patch 中心 owner 顺序错误：0。
5. 累计重复覆盖面积：2,842.239 平方千米。
6. 重复覆盖面积占全部父网格面积：0.029983%，低于 0.1% 上限。

最终成员审计采用 `adjacent-owner-zone-seam-v1` 规则，状态为通过。旧的 pair 阈值结果作为历史证据
保存在 `china_full_grid_membership_audit_legacy_pair_threshold.json`，没有删除或覆盖。

## 四、独立读取检查

1. 读取全部 189 个 GeoParquet 分片的元数据，累计行数分别为 all 5,785,781、sampled 62,000、
   unsampled 5,723,781。
2. 全部 GeoParquet 的 geometry CRS 为 EPSG:4326。
3. 对 all、sampled、unsampled 三个集合的 11 个 UTM 分区各读取一个代表 Shapefile，共 33 组，
   文件均可打开且 CRS 为 EPSG:4326。
4. `SHA256SUMS.json` 记录交付目录内全部文件的相对路径、大小和 SHA256。
5. ZIP 使用 `zipfile.testzip()` 检查，未发现损坏成员。

## 五、交付内容

交付包包括全国 all、sampled、unsampled 三套 GeoParquet 与 Shapefile，宏网格 GeoPackage，三张检查
预览图，最终成员审计、原始审计和 UTM 接缝审计，`README.md`、合并版 `先读我.docx`、原 1% 采样版
`先读我.docx`，以及边界、修正版宏网格清单、62,000 patch 原始 registry 和生产配置。

最终发布目标为 ModelScope 数据集 `WeijieWu/xuannv_china_full_grid`。
