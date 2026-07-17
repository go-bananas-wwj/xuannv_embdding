# 玄女 China V1：数据下载、存储与处理实施计划

## 1. 这次要做什么

当前已经生成 **60,500 个全国候选 patch**。每个 patch 是 `1280 m x
1280 m`，输出空间大小保持 `128 x 128`，等效 10 m 分辨率。

候选点由三部分组成：

| 类型 | 数量 | 含义 |
| --- | ---: | --- |
| 全国空间主样本 | 57,405 | 约 1% 的陆地空间骨架，保证全国基本覆盖。 |
| 空间分层补样 | 2,595 | 对较少覆盖的 UTM 空间分区进行补充。 |
| 海岸补样 | 500 | 用独立海洋面数据确认的临海点，不把陆地国界误当海岸。 |
| 合计 | 60,500 | 这是下载前的候选点，不等于最终可训练点。 |

要建立的影像档案为 **2025 年 4 月到 2026 年 4 月，共 13 个月**。模型一次仍只看连续 6 个月；13 个月的作用是提供多个训练窗口和更多可用观测。最终某个点若云量、掩膜或数据缺失不达标，会被剔除或替换，不能为了凑数量保留坏影像。

## 2. 数据从哪里来

所有网络任务都通过 `scripts/data/run_china_v1_direct.sh` 启动。它会清除本机的代理环境变量，因此**不使用梯子**。本机已用无代理直连测试过 Planetary Computer、Copernicus Data Space 和 USGS 的目录接口。

| 数据 | 首选来源 | 备用来源 | 下载内容与用途 | 访问注意事项 |
| --- | --- | --- | --- | --- |
| Sentinel-2 L2A | Microsoft Planetary Computer：`sentinel-2-l2a` | Copernicus Data Space：`sentinel-2-l2a` | 光学反射率和 SCL 场景分类，用作主光学输入与云掩膜 | Planetary Computer 的目录与多数资产可匿名访问，但会限流。 |
| Sentinel-1 | Planetary Computer：`sentinel-1-rtc`，**仅在已有账户授权时使用** | Copernicus Data Space Sentinel-1 GRD | VV/VH 雷达输入，云天情况下提供补充信息 | 官方文档说明 PC 的 RTC 集合需要账户。没有授权时，必须先在 2,000 点试点验证 CDSE GRD 的处理路径，不能直接混用。 |
| Landsat Collection 2 L2 | Planetary Computer：`landsat-c2-l2` | USGS Landsat STAC | 独立光学观测，读取表面反射率与 `QA_PIXEL` | 用 QA 位图生成有效像素掩膜。 |
| ESA WorldCover、Copernicus DEM | Planetary Computer 或官方静态产品 | 无 | 只做抽样分层、地形统计和审计 | 不作为精细建筑/道路真值。 |
| OSM | Geofabrik 中国 `2025-01-01` 快照 | 无 | 广义弱语义、稀有类别抽样 | 已下载并锁定 SHA-256；空白区域一律视为未知。 |
| 海洋面 | Natural Earth 10m Ocean | 无 | 仅识别海岸补样 | 已下载并校验。 |

权威接口说明：

- [Planetary Computer 数据访问说明](https://planetarycomputer.microsoft.com/docs)
- [Planetary Computer 的 STAC 云量筛选示例](https://planetarycomputer.microsoft.com/docs/quickstarts/reading-stac/)
- [Planetary Computer Sentinel-1 RTC 访问要求](https://planetarycomputer.microsoft.com/docs/tutorials/customizable-rtc-sentinel1/)
- [Copernicus Data Space STAC 接口](https://documentation.dataspace.copernicus.eu/APIs/STAC.html)
- [Copernicus Data Space S3 下载接口](https://documentation.dataspace.copernicus.eu/APIs/S3.html)
- [USGS Landsat STAC 接口](https://www.usgs.gov/landsat-missions/spatiotemporal-asset-catalog-stac)

## 3. 数据如何存储

所有全国数据只放在第二块盘：`/data2/xuannv_embedding/china_v1/`。代码、配置和文档只放 Git 仓库 `/root/workspace/xuannv/`；大影像、权重和日志不进 Git。

```text
/data2/xuannv_embedding/china_v1/
├── source_snapshots/             # 小型、可追溯的原始静态数据；永久保留
│   ├── osm/                      # 2025-01 OSM PBF、SHA-256 锁文件
│   ├── geoboundaries/            # 国界、省界、SHA-256
│   └── natural_earth/            # 海洋面、SHA-256
├── atlas/                        # 候选点、最终质量合格点、空间统计
├── stac_catalogs/                # 按“数据源/月份/UTM 分区”缓存的精简 STAC JSONL
├── shards/                       # 实际影像分片，每个分片 500 个 patch
│   ├── pending/                  # 尚未完成或需重试的分片
│   ├── complete/shard_00000/     # 已验证分片
│   │   ├── s2/ s1/ landsat/      # 月度复合影像
│   │   ├── masks/                # 每源每月有效像素掩膜
│   │   ├── quality/              # 每个候选景的分数、云量、选中理由
│   │   └── manifest.jsonl        # 分片内的相对路径和来源信息
│   └── failed/                   # 失败原因、可重试任务；不混入 complete
├── labels/                       # OSM 置信掩膜、弱语义栅格、静态统计
├── manifests/                    # 合并后的最终训练清单、替换记录、哈希
├── statistics/                   # 各源均值/方差、有效像素比例
├── qa/                           # 质量报告、随机 patch 可视化、覆盖地图
├── logs/                         # 每个分片的命令、Git commit、吞吐量日志
├── tmp/                          # 仅存短暂 COG 窗口；分片验收后删除
└── quarantine/                   # 中断/过期/不可用文件；永不参与训练
```

### 容量预算

海淀 P10C 已测得六个月 S2+S1+Landsat 的质量筛选后数据约为 `8.18 MB/patch`。按 13 个月和 60,500 点换算：

| 内容 | 预计占用 | 保存策略 |
| --- | ---: | --- |
| S2 月度复合 | 约 235 GB | 仅保留质量最好的复合结果。 |
| S1 月度复合 | 约 336 GB | 仅保留复合结果和有效掩膜。 |
| Landsat 月度复合 | 约 502 GB | 仅保留复合结果和 QA 掩膜。 |
| 三类核心传感器 | 约 1.07 TB | 60,500 点 x 13 个月。 |
| 掩膜、OSM/静态层、清单、统计、QA | 70-120 GB | 压缩存储。 |
| 临时目录峰值 | 不超过 150 GB | 每个分片验收后删除。 |
| 总峰值 | 约 1.30-1.40 TB | 不保留全国原始整景副本。 |

当前数据盘可用空间超过 3 TiB，足够执行并保留后续嵌入产物。

## 4. 怎么下载和处理

### 第一步：冻结候选点与静态层

1. 固定 60,500 点、抽样种子、海岸补样规则、国界/海洋面/OSM 的 SHA-256。
2. 对每个点记录 `patch_id`、UTM 分区、坐标、抽样原因和来源版本。
3. 不下载任何全国整景影像，不生成全国 NetCDF。

### 第二步：先只下载目录信息

按 `月份 x UTM 分区 x 数据源` 请求 STAC 元数据，只保存必要字段：景 ID、时间、边界、云量、投影、资产 URL、版本号。

1. S2 和 Landsat 先用 `eo:cloud_cover` 做低成本预筛，例如优先请求云量较低景。
2. 这个筛选只是减少无效请求，最终是否保留由 patch 内的 SCL/QA 像素掩膜决定。
3. 每次查询保存 JSONL、查询参数和返回哈希，保证之后可重放。
4. S1 在这里完成账号/授权探测；没有稳定的 RTC 读取能力就停止，不进入全量下载。

### 第三步：2,000 个 patch 试点

先选 2,000 个点，分为 4 个、每个 500 点的分片。每个分片按以下顺序执行：

1. 从 STAC 选出与该 patch 相交的候选景。
2. 对签名 COG 只读取 patch 周边窗口，不下载整景。
3. 将窗口重投影到该点所属 UTM 分区的 10 m 网格。
4. S2：连续反射率用双线性重采样，SCL 用最近邻重采样；根据 SCL 生成云/阴影/雪等有效像素掩膜。
5. Landsat：六个反射率波段应用比例系数；`QA_PIXEL` 原样保留，绝不缩放或双线性插值；根据位标记生成掩膜。
6. S1：统一 VV/VH 的标度、投影和无效值定义；记录是否来自 RTC 或经过验证的 GRD 流程。
7. 每个 patch 每月最多保留 S2/Landsat 质量最好的两景，生成月度复合及每像素有效掩膜。
8. 写入分片临时目录，运行哈希、空间对齐、有效像素比例检查；通过后原子移动至 `complete/`，删除 `tmp/`。

### 第四步：全量物化

试点通过后，再处理剩余 58,500 点。每次最多同时运行 6 个 CPU/I/O worker，总远程请求从 6 开始，稳定后最多提高到 12；NPU 不参与下载，留给后续训练。

每完成一个 500 点分片，就生成：

- `manifest.jsonl`：影像相对路径、输入景 ID、日期、投影和哈希；
- `quality.jsonl`：候选景分数、云量、像素有效比例、被保留/拒绝的原因；
- `shard_report.json`：字节数、耗时、失败数、重试数、Git commit、实际导入包路径；
- 2 个随机 patch 的“每月 x 每源”可视化大图。

## 5. 质量门禁

一个点只有满足以下条件才进入最终训练清单：

1. 对任意连续 6 个月训练窗口，S2、S1、Landsat 每个源至少有 4 个完整月份。
2. 六个月窗口总计至少有 14 个“源-月份”可用记录。
3. 掩膜缺失直接失败，不把全图默认当有效。
4. 低质量景不能作为兜底景保留；一个月可为空。
5. OSM 未覆盖区域是未知区域，弱语义损失在这些位置关闭。
6. 每个分片必须通过随机可视化对齐检查：S2、S1、Landsat、掩膜和 OSM 边界不允许出现系统性偏移。

## 6. 时间与 ETA

| 阶段 | 内容 | 并行方式 | 预计时间 |
| --- | --- | ---: | ---:|
| 0 | 冻结候选、静态层、哈希 | CPU | 已完成 |
| 1 | 13 个月 STAC 元数据目录 | 6 worker，最多 12 个请求 | 2-6 小时 |
| 2 | 2,000 点试点物化与处理 | 6 个 CPU/I/O worker | 4-8 小时 |
| 3 | 试点 QA、吞吐量和容量复核 | CPU | 0.5-1 小时 |
| 4 | 剩余 58,500 点全量物化 | 6 worker，自适应限流 | 30-60 小时 |
| 5 | 最终清单、替换点、统计、全域可视化 | CPU | 4-8 小时 |
| 6 | 人工验收后启动 NPU 训练 | 6 NPU | 另行确认 |

全量阶段的保守估计为 30-60 小时，原因是需要传输约 1.5-2.2 TB 的 COG 窗口数据，并为远端限流、重试、重投影和磁盘校验预留时间。试点结束后用实测速度重算：

```text
全量 ETA = 2,000 点试点实际耗时 x 60,500 / 2,000 x 1.3（重试系数）
```

## 7. 当前不做的事情

- 不下载全国高分光学或高分 SAR：没有冻结的全国许可、时间覆盖和存储预算前不引入。
- 不下载全国完整 Sentinel/Landsat 整景或拼成全国 NetCDF。
- 不在下载过程中启动训练。
- 不把 OSM 或未来日期标签当成下游真值。

## 8. 启动顺序

当前可直接开始的是“STAC 元数据目录 + 2,000 点试点”。只有试点报告确认 S1 路径、云掩膜、空间对齐、容量和真实 ETA 均正常，才开始 58,500 点全量下载。
