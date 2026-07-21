# 中国版 Alpha Earth 技术方案：现有资源季度版

> **一句话结论：** 使用现有 300 TB 在线存储和 16 张 Ascend 910B，可建设
> 2020Q1--2021Q4 共 8 期、全国 10 m、64 维 int8 地理 embedding；完整源档案、
> 训练工作区、模型权重和季度产品合计按 **250--300 TB** 管理，20 万 updates
> 正式训练约 **36 天**，8 期导出与验收约 **17--27 天**。

版本：2026-07-21  
目标里程碑：2026-10-30，DTDC 大会发布  
拟启动：2026-08-01  
模型研发：赵龙、伍炜杰、陆君言  
数据支撑：CB 项目  
硬件支撑：LZ 项目

## 一、建设目标

建设一个面向中国陆地的季度地理 embedding 底座。产品不是某一个建筑、道路
或水体分割模型，而是每个地理位置一个可复用的 64 维向量，下游只需少量标注
和轻量头即可完成土地覆盖、建筑、道路、水体、绿地、农业、检索和变化分析。

| 项目 | 本方案定义 |
| --- | --- |
| 空间范围 | 中国陆地及主要近岸岛屿 |
| 训练位置 | 62,000 个 `1280 m x 1280 m` patch，约占中国陆地 1.06% |
| 时间范围 | 2020、2021 两年 |
| 产品频率 | 每季度一期，共 8 期 |
| 产品分辨率 | 10 m |
| embedding 维度 | 64 维 |
| 发布编码 | int8，保留分块/分通道量化参数 |
| 模型输出 | 每个 patch 为 `128 x 128 x 64` |
| 训练硬件 | 16 张 Ascend 910B 64 GB |

季度定义固定为 Q1（1--3 月）、Q2（4--6 月）、Q3（7--9 月）、Q4（10--12 月）。
每一期 embedding 只能使用该季度及静态数据，禁止看到未来季度影像。

## 二、与 AlphaEarth 和现有底座的关系

[AlphaEarth Foundations](https://arxiv.org/abs/2507.22291) 证明了约 10 m、64 维
通用地理 embedding 可以支持少样本制图、检索和变化任务。玄女海淀/哈尔滨
底座已经验证 S2、S1、Landsat、稀疏高分影像和 OSM 弱语义的多模态训练路径，
并测得 6 张 910B 约 2,394 patch-visits/卡·小时。

本方案沿用以下已验证原则：

1. 主 embedding 保持 64 维和 10 m 网格；
2. 月/季度光学必须使用像素级云、阴影、雪和无效值掩膜；
3. OSM 仅作宽泛弱语义，空白区域是 unknown，不是 background；
4. 不使用最终下游人工测试标签训练 embedding；
5. S1/SAR 重建权重低于高质量光学，高分目标负责边界细节；
6. 模型按固定 updates 训练，不再用小数据的 800 epoch 口径。

## 三、数据方案

### 3.1 现有清单内的主数据

输入清单为《遥感影像（含产品、地面站数据）》，SHA-256：
`91fc81e02ac836683325e9edb1c9312c7c4299c24f2088a324f11ee60b4f1907`。
清单“数据大小（MB）”单位不统一，本方案以“数据大小（TB）”列和实际文件字节
复核为准。

| 数据 | 2020/2021 可用性 | 分辨率 | 训练用途 | 状态 |
| --- | --- | ---: | --- | --- |
| Sentinel-2 L2A | 两年完整 | 10/20/60 m | 主光学、季度反射率、云掩膜 | 核心 |
| Sentinel-1 GRD/RTC | 两年完整 | 约 10 m | 全天候结构和散射信息 | 核心 |
| Landsat-8 C2 L2 | 两年完整 | 30 m | 独立光学、热红外和 QA | 核心 |
| Landsat-9 C2 L2 | 仅 2021Q4 | 30 m | Q4 补充，不作为跨年必需源 | 可选 |
| GF1/GF6/HJ2 | 清单覆盖 2021，2020 需再核验 | 2/16 m | 稀疏高分光学目标 | 辅助 |
| JL1GP02 黑土地 | 2020-10 至 2021-10 | 5 m | 东北农业和黑土地细节 | 辅助 |
| Hi-GLASS 地表反射率 | 2013--2022 | 30 m | 光谱一致性和物理目标 | 辅助 |
| MOD/MYD 系列 | 两年完整 | 250 m--1 km | NDVI/EVI、LAI、GPP、LST | 辅助 |

Sentinel 和 Landsat 的正式访问分别采用
[Copernicus Data Space](https://www.copernicus.eu/en/access-data) 与
[USGS Landsat Collection 2](https://www.usgs.gov/landsat-missions/landsat-collection-2)。
只读取 62,000 点对应窗口，不通过公网逐景下载全国整景作为主生产路径。

### 3.2 建议补充的土地、地形和植被产品

| 产品 | 年份/频率 | 分辨率 | 如何进入训练 | 访问与注意事项 |
| --- | --- | ---: | --- | --- |
| ESA WorldCover | 2020、2021 年度 | 10 m | 11 类宽语义弱标签 | [官方数据页](https://esa-worldcover.org/en/data-access)；两年算法版本不同，不能直接作为变化真值 |
| WorldCover S1/S2/NDVI composites | 2020、2021 年度 | 10/20 m | 年度稳健光谱和 NDVI 教师 | 官方 AWS COG；与 S1/S2 同源，不能作独立评测 |
| Dynamic World | 两年逐景 | 10 m | 9 类概率弱标签和置信度 | [官方说明](https://www.dynamicworld.app/about/index.html)；只在高置信像素监督 |
| GlobeLand30 | 2020 | 30 m | 2020 宽类土地覆盖交叉约束 | [自然资源部介绍](https://www.ngcc.cn/xwzx/ywcg/202401/t20240103_1270.html) |
| GLC_FCS30D | 覆盖 2020、2021 | 30 m | 细类土地覆盖和一致性筛选 | [ESSD 数据说明](https://essd.copernicus.org/articles/16/1353/2024/) |
| CLCD | 至少覆盖 2020 | 30 m | 中国年度土地覆盖辅助 | [数据与论文](https://essd.copernicus.org/articles/13/3907/2021/)；2021 可用性单独验收 |
| Copernicus DEM GLO-30 | 静态 | 30 m | 高程、坡度、坡向连续目标 | [官方产品页](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM) |
| MOD13Q1/MYD13Q1 | 16 天 | 250 m | 季度 NDVI/EVI 时序目标 | [NASA MODIS](https://modis.gsfc.nasa.gov/data/dataprod/mod13.php) |
| NODA 中国 NDVI/FVC | 2000--2024 | 250 m | 中国区域植被指数交叉约束 | [NODA 公告](https://www.noda.ac.cn/rsgs/news/showNewsById?id=6916eb157575fb4d046058df)；需在线申请 |
| WorldCereal | 2021 季节产品 | 10 m | 作物、灌溉和农业语义 | [官方产品页](https://esa-worldcereal.org/en/products/global-maps)；仅用于 2021 |
| JRC Global Surface Water | 2020/2021 月度 | 30 m | 季节水体连续目标 | [官方入口](https://global-surface-water.appspot.com/download) |
| 历史 OSM | 2020/2021 季末快照 | 矢量 | 建筑、道路、水体、土地利用弱语义 | [ohsome API](https://docs.ohsome.org/ohsome-api/stable/endpoints.html)；保留 known mask |

数据优先级为：10 m 动态观测 > 可靠像素掩膜 > 静态 DEM > 宽语义弱标签 >
250 m--1 km 物理目标。粗分辨率产品不能上采样后冒充 10 m 细节，只能提供
区域级语义或连续物理约束。

### 3.3 NODA 专题入口

- 用户指定专题一：[NODA 主题 6690d6...](https://www.noda.ac.cn/datasharing/theme/viewThemeById?id=6690d6b3494be1036384ea58)；
- 用户指定专题二：[NODA 主题 63ef0d...](https://www.noda.ac.cn/datasharing/theme/viewThemeById?id=63ef0d710e2b985f5244c382)；
- 2020 年 2 m 一张图专题：[NODA 主题 64ffc7...](https://www.noda.ac.cn/datasharing/theme/viewThemeById?id=64ffc7a0ae06c00d8763ec3f)。

NODA 页面部分元数据需要登录后动态加载。正式冻结数据前必须导出每个数据集的
景数、时间范围、波段、位深、覆盖 footprint、许可和真实字节数；网页可访问
不等于已经获得批量训练许可。

## 四、全国 62,000 点抽样

基础抽样沿用已经实现的地理网格方案：`10 x 10` 个 1,280 m patch 构成一个
宏网格，通过固定种子 SHA-256 选择候选；边界残缺宏网格按有效候选数修正接纳
概率，保证每个有效 patch 的期望入选概率约为 1%。

![全国基础1%空间抽样分布](assets/china_v1_2021_multisource/national_static_1pct_sampling_preview.png)

**图 1  全国基础 1% 空间抽样分布。** 红点为 57,405 个基础候选；再增加
2,595 个空间均衡样本、500 个海岸样本和 1,500 个城市/稀有地物/困难观测样本，
最终为 62,000 个，不删除、不重采样现有候选。

训练期间按 sampling reason 控制 batch 组成，避免城市和易获取地区占满训练。
验证使用空间 block split；最终配方确认后允许用 62,000 点全量再训练生产权重，
但独立评测区始终不参与生产指标报告。

## 五、季度数据处理和模型训练

### 5.1 季度合成

每个 patch、每个季度执行以下流程：

1. S2 用 SCL 掩膜云、阴影和无效像素，Landsat 用 QA_PIXEL，S1 保留有效掩膜；
2. 先按 patch 内有效像素比例筛场景，再对质量合格观测做中位数/分位数组合；
3. 不把低质量景作为兜底，允许某源某季度缺失；
4. 每个源保留观测日期、场景 ID、质量分数和每像素掩膜；
5. 土地覆盖、DEM 和 OSM 使用 known/confidence mask；冲突位置关闭语义损失；
6. 2020 与 2021 使用同一投影、网格原点、波段标度和类别映射。

### 5.2 模型和损失

模型保持 P10C 已验证的多源编码器、64 维 bottleneck 和 `128 x 128` 输出，按
模态独立归一化和独立解码。主要损失保持简洁：

- 掩膜内 S2/Landsat 光学重建；
- 低权重 S1 重建；
- DEM/NDVI/LAI 等连续物理目标；
- OSM/土地覆盖 masked weak semantic loss；
- modality dropout 困难重建；
- 低权重方差/协方差约束，防止 64 维坍缩。

所有损失先做梯度尺度审计。某个辅助损失若降低但 downstream probe 持续下降，
应移除或降权，不以“损失越多越先进”为目标。

## 六、存储和算力

### 6.1 10 m 季度 embedding 容量

```text
中国陆地面积 = 9.6 x 10^12 m2
10 m 像素数 = 9.6 x 10^12 / 100 = 96,000,000,000
单季度 int8 = 960 亿 x 64 Byte = 6.144 TB
8 个季度 = 49.152 TB
```

| 内容 | 单份净载荷 | 工程规划 |
| --- | ---: | ---: |
| 8 期全国 int8 embedding | 49.15 TB | 45--55 TB（分块压缩、索引和质量掩膜） |
| 62,000 点季度训练数据包 | 约 1.5--2.5 TB | 3 TB 验收线 |
| best/last 权重与下游头 | 小于 0.01 TB | 0.05 TB |
| 完整 2020/2021 源档案 | 约 190--220 TB | 以真实文件审计更新 |
| 缓存、双处理版本、QA 和日志 | - | 15--25 TB |
| **项目在线存储合计** | - | **约 250--300 TB** |

结论：现有 300 TB 可以支撑，但必须采用“源档案只保留一份、训练裁窗分片、
embedding int8、临时缓存按 shard 回收”的策略。若还要求 embedding 双副本，
应再增加至少 60 TB；更稳妥的物理容量为 350--400 TB。

### 6.2 训练和导出时间

| 阶段 | 16 张 910B 预计时间 | 通过条件 |
| --- | ---: | --- |
| 2,000 点试点 | 5--7 天 | 对齐、掩膜、季度覆盖和容量通过 |
| 1,000-step 性能基准 | 1--2 天 | 每卡 HBM <=48 GB，扩展效率 >=70% |
| 20 万 updates 正式训练 | 约 36 天 | 固定 probe 和恢复演练通过 |
| 8 期全国 embedding 导出 | 约 10--17 天 | 16 卡端到端 >=35 patch/s |
| QA、索引和下游验收 | 约 7--10 天 | 无缺片、无接缝、量化精度通过 |

模型训练预计实际峰值 20--32 GB HBM/卡，工程控制线 48 GB/卡。16 张 64 GB
910B 总 HBM 1,024 GB，满足本方案。

## 七、实施计划

| 日期 | 工作 | 交付门槛 |
| --- | --- | --- |
| 08-01 至 08-10 | 数据清单、许可、真实容量和季度覆盖审计 | 2020/2021 各源 coverage matrix |
| 08-11 至 08-20 | 2,000 点物化、可视化和时序方案消融 | 数据 QA 报告、冻结配置 |
| 08-21 至 08-25 | 16 卡性能与恢复基准 | HBM、吞吐和 checkpoint 通过 |
| 08-26 至 10-01 | 20 万 updates 正式训练 | best/last 权重和周期 probe |
| 10-02 至 10-18 | 8 期 embedding 并行导出 | 分区分片验收、失败自动恢复 |
| 10-19 至 10-25 | 下游评测、检索、变化分析和报告 | 固定协议结果与全域 QA |
| 10-26 至 10-30 | 补片、文档和发布缓冲 | 不新增大实验 |

如果 16 卡直到 8 月底才可用，正式训练和导出将直接进入关键路径，10 月 30 日
仅有极小缓冲。此时必须在 8 月内提前完成全部数据物化和 CPU 侧 QA，并禁止在
训练启动后临时增加新模态。

## 八、验收指标

1. 8 个季度全国分片数量完整，checksum、索引和质量掩膜齐全；
2. 全域 PCA 不出现系统性 patch 接缝、省域颜色漂移和云块；
3. 同协议 linear/MLP/conv3x3 probe 覆盖建筑、道路、水体、土地覆盖和农业；
4. 报告 F1、AP、AUC、mIoU、Recall@K 和 5-fold x 3-seed 均值方差；
5. 2020Q4--2021Q1 的变化结果不能由季节、云或标签版本差异主导；
6. int8 相对 FP16 的固定下游指标下降不超过预设容差；
7. 训练数据、权重、季度 embedding 和下游头都有版本、哈希和数据许可清单。

## 九、主要风险与决策门

- **WorldCover 版本差异：** 2020 v100 与 2021 v200 不能直接作为变化真值；
- **NODA 批量许可：** 页面可访问不等于可批量训练，8 月 10 日前必须书面确认；
- **2020 国产高分覆盖：** 缺失区域不做虚假填充，以 S2/S1/L8 为主；
- **公网下载：** 公网只用于小型标签和补充，不承担全国主数据迁移；
- **300 TB 容量紧张：** 不保留重复整景和数百万 `.pt` 小文件；
- **时间节点：** 任一核心源 8 月 20 日仍未通过 QA，则从首版移出而不是拖延全局。

## 十、公开依据

- [AlphaEarth Foundations 论文](https://arxiv.org/abs/2507.22291)
- [ESA WorldCover 2020/2021 数据与年度 composites](https://esa-worldcover.org/en/data-access)
- [Copernicus Sentinel 数据访问](https://www.copernicus.eu/en/access-data)
- [USGS Landsat Collection 2](https://www.usgs.gov/landsat-missions/landsat-collection-2)
- [Copernicus DEM GLO-30](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM)
- [NASA MODIS Vegetation Indices](https://modis.gsfc.nasa.gov/data/dataprod/mod13.php)
- [ESA WorldCereal 2021](https://esa-worldcereal.org/en/products/global-maps)
- [ohsome 历史 OSM API](https://docs.ohsome.org/ohsome-api/stable/endpoints.html)
- [NODA 2021 高分一张图发布说明](https://www.noda.ac.cn/aircas/news/showNewsById?id=6694ccca4782da475b5c8ffa)

## 十一、项目实现依据

- [全国基础 1% 抽样代码](https://github.com/go-bananas-wwj/xuannv_embdding/blob/v3-semantic-64d/scripts/data/visualize_national_sampling.py)
- [空间补采样代码](https://github.com/go-bananas-wwj/xuannv_embdding/blob/v3-semantic-64d/scripts/data/build_national_spatial_supplement.py)
- [海岸补采样代码](https://github.com/go-bananas-wwj/xuannv_embdding/blob/v3-semantic-64d/scripts/data/build_national_coastal_supplement.py)
- [全国数据物化器](https://github.com/go-bananas-wwj/xuannv_embdding/blob/v3-semantic-64d/scripts/data/materialize_china_v1_shard.py)
- [全国下载 watchdog](https://github.com/go-bananas-wwj/xuannv_embdding/blob/v3-semantic-64d/scripts/data/run_china_v1_watchdog.py)
