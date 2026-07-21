# 全国地理 Embedding 会前调研与两天执行清单

版本：2026 年 7 月 21 日

用途：用于两至三天后的项目同步会。会议重点回答模型规模、境外数据下载、10 米数据准备，以及 2 米产品两条路线的可行性和成本。

## 一、会前需要形成的四个结论

1、全国 10 米底座按约 1 亿参数设计，使用 16 张 Ascend 910B 训练。

2、境外下载节点暂定为 Frankfurt 下载 Sentinel-2，Oregon 下载 Sentinel-1 和 Landsat。两地都需要完成真实速度测试。

3、10 米版要形成逐源数据清单，写清楚从哪里下载、下载多少、怎样生成像素掩膜、怎样配准和怎样形成季度输入。

4、2 米版同时保留两条路线：生成式嵌入升级，以及直接采购高分辨率影像。会议上需要比较两条路线的精度、成本、存储和后续更新费用。

## 二、1 亿参数模型方案

### 1. 当前模型规模

海淀 P10C 生产模型实测为 33,936,230 个参数。

纯模型 FP32 权重约 129.5 MiB，FP16 权重约 64.7 MiB，包含 Adam 优化器和训练状态的 epoch 800 checkpoint 为 407.7 MB。

全国版不改变 64 维输出，也不改变 `128 x 128` 空间输出。参数主要增加在 STP 主干，不通过增加最终 embedding 维度扩张模型。

### 2. 已核算的 1 亿参数候选

| 配置项 | 海淀 P10C | 全国 1 亿参数候选 |
| --- | ---: | ---: |
| stem_dim | 32 | 48 |
| space_dim | 512 | 768 |
| time_dim | 256 | 384 |
| precision_dim | 128 | 192 |
| STP blocks | 6 | 8 |
| attention heads | 8 | 8 |
| embed_dim | 64 | 64 |
| 实际参数量 | 33.94 M | 101.15 M |
| FP16 纯模型 | 67.9 MB | 202.3 MB |
| FP32 模型加 Adam 状态理论下限 | 0.41 GB | 1.21 GB |

101.15 M 是按照当前代码实际实例化后统计的参数量，不是按比例估算。

### 3. 启动条件

1、先在单卡运行前向、反向和一个优化步骤，检查维度及损失。

2、再用 6 卡运行 1,000 step，测量单卡 HBM、吞吐和通信占比。

3、最后用 16 卡运行 1,000 step，确认两台 8 卡节点之间的 HCCL 扩展效率。

4、单卡峰值显存控制在 48 GB 以内。超出后依次使用梯度检查点、降低单卡 batch 和增加梯度累积。

5、1 亿参数模型只有在固定下游 probe 不低于 3,394 万参数模型时才进入正式训练。不能只因为参数更多就直接替换生产结构。

## 三、境外下载节点的结论

### 1. Sentinel-1 不需要在阿拉斯加开服务器

Alaska Satellite Facility（ASF）是 Sentinel-1 的检索、授权和数据服务机构。ASF 的 Sentinel-1 云数据位于 NASA Earthdata Cloud，直接 S3 访问要求计算资源位于 AWS `us-west-2`，即 Oregon。

推荐流程为：

1、注册免费的 NASA Earthdata Login。

2、通过 ASF Vertex 或 `asf_search` 查询 2020--2021 年与 62,000 个 patch 相交的 Sentinel-1 产品。

3、在 Oregon `us-west-2` 创建 EC2 服务器。

4、通过 ASF 的临时 S3 credential 接口获得短期访问密钥。

5、在同区直接读取 S3 对象，完成 RTC 或裁窗后，只把训练分片传回国内。

正式依据：[ASF S3 credentials](https://sentinel1.asf.alaska.edu/s3credentialsREADME)、[NASA Earthdata Cloud 同区访问说明](https://www.earthdata.nasa.gov/s3fs-public/2023-05/LAADS%20DAAC%20Webinar%20_Final%20Presentation_4_26_23_0.pdf)。

### 2. Sentinel-2 优先测试 Frankfurt

AWS 的 Sentinel-2 L2A 公开桶 `s3://sentinel-s2-l2a` 位于 `eu-central-1`，即 Frankfurt，可以匿名访问。

Copernicus Data Space 也提供 S3 兼容接口 `https://eodata.dataspace.copernicus.eu/`。该接口使用欧洲对象存储并根据访问位置分流。

Frankfurt 需要比较两个来源：

1、AWS Sentinel-2 L2A 公开桶，测试匿名整产品下载和窗口读取。

2、Copernicus Data Space S3，测试同一批产品的并行下载。

正式依据：[AWS Sentinel-2 Open Data](https://registry.opendata.aws/sentinel-2/)、[Copernicus Data Space S3](https://documentation.dataspace.copernicus.eu/APIs/S3.html)。

### 3. Landsat 优先测试 Oregon，不优先放在 Frankfurt

USGS Landsat Collection 2 的官方 S3 桶为 `s3://usgs-landsat`，位于 Oregon `us-west-2`，并启用了 Requester Pays。

Landsat 与 Sentinel-1 可以共用 Oregon 服务器。这样可以减少跨区域读取和跨区域传输费用。

正式依据：[USGS Landsat Cloud Access](https://www.usgs.gov/software/introduction-landsat-cloud-access-direct-requester-pays)。

### 4. 亚马逊付费桶是否更快

Requester Pays 的含义是请求方承担 GET、LIST 和数据传输费用，不是购买更高下载优先级。

下载速度主要取决于以下因素：

1、计算服务器与对象桶是否在同一个 AWS region。

2、是否直接读取 COG、JP2 或 GeoTIFF 的窗口，还是把完整产品先下载下来。

3、并发数、单对象大小、EBS 写入速度和服务器网络带宽。

4、是否反复进行小文件 GET 和 LIST 请求。

对本项目最有价值的付费方式是“在同区租 CPU 服务器，把代码搬到数据旁边”，而不是从国内直接购买 Requester Pays 流量。

AWS 官方说明 Requester Pays 由请求方支付请求和传输费用：[AWS Requester Pays](https://docs.aws.amazon.com/AmazonS3/latest/userguide/troubleshoot-403-errors.html)。

## 四、两地服务器怎样开

### 1. 账号准备

1、申请 AWS 国际站账号，不使用 AWS 中国区账号。国际站与中国区账号体系和 region 不互通。

2、绑定可用的国际支付方式，开启 MFA。

3、创建 IAM 管理账号和运行角色，不在程序中保存 root 密钥。

4、在 AWS Budgets 设置 200 美元试验告警和 500 美元硬限制提醒。

5、另外注册 NASA Earthdata Login 和 Copernicus Data Space 账号。

6、所有密钥保存到服务器的 secret/environment 中，不写进 GitHub、文档或聊天记录。

### 2. Frankfurt 服务器

1、AWS Console 右上角选择 `Europe (Frankfurt) eu-central-1`。

2、创建 Ubuntu 22.04 或 24.04 EC2。

3、试验机建议 `m7i.4xlarge` 或同等级实例，至少 16 vCPU、64 GB 内存。

4、挂载 1--2 TB gp3 EBS，吞吐先设置 500 MB/s。

5、安全组只开放 SSH 22，并限制为单位出口 IP。

6、安装 AWS CLI、GDAL、rasterio、boto3、s3cmd、rclone 和项目裁窗代码。

7、分别测试 AWS Sentinel-2 L2A 和 Copernicus Data Space S3。

匿名检查 AWS L2A 桶的命令为：

```bash
aws s3 ls --no-sign-request --region eu-central-1 s3://sentinel-s2-l2a/
```

Copernicus Data Space 使用该平台生成的 S3 key，并指定 endpoint：

```bash
aws --endpoint-url https://eodata.dataspace.copernicus.eu s3 ls s3://eodata/Sentinel-2/
```

### 3. Oregon 服务器

1、AWS Console 选择 `US West (Oregon) us-west-2`。

2、创建与 Frankfurt 相同规格的 EC2 和 gp3 EBS。

3、为 Landsat 创建仅允许读取 `usgs-landsat` 的 IAM role。

4、Landsat 请求必须带 `--request-payer requester`。

```bash
aws s3 cp --request-payer requester \
  s3://usgs-landsat/<collection-2-object> ./
```

5、Sentinel-1 使用 Earthdata Login 登录 ASF，获得临时 S3 credentials。

6、通过 `asf_search` 查询产品，并读取查询结果中的 S3/HTTPS 地址。

7、如暂时拿不到 S3 路径，同时测试 ASF HTTPS 下载，作为对照组。

### 4. 服务器关闭和费用控制

1、试验完成后先导出日志和结果，再 terminate EC2。

2、单独删除不再使用的 EBS、snapshot、Elastic IP 和临时 S3 bucket。

3、每天检查 Cost Explorer。Requester Pays 需要同时检查请求费和数据传输费。

4、境外服务器只保存临时原始数据。裁窗、掩膜和分片完成后立即删除完整场景。

5、正式费用必须使用启动当天的 [AWS Pricing Calculator](https://calculator.aws/) 重新计算。当前文档不把公开价当成长期合同价。

## 五、境外下载速度怎样测试

### 1. 测试样本

从全国清单固定选择 120 个 patch：

| 类型 | 数量 |
| --- | ---: |
| 东部城市 | 20 |
| 东北农业 | 20 |
| 西北荒漠 | 20 |
| 西南山区 | 20 |
| 华南多云区 | 20 |
| 海岸和岛屿 | 20 |

每个 patch 查询 2020Q1--2021Q4。三种数据源使用完全相同的 patch 和季度，保证速度结果可比较。

### 2. 每个来源测试三种方式

1、单线程下载完整产品。

2、16 线程下载完整产品。

3、远程窗口读取，只读取 patch 所需波段和像素范围。

窗口读取是正式方案的首选。完整产品下载用于判断服务端吞吐上限和窗口读取是否真的节省流量。

### 3. 必须记录的指标

| 指标 | 记录方法 |
| --- | --- |
| 产品查询时间 | STAC/OData/ASF 查询开始到返回完成 |
| 首字节延迟 | 发起 GET 到收到首批数据 |
| 下载吞吐 | 有效字节除以下载时间，记录 P50/P90 |
| 失败率 | 失败请求数除以总请求数 |
| 重试后成功率 | 自动重试后成功的比例 |
| 请求数量 | GET、HEAD、LIST 数量 |
| 原始下载量 | 云平台实际传输字节 |
| 处理后数据量 | 最终 WebDataset/Zarr shard 字节 |
| 单 patch 成本 | 计算、EBS、请求和传输费用合计 |
| 全国 ETA | 按 62,000 patch 和 8 个季度外推 |

### 4. 通过标准

1、连续 24 小时任务失败率低于 1%。

2、失败任务可以从 manifest 自动恢复，不重复下载已经完成的窗口。

3、境外节点到处理后分片的平均吞吐达到 50 MB/s 以上。

4、120 个 patch 的三源处理结果全部通过坐标和像素掩膜检查。

5、根据真实账单外推后，成本在项目预算范围内。

## 六、10 米版本需要哪些数据

### 1. 核心动态数据

| 数据 | 下载位置 | 使用内容 | 时间 |
| --- | --- | --- | --- |
| Sentinel-2 L2A | Frankfurt AWS / Copernicus Data Space | 12 通道、SCL、AOT、WVP 和元数据 | 2020--2021 |
| Sentinel-1 | Oregon ASF / Earthdata Cloud | VV、VH、轨道、入射角和有效掩膜 | 2020--2021 |
| Landsat-8 C2 L2 | Oregon `usgs-landsat` | 7 通道、QA_PIXEL、QA_RADSAT 和 MTL | 2020--2021 |
| Landsat-9 C2 L2 | Oregon `usgs-landsat` | 仅补充 2021Q4 | 2021Q4 |

### 2. 年度和静态数据

| 数据 | 主要用途 | 来源 |
| --- | --- | --- |
| ESA WorldCover 2020/2021 | 宽类别土地覆盖弱标签 | [WorldCover](https://esa-worldcover.org/en/data-access) |
| Dynamic World | 逐景土地覆盖概率和置信度 | [Dynamic World](https://www.dynamicworld.app/about/index.html) |
| Copernicus DEM GLO-30 | 高程、坡度和坡向 | [Copernicus DEM](https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM) |
| GlobeLand30 / GLC_FCS30D / CLCD | 中国土地利用辅助信息 | 各数据集官方入口 |
| WorldCereal 2021 | 农业和灌溉信息 | [WorldCereal](https://esa-worldcereal.org/en/products/global-maps) |
| 历史 OSM | 道路、建筑、水体和土地利用弱标签 | [ohsome API](https://docs.ohsome.org/ohsome-api/stable/endpoints.html) |
| MODIS NDVI/EVI | 季度植被状态辅助信息 | [NASA MODIS](https://modis.gsfc.nasa.gov/data/dataprod/mod13.php) |

### 3. 10 米数据量口径

62,000 个 patch 的实际陆地面积约为：

```text
62,000 x 1.28 km x 1.28 km = 101,580.8 km2
```

只按窗口读取并保留筛选后的季度观测，核心动态数据、掩膜、标签和索引预计形成 3--5 TB 可训练数据包。

如果把 2020--2021 年完整场景全部回传，容量可能达到 190--220 TB。正式路线必须在境外完成查询、质量筛选和裁窗，避免把完整产品全部传回国内。

## 七、10 米版本怎样预处理

### 1. Sentinel-2

1、使用 L2A 地表反射率，不混用 L1C DN。

2、使用 SCL 去除 no-data、饱和、云影、中高概率云、卷云和雪。

3、云和云影外扩 1--3 个像素，降低薄云边缘残留。

4、20 米和 60 米通道根据物理含义重采样到 10 米，保留原始分辨率字段。重采样不等于增加真实空间细节。

5、按 patch 内有效像素比例、薄云、AOT 和观测日期给场景评分。

6、每季度保留质量最好的 2--4 景，使用加权中位数形成季度输入，同时保存观测数量和季度内日期。

Sentinel-2 L2A 包含 Scene Classification、云影和雪信息：[Sentinel-2 L2A 产品说明](https://sentinels.copernicus.eu/web/sentinel/sentinel-data-access/sentinel-products/collection-1-level-2a)。

### 2. Sentinel-1

1、优先使用已经完成 radiometric terrain correction 的 RTC 产品。

2、如果只能获得 GRD，依次执行轨道更新、热噪声去除、辐射定标、地形校正和正射校正。

3、保存 VV、VH、入射角、轨道方向和有效区域掩膜。

4、统一转为线性功率或 dB，整个项目只能选一种统计空间，不能在不同区域混用。

5、同季度多景使用稳健中位数，避免单景斑点噪声主导结果。

6、S1 重建权重低于光学，不让雷达噪声控制 64 维 embedding。

标准处理链包括定标、热噪声去除、可选斑点滤波、RTC 和正射校正：[Sentinel-1 GRD processing](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S1GRD.html)。

### 3. Landsat

1、使用 Collection 2 Level-2 Surface Reflectance。

2、应用官方比例系数 `reflectance = DN x 0.0000275 - 0.2`。

3、使用 QA_PIXEL 去除填充值、膨胀云、卷云、云、云影和雪。

4、使用 QA_RADSAT 去除饱和像素，并记录 SR_QA_AEROSOL。

5、Landsat 保留 30 米原始信息支持。可以对齐到 10 米网格，但不能将其描述为真实 10 米细节。

官方比例系数和质量波段见 [Landsat Collection 2 Level-2](https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products) 与 [QA bands](https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands)。

### 4. 跨源统一处理

1、每个 patch 固定 CRS、网格原点、`128 x 128` 像素和 affine transform。

2、不同数据源先重投影到同一网格，再进入模型。

3、所有数据保留 source、scene ID、timestamp、scale、offset、nodata 和质量分数。

4、缺失模态使用 availability mask，不使用全零影像冒充有效观测。

5、年度数据只监督年度汇总或稳定属性，不复制为四个季度的变化真值。

6、最终保存为较大的 WebDataset/Zarr shard，避免数百万 `.pt` 小文件。

7、先生成 120 个测试 patch 的逐季度训练输入大图，人工检查云、配准、分类标签和时间归属。

## 八、2 米路线一：生成式 Embedding 升级

### 1. 基本结构

10 米底座先生成季度 embedding：

```text
Z10(q) = 128 x 128 x 64
```

增强模型把每个 10 米像素细化为 `5 x 5` 个 2 米像素，目标为：

```text
Z2(q) = 640 x 640 x 64
```

为保持现有模型单次输出 `128 x 128` 的约束，工程上将 1.28 km patch 划分为 `5 x 5` 个 256 m 子块，每次输出一个 `128 x 128 x 64` 的 2 米子块，最后进行重叠裁边和拼接。

### 2. 两种输入方式

| 方式 | 输入 | 优点 | 问题 |
| --- | --- | --- | --- |
| 纯生成式 | 10 米 embedding | 更新成本最低 | 不能知道 10 米像素内部真实道路和建筑位置，可能生成合理但错误的细节 |
| 高分引导式 | 10 米 embedding + 当年 2/5 米影像 | 2 米边界有真实影像依据 | 每年仍需要购买或获取一次高分覆盖 |

建议把高分引导式作为主方案，纯生成式只作为低成本对照。

### 3. 训练方法

1、在有高分影像的区域训练一个高分 teacher，得到参考 2 米 embedding。

2、student 输入季度 10 米 embedding。高分引导式再输入同年的 2/5 米光学特征。

3、输出使用 residual 结构：上采样后的 10 米向量加上高分细节修正量。

4、将预测的 2 米 embedding 下采样回 10 米，要求与原 10 米 embedding 一致。

5、使用高分光学重建、边界梯度、embedding 蒸馏、方差/协方差和时间一致性损失。

6、生成模型输出 uncertainty。没有高分引导、云覆盖严重或超出训练分布的区域标记为低可信。

7、评测不能只看生成图是否清晰，必须比较建筑、道路、水体、田块边界、检索和变化任务。

### 4. 相关论文

1、[DSen2](https://arxiv.org/abs/1803.04271) 通过降采样真实 Sentinel-2 数据构造监督，学习不同空间分辨率间的超分关系。

2、[Flow matching for Sentinel-2 super-resolution](https://arxiv.org/abs/2605.00367) 使用同日 10 米 Sentinel-2 与 2.5 米 NAIP 配对，完成 4 倍超分，并讨论感知质量和像素真实性之间的取舍。

3、[SEED-SR](https://openaccess.thecvf.com/content/WACV2026/html/Agarwal_Segmentation-Aware_Latent_Diffusion_for_Satellite_Image_Super-Resolution_Enabling_Smallholder_Farm_WACV_2026_paper.html) 不直接在 RGB 像素空间超分，而是在面向分割任务的 latent space 中进行超分，与“Embedding 超分”最接近。

4、[TexADiff](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Remote_Sensing_Image_Super-Resolution_for_Imbalanced_Textures_A_Texture-Aware_Diffusion_CVPR_2026_paper.html) 专门处理遥感纹理不均衡和生成细节幻觉问题。

### 5. 必须说明的限制

只输入 10 米 embedding 时，模型不可能从信息上恢复真实存在但在 10 米输入中不可见的细节。它能生成的是“统计上可能的 2 米 embedding”，不能直接作为真实 2 米观测产品宣传。

只有使用同位置高分影像引导，或者通过独立 2 米标签验证后，才能说明输出具有真实 2 米制图能力。

### 6. 后续更新成本

纯生成式方案每季度只需要更新免费的 S2、S1 和 Landsat，然后重新运行增强模型。主要成本是计算和 2 米 embedding 存储。

高分引导式方案可以每季度更新 10 米动态 embedding，每年购买一次 2/5 米高分影像作为空间底图。它不能表达当年高分底图之后新增的小建筑边界，但比每季度采购全国高分影像便宜得多。

## 九、2 米路线二：直接使用高分辨率影像

### 1. 数据量

四通道 uint16 影像的理论净数据量如下：

| 范围 | 单次 2 米覆盖 | 8 个季度净数据 | 考虑多景选云和工程副本 |
| --- | ---: | ---: | ---: |
| 62,000 个训练 patch | 约 0.20 TB | 约 1.63 TB | 约 5--10 TB |
| 全国 960 万 km2 | 约 19.2 TB | 约 153.6 TB | 约 0.5--1.0 PB |

上表只是高分影像，不包含 2 米 embedding。8 期全国 64 维 int8 embedding 另需约 1.229 PB 净数据。

### 2. 公开价格只能用于初步测算

会议中提到的存档数据参考价约为 10 元/km2。公开经销商给出的国产 1--3 米历史影像区间约为 1--30 元/km2，高分辨率处理产品也有 50--100 元/km2 的报价。最终价格受年份、最小起订面积、整景购买、波段、处理级别和发布许可影响。

| 采购范围 | 10 元/km2 | 30 元/km2 | 50 元/km2 |
| --- | ---: | ---: | ---: |
| 62,000 个训练 patch，单次覆盖 | 约 102 万元 | 约 305 万元 | 约 508 万元 |
| 62,000 个训练 patch，8 个季度 | 约 813 万元 | 约 2,438 万元 | 约 4,063 万元 |
| 全国单次覆盖 | 约 9,600 万元 | 约 2.88 亿元 | 约 4.8 亿元 |
| 全国 8 个季度 | 约 7.68 亿元 | 约 23.04 亿元 | 约 38.4 亿元 |

这些数字没有考虑大客户折扣，也没有考虑很多供应商不允许按 1.6384 km2 小窗口购买。会议上只能作为数量级说明，不能作为采购预算。

公开参考：[国产影像价格范围](https://www.o-map.cn/)、[Planet 定价页面](https://www.planet.com/pricing/?tab=platform)。

### 3. 必须向供应商询价的内容

1、2020 和 2021 年全国 2 米、3 米、5 米影像分别有多少覆盖。

2、是否能按 62,000 个 `1.28 km x 1.28 km` AOI 购买，还是必须按整景或最小面积购买。

3、存档影像和新拍影像的单价。

4、每个季度形成一次无云覆盖需要多少景，实际覆盖率是多少。

5、全色、多光谱波段数、位深、产品级别和正射精度。

6、是否允许用于模型训练。

7、是否允许发布模型权重、下游结果和衍生 embedding。

8、后续每季度或每年更新的价格。

9、是否提供批量 API、对象存储或离线硬盘交付。

10、采购后的数据能否长期保存和内部共享。

### 4. 后续更新成本

直接高分路线不是一次性费用。若产品每季度更新一次，就需要持续购买新增高分覆盖、处理云和跨传感器差异，并重新导出 2 米 embedding。

按全国每次覆盖 10 元/km2 的会议参考值，每增加一个季度的理论影像费就是约 9,600 万元。实际合同可能有较大折扣，但如果没有长期框架协议，该路线很难持续运营。

因此直接高分路线更适合以下两种情况：

1、数据中心已经拥有全国年度 2 米影像和训练使用权。

2、只对重点城市、海岸、农业区和业务区生成 2 米产品，而不是每季度覆盖全国。

## 十、两条 2 米路线的会议比较

| 项目 | 生成式升级 | 高分引导式升级 | 直接高分训练 |
| --- | --- | --- | --- |
| 主要输入 | 10 米 embedding | 10 米 embedding + 年度高分 | 每季度高分影像 |
| 是否是真实观测细节 | 否 | 部分是 | 是 |
| 首次数据采购 | 最低 | 中等 | 最高 |
| 后续季度更新 | 低 | 中低 | 高 |
| 幻觉风险 | 高 | 中 | 低 |
| 全国运行可持续性 | 高 | 较高 | 取决于长期采购预算 |
| 建议定位 | 研究对照 | 首选高分增强路线 | 有稳定数据和预算时采用 |

会前建议形成的决策为：先完成全国 10 米底座，同时在四类区域开展“纯生成式、高分引导式、直接高分 teacher”三组小实验。没有区域实验证据前，不承诺全国生成式产品等同于真实 2 米观测。

## 十一、未来两天执行清单

### 第一天：账号、数据清单和下载节点

1、建立数据清单表，字段包括 source、年份、季度、产品级别、分辨率、波段、覆盖率、本地路径、下载入口、授权、预计容量和负责人。

2、从全国样本中冻结 120 个下载测试 patch，并提交 patch ID 和坐标文件。

3、申请或确认 AWS 国际站、Earthdata Login 和 Copernicus Data Space 账号。

4、在 AWS 设置预算告警、IAM role 和 SSH key。

5、创建 Frankfurt 和 Oregon 两台试验服务器。

6、Frankfurt 完成 Sentinel-2 L2A 匿名桶和 Copernicus S3 连通测试。

7、Oregon 完成 ASF Sentinel-1 检索、临时 S3 credential 和 Landsat Requester Pays 连通测试。

8、联系济南、怀柔和数据中心，核实本地 2020/2021 数据的真实文件清单和字节数。

### 第二天：速度、预处理和成本

1、对 120 个 patch 跑三源查询和下载测试。

2、比较单线程、16 线程和远程窗口读取。

3、生成 S2 云掩膜、S1 RTC、Landsat QA 和跨源配准结果。

4、随机选择 12 个 patch 生成季度输入大图，人工检查云、时间和配准。

5、根据真实吞吐外推 62,000 patch 的下载 ETA。

6、根据 AWS 账单外推完整数据准备成本。

7、向至少两家国产高分供应商和 Planet 询价。

8、完成 1 亿参数模型单卡 dry-run；如服务器可用，再完成 6 卡 1,000-step 基准。

9、整理 2 米三组小实验的数据需求和样本区。

### 第三天会前：形成汇报材料

1、提交一张数据 availability matrix，明确每个源哪些季度有数据。

2、提交一张境外节点速度和费用对比表。

3、提交 12 个 patch 的预处理可视化。

4、提交 10 米版总数据量、下载时间、训练时间和存储量。

5、提交 2 米生成式、高分引导式和直接高分三路线对比表。

6、提交供应商询价状态，未获得正式报价的项目标记为“待书面报价”。

7、提交 1 亿参数模型实测参数量、显存和 16 卡预计训练时间。

8、给出明确建议：立即做什么、需要领导协调什么、暂时不承诺什么。

## 十二、需要你推动的外部事项

1、提供或批准 AWS 国际站付费账号。没有付费账号无法测试 Landsat Requester Pays，也无法创建 Frankfurt 和 Oregon EC2。

2、确认试验费用上限。建议先批准 500 美元以内的两地测试预算，超过后重新申请。

3、联系数据中心，要求提供 2020/2021 年真实文件清单，不只提供口头容量。

4、联系国产高分供应商、长光或现有合作单位，取得书面报价和授权范围。

5、协调济南、郑州和现有 16 张 910B 的交付时间、网络和存储权限。

6、确认会议上是否将“全国真实 2 米”作为正式交付目标，还是将其调整为“全国 10 米底座加重点区域高分增强”。两者成本相差很大。

## 十三、我可以继续执行的工作

1、生成 101.15 M 参数的自包含全国实验配置。

2、编写 S1、S2、Landsat 统一 STAC/ASF 查询与下载 benchmark 脚本。

3、记录吞吐、失败率、字节数和费用，自动输出比较表。

4、复用全国 patch 清单生成 120 个固定测试位置。

5、实现三源预处理和训练输入 QA 大图。

6、设计 10 米 embedding 到 2 米 embedding 的小规模生成式实验。

7、把会前结果整理成 Markdown、DOCX 或汇报页面。

上述执行需要 AWS、Earthdata 和 Copernicus 凭证通过服务器 secret 提供，不能把长期密钥写入仓库。
