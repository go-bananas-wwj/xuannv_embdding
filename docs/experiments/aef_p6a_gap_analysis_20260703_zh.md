# AEF 与 P6A 嵌入差距核查

日期：2026-07-03

## 结论

当前 P6A 不是简单的坐标未对齐问题。更主要的问题是：64 维表达发生了低维塌缩，空间边界被 1/2 分辨率主干和上采样抹平，高分辨率影像没有真正注入最终 embedding。

## 64 维是否都被用到

统计对象：

- AEF：`/data/xuannv_embedding/embeddings/aef_official_2025_annual/haidian`
- P6A：`/data/xuannv_embedding/embeddings/v2_202512_202605/20260703_v2_p6a_haidian_202512_202605_pixelmask_full_20260703_best_p6a_osm_landcover_best/haidian`
- 采样：120 个 patch，每个 patch 512 个像素，共 61,440 个像素。

| 模型 | 死维数量 | 90% 方差维数 | 95% 方差维数 | 99% 方差维数 | 熵有效秩 | Top3 方差占比 | Top8 方差占比 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AEF 2025 annual | 0 | 15 | 22 | 42 | 12.60 | 55.85% | 78.75% |
| P6A 202604 | 0 | 6 | 8 | 25 | 5.11 | 81.47% | 95.66% |
| P6A 202512 | 0 | 5 | 7 | 24 | 4.78 | 83.13% | 96.16% |

解释：

- AEF 的 64 个维度不是 64 个独立语义类别，而是 64D 球面坐标；官方说明所有 band 应一起使用，单 band 不可独立解释。
- P6A 没有死维，但主要变化被压到前 7-8 个主方向里；这说明 embedding 空间不够展开，下游任务看到的语义分布更单薄。

## AEF 公开资料要点

公开资料来源：

- Google Earth Engine Satellite Embedding V1 Annual dataset
- AlphaEarth Foundations paper, arXiv:2507.22291

关键点：

- AEF 公开产品是 10m、年度、64 维、单位长度 embedding。
- 64 个 band 是一个整体坐标，不能把单个 band 当成具体类别。
- 年份之间的 embedding 空间一致，可以用点积/角度做变化检测。
- AEF 论文中 STP 也有 1/16、1/8、1/2 多分辨率路径，但强调保持 spatial precision，并最终生产 10m embedding fields。
- AEF 使用 teacher/student/text alignment 框架、vMF bottleneck、batch uniformity。
- AEF 训练数据规模远大于本项目：论文写到超过 30 亿观测、九类 gridded 数据源和一个文本源，覆盖约 1.1% 陆地。
- AEF 的数据源包括 Sentinel-2、Landsat、Sentinel-1、PALSAR、DEM、GEDI、ERA5-Land、GRACE、NLCD、Wikipedia/GBIF 等。所有 raster 数据统一到 10m UTM 网格。
- AEF 对云不是简单输入合成，而是在训练损失中用 valid data mask 排除无效像素。

## 当前 P6A 与 AEF 的主要差异

1. 空间分辨率路径不同

当前代码的 `STPEncoder` 固定 `PRECISION_SCALE = 2`，输出 `H//2, W//2` 特征；随后通过 `EmbeddingUpsampleHead` 上采样回 128x128。文件 shape 是 128x128，但细节信息在主干里已经降到 64x64。

相关代码：

- `src/xuannv_embedding/models/blocks.py`
- `src/xuannv_embedding/models/model.py`

2. 高分辨率影像没有注入 embedding

当前配置：

```yaml
model:
  stp:
    highres_fusion_to_embedding: false
```

所以高分影像主要作为重建目标，不会直接进入最终 embedding。模型很难从 10m/30m 月度输入里恢复小岛、窄路、岸线等细节。

3. 高分编码器本身也会强下采样

`NativeResolutionHighResEncoder` 用三层 stride=2 卷积，然后 `AdaptiveAvgPool2d` 到目标网格。即使打开高分融合，这个版本也可能过度平均 3m 影像细节。

4. 训练规模和数据多样性差异极大

P6A 只训练海淀 320 个 patch、6 个月窗口；AEF 是全球级、多年、多源、多任务训练。AEF 可以学习稳定地物边界和跨传感器鲁棒性，P6A 更容易被局部数据、云雾、季节差异牵着走。

5. 训练目标不同

P6A 有 OSM 弱语义和重建，但没有 AEF 的完整 teacher/student/text 对齐框架。OSM 标签有噪声，且语义 supervision 权重较低，更像辅助正则。

## 幻觉/偏差纠正

- “P6A 已经是 128x128，所以就是 10m 细节”：不准确。保存 shape 是 128x128，但主干有效表达先降到 64x64，再上采样。
- “高分辨率重建等于高分辨率注入 embedding”：不准确。当前 `highres_fusion_to_embedding: false`，高分不进入最终 embedding。
- “AEF 64 维每一维都有可解释语义”：不准确。官方说明单 band 不独立可解释，必须整体使用。
- “只要重建好，变化检测自然好”：不完全。若重建目标被云雾、季节或模糊平均主导，embedding 会学到影像外观而不是稳定地物语义。
- “官方训练代码已经公开可逐行复现”：目前未确认。公开的是论文、数据集和教程；GitHub 上主要是第三方复现或使用示例。

## 下一步建议

P0：修成真正 128x128 embedding 主干。

- 增加配置项让 precision path 可保持 1x，即 `PRECISION_SCALE = 1` 或新增 high-res precision branch。
- 不能只靠最后上采样补 128x128。

P0：让高分影像进入 embedding。

- 打开 `highres_fusion_to_embedding` 做 smoke test。
- 但要重写高分 encoder，避免三层 stride=2 + adaptive average 过度抹平。
- 推荐用轻量 CNN/FPN，把 3m 高分特征先提边界，再投到 10m embedding 网格。

P1：加强 embedding 空间展开。

- 提高 batch uniformity / hyperspherical uniformity 的有效性。
- 监控 PCA 有效秩、Top8 方差占比、99% 方差维数，避免 64 维塌缩到 7-8 个方向。

P1：增加边界保持目标。

- 从 OSM water/building/road 生成边界带。
- 加轻量 boundary contrast loss：同类内部相似，边界两侧 embedding 拉开。

P1：学习 AEF 的数据处理习惯。

- 数据全部统一到 10m UTM 网格。
- 对云/无效像素只在 loss mask 中排除，不把云当成正常重建目标。
- 训练中记录每个 source 的 valid mask 覆盖率。

