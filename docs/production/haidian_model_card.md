# 海淀区地理嵌入模型卡

模型名：`haidian-embedding-v1`

候选来源：`v2_p10c_haidian_202512_202605_osm_semantic_hardneg_20260704 / epoch_800.pt`

## 用途

为北京市海淀区生成月度 64 维地理 embedding，用于少样本或浅层头下游任务，例如建筑提取、道路提取、水体提取和后续土地利用分析。

## 输入

- Sentinel-2
- Sentinel-1
- Landsat
- 高分辨率光学
- 高分辨率 SAR
- 月份编码和空间位置

## 输出

- 每个 patch 每个月一张 `64×128×128` embedding map。
- 每个 patch 每个月一个 scene-level embedding。

## 训练策略

- 多源重建：约束 embedding 保留真实影像信息。
- 高分辨率重建：增强边界和纹理。
- OSM 弱语义：引入建筑、道路、水体等区域弱标签。
- 困难重建：随机遮挡部分输入源和空间块。

## 当前指标

202604 月 fold-0 quick eval：

| 任务 | F1_best |
|---|---:|
| 建筑 | 0.4832 |
| 道路 | 0.5232 |
| 水体 | 0.6306 |
| 平均 | 0.5457 |

## 权重

权重保存在 ModelScope：

`artifacts/haidian-embedding-v1/checkpoints/haidian_embedding_v1_p10c_epoch800.pt`

SHA256：

`69dfd81c898544413a747f5c7304cc9210ad1cf420ce724864b8bd7deb6ed790`

## 限制

该版本针对海淀区生产场景整理，不应直接视为全国或跨城市通用模型。
