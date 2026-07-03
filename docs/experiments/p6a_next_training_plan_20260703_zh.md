# P6A 下一轮训练计划

日期：2026-07-03

## 现在已经完成的升级

### 1. 数据时间范围收窄

本轮只做海淀区，时间范围固定为 2025-12 到 2026-05，共 6 个月。

这样做的原因是这 6 个月多源数据更完整，适合先把海淀区 embedding 做精，而不是继续被跨区域和缺数据问题干扰。

### 2. 高分辨率数据补齐

高分光学中 2026-04-30 这景已经重新按日期切片，并映射到 2026-05 使用。

训练中每个月有对应的高分光学监督；高分 SAR 仍按实际可用月份进入，缺少月份不会强制补假标签。

### 3. 像素级 mask 接入训练

训练数据现在会读取每个 source 的 `*_mask.tif`。

S2、Landsat、多景月度平均时，不再简单平均所有影像，而是按像素 mask 做有效像素加权平均。重建 loss 也只在有效像素上计算。

### 4. 质量过滤后的 clean manifest

S2 和 Landsat 同月多景影像已经做质量筛选。训练视角上剔除差影像，只保留质量较好的影像参与月度合成。

当前训练 manifest：

`/data/xuannv_embedding/processed/haidian/manifest_p6a_202512_202605_pixelmask_clean_osm_landcover.json`

### 5. 移除明显错误的 OSM 标签

`osm_activity_poi` 已移除。它来自 POI/活动点缓冲区，容易形成大圆，边界不可靠，会污染背景。

### 6. OSM 替换 WorldCover

原始 WorldCover 太粗，很多城市 patch 被整块标成 built-up，不能区分建筑、道路、绿地、水体等细节。

现在已生成 `osm_landcover`，作为类似 WorldCover 的单通道土地/地物分类图：

| ID | 类别 | 来源 |
| --- | --- | --- |
| 0 | background | OSM 未覆盖区域 |
| 1 | residential | `osm_residential` |
| 2 | commercial | `osm_commercial` |
| 3 | industrial | `osm_industrial` |
| 4 | agriculture | `osm_agriculture` |
| 5 | green | `osm_green` |
| 6 | recreation | `osm_playground` |
| 7 | construction | `osm_construction` |
| 8 | water | `osm_water` |
| 9 | building | `osm_building` |
| 10 | transport | major road + minor road + path/walk + rail |

配置中仍复用 `worldcover` target head，但实际读取的是 `osm_landcover`。分类通道数已从 9 改为 11。

### 7. OSM 弱标签作为辅助语义 probe

除了合并后的 `osm_landcover`，训练还保留多个细粒度 OSM 弱标签，作为 semantic probe：

- building
- major road
- minor road
- rail
- water
- green
- agriculture
- residential
- commercial
- industrial
- construction
- path/walk
- playground

这些弱标签不是强制最终分类，而是在 embedding 上接一个简单 probe，轻量地推动 embedding 学会地物语义。

当前权重：`semantic_probe_weight = 0.05`，前 20 epoch warmup。

### 8. 困难重建

训练中已开启输入遮挡：

- S2 dropout 0.1
- S1 dropout 0.1
- Landsat dropout 0.2
- 每个样本最多看 4 个月
- 空间 block mask 概率 0.5

这会逼模型不能只记住输入，而要学会更稳的跨源、跨月语义表达。

## 当前配置摘要

配置文件：

`configs/v2_p6a_haidian_202512_202605_pixelmask_full_20260703.yaml`

核心参数：

- embed dim：64
- 月份：2025-12 到 2026-05
- patch 数：320
- 训练轮数：300 epoch
- batch size：单卡 3
- 6 卡总 batch：18
- gradient accumulation：2
- 有效 batch：约 36
- AMP：开启
- gradient checkpointing：开启
- NPU：0-5

## 下一步训练计划

### Step 0：训练前最后检查

先不直接跑满 300 epoch。训练前先做 4 个检查：

1. 配置读取检查。
2. dataloader 抽样检查。
3. `prepare_batch` 检查，确认 `worldcover` target 实际是 OSM landcover，类别范围 0-10。
4. post-training eval 脚本检查，确认下游测评流程中的脚本都存在。

通过条件：

- batch 能正常构造；
- `worldcover` target shape 为 `[B, 6, 128, 128]`；
- target max 小于 11；
- `worldcover` 不进入模型输入，只作为目标；
- 下游评估脚本链路没有缺失文件。

### Step 1：短跑 smoke 训练

先跑 10-20 epoch，用来确认训练过程没问题。

观察项：

- loss 是否正常下降；
- `recon_worldcover` 是否有效；
- `semantic_probe_*_loss` 是否正常；
- 没有 NPU OOM；
- 没有 NaN/Inf；
- checkpoint 能保存；
- wandb 能记录。

如果 smoke 有异常，立即停，不进入长训。

### Step 2：正式 6 卡长训

通过 smoke 后，使用 NPU 0-5 进入正式训练。

建议命令：

```bash
tmux new -s p6a_osm_landcover
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5
export WANDB_API_KEY=***
bash scripts/train/launch_6card.sh configs/v2_p6a_haidian_202512_202605_pixelmask_full_20260703.yaml
```

训练目标：

- 先跑满 300 epoch；
- 中间每 10 epoch eval；
- 每 20 epoch 保存 checkpoint；
- 重点观察 best checkpoint 是否持续变好。

### Step 3：训练中监控

每隔一段时间检查：

- 当前 epoch / 总 epoch；
- ETA；
- total loss；
- S2/S1/Landsat 重建 loss；
- highres optical / SAR 重建 loss；
- OSM landcover 分类 loss；
- semantic probe loss；
- GPU/NPU 显存和进程状态；
- 最近 checkpoint 是否正常保存。

异常处理：

- loss NaN：停训，保留日志和最近 checkpoint；
- OOM：降低 batch size 或关闭部分高分输入；
- worldcover/osm_landcover loss 不下降：检查标签类别分布和 loss 权重；
- semantic probe loss 很高且主 loss 被拖住：降低 `semantic_probe_weight`。

### Step 4：训练后导出 embedding

使用 best checkpoint 导出 2025-12 到 2026-05 的海淀 embedding。

导出后必须检查：

- embedding 是否 128x128；
- 每个月是否都有输出；
- PCA 可视化是否比旧版本更连续；
- 云雾区域是否仍然产生突兀颜色斑点；
- 建筑、道路、绿地、水体是否在 PCA 上有更一致的语义分布。

### Step 5：统一下游评估

下游任务继续使用 OSM 弱标签和已有人工标签，不用合并后的 `transport` 直接替代全部道路任务。

建议任务：

- 建筑物提取：`osm_building` / `building_osm`
- 道路提取：`osm_major_road + osm_minor_road + osm_path_walk`
- 交通线网：道路 + `osm_rail`
- 水体提取：`osm_water`
- 施工地：`osm_construction` 与已有 construction 标注交叉检查
- 土地利用分类：使用 `osm_landcover`

每个任务至少跑：

- linear head；
- MLP head；
- 最佳阈值搜索；
- 可视化预测概率和二值结果。

### Step 6：对比验收

对比对象：

- 旧 P6A / P5H 模型；
- AEF embedding；
- 当前 OSM landcover 新模型。

验收标准：

1. PCA 可视化更连续，少云雾噪声斑点。
2. 建筑、道路、水体、绿地在 embedding 上更可分。
3. linear/MLP 下游头效果提升，而不是只靠复杂 head。
4. 道路和建筑不再大面积预测背景。
5. 至少在建筑、道路、水体三个任务上接近或超过 AEF。

## 我建议的执行顺序

1. 先做 Step 0 检查。
2. 如果通过，跑 10-20 epoch smoke。
3. smoke 正常后，启动 300 epoch 6 卡正式训练。
4. 训练完成后自动导出 embedding。
5. 跑统一下游评估和 AEF 对比。
6. 汇总指标、可视化和失败样例。

当前不建议再改模型结构。先把这版“clean data + OSM landcover + OSM semantic probe + hard reconstruction”跑出可信 baseline，再决定是否升级网络。
