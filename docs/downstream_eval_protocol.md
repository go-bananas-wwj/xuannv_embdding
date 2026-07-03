# 下游测评统一流程

本文档定义冻结 embedding 的下游测评流程，用来公平比较不同 Baseline。

## 1. 测评目的

下游测评只回答一个问题：固定一个 embedding 模型以后，不更新 embedding 主干，只训练一个很小的下游头，能不能稳定完成建筑、道路、水体、施工地等任务。

因此，每次比较 Baseline 时，必须保证：

- 使用同一批 patch；
- 使用同一份 `split_5fold.json`；
- 使用同一月份或同一双时相组合；
- 使用同一种下游头配置；
- 使用同一套阈值选择和指标汇总规则。

## 2. 输入数据

先把 checkpoint 导出成每个 patch 的 embedding map，再训练下游头。

单月语义任务：

- 例如 202604；
- `training.months: [202604]`；
- `temporal_mode: single`；
- 输入维度是 embedding 原始维度，例如 64。

双时相变化任务：

- 例如 202512 和 202605；
- `training.months: [202512, 202605]`；
- `temporal_mode: concat_diff`；
- 输入是 `t1 embedding + t2 embedding + abs(t2 - t1)`，如果原始维度是 64，则下游输入维度是 192。

单月任务和双时相任务不能直接混为一个结论，只能在各自任务内比较。

## 3. 任务集合

海淀区当前正式任务以已经有固定 split 的任务为准：

- 建筑提取：`haidian_building_osm`
- 道路提取：`haidian_road_osm`
- 水体提取：`haidian_water_osm`
- 施工地：`construction`，目前可作为补充任务，不作为默认主结论

如果新增 OSM 类别，必须先生成并检查 `split_5fold.json`，再进入正式测评。没有固定 split 的任务只能算探索性结果。

## 4. 下游头训练

当前默认使用二分类分割头：

- embedding 主干冻结，只读取预导出的 `*_embedding_map.pt`；
- 每个任务单独训练一个二分类 head；
- 当前默认 head 是 `unet`；
- 当前默认 loss 是 `bce_dice_tversky`；
- 训练集只做同步水平/垂直翻转；
- 早停指标默认是验证集 `f1_best`。

注意：`unet` head 能看应用上限，但不是最纯粹的“简单探针”。为了判断 embedding 本身是不是强，正式报告后续应同时增加一组 `linear/mlp probe`，作为表示能力的主证据；`unet` 结果作为应用上限。

## 5. 数据划分

正式测评必须使用已有的 `split_5fold.json`：

- 训练集：训练下游头；
- 验证集：选最佳 epoch 和最佳阈值；
- 测试集：只做最终汇报。

快速开发可以只跑 `fold0`，但不能把 `fold0` 当最终结论。正式对比必须跑完整 5 folds，并报告均值。

脚本默认禁止自动创建 split。只有调试时才允许显式传 `--allow-create-split`。

## 6. 阈值和指标

二分类分割不能默认只看 0.5 阈值。正式规则如下：

1. 每个任务、每个 fold 在验证集 PR 曲线上找 F1 最优阈值；
2. 把这个验证集阈值固定到测试集；
3. 正式主指标使用测试集 `f1_at_threshold`、`miou`、`ap/auprc`、`auc_roc`；
4. `f1_0.5` 只作为校准诊断；
5. 测试集 `f1_best` 是 oracle 诊断项，不作为正式主指标。

每个任务都要保存：

- 概率预测图；
- 阈值化预测图，前景红色、背景白色；
- GT 标签图；
- 关键 patch 的横向拼图。

## 7. AEF 对比规则

只有在 AEF 使用同一任务、同一 split、同一月份或同一双时相输入时，才允许做直接指标对比。

当前 AEF benchmark 对海淀 OSM 建筑、道路、水体任务不完整，所以这些任务不能声称已经和 AEF 做了严格直接对比。可以报告“AEF 缺少同任务结果”，但不能强行横比。

## 8. 当前已知偏差

最近若干次结果主要是快速开发结果：

- 多数只跑了 `fold0`；
- 当前 head 是 `unet`，不是简单 `linear/mlp probe`；
- 海淀 OSM 建筑、道路、水体缺少可直接比较的 AEF 结果；
- 旧版可视化流程默认使用 `202512/202605`，单月 202604 任务可能会误配月份；现在 `post_training_eval.py` 已改为从配置自动推断，也可以显式传 `--visualization-months`；
- `test f1_best` 不能作为正式结论，只能看 `f1_at_threshold`。

## 9. 标准命令

快速开发，单月 202604：

```bash
PYTHONPATH=/root/workspace/xuannv/src:/root/workspace/xuannv/downstreams \
python scripts/scale/post_training_eval.py \
  --embedding-root /data/xuannv_embedding/embeddings/v2_202512_202605/EXPORT_DIR \
  --run-name RUN_NAME \
  --config downstreams/configs/v2_acceptance_quick_single_202604.yaml \
  --tasks haidian_building_osm haidian_road_osm haidian_water_osm \
  --fold 0 \
  --skip-v1-comparison
```

正式 5-fold：

```bash
PYTHONPATH=/root/workspace/xuannv/src:/root/workspace/xuannv/downstreams \
python scripts/scale/post_training_eval.py \
  --embedding-root /data/xuannv_embedding/embeddings/v2_202512_202605/EXPORT_DIR \
  --run-name RUN_NAME \
  --config downstreams/configs/v2_acceptance_quick_single_202604.yaml \
  --tasks haidian_building_osm haidian_road_osm haidian_water_osm \
  --all-folds \
  --skip-v1-comparison
```

双时相变化任务应把 config 换成 `v2_acceptance_quick_concat_diff.yaml`，并确保 embedding root 里有两个对应月份。

## 10. 后续补强

为了更符合“好的 embedding 用简单头也应该好”的判断，下一步建议增加两套固定配置：

- `linear_probe`：1x1 conv 或像素级线性分类，作为最严格表示测评；
- `mlp_probe`：轻量 2-3 层 MLP/1x1 conv，作为 few-shot 友好测评；
- `unet_head`：保留当前配置，作为应用上限测评。

三套 head 共用同一 split、同一阈值规则、同一可视化流程。
