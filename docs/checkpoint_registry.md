# Checkpoint 权重清单

> 最后更新：2026-06-26 UTC，存储清理后。

本文档记录需要长期保护的模型权重。`embeddings/` 和 `cache/` 不列为核心资产，
因为它们可以通过 checkpoint、输入数据、配置和脚本重新生成。

## 存储策略

- 有意义的训练实验保留 `best.pt`。
- 常规 `epoch_*.pt` 中间 checkpoint 默认不长期保存，除非明确需要断点恢复或对照实验。
- 原始输入和预处理输入保留在 `/data/xuannv_embedding/raw` 与 `/data/xuannv_embedding/processed`。
- `/data/xuannv_embedding/embeddings` 视为可重建的推理产物。
- `/data/xuannv_embedding/cache` 视为可重建的预处理缓存。
- 后续删除任何新的数据目录或 checkpoint 前，必须先获得明确确认。

## 当前保护权重

| 实验 | 保护的 checkpoint | 大小 | 修改时间 | 说明 |
|------------|----------------------|------|---------------|-------|
| `harbin_128_v1` | `/data/xuannv_embedding/outputs/harbin_128_v1/best.pt` | 1.8G | `2026-06-20 08:07 UTC` | 哈尔滨 Stage 1 模型。 |
| `harbin_128_stage2_v1_20260620_1628` | `/data/xuannv_embedding/outputs/harbin_128_stage2_v1_20260620_1628/best.pt` | 1.8G | `2026-06-20 16:28 UTC` | 哈尔滨 Stage 2 模型。 |
| `v1.1_distill_quick_harbin` | `/data/xuannv_embedding/outputs/v1.1_distill_quick_harbin/best.pt` | 1.8G | `2026-06-21 16:16 UTC` | v1.1 quick distillation。 |
| `v1.1_distill_long_smoke` | `/data/xuannv_embedding/outputs/v1.1_distill_long_smoke/best.pt` | 1.8G | `2026-06-21 20:27 UTC` | smoke 运行；v1.1 版本选型完成前暂留。 |
| `v1.1_distill_long_stable_50ep_fast_smoke` | `/data/xuannv_embedding/outputs/v1.1_distill_long_stable_50ep_fast_smoke/best.pt` | 1.8G | `2026-06-21 20:58 UTC` | smoke 运行；v1.1 版本选型完成前暂留。 |
| `v1.1_distill_long_stable_10ep` | `/data/xuannv_embedding/outputs/v1.1_distill_long_stable_10ep/best.pt` | 1.8G | `2026-06-21 23:39 UTC` | stable 10 epoch。 |
| `v1.1_distill_long_stable_50ep` | `/data/xuannv_embedding/outputs/v1.1_distill_long_stable_50ep/best.pt` | 1.8G | `2026-06-22 01:56 UTC` | stable 50 epoch。 |
| `v1.1_distill_long_stage2` | `/data/xuannv_embedding/outputs/v1.1_distill_long_stage2/best.pt` | 1.8G | `2026-06-22 05:09 UTC` | v1.1 stage2。 |
| `v1.1_distill_long_stage2_v2` | `/data/xuannv_embedding/outputs/v1.1_distill_long_stage2_v2/best.pt` | 1.8G | `2026-06-22 08:33 UTC` | v1.1 stage2 v2。 |
| `v1.1_distill_long_stage2_rkd` | `/data/xuannv_embedding/outputs/v1.1_distill_long_stage2_rkd/best.pt` | 1.8G | `2026-06-22 12:54 UTC` | v1.1 stage2 RKD。 |

## 生产版定版（2026-07-12）

**海淀区生产版 = P10C epoch_800**（`v2_p10c_haidian_202512_202605_osm_semantic_hardneg_20260704/epoch_800.pt`）

- 稳定别名：`/data/xuannv_embedding/outputs/production/haidian_production_v1.pt`
- 配置：`configs/v2_p10c_haidian_202512_202605_osm_semantic_hardneg_20260704.yaml`
- 选型依据（同协议横评 P10C / P11 / P12 / P13 / P14A / V4A）：
  - few-shot 6 任务 × 3 shots 全面最优（P14A 0/18 胜、V4A 1/18 胜）；
  - 向量检索三任务最优（building 0.3546 / road 0.4492 / water 0.5858）；
  - conv3x3 building 0.4842 / road F1 0.5218 / water 0.6237 最优；
  - 唯一短板 construction（0.5495/0.5435，V4A 0.5988/0.6105 更高），
    但 V4A 其余指标全面退化，不适合生产。
- 候选升级：V4B（蒸馏衰减 + 原型对比加强）训练中，若全面达标再升级生产版。

## 稳定别名

| 别名 | 指向 | 用途 |
|-------|--------|---------|
| `/data/xuannv_embedding/outputs/production/haidian_production_v1.pt` | `v2_p10c_.../epoch_800.pt` | 海淀区生产版稳定路径。 |
| `/data/xuannv_embedding/outputs/harbin_128_stage2_v1` | `harbin_128_stage2_v1_20260620_1628` | 供脚本和文档引用 Stage 2 哈尔滨模型的稳定路径。 |

## 2026-06-26 清理记录

- 删除 `/data/xuannv_embedding/embeddings` 下的生成式 embedding 目录，保留空目录占位。
- 删除 `/data/xuannv_embedding/outputs` 下的中间 `epoch_*.pt`，保留上表所有 `best.pt`。
- 删除 `/data/xuannv_embedding/cache` 下的预处理缓存目录，保留空目录占位。
- `/data/xuannv_embedding` 当前约占用 262G。
- `/data` 当前约剩余 1.9T。

## 重新生成 Embedding

只有在下游评估或报告确实需要时，才重新生成 embedding。示例命令：

```bash
python downstreams/scripts/precompute_embeddings.py \
  --config configs/haidian_128_stage2_infer_lowres.yaml \
  --checkpoint /data/xuannv_embedding/outputs/harbin_128_stage2_v1/best.pt \
  --regions haidian \
  --output-root /data/xuannv_embedding/embeddings \
  --suffix stage2_v1
```

重新生成 embedding 时，需要保留输出目录中的 `meta.json`，确保配置路径、checkpoint
路径、git commit 和命令行参数都可追溯。
