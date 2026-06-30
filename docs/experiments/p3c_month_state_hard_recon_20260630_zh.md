# P3C 月度状态嵌入困难重建实验

生成时间：2026-06-30

## 目标

P3C 用来修正 P3A/P3B 的核心问题：导出的月度 embedding 不应被跨月 attention 和静态高分辨率特征抹平。每个月的主 embedding 应主要表达该月 S2/S1/Landsat 观测到的地表状态，从而让变化检测可以通过两个时相 embedding 的差异自然体现出来。

## 相对 P3B 的关键变化

- `stp.time_attention_mode: none`：关闭 STP 时间分支里的跨月自注意力，保留逐月时间编码和逐 token MLP。
- `stp.highres_fusion_to_embedding: false`：高分辨率光学/SAR 仍作为重建目标，但不再复制融合进每个月主 embedding。
- `epochs: 600`：训练轮次加大，接近 P3A 8 小时长训尺度。
- `source_dropout_probs` 提升到低分源 0.2、高分源 0.5：让模型在缺源情况下仍能重建，形成困难重建。
- `temporal_contrast_weight: 0.08`：加强由 S2/S1/Landsat 影像差异产生的弱时序约束。
- `semantic_probe_weight: 0.025`：保留 OSM 弱语义，但降低权重，避免模型过度学习静态地图分布。

## 训练入口

配置文件：

```bash
configs/v2_p3c_month_state_hard_recon_20260630.yaml
```

推荐从 P3A best 初始化：

```bash
bash scripts/train/launch_6card.sh \
  configs/v2_p3c_month_state_hard_recon_20260630.yaml \
  --init-from /data/xuannv_embedding/outputs/v2_p3a_osm_weak_semantic_8h_20260629/best.pt
```

## 预期观察

- 如果 P3C 有效，变化类任务中 `abs(emb_t2 - emb_t1)` 应更明显，`building_change` 和 `farm_change` 的 F1/AP 应优先改善。
- 如果重建 loss 下降但变化类仍不提升，说明仅靠月度局部 embedding 与困难重建还不够，需要进一步引入更强的月度状态重建目标。
- P3C 的高分重建可能略弱于 P3B，因为高分信息不再直接进入主 embedding，这是为了换取更干净的月度状态表示。
