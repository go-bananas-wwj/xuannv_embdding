# P1B 简单下游头探针实验（2026-06-27）

## 实验目的

用户指出：如果 embedding 真的很好，下游任务不应该依赖复杂 decoder；用很简单的线性层或 MLP 层也应该能取得不错效果。因此本次新增并运行 simple probe，对 P1B embedding 的“可读性”做检查。

## 实验设置

输出目录：

`/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p1b_simple_probe_20260627`

对比对象：

- `unet_quick`：之前的轻量 U-Net 下游头。
- `linear`：单个 1x1 Conv，严格逐像素线性探针，不看邻域。
- `mlp`：两层 1x1 Conv + ReLU，只做逐像素通道非线性混合，不引入空间上下文。

新增配置：

- `downstreams/configs/v2_probe_linear_concat_diff.yaml`
- `downstreams/configs/v2_probe_mlp_concat_diff.yaml`

诊断可视化：

- Linear: `/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p1b_simple_probe_20260627/diagnostics_linear/index.md`
- MLP: `/data/xuannv_embedding/experiments/v2_202512_202605/expanded_downstream/p1b_simple_probe_20260627/diagnostics_mlp/index.md`

## 指标对比（fold0 quick）

| Task | Head | AUC | F1_best | F1@0.5 | F1@val_thr | mIoU | thr |
|---|---|---:|---:|---:|---:|---:|---:|
| haidian_building_osm | unet_quick | 0.8336 | 0.3590 | 0.2053 | 0.3525 | 0.2140 | 0.9743 |
| haidian_building_osm | linear | 0.7773 | 0.3057 | 0.1588 | 0.3045 | 0.1796 | 0.5488 |
| haidian_building_osm | mlp | 0.7652 | 0.2938 | 0.1588 | 0.2924 | 0.1712 | 0.6580 |
| haidian_road_osm | unet_quick | 0.7219 | 0.3869 | 0.2736 | 0.3864 | 0.2394 | 0.9600 |
| haidian_road_osm | linear | 0.6127 | 0.3176 | 0.2736 | 0.3176 | 0.1888 | 0.5428 |
| haidian_road_osm | mlp | 0.6021 | 0.3020 | 0.2736 | 0.3003 | 0.1767 | 0.7018 |
| harbin_building_osm | unet_quick | 0.9515 | 0.4283 | 0.2446 | 0.4129 | 0.2601 | 0.9528 |
| harbin_building_osm | linear | 0.7408 | 0.1373 | 0.0000 | 0.1249 | 0.0666 | 0.4644 |
| harbin_building_osm | mlp | 0.8981 | 0.2860 | 0.2694 | 0.2856 | 0.1666 | 0.5038 |
| harbin_road_osm | unet_quick | 0.8423 | 0.3654 | 0.1802 | 0.3644 | 0.2228 | 0.9717 |
| harbin_road_osm | linear | 0.6610 | 0.1887 | 0.1357 | 0.1885 | 0.1041 | 0.5207 |
| harbin_road_osm | mlp | 0.7973 | 0.3183 | 0.2206 | 0.3118 | 0.1847 | 0.6719 |

## 结论

1. Simple head 整体低于 U-Net，说明当前 P1B embedding 对 OSM 建筑/道路任务不是“简单可读”的理想状态。
2. Linear probe 在哈尔滨建筑、哈尔滨道路上明显失败，尤其 `harbin_building_osm` 的 `F1@0.5=0.0`、`F1_best=0.1373`，说明建筑信息不能被简单线性边界稳定读出。
3. MLP 在哈尔滨两个任务上明显优于 linear，说明 embedding 中存在一部分非线性可读信息；但 MLP 仍低于 U-Net，距离“embedding 本身足够好”还有差距。
4. U-Net 的阈值普遍在 0.95 到 0.97，simple probe 的阈值多在 0.46 到 0.70，说明 U-Net 之前的概率校准更差，而 simple probe 虽然校准正常一些，但分割能力不足。
5. 可视化统计显示 simple probe 仍有前景泛滥问题。例如海淀道路在 linear/MLP 的 `Pred>=0.5` 平均预测正类比例都接近 1.0，说明问题不只是 U-Net decoder，而是 embedding + loss + OSM 标签三者共同造成。

## 对“embedding 好不好”的判断

当前 P1B embedding 对建筑/道路有一定信号，但还不够强。它不是完全没学到，因为 U-Net 和 MLP 能提取一些信息；但也不是理想的通用地物 embedding，因为简单线性层和小 MLP 不能稳定取得接近 U-Net 的效果，更不能做到干净边界和良好阈值表现。

后续升级应把 simple probe 作为固定验收项：每轮 embedding 训练完，不只看复杂 decoder，还必须看 linear/MLP probe。如果 embedding 真的升级成功，linear/MLP 的 F1 和 mIoU 应该同步提升，且预测正类比例不能再大面积失控。
