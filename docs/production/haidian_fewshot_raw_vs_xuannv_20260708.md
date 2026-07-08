# 玄女海淀 V1：Few-shot 原始影像基线对比

## 结论

- 在 5-shot 和 10-shot 设置下，玄女 embedding 的最佳下游头在建筑、道路、水体三类任务上均超过 raw 2026-04 原始影像的最佳强头。
- 在 50-shot 设置下，玄女在建筑、道路仍领先；水体任务 raw UNet 与玄女 conv3x3 基本打平。
- 这说明 embedding 的主要价值不是在 full-label 强监督场景里一定压过 raw+UNet，而是在少量标注、轻量训练、快速制图场景中更稳、更省标注、更容易复用。

## 公平性说明

- 标签：建筑、道路、水体均使用同一套 OSM 二值标签。
- 划分：同一 fold，训练/验证/测试 patch 完全一致。
- Shot：5/10/50-shot 表示每个任务只从训练集中选取少量正样本 patch，并配同数量负样本 patch 训练下游头。
- 原始影像输入：2026 年 4 月 Sentinel-2、Sentinel-1、Landsat、高分光学、高分 SAR 派生的 42 通道特征。
- 玄女输入：同月 64 维 embedding。
- 阈值：均在验证集上选择最佳阈值，再报告测试集 F1、AP、AUC。

## 最佳模型对比

![Few-shot best F1](/data/xuannv_embedding/experiments/p12_context_eval_20260707/fewshot_raw_vs_xuannv_20260708/report/fewshot_best_f1.png)

| 任务 | Shot | 玄女最佳头 | F1 | Raw最佳头 | RawF1 | 提升 | AP | RawAP | AUC | RawAUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 建筑 Building | 5 | conv3x3 | 0.459 | conv3x3 | 0.418 | +9.8% | 0.413 | 0.354 | 0.879 | 0.858 |
| 建筑 Building | 10 | conv3x3 | 0.454 | conv3x3 | 0.439 | +3.5% | 0.405 | 0.375 | 0.883 | 0.874 |
| 建筑 Building | 50 | conv3x3 | 0.490 | conv3x3 | 0.467 | +5.0% | 0.448 | 0.409 | 0.898 | 0.888 |
| 道路 Road | 5 | conv3x3 | 0.474 | conv3x3 | 0.401 | +18.3% | 0.507 | 0.329 | 0.795 | 0.740 |
| 道路 Road | 10 | conv3x3 | 0.487 | conv3x3 | 0.435 | +11.8% | 0.511 | 0.374 | 0.803 | 0.768 |
| 道路 Road | 50 | conv3x3 | 0.517 | conv3x3 | 0.501 | +3.3% | 0.561 | 0.523 | 0.828 | 0.820 |
| 水体 Water | 5 | conv3x3 | 0.613 | unet | 0.433 | +41.5% | 0.599 | 0.468 | 0.878 | 0.816 |
| 水体 Water | 10 | conv3x3 | 0.612 | deeplab_lite | 0.487 | +25.6% | 0.583 | 0.488 | 0.874 | 0.810 |
| 水体 Water | 50 | conv3x3 | 0.631 | unet | 0.631 | -0.0% | 0.613 | 0.625 | 0.894 | 0.878 |

## 下游头完整热力图

![Few-shot head heatmap](/data/xuannv_embedding/experiments/p12_context_eval_20260707/fewshot_raw_vs_xuannv_20260708/report/fewshot_head_heatmap.png)

## 5-shot 代表性可视化

![Few-shot examples](/data/xuannv_embedding/experiments/p12_context_eval_20260707/fewshot_raw_vs_xuannv_20260708/report/fewshot_5shot_examples.png)

## 全部组合指标

| 任务 | Shot | 输入 | 头 | F1 | AP | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| 建筑 Building | 5 | Raw 202604 image | conv3x3 | 0.418 | 0.354 | 0.858 |
| 建筑 Building | 5 | Raw 202604 image | deep_wide_mlp | 0.281 | 0.184 | 0.759 |
| 建筑 Building | 5 | Raw 202604 image | deeplab_lite | 0.365 | 0.292 | 0.818 |
| 建筑 Building | 5 | Raw 202604 image | unet | 0.416 | 0.353 | 0.840 |
| 建筑 Building | 5 | Raw 202604 image | wide_mlp | 0.290 | 0.209 | 0.763 |
| 建筑 Building | 5 | Xuannv embedding | conv3x3 | 0.459 | 0.413 | 0.879 |
| 建筑 Building | 5 | Xuannv embedding | deep_wide_mlp | 0.354 | 0.295 | 0.840 |
| 建筑 Building | 5 | Xuannv embedding | deeplab_lite | 0.406 | 0.338 | 0.829 |
| 建筑 Building | 5 | Xuannv embedding | unet | 0.425 | 0.373 | 0.863 |
| 建筑 Building | 5 | Xuannv embedding | wide_mlp | 0.326 | 0.250 | 0.822 |
| 建筑 Building | 10 | Raw 202604 image | conv3x3 | 0.439 | 0.375 | 0.874 |
| 建筑 Building | 10 | Raw 202604 image | deep_wide_mlp | 0.308 | 0.210 | 0.786 |
| 建筑 Building | 10 | Raw 202604 image | deeplab_lite | 0.385 | 0.307 | 0.837 |
| 建筑 Building | 10 | Raw 202604 image | unet | 0.415 | 0.343 | 0.851 |
| 建筑 Building | 10 | Raw 202604 image | wide_mlp | 0.279 | 0.196 | 0.759 |
| 建筑 Building | 10 | Xuannv embedding | conv3x3 | 0.454 | 0.405 | 0.883 |
| 建筑 Building | 10 | Xuannv embedding | deep_wide_mlp | 0.347 | 0.285 | 0.833 |
| 建筑 Building | 10 | Xuannv embedding | deeplab_lite | 0.403 | 0.315 | 0.842 |
| 建筑 Building | 10 | Xuannv embedding | unet | 0.431 | 0.372 | 0.861 |
| 建筑 Building | 10 | Xuannv embedding | wide_mlp | 0.336 | 0.269 | 0.827 |
| 建筑 Building | 50 | Raw 202604 image | conv3x3 | 0.467 | 0.409 | 0.888 |
| 建筑 Building | 50 | Raw 202604 image | deep_wide_mlp | 0.324 | 0.225 | 0.801 |
| 建筑 Building | 50 | Raw 202604 image | deeplab_lite | 0.425 | 0.362 | 0.865 |
| 建筑 Building | 50 | Raw 202604 image | unet | 0.458 | 0.405 | 0.881 |
| 建筑 Building | 50 | Raw 202604 image | wide_mlp | 0.318 | 0.241 | 0.802 |
| 建筑 Building | 50 | Xuannv embedding | conv3x3 | 0.490 | 0.448 | 0.898 |
| 建筑 Building | 50 | Xuannv embedding | deep_wide_mlp | 0.397 | 0.336 | 0.855 |
| 建筑 Building | 50 | Xuannv embedding | deeplab_lite | 0.425 | 0.363 | 0.871 |
| 建筑 Building | 50 | Xuannv embedding | unet | 0.478 | 0.432 | 0.891 |
| 建筑 Building | 50 | Xuannv embedding | wide_mlp | 0.376 | 0.318 | 0.847 |
| 水体 Water | 5 | Raw 202604 image | conv3x3 | 0.396 | 0.429 | 0.825 |
| 水体 Water | 5 | Raw 202604 image | deep_wide_mlp | 0.353 | 0.312 | 0.748 |
| 水体 Water | 5 | Raw 202604 image | deeplab_lite | 0.391 | 0.383 | 0.780 |
| 水体 Water | 5 | Raw 202604 image | unet | 0.433 | 0.468 | 0.816 |
| 水体 Water | 5 | Raw 202604 image | wide_mlp | 0.294 | 0.190 | 0.762 |
| 水体 Water | 5 | Xuannv embedding | conv3x3 | 0.613 | 0.599 | 0.878 |
| 水体 Water | 5 | Xuannv embedding | deep_wide_mlp | 0.554 | 0.535 | 0.822 |
| 水体 Water | 5 | Xuannv embedding | deeplab_lite | 0.547 | 0.525 | 0.846 |
| 水体 Water | 5 | Xuannv embedding | unet | 0.557 | 0.540 | 0.835 |
| 水体 Water | 5 | Xuannv embedding | wide_mlp | 0.524 | 0.518 | 0.811 |
| 水体 Water | 10 | Raw 202604 image | conv3x3 | 0.452 | 0.505 | 0.843 |
| 水体 Water | 10 | Raw 202604 image | deep_wide_mlp | 0.398 | 0.322 | 0.797 |
| 水体 Water | 10 | Raw 202604 image | deeplab_lite | 0.487 | 0.488 | 0.810 |
| 水体 Water | 10 | Raw 202604 image | unet | 0.484 | 0.496 | 0.811 |
| 水体 Water | 10 | Raw 202604 image | wide_mlp | 0.325 | 0.186 | 0.777 |
| 水体 Water | 10 | Xuannv embedding | conv3x3 | 0.612 | 0.583 | 0.874 |
| 水体 Water | 10 | Xuannv embedding | deep_wide_mlp | 0.537 | 0.531 | 0.824 |
| 水体 Water | 10 | Xuannv embedding | deeplab_lite | 0.555 | 0.537 | 0.851 |
| 水体 Water | 10 | Xuannv embedding | unet | 0.586 | 0.550 | 0.848 |
| 水体 Water | 10 | Xuannv embedding | wide_mlp | 0.520 | 0.512 | 0.808 |
| 水体 Water | 50 | Raw 202604 image | conv3x3 | 0.612 | 0.607 | 0.889 |
| 水体 Water | 50 | Raw 202604 image | deep_wide_mlp | 0.384 | 0.289 | 0.784 |
| 水体 Water | 50 | Raw 202604 image | deeplab_lite | 0.612 | 0.597 | 0.867 |
| 水体 Water | 50 | Raw 202604 image | unet | 0.631 | 0.625 | 0.878 |
| 水体 Water | 50 | Raw 202604 image | wide_mlp | 0.472 | 0.408 | 0.833 |
| 水体 Water | 50 | Xuannv embedding | conv3x3 | 0.631 | 0.613 | 0.894 |
| 水体 Water | 50 | Xuannv embedding | deep_wide_mlp | 0.600 | 0.572 | 0.866 |
| 水体 Water | 50 | Xuannv embedding | deeplab_lite | 0.618 | 0.589 | 0.866 |
| 水体 Water | 50 | Xuannv embedding | unet | 0.628 | 0.591 | 0.854 |
| 水体 Water | 50 | Xuannv embedding | wide_mlp | 0.594 | 0.569 | 0.852 |
| 道路 Road | 5 | Raw 202604 image | conv3x3 | 0.401 | 0.329 | 0.740 |
| 道路 Road | 5 | Raw 202604 image | deep_wide_mlp | 0.330 | 0.207 | 0.636 |
| 道路 Road | 5 | Raw 202604 image | deeplab_lite | 0.343 | 0.264 | 0.677 |
| 道路 Road | 5 | Raw 202604 image | unet | 0.342 | 0.227 | 0.651 |
| 道路 Road | 5 | Raw 202604 image | wide_mlp | 0.354 | 0.223 | 0.668 |
| 道路 Road | 5 | Xuannv embedding | conv3x3 | 0.474 | 0.507 | 0.795 |
| 道路 Road | 5 | Xuannv embedding | deep_wide_mlp | 0.381 | 0.260 | 0.711 |
| 道路 Road | 5 | Xuannv embedding | deeplab_lite | 0.373 | 0.356 | 0.695 |
| 道路 Road | 5 | Xuannv embedding | unet | 0.375 | 0.274 | 0.704 |
| 道路 Road | 5 | Xuannv embedding | wide_mlp | 0.408 | 0.307 | 0.742 |
| 道路 Road | 10 | Raw 202604 image | conv3x3 | 0.435 | 0.374 | 0.768 |
| 道路 Road | 10 | Raw 202604 image | deep_wide_mlp | 0.363 | 0.235 | 0.675 |
| 道路 Road | 10 | Raw 202604 image | deeplab_lite | 0.383 | 0.330 | 0.716 |
| 道路 Road | 10 | Raw 202604 image | unet | 0.406 | 0.328 | 0.741 |
| 道路 Road | 10 | Raw 202604 image | wide_mlp | 0.362 | 0.246 | 0.686 |
| 道路 Road | 10 | Xuannv embedding | conv3x3 | 0.487 | 0.511 | 0.803 |
| 道路 Road | 10 | Xuannv embedding | deep_wide_mlp | 0.406 | 0.307 | 0.742 |
| 道路 Road | 10 | Xuannv embedding | deeplab_lite | 0.447 | 0.468 | 0.757 |
| 道路 Road | 10 | Xuannv embedding | unet | 0.456 | 0.438 | 0.782 |
| 道路 Road | 10 | Xuannv embedding | wide_mlp | 0.392 | 0.275 | 0.725 |
| 道路 Road | 50 | Raw 202604 image | conv3x3 | 0.501 | 0.523 | 0.820 |
| 道路 Road | 50 | Raw 202604 image | deep_wide_mlp | 0.400 | 0.298 | 0.734 |
| 道路 Road | 50 | Raw 202604 image | deeplab_lite | 0.475 | 0.500 | 0.789 |
| 道路 Road | 50 | Raw 202604 image | unet | 0.482 | 0.520 | 0.786 |
| 道路 Road | 50 | Raw 202604 image | wide_mlp | 0.379 | 0.269 | 0.704 |
| 道路 Road | 50 | Xuannv embedding | conv3x3 | 0.517 | 0.561 | 0.828 |
| 道路 Road | 50 | Xuannv embedding | deep_wide_mlp | 0.480 | 0.485 | 0.802 |
| 道路 Road | 50 | Xuannv embedding | deeplab_lite | 0.489 | 0.540 | 0.800 |
| 道路 Road | 50 | Xuannv embedding | unet | 0.503 | 0.557 | 0.810 |
| 道路 Road | 50 | Xuannv embedding | wide_mlp | 0.448 | 0.428 | 0.782 |
