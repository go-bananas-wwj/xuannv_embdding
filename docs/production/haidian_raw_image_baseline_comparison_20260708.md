# 玄女海淀 V1：下游头与原始影像基线补充评测

## 结论先行

- 只换下游头时，`conv3x3` 空间头在建筑、道路、水体三类任务上稳定优于原来的局部头、wide MLP 和 deep-wide MLP，说明玄女 embedding 中存在可被轻量空间头释放的局部边界信息。
- 使用 2026 年 4 月原始多源影像直接训练下游头时，强监督的 UNet/DeepLab-lite 在 full train split 场景下可以达到或超过玄女轻量头；但这类方法依赖更重的输入、更大的下游模型和完整监督训练。
- 公平解释是：玄女 embedding 的价值不应只看 full-label 强监督上限，更应强调少量标注、轻量头、快速制图和跨任务复用；raw+UNet 是强基线，适合展示“如果不用 embedding，需要更重下游模型才能追上”。

## 评测设置

- 区域：海淀区 320 个 patch。
- 月份：原始影像基线使用 2026 年 4 月多源输入。
- 任务：建筑、道路、水体。
- 玄女输入：64 维 embedding，使用同一训练/验证/测试划分，重点比较 P10C/P12C 与多种下游头。
- 原始影像输入：Sentinel-2、Sentinel-1、Landsat、高分光学和高分 SAR 派生的 42 通道特征，不使用 embedding。
- 阈值：沿用验证集选择的最佳阈值，再在测试集报告 F1、AP、AUC。

## 核心指标图

![F1 comparison](/data/xuannv_embedding/experiments/p12_context_eval_20260707/raw_202604_head_sweep_20260708_report/raw_vs_xuannv_f1.png)

![AP comparison](/data/xuannv_embedding/experiments/p12_context_eval_20260707/raw_202604_head_sweep_20260708_report/raw_vs_xuannv_ap.png)

## 玄女下游头消融

| task | source | F1 | AP | AUC | threshold | best_epoch |
| --- | --- | --- | --- | --- | --- | --- |
| building | P10C conv3x3 | 0.484 | 0.450 | 0.896 | 0.600 | 49 |
| building | P10C deep-wide MLP | 0.418 | 0.366 | 0.862 | 0.534 | 49 |
| building | P10C old local | 0.366 | 0.296 | 0.836 | 0.475 | 9 |
| building | P10C wide MLP | 0.394 | 0.344 | 0.853 | 0.582 | 49 |
| building | P12C conv3x3 | 0.481 | 0.440 | 0.896 | 0.553 | 49 |
| building | P12C deep-wide MLP | 0.421 | 0.364 | 0.865 | 0.524 | 49 |
| building | P12C old local | 0.361 | 0.292 | 0.831 | 0.468 | 49 |
| building | P12C wide MLP | 0.401 | 0.339 | 0.856 | 0.501 | 49 |
| road | P10C conv3x3 | 0.520 | 0.566 | 0.834 | 0.578 | 44 |
| road | P10C deep-wide MLP | 0.502 | 0.532 | 0.815 | 0.504 | 49 |
| road | P10C old local | 0.419 | 0.404 | 0.751 | 0.468 | 49 |
| road | P10C wide MLP | 0.491 | 0.515 | 0.808 | 0.469 | 49 |
| road | P12C conv3x3 | 0.516 | 0.555 | 0.830 | 0.546 | 44 |
| road | P12C deep-wide MLP | 0.494 | 0.515 | 0.810 | 0.410 | 49 |
| road | P12C old local | 0.419 | 0.394 | 0.747 | 0.468 | 49 |
| road | P12C wide MLP | 0.476 | 0.484 | 0.796 | 0.386 | 49 |
| water | P10C conv3x3 | 0.623 | 0.604 | 0.891 | 0.818 | 44 |
| water | P10C deep-wide MLP | 0.604 | 0.586 | 0.858 | 0.322 | 49 |
| water | P10C old local | 0.431 | 0.441 | 0.709 | 0.467 | 49 |
| water | P10C wide MLP | 0.590 | 0.558 | 0.832 | 0.294 | 49 |
| water | P12C conv3x3 | 0.614 | 0.604 | 0.887 | 0.868 | 44 |
| water | P12C deep-wide MLP | 0.595 | 0.577 | 0.865 | 0.438 | 49 |
| water | P12C old local | 0.465 | 0.442 | 0.731 | 0.467 | 49 |
| water | P12C wide MLP | 0.580 | 0.557 | 0.850 | 0.340 | 49 |

## 原始影像强基线最佳结果

| task | source | F1 | AP | AUC | threshold | best_epoch |
| --- | --- | --- | --- | --- | --- | --- |
| building | Raw 202604 unet | 0.503 | 0.451 | 0.902 | 0.874 | 25 |
| road | Raw 202604 unet | 0.580 | 0.643 | 0.838 | 0.868 | 20 |
| water | Raw 202604 deeplab_lite | 0.667 | 0.673 | 0.897 | 0.805 | 40 |

## 玄女 conv3x3 与原始影像强头对比

| task | source | F1 | AP | AUC | threshold | best_epoch |
| --- | --- | --- | --- | --- | --- | --- |
| building | P10C conv3x3 | 0.484 | 0.450 | 0.896 | 0.600 | 49 |
| building | P12C conv3x3 | 0.481 | 0.440 | 0.896 | 0.553 | 49 |
| building | Raw 202604 unet | 0.503 | 0.451 | 0.902 | 0.874 | 25 |
| road | P10C conv3x3 | 0.520 | 0.566 | 0.834 | 0.578 | 44 |
| road | P12C conv3x3 | 0.516 | 0.555 | 0.830 | 0.546 | 44 |
| road | Raw 202604 unet | 0.580 | 0.643 | 0.838 | 0.868 | 20 |
| water | P10C conv3x3 | 0.623 | 0.604 | 0.891 | 0.818 | 44 |
| water | P12C conv3x3 | 0.614 | 0.604 | 0.887 | 0.868 | 44 |
| water | Raw 202604 deeplab_lite | 0.667 | 0.673 | 0.897 | 0.805 | 40 |

## 代表性可视化

### Building

![Building examples](/data/xuannv_embedding/experiments/p12_context_eval_20260707/raw_202604_head_sweep_20260708_report/building_xuannv_vs_raw_examples.png)

### Road

![Road examples](/data/xuannv_embedding/experiments/p12_context_eval_20260707/raw_202604_head_sweep_20260708_report/road_xuannv_vs_raw_examples.png)

### Water

![Water examples](/data/xuannv_embedding/experiments/p12_context_eval_20260707/raw_202604_head_sweep_20260708_report/water_xuannv_vs_raw_examples.png)

## 分析

1. 原始影像直接训练强头的建筑和道路结果较强，说明局部纹理和邻域上下文对这两类任务非常重要；这也解释了为什么纯 MLP 下游头容易产生假正例。
2. 玄女 embedding 接入 `conv3x3` 后显著好于纯通道 MLP，说明 embedding 不是完全坍缩，但下游头需要一点局部空间建模才能把边界信息释放出来。
3. 下一步若要进一步提高 embedding 本体，应继续让主模型在训练阶段学习更清晰的局部边界和多维语义，而不是只在下游端堆大模型。
