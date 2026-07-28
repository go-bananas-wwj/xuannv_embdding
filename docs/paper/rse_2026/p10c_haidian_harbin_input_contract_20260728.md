# P10C 海淀冻结编码器到哈尔滨：输入合同审计

日期：2026-07-28  
状态：**通过架构兼容性闸门；尚未构成下游迁移结果。**

## 目的

本审计只回答一个问题：在不修改 P10C epoch 800 编码器参数的前提下，哈尔滨的
真实输入能否按显式缺模态规则产生有效的 64×128×128 月度 embedding。它不训练
编码器，不读取下游标签，也不报告下游分数。

## 冻结合同

| P10C 输入槽位 | 通道 | 哈尔滨处理 | 是否可用 |
| --- | ---: | --- | --- |
| Sentinel-2 | 12 | 真实哈尔滨 S2，使用海淀 P10C 训练统计量 | 是 |
| Sentinel-1 | 2 | 真实哈尔滨 S1，使用海淀 P10C 训练统计量 | 是 |
| Landsat | 7 | 真实哈尔滨 Landsat，使用海淀 P10C 训练统计量 | 是 |
| 高分光学 | 3 | 映射到 checkpoint 的 `highres_optical_haidian` schema 槽位；原始来源保留为 `highres_optical_harbin` | 是 |
| 高分 SAR | 1 | 哈尔滨没有该数据源；保持空帧和零 availability，不填零影像冒充观测 | 否，显式缺失 |

该 schema 映射仅适配 checkpoint 的模块键名；不宣称海淀和哈尔滨的高分光学是同一
物理产品。冻结迁移不允许 encoder fine-tuning、学习型 adapter、目标区统计量拟合、
静默传感器重命名或把缺失传感器作为真实观测。

合同配置：`configs/paper_p10c_haidian_frozen_to_harbin_20260728.yaml`。  
转换后的哈尔滨 manifest：
`/data/xuannv_embedding/processed/harbin/manifest_p10c_haidian_frozen_to_harbin_20260728.json`。

## 真实数据 NPU dry-run

- checkpoint：P10C epoch 800，SHA-256 `69dfd81c898544413a747f5c7304cc9210ad1cf420ce724864b8bd7deb6ed790`。
- 数据：哈尔滨 `harbin_patch_000000`、`harbin_patch_000001`，2025-12 至 2026-05。
- 设备：`npu:0`。
- 输出：`[2, 6, 64, 128, 128]`，全部有限值。
- embedding 像素向量范数：均值 `1.000000`，标准差 `8.18e-08`，符合 vMF 单位球面输出。
- 高分输入字典仅包含 `highres_optical_haidian`；`highres_sar_haidian` 未出现，证明其
  没有被静默注入模型。

完整机器可读记录：
`/data/xuannv_embedding/experiments/paper_p10c_haidian_frozen_to_harbin_20260728/compatibility/dry_run_2patch_npu0.json`。

## 结论与边界

P10C 可在哈尔滨的共同模态加显式缺失高分 SAR 条件下稳定前向，因此允许继续执行
“冻结海淀 P10C → 哈尔滨 embedding → 仅训练哈尔滨 Conv3x3 reader”的评测。该结果
只证明输入和数值兼容，**不证明**跨城下游迁移性能；性能结论须由固定空间划分、
相同 shot 日程、验证集阈值和 AEF 同头对照给出。
