# xuannv_embedding 数据目录

本目录与代码目录 `/root/workspace/xuannv` 分离，专门存放原始数据、预处理结果、统计量与训练输出。

## 目录结构

```
/data/xuannv_embedding/
├── raw/                 # 原始遥感影像（NetCDF、GeoTIFF、压缩包）
│   ├── harbin/          # 哈尔滨新区
│   ├── haidian/         # 海淀区
│   └── beijing/         # 北京/Planet 高分辨率数据
├── processed/           # 对齐后的 patches（GeoTIFF）
│   ├── harbin/scenes/
│   └── haidian/scenes/
├── statistics/          # 各 source 的均值/标准差等统计量
│   ├── harbin/
│   └── haidian/
└── outputs/             # 训练 checkpoint、日志、推理结果
    ├── harbin_v1/
    └── haidian_v1/
```

## 当前状态

- 已删除 smoke 测试产物与旧的预处理结果。
- `raw/haidian/highres_sar/`：保留（ModelScope 天仪 SAR 部分下载中）。
- `raw/harbin/highres_optical/`：保留（百度网盘高分辨率光学数据）。
- `raw/harbin/{s1,s2,landsat}` 与 `raw/haidian/{s1,s2,landsat}`：已清空，等待重新下载 2025-01 至 2026-05 全量数据。
