# xuannv_embedding 数据目录布局

> 本文档与 `/data/xuannv_embedding/README.md` 保持同步。若数据盘文档已更新，以数据盘版本为准。

数据根目录：`/data/xuannv_embedding/`

## 核心区分

- **`raw/<region>/<source>/`**：从外部源**原样下载**的数据，未对齐、未切 patch。
- **`processed/<region>/`**：已对齐到统一网格并切成 128×128 patch 的数据，可直接输入模型。
- **`outputs/`**：训练产物（checkpoints、logs、eval）。需要长期保护的权重见 `docs/checkpoint_registry.md`。
- **`embeddings/`**：由 checkpoint + 输入数据重新生成的推理产物，默认不长期保存。
- **`cache/`**：可重新生成的预处理缓存，默认不长期保存。
- **`logs/`**：下载/预处理原始日志。
- **`statistics/<region>/`**：各 source 的 mean/std 统计量。

## 目录树

```
/data/xuannv_embedding/
├── logs/                          # 下载/预处理日志
├── outputs/                       # 训练输出
├── embeddings/                    # 预生成 embedding（可重建，按需生成）
├── cache/                         # 预处理缓存（可重建，按需生成）
├── processed/                     # 预处理后的数据
│   ├── harbin/
│   │   ├── patches/               # 时序影像 patch（当前仅 s2）
│   │   │   └── s2/
│   │   ├── labels/
│   │   │   └── worldcover/
│   │   ├── geography/             # 静态地理辅助数据（预留）
│   │   ├── metadata/              # 统计量副本、配置快照
│   │   └── manifest.json          # patch 索引清单
│   └── haidian/
│       ├── patches/               # 时序影像 patch（当前仅 s2）
│       │   └── s2/
│       ├── labels/
│       │   └── worldcover/
│       ├── geography/             # 静态地理辅助数据（预留）
│       ├── metadata/              # 统计量副本、配置快照
│       └── manifest.json          # patch 索引清单
├── raw/                           # 原始下载数据
│   ├── harbin/
│   │   ├── s1/
│   │   ├── s2/                    # S2 2025 按月拆分 NetCDF
│   │   ├── landsat/
│   │   ├── esri_lulc_2023/
│   │   ├── highres_optical/
│   │   ├── dem/
│   │   ├── jrc_water/
│   │   └── dynamic_world/
│   └── haidian/
│       ├── s1/
│       ├── s2/
│       ├── landsat/
│       ├── esri_lulc_2023/
│       ├── highres_optical/       # PlanetScene 解压/镶嵌产物
│       ├── highres_sar/           # 天仪 SAR 原始数据
│       ├── dem/
│       ├── jrc_water/
│       └── dynamic_world/
├── statistics/
│   ├── harbin/
│   └── haidian/
└── README.md
```

> 说明：只有 **harbin** 和 **haidian** 两个实验区，不存在 `raw/beijing`。

## 关键路径对应表

| 用途 | 哈尔滨路径 | 海淀路径 |
|------|-----------|----------|
| Region 根目录 | `/data/xuannv_embedding/processed/harbin` | `/data/xuannv_embedding/processed/haidian` |
| Manifest | `.../harbin/manifest.json` | `.../haidian/manifest.json` |
| 统计量 | `/data/xuannv_embedding/statistics/harbin` | `/data/xuannv_embedding/statistics/haidian` |
| 训练配置 | `configs/harbin_128.yaml` | `configs/haidian_128.yaml` |
| 下载日志 | `logs/download_pc_retry.log`、`logs/download_baidu.log` | 同上 |

## 下游任务与 Embedding 产物

- **`downstream/labels/<region>/<task>/`**：labelme 原始标注、对齐后的 mask、`split_5fold.json`、`label_meta.json`。
- **`embeddings/YYYYMMDD_<exp>_<sha>/`**：预生成 embedding，含 per-patch `{month}_embedding_map.pt`、`{month}_scene_embedding.pt` 与 `meta.json`。该目录是派生产物，可由 checkpoint、配置和输入数据重建。
- **`outputs/downstream/<task>_<region>_<date>_<suffix>/`**：下游训练输出，含 `fold_*/{metrics.json,predictions/,checkpoints/,visualizations/}` 与 `summary_5fold.json`。
