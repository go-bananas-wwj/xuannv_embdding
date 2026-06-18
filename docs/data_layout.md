# xuannv_embedding 数据目录

本目录是 `xuannv_embedding` 项目的离线数据根，代码位于 `/root/workspace/xuannv`。本文档记录数据目录的**当前实际结构**，方便在清理、重建或跨机器同步时核对。

## 目录结构

```
/data/xuannv_embedding/
├── outputs/                        # 训练输出（checkpoints / logs / eval），当前为空
├── processed/                      # 预处理后的 patch 数据 + manifest，当前为空
├── statistics/                     # 归一化统计量，当前为空
└── raw/                            # 原始下载数据
    ├── beijing/                    # 北京示范区完整原始数据
    │   ├── dem/
    │   ├── dynamic_world/
    │   ├── highres/
    │   ├── jrc_water/
    │   ├── landsat/
    │   ├── s1/
    │   ├── s2/
    │   └── worldcover/
    ├── haidian/                    # 海淀实验区（哈尔滨训练对比区）
    │   ├── dem/
    │   ├── dynamic_world/
    │   ├── highres/
    │   ├── highres_sar/            # 天仪 SAR 部分下载保留
    │   ├── jrc_water/
    │   ├── landsat/                # 空占位，等待完整下载
    │   ├── s1/                     # 空占位，等待完整下载
    │   ├── s2/                     # 空占位，等待完整下载
    │   └── worldcover/
    └── harbin/                     # 哈尔滨新区实验区
        ├── dem/
        ├── dynamic_world/
        ├── highres/
        ├── highres_optical/        # 百度网盘高分辨率光学部分下载保留
        ├── jrc_water/
        └── worldcover/
```

## 当前状态

- 已删除旧数据：`smoke` 目录、旧 `processed/`、旧 `statistics/`、旧 `outputs/` 均已被清理。
- `raw/haidian/highres_sar/` 保留：天仪 SAR 通过 ModelScope 的部分下载，尚未完成全部补齐。
- `raw/harbin/highres_optical/` 保留：百度网盘高分辨率光学数据的部分下载，尚未完成全部补齐。
- `raw/harbin/{s1,s2,landsat}` 已删除；`raw/haidian/{s1,s2,landsat}` 目前为空占位目录，等待后续完整下载。
- `processed/`、`statistics/`、`outputs/` 当前为空，将由数据预处理与训练流程逐步填充。
