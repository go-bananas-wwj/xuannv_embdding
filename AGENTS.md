# AGENTS.md — xuannv_embedding

## 强制规则

1. **与用户交流使用中文**。
2. **每次修改后执行**: `git add -A && git commit -m "描述" && git push origin main`。
3. **训练前检查 NPU**: `npu-smi info`。
4. **禁止 `nohup` 跑训练**，使用 `tmux`。
5. **文件操作限制在 `/root/workspace/xuannv` 和 `/data/xuannv_embedding/` 内**。
6. **数据目录**: `/data/xuannv_embedding/` 是数据根，代码中不可写死其他路径。
7. **设备选择**: 统一走 `xuannv_embedding.utils.device.get_device()`，优先 NPU。
8. **代码风格**: `from __future__ import annotations`；类型注解完整；模块内注释中文。
9. **测试**: 每次重大修改后跑 `pytest tests/test_smoke.py`。
10. **不要修改数据目录结构**: 下载脚本输出到 `raw/`，预处理输出到 `processed/`，统计量到 `statistics/`，训练输出到 `outputs/`。

## 项目目标

- 基于 AEF 论文 clean re-implementation，生成月度地理嵌入。
- 支持 S1/S2/Landsat + 稀疏高分辨率数据 availability-aware 融合。
- 在北京海淀（320 patches）和哈尔滨新区（424 patches）训练。

## 关键路径

1. 数据下载脚本（PC / ModelScope / BaiduPCS-Go）。
2. 预处理：对齐 → patchify → 统计量 → manifest。
3. 模型：sensor encoders → space/time operators → VMF bottleneck → decoders。
4. 训练：DDP on NPU，重建 + uniformity 损失。
5. 评估：embedding 提取 → KNN / MLP / 变化检测 AUC。

## 目录约定

```
/root/workspace/xuannv/          # 代码根
/data/xuannv_embedding/          # 数据根
  raw/                           # 原始下载数据
  processed/                     # patch 化数据 + manifest
  statistics/                    # 归一化统计量
  outputs/                       # checkpoint / log / eval
```

## 依赖安装

```bash
cd /root/workspace/xuannv
conda create -n xuannv_emb python=3.11 -y
conda activate xuannv_emb
pip install -e .
```

## 训练启动示例

```bash
conda activate xuannv_emb
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 scripts/train/train.py --config configs/harbin_monthly.yaml
```

## 数据下载示例

```bash
# S1/S2/Landsat via Planetary Computer
python scripts/data/download_pc.py --region harbin --start 2025-01-01 --end 2026-05-31

# 天仪 SAR via ModelScope
python scripts/data/download_modelscope.py --dataset WeijieWu/haidian_sar_2025 --output /data/xuannv_embedding/raw/haidian/highres_sar

# 百度网盘
python scripts/data/download_baidu.py --links-file links.txt --output /data/xuannv_embedding/raw/beijing/highres_optical
```

## 执行计划

详见 `docs/superpowers/plans/2026-06-17-xuannv-embedding-rebuild.md`。
