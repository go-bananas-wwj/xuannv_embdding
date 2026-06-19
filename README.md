# xuannv_embedding

基于 AEF（AlphaEarth Foundations）论文的 clean re-implementation，用于生成月度地理嵌入。

## 核心目标

- 输入：Sentinel-1、Sentinel-2、Landsat，以及稀疏可得的高分辨率数据（SAR / 光学）。
- 输出：每个区域每个月的地理嵌入（embedding）。
- 下游：用简单头（kNN / 几层 MLP）即可做变化检测、土地覆盖分类等任务。

## 项目状态

- [x] 项目计划已制定：见 `docs/superpowers/plans/`
- [x] 数据下载脚本：PC / ModelScope / 百度网盘三类脚本已完成
- [x] 预处理 pipeline：对齐、patchify、统计量、manifest 生成已完成
- [x] 模型实现：多分辨率 STP + TemporalSummarizer + EmbeddingUpsampleHead，128×128 输入输出
- [x] 数据目录整理：patch / 标签 / 地理辅助 / 元数据 分类目录已建立
- [x] 全量低分辨率数据已下载并预处理为 128×128 patch
- [x] 高分辨率数据已下载（哈尔滨 DOM、海淀 PlanetScene、天仪 SAR）
- [ ] 高分辨率 SAR/光学 patch 预处理（需配准/重采样）
- [ ] 128×128 全量训练启动

## 目录约定

```
/root/workspace/xuannv/          # 代码根
/data/xuannv_embedding/          # 数据根
```

数据分类与状态详见：

- 数据盘权威文档：`/data/xuannv_embedding/README.md`
- 项目侧目录说明：`docs/data_layout.md`
- 实时数据清单：`docs/data_inventory.md`

## 快速开始

```bash
cd /root/workspace/xuannv
conda create -n xuannv_emb python=3.11 -y
conda activate xuannv_emb
pip install -e .
pytest tests/test_manifest.py tests/test_model.py tests/test_train_entry.py -q
```

## 训练

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 scripts/train/train.py --config configs/harbin_128.yaml
```

> 当前 `configs/harbin_128.yaml` 与 `configs/haidian_128.yaml` 已按 `processed/<region>/patches/`、`labels/`、`metadata/` 的新目录结构配置。

