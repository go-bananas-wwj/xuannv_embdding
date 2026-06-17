# xuannv_embedding

基于 AEF（AlphaEarth Foundations）论文的 clean re-implementation，用于生成月度地理嵌入。

## 核心目标

- 输入：Sentinel-1、Sentinel-2、Landsat，以及稀疏可得的高分辨率数据（SAR / 光学）。
- 输出：每个区域每个月的地理嵌入（embedding）。
- 下游：用简单头（kNN / 几层 MLP）即可做变化检测、土地覆盖分类等任务。

## 项目状态

- [x] 项目计划已制定：见 `docs/superpowers/plans/2026-06-17-xuannv-embedding-rebuild.md`
- [ ] 数据下载脚本
- [ ] 预处理 pipeline
- [ ] 模型实现
- [ ] 训练与评估

## 目录约定

```
/root/workspace/xuannv/          # 代码根
/data/xuannv_embedding/          # 数据根
```

## 快速开始

```bash
cd /root/workspace/xuannv
conda create -n xuannv_emb python=3.11 -y
conda activate xuannv_emb
pip install -e .
pytest tests/test_smoke.py -v
```

## 训练

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 scripts/train/train.py --config configs/harbin_monthly.yaml
```
