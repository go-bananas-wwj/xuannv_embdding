# Phase 8：补全高分辨率数据并完成哈尔滨/海淀区正式训练

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成 chat1 中要求的高分辨率数据（百度网盘北京/哈尔滨光学、ModelScope 海淀 SAR）下载与预处理；为哈尔滨和海淀区生成完整训练数据；基于 128×128 patches 训练最终嵌入模型并提取 embedding 做下游评估。

**Architecture:** 复用现有 `scripts/data/download_baidu.py`、`download_modelscope.py`、`preprocess.py` 与 `scripts/train/train.py`；通过 `spatial_stride=2` 控制显存；高分辨率数据通过 `AvailabilityAwareFusion` 以稀疏帧形式融入模型。

**Tech Stack:** Python 3.11, PyTorch, torch_npu, rasterio, xarray, BaiduPCS-Go, ModelScope SDK, scikit-learn.

---

## File Structure

| 路径 | 职责 |
|------|------|
| `scripts/data/download_baidu.py` | 调用 BaiduPCS-Go 批量下载百度网盘数据 |
| `scripts/data/download_modelscope.py` | 下载 ModelScope 天仪 SAR 数据集 |
| `scripts/data/preprocess.py` | 对齐、切 patch、填充缺失 |
| `scripts/data/generate_manifest.py` | 生成训练 manifest |
| `scripts/data/compute_statistics.py` | 计算各 source 的 mean/std |
| `src/xuannv_embedding/models/model.py` | AEFModel，含高分辨率融合 |
| `src/xuannv_embedding/training/batch_preparation.py` | 构造模型输入与 target |
| `scripts/train/train.py` | 训练入口 |
| `configs/haidian_monthly.yaml` | 海淀区完整训练配置 |
| `configs/harbin_monthly.yaml` | 哈尔滨新区完整训练配置 |
| `scripts/eval/extract_embeddings.py` | 提取 embedding |
| `scripts/eval/knn_eval.py` | KNN 下游评估 |
| `scripts/eval/change_detection_eval.py` | 变化检测 AUC 评估 |

---

### Task 1: 列出百度网盘 `/玄女科技/` 目录并生成可下载的 links 文件

**Files:**
- Create: `scripts/data/baidu_links_beijing.txt`
- Create: `scripts/data/baidu_links_harbin.txt`
- Modify: `scripts/data/baidu_links_haidian.txt`（若已存在则更新）

**Context:** 百度网盘高分辨率数据位于 `/玄女科技/`。`download_baidu.py` 要求 links 文件每行一个远程路径。需要先用 `BaiduPCS-Go ls` 找到北京、哈尔滨、海淀数据的真实远程目录或文件路径。

- [ ] **Step 1: 列出远程目录**

```bash
cd /root/workspace/xuannv
export BAIDUPCS_GO_CONFIG_DIR=/root/workspace/xuannv/tools/baidupcs
./tools/baidupcs/BaiduPCS-Go ls /玄女科技/
```

Expected: 输出 `/玄女科技/` 下的文件/目录列表，例如 `beijing_rs_psscene_analytic_sr_udm2.zip`、`harbin_aoi.zip` 等。

- [ ] **Step 2: 将真实远程路径写入 links 文件**

假设 Step 1 发现北京数据文件为 `/玄女科技/beijing_rs_psscene_analytic_sr_udm2.zip`，哈尔滨数据为 `/玄女科技/harbin_highres.zip`，海淀数据为 `/玄女科技/haidian_highres.zip`：

```bash
cd /root/workspace/xuannv
echo '/玄女科技/beijing_rs_psscene_analytic_sr_udm2.zip' > scripts/data/baidu_links_beijing.txt
echo '/玄女科技/harbin_highres.zip' > scripts/data/baidu_links_harbin.txt
echo '/玄女科技/haidian_highres.zip' >> scripts/data/baidu_links_haidian.txt
```

- [ ] **Step 3: 提交 links 文件**

```bash
cd /root/workspace/xuannv
git add scripts/data/baidu_links_*.txt
git commit -m "docs(data): record baidu high-res remote paths"
git push origin main
```

---

### Task 2: 下载百度网盘北京/哈尔滨/海淀高分辨率数据

**Files:**
- Modify: `docs/data_status/2026-06-18_highres_sample.md`

**Context:** 使用已登录的 BaiduPCS-Go session 下载。大文件可能耗时较长，超时失败可重试。

- [ ] **Step 1: 北京海淀光学数据**

```bash
cd /root/workspace/xuannv
python scripts/data/download_baidu.py \
    --links-file scripts/data/baidu_links_beijing.txt \
    --output /data/xuannv_embedding/raw/beijing/highres_optical
```

Expected: 日志显示下载完成，输出目录下存在 `.zip` 或已解压的 `.tif` 文件。

- [ ] **Step 2: 哈尔滨新区光学数据**

```bash
cd /root/workspace/xuannv
python scripts/data/download_baidu.py \
    --links-file scripts/data/baidu_links_harbin.txt \
    --output /data/xuannv_embedding/raw/harbin/highres_optical
```

Expected: 同上。

- [ ] **Step 3: 海淀高分辨率光学补充数据**

```bash
cd /root/workspace/xuannv
python scripts/data/download_baidu.py \
    --links-file scripts/data/baidu_links_haidian.txt \
    --output /data/xuannv_embedding/raw/haidian/highres_optical
```

Expected: 同上。

- [ ] **Step 4: 记录下载状态并提交**

更新 `docs/data_status/2026-06-18_highres_sample.md`，补充北京/哈尔滨/海淀高分辨率光学数据下载状态，然后：

```bash
cd /root/workspace/xuannv
git add docs/data_status/2026-06-18_highres_sample.md
git commit -m "docs: update baidu high-res download status"
git push origin main
```

---

### Task 3: 继续并完成 ModelScope 海淀 SAR 数据下载

**Files:**
- Modify: `docs/data_status/2026-06-18_highres_sample.md`

**Context:** 此前已 partial 下载约 33 GB 到 `/data/xuannv_embedding/raw/haidian/highres_sar/`。脚本支持断点续传。

- [ ] **Step 1: 续传下载**

```bash
cd /root/workspace/xuannv
export MODELSCOPE_TOKEN="ms-399d1804-1cb3-446a-a3f7-dfc4dc70d977"
python scripts/data/download_modelscope.py \
    --dataset WeijieWu/haidian_sar_2025 \
    --output /data/xuannv_embedding/raw/haidian/highres_sar \
    --cache-dir /tmp/modelscope_cache
```

Expected: 日志显示文件已存在则跳过，缺失文件继续下载，直至完成或再次超时。若超时，记录已下载大小。

- [ ] **Step 2: 过滤非海淀区数据**

海淀区范围见 `configs/regions/haidian.geojson`。若下载结果中包含非海淀点位，需写过滤脚本（或在预处理时通过地理范围裁剪）。

```bash
cd /root/workspace/xuannv
find /data/xuannv_embedding/raw/haidian/highres_sar -name '*.zip' | wc -l
```

Expected: 输出 zip 文件总数。

- [ ] **Step 3: 解压 zip（如需要）**

```bash
cd /data/xuannv_embedding/raw/haidian/highres_sar
find . -name '*.zip' -exec unzip -o {} \;
```

Expected: 解压出 `.tif` 或 `.SAFE` 目录。

- [ ] **Step 4: 提交状态更新**

```bash
cd /root/workspace/xuannv
git add docs/data_status/2026-06-18_highres_sample.md
git commit -m "docs: update modelscope haidian sar download status"
git push origin main
```

---

### Task 4: 为高分辨率数据编写并运行预处理

**Files:**
- Create: `scripts/data/preprocess_haidian_highres.json`
- Create: `scripts/data/preprocess_harbin_highres.json`
- Create/Modify: `scripts/data/preprocess_haidian.json`、`scripts/data/preprocess_harbin.json`（主配置加入 `highres` source）

**Context:** 高分辨率数据的空间分辨率、坐标系与 S2/S1/Landsat 不同，需要重采样到与主数据相同的 CRS 与 patch grid。`preprocess.py` 已支持多 source 对齐。

- [ ] **Step 1: 哈尔滨高分辨率预处理配置**

创建 `scripts/data/preprocess_harbin_highres.json`：

```json
{
  "_comment": "harbin highres optical aligned to 128x128 patch grid",
  "region": "harbin",
  "raw_root": "/data/xuannv_embedding/raw/harbin/highres_optical",
  "output_root": "/data/xuannv_embedding/processed/harbin_128/scenes",
  "crs": "EPSG:32652",
  "patch_size_m": 1280,
  "patch_size_px": 128,
  "sources": ["highres"],
  "fill_missing": "zero"
}
```

- [ ] **Step 2: 运行哈尔滨高分辨率预处理**

```bash
cd /root/workspace/xuannv
python scripts/data/preprocess.py --config scripts/data/preprocess_harbin_highres.json
```

Expected: 在 `/data/xuannv_embedding/processed/harbin_128/scenes/highres/` 下生成与 s2/s1/landsat 同名的 patch `.tif` 文件。

- [ ] **Step 3: 海淀区高分辨率预处理配置与执行**

创建 `scripts/data/preprocess_haidian_highres.json`：

```json
{
  "_comment": "haidian highres optical aligned to 128x128 patch grid",
  "region": "haidian",
  "raw_root": "/data/xuannv_embedding/raw/haidian/highres_optical",
  "output_root": "/data/xuannv_embedding/processed/haidian_128/scenes",
  "crs": "EPSG:32650",
  "patch_size_m": 1280,
  "patch_size_px": 128,
  "sources": ["highres"],
  "fill_missing": "zero"
}
```

注意：海淀区 CRS 需根据 `configs/regions/haidian.geojson` 的实际坐标系调整（当前为 `EPSG:32650`，若实际不同请替换）。

```bash
cd /root/workspace/xuannv
python scripts/data/preprocess.py --config scripts/data/preprocess_haidian_highres.json
```

Expected: 同上，输出到 `/data/xuannv_embedding/processed/haidian_128/scenes/highres/`。

- [ ] **Step 4: 海淀 SAR 预处理**

创建 `scripts/data/preprocess_haidian_sar.json`：

```json
{
  "_comment": "haidian highres SAR aligned to 128x128 patch grid",
  "region": "haidian",
  "raw_root": "/data/xuannv_embedding/raw/haidian/highres_sar",
  "output_root": "/data/xuannv_embedding/processed/haidian_128/scenes",
  "crs": "EPSG:32650",
  "patch_size_m": 1280,
  "patch_size_px": 128,
  "sources": ["highres_sar"],
  "fill_missing": "zero"
}
```

注意：`highres_sar` 作为额外 source，需在模型 `sensor_channels` 中注册。

```bash
cd /root/workspace/xuannv
python scripts/data/preprocess.py --config scripts/data/preprocess_haidian_sar.json
```

- [ ] **Step 5: 提交预处理配置**

```bash
cd /root/workspace/xuannv
git add scripts/data/preprocess_*_highres*.json
git commit -m "feat(data): add highres preprocessing configs for harbin/haidian"
git push origin main
```

---

### Task 5: 生成完整 manifest 与统计量

**Files:**
- Modify: `docs/data_status/2026-06-18_harbin_sample.md`
- Create/Modify: `docs/data_status/2026-06-18_haidian_sample.md`

- [ ] **Step 1: 哈尔滨 128 完整 manifest（含 highres）**

```bash
cd /root/workspace/xuannv
python scripts/data/generate_manifest.py \
    --processed-dir /data/xuannv_embedding/processed/harbin_128/scenes \
    --output /data/xuannv_embedding/processed/harbin_128/scenes/manifest.json \
    --sources s2 s1 landsat highres
```

Expected: `manifest.json` 中每条记录包含 `s2/s1/landsat/highres` 四个字段。

- [ ] **Step 2: 哈尔滨 128 统计量**

```bash
cd /root/workspace/xuannv
python scripts/data/compute_statistics.py \
    --processed-dir /data/xuannv_embedding/processed/harbin_128/scenes \
    --output-dir /data/xuannv_embedding/statistics/harbin_128 \
    --sources s2 s1 landsat highres
```

Expected: `/data/xuannv_embedding/statistics/harbin_128/` 下存在四个 `_stats.json`。

- [ ] **Step 3: 海淀区 128 manifest 与统计量**

```bash
cd /root/workspace/xuannv
python scripts/data/generate_manifest.py \
    --processed-dir /data/xuannv_embedding/processed/haidian_128/scenes \
    --output /data/xuannv_embedding/processed/haidian_128/scenes/manifest.json \
    --sources s2 s1 landsat highres highres_sar

python scripts/data/compute_statistics.py \
    --processed-dir /data/xuannv_embedding/processed/haidian_128/scenes \
    --output-dir /data/xuannv_embedding/statistics/haidian_128 \
    --sources s2 s1 landsat highres highres_sar
```

Expected: 同上，统计量目录下存在对应文件。

- [ ] **Step 4: 验证并提交状态文档**

```bash
cd /root/workspace/xuannv
python - <<'PY'
import json
from pathlib import Path
for region in ["harbin_128", "haidian_128"]:
    p = Path(f"/data/xuannv_embedding/processed/{region}/scenes/manifest.json")
    if p.exists():
        data = json.loads(p.read_text())
        print(region, "records:", len(data), "sources:", list(data[0].keys()) if data else [])
    else:
        print(region, "manifest missing")
PY

git add docs/data_status/
git commit -m "docs: update full manifest/statistics status for harbin and haidian"
git push origin main
```

Expected: 打印两个区域的 patch 数量与 source 字段列表。

---

### Task 6: 更新训练配置以支持高分辨率 source

**Files:**
- Create: `configs/harbin_128.yaml`（已存在，需加入 `highres` source）
- Create: `configs/haidian_128.yaml`
- Modify: `configs/base.yaml`（加入 `highres_sar` sensor_channels 与 target head）

- [ ] **Step 1: 更新 base.yaml 支持高分辨率 SAR source**

在 `configs/base.yaml` 中：

```yaml
model:
  embed_dim: 64
  sensor_channels:
    s2: 12
    s1: 2
    landsat: 6
    highres: 3
    highres_sar: 1
  target_heads:
    s2_recon:
      loss_type: continuous
      channels: 12
      weight: 1.0
    s1_recon:
      loss_type: continuous
      channels: 2
      weight: 1.0
    landsat_recon:
      loss_type: continuous
      channels: 6
      weight: 1.0
    highres_recon:
      loss_type: continuous
      channels: 3
      weight: 0.5
    worldcover:
      loss_type: categorical
      channels: 11
      weight: 0.5
```

- [ ] **Step 2: 哈尔滨 128 配置**

确保 `configs/harbin_128.yaml` 内容：

```yaml
_base_: base.yaml

experiment:
  name: harbin_128_v1

model:
  spatial_stride: 2

training:
  epochs: 100
  lr: 1.0e-4
  weight_decay: 0.05
  warmup_epochs: 5
  gradient_accumulation_steps: 1
  save_every: 10
  eval_every: 10

data:
  region: harbin_128
  root: /data/xuannv_embedding/processed/harbin_128/scenes
  manifest_path: /data/xuannv_embedding/processed/harbin_128/scenes/manifest.json
  statistics_dir: /data/xuannv_embedding/statistics/harbin_128
  patch_size: 128
  num_samples: 424
  batch_size: 4
  max_patches: 424
  sources:
    - s2
    - s1
    - landsat
    - highres
```

- [ ] **Step 3: 海淀区 128 配置**

创建 `configs/haidian_128.yaml`：

```yaml
_base_: base.yaml

experiment:
  name: haidian_128_v1

model:
  spatial_stride: 2

training:
  epochs: 100
  lr: 1.0e-4
  weight_decay: 0.05
  warmup_epochs: 5
  gradient_accumulation_steps: 1
  save_every: 10
  eval_every: 10

data:
  region: haidian_128
  root: /data/xuannv_embedding/processed/haidian_128/scenes
  manifest_path: /data/xuannv_embedding/processed/haidian_128/scenes/manifest.json
  statistics_dir: /data/xuannv_embedding/statistics/haidian_128
  patch_size: 128
  num_samples: 320
  batch_size: 4
  max_patches: 320
  sources:
    - s2
    - s1
    - landsat
    - highres
    - highres_sar
```

- [ ] **Step 4: 验证配置可加载**

```bash
cd /root/workspace/xuannv
python - <<'PY'
from xuannv_embedding.config import Config
for p in ["configs/harbin_128.yaml", "configs/haidian_128.yaml"]:
    cfg = Config.from_yaml(p)
    print(p, cfg.experiment.name, cfg.data.sources, cfg.model.spatial_stride)
PY
```

Expected: 两个配置均成功加载，source 列表与上述一致。

- [ ] **Step 5: 提交配置**

```bash
cd /root/workspace/xuannv
git add configs/base.yaml configs/harbin_128.yaml configs/haidian_128.yaml
git commit -m "feat(config): add haidian_128 config and highres_sar support"
git push origin main
```

---

### Task 7: 修复高分辨率 source 在训练入口的拼接逻辑

**Files:**
- Modify: `src/xuannv_embedding/training/batch_preparation.py`
- Modify: `scripts/train/train.py`

**Context:** 当前 `prepare_batch` 把 `highres` 从时序源中抽出并聚合为单帧。若同时存在 `highres` 与 `highres_sar`，需分别处理。

- [ ] **Step 1: 修改 `prepare_batch` 支持多个高分辨率 source**

在 `prepare_batch` 中，把所有 `source_name.startswith("highres")` 的 source 从时序源中分离，分别生成 `highres_frame_{name}` 与 `highres_mask_{name}`。为保持接口兼容，默认保留 `highres_frame` / `highres_mask`（对应 `highres` source），新增 `highres_frames: dict[str, Tensor]` 字段。

- [ ] **Step 2: 修改 `Trainer._forward` 读取高分辨率帧**

`Trainer._forward` 当前读取 `batch["highres_frame"]` 与 `batch["highres_mask"]`。保留该行为，并增加：

```python
highres_frames = batch.get("highres_frames", {})
```

目前 AEFModel 只接受单个 `highres_frame`，因此 `highres_sar` 可暂不参与 forward（mask 全 0 或作为后续扩展）。

- [ ] **Step 3: 运行测试**

```bash
cd /root/workspace/xuannv
pytest tests/test_train_entry.py tests/test_training.py -v
```

Expected: 全部通过。

- [ ] **Step 4: 提交**

```bash
cd /root/workspace/xuannv
git add src/xuannv_embedding/training/batch_preparation.py scripts/train/train.py
git commit -m "feat(train): support multiple highres sources in batch preparation"
git push origin main
```

---

### Task 8: 运行哈尔滨/海淀区正式训练

**Files:**
- Modify: `docs/data_status/2026-06-18_harbin_sample.md`
- Modify: `docs/data_status/2026-06-18_haidian_sample.md`

- [ ] **Step 1: 哈尔滨 128 训练**

```bash
cd /root/workspace/xuannv
python scripts/train/train.py --config configs/harbin_128.yaml --device npu:0
```

Expected: 训练跑完 100 epochs，每 10 epoch 保存 checkpoint 到 `/data/xuannv_embedding/outputs/harbin_128_v1/`。

- [ ] **Step 2: 海淀区 128 训练**

```bash
cd /root/workspace/xuannv
python scripts/train/train.py --config configs/haidian_128.yaml --device npu:0
```

Expected: 同上，输出到 `/data/xuannv_embedding/outputs/haidian_128_v1/`。

- [ ] **Step 3: 提交训练状态**

```bash
cd /root/workspace/xuannv
git add docs/data_status/
git commit -m "docs: record harbin/haidian training completion"
git push origin main
```

---

### Task 9: 提取 embedding

**Files:**
- Modify: `docs/data_status/2026-06-18_harbin_sample.md`
- Modify: `docs/data_status/2026-06-18_haidian_sample.md`

- [ ] **Step 1: 提取哈尔滨 embedding**

```bash
cd /root/workspace/xuannv
python scripts/eval/extract_embeddings.py \
    --config configs/harbin_128.yaml \
    --checkpoint /data/xuannv_embedding/outputs/harbin_128_v1/epoch_99.pt \
    --output /data/xuannv_embedding/outputs/harbin_128_v1/embeddings_train.npz \
    --device npu:0
```

Expected: 生成 `.npz` 文件，包含 `embeddings`、`embedding_maps`、`patch_ids`。

- [ ] **Step 2: 提取海淀区 embedding**

```bash
cd /root/workspace/xuannv
python scripts/eval/extract_embeddings.py \
    --config configs/haidian_128.yaml \
    --checkpoint /data/xuannv_embedding/outputs/haidian_128_v1/epoch_99.pt \
    --output /data/xuannv_embedding/outputs/haidian_128_v1/embeddings_train.npz \
    --device npu:0
```

Expected: 同上。

---

### Task 10: 下游评估（变化检测 / KNN 分类）

**Files:**
- Create: `docs/eval_results/2026-06-18_eval_report.md`

- [ ] **Step 1: 双时相 embedding 变化检测 AUC**

使用不同月份的 embedding 计算 AUC（需构造 labels，可用差分阈值作为伪标签）：

```bash
cd /root/workspace/xuannv
python scripts/eval/change_detection_eval.py \
    --before /data/xuannv_embedding/outputs/harbin_128_v1/embeddings_202501.npz \
    --after /data/xuannv_embedding/outputs/harbin_128_v1/embeddings_202502.npz \
    --label /data/xuannv_embedding/outputs/harbin_128_v1/labels_change.npz \
    --output /data/xuannv_embedding/outputs/harbin_128_v1/change_detection_auc.npz
```

注意：若还没有真实变化标签，可先使用 embedding 差分的中位数作为伪标签跑通脚本。

- [ ] **Step 2: KNN 分类评估**

若已有 WorldCover 或 JRC Water 标签，可运行：

```bash
cd /root/workspace/xuannv
python scripts/eval/knn_eval.py \
    --embedding-train /data/xuannv_embedding/outputs/harbin_128_v1/embeddings_train.npz \
    --label-train /data/xuannv_embedding/outputs/harbin_128_v1/labels_train.npz \
    --embedding-test /data/xuannv_embedding/outputs/harbin_128_v1/embeddings_val.npz \
    --label-test /data/xuannv_embedding/outputs/harbin_128_v1/labels_val.npz \
    --k 5 \
    --output /data/xuannv_embedding/outputs/harbin_128_v1/knn_metrics.npz
```

- [ ] **Step 3: 撰写评估报告并提交**

创建 `docs/eval_results/2026-06-18_eval_report.md`，记录 AUC 与 KNN 指标，然后：

```bash
cd /root/workspace/xuannv
git add docs/eval_results/
git commit -m "docs: add downstream evaluation report"
git push origin main
```

---

## Self-Review

**1. Spec coverage：** chat1 中要求的核心目标均已覆盖：
- 高分辨率数据引入 → Task 2/3/4/7
- 哈尔滨/海淀区月度 embedding → Task 6/8/9
- 下游任务评估 → Task 10
- 数据/代码目录分离 → 所有输出限定在 `/data/xuannv_embedding/`，代码在 `/root/workspace/xuannv`

**2. Placeholder scan：** 本计划使用 `/玄女科技/` 下的示例文件名（如 `beijing_rs_psscene_analytic_sr_udm2.zip`）作为占位，实际执行前需用 `BaiduPCS-Go ls` 替换为真实远程路径。

**3. Type consistency：** `highres_sar` 作为新 source 名称需在 `base.yaml`、`configs/haidian_128.yaml`、`batch_preparation.py` 中保持一致。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-18-complete-highres-data-and-final-training.md`.

Two execution options:

1. **Subagent-Driven (recommended)** - 我按 Task 逐个 dispatch 子代理实现，每个 Task 后做 spec + quality review。
2. **Inline Execution** - 我在当前会话中按 Task 顺序直接执行。

你确认后我开始执行。
