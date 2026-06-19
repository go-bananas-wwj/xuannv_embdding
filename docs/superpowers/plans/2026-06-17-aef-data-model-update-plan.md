# AEF 数据与模型更新实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完成哈尔滨与海淀全量数据的多源 patch 生成与 manifest 更新，并将 AEF 模型改造为逐月 embedding、无 L2 归一化 bottleneck、高分辨率原生分辨率独立模态，最终跑通双区域单模型最小规模训练。

**Architecture:** 数据层补齐 S1/Landsat/高分辨率 patch 与逐月可用性 manifest；模型层用 `MonthlyEmbeddingModule` 替换全局时序池化，用 `LearnedBottleneck` 替换 vMF，用原生分辨率 highres encoder + `AvailabilityAwareFusion` 替换同尺寸假设；训练层改为逐月 masked loss + 多 NPU DDP/AMP/checkpointing。

**Tech Stack:** Python 3.11, PyTorch, torch_npu, rioxarray/rasterio, xarray, pytest, DDP.

---

## File Structure

| 文件 | 职责 |
|------|------|
| `src/xuannv_embedding/models/bottleneck.py` | `LearnedBottleneck`（无 L2 归一化） |
| `src/xuannv_embedding/models/blocks.py` | `MonthlyEmbeddingModule` + STP 组件 |
| `src/xuannv_embedding/models/sensor_encoders.py` | `NativeResolutionHighResEncoder` |
| `src/xuannv_embedding/models/highres_fusion.py` | availability-aware 融合（处理不同尺寸） |
| `src/xuannv_embedding/models/decoders.py` | 支持时序维度的连续/分类解码头 |
| `src/xuannv_embedding/models/model.py` | 整合逐月 embedding 与高分辨率分支的 AEFModel |
| `src/xuannv_embedding/training/batch_preparation.py` | 构造月度 target 与 highres 输入 |
| `src/xuannv_embedding/training/loss.py` | 逐月 masked 损失 |
| `src/xuannv_embedding/training/trainer.py` | 多 NPU 训练循环与优化 |
| `src/xuannv_embedding/data/dataset.py` | 月度窗口采样 |
| `src/xuannv_embedding/config.py` | 新增配置项 |
| `scripts/data/generate_manifest.py` | 多源逐月 manifest |
| `scripts/data/preprocess.py` | 切 patch 主流程 |
| `configs/base.yaml` / `harbin_128.yaml` / `haidian_128.yaml` | 训练配置 |
| `tests/test_model.py` | 模型输出形状与归一化测试 |
| `tests/test_batch_preparation.py` | 月度 target 构造测试 |
| `tests/test_train_entry.py` | 训练入口冒烟测试 |

---

## Task 1: S1 / Landsat Patch 生成

**Files:**
- Modify: `scripts/data/preprocess.py`
- Modify: `scripts/data/preprocess_harbin.json`, `scripts/data/preprocess_haidian.json`
- Test: 检查 `processed/*/patches/s1/` 与 `processed/*/patches/landsat/` 存在

- [ ] **Step 1: 确认 S1 / Landsat 原始数据路径**

在数据盘确认：
- `raw/harbin/s1/`, `raw/harbin/landsat/`
- `raw/haidian/s1/`, `raw/haidian/landsat/`

若缺失，先执行对应的下载脚本。

- [ ] **Step 2: 在预处理配置中新增 s1 / landsat 条目**

示例（`scripts/data/preprocess_harbin.json`）：

```json
{
  "region": "harbin",
  "patch_size": 128,
  "resolution": 10,
  "sources": {
    "s2": {"raw_glob": "raw/harbin/s2/**/*.nc", "bands": 13},
    "s1": {"raw_glob": "raw/harbin/s1/**/*.tif", "bands": 2},
    "landsat": {"raw_glob": "raw/harbin/landsat/**/*.tif", "bands": 11}
  }
}
```

- [ ] **Step 3: 运行 S1 / Landsat 切 patch**

```bash
cd /root/workspace/xuannv
python scripts/data/preprocess.py --config scripts/data/preprocess_harbin.json --source s1
python scripts/data/preprocess.py --config scripts/data/preprocess_harbin.json --source landsat
python scripts/data/preprocess.py --config scripts/data/preprocess_haidian.json --source s1
python scripts/data/preprocess.py --config scripts/data/preprocess_haidian.json --source landsat
```

- [ ] **Step 4: 验证输出目录与数量**

```bash
find processed/harbin/patches/s1 -name '*.tif' | wc -l
find processed/harbin/patches/landsat -name '*.tif' | wc -l
find processed/haidian/patches/s1 -name '*.tif' | wc -l
find processed/haidian/patches/landsat -name '*.tif' | wc -l
```

- [ ] **Step 5: Commit & push**

```bash
git add scripts/data/preprocess_*.json
# 不提交数据，只提交脚本/配置
git commit -m "data: add s1/landsat preprocessing configs for harbin and haidian"
git push origin feat/data-and-model-rework
```

---

## Task 2: 高分辨率 Patch 生成

**Files:**
- Create: `scripts/data/preprocess_highres.py`
- Create: `scripts/data/preprocess_highres_harbin.json`
- Create: `scripts/data/preprocess_highres_haidian.json`
- Test: 检查 `processed/*/patches/highres_optical/` 与 `processed/*/patches/highres_sar/`

- [ ] **Step 1: 编写高分辨率切 patch 脚本**

`scripts/data/preprocess_highres.py` 核心逻辑：

```python
import argparse, json, pathlib
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import from_origin


def crop_to_patch(input_path, output_path, bounds, crs, target_res):
    with rasterio.open(input_path) as src:
        if src.crs.to_epsg() != crs:
            raise ValueError(f"CRS mismatch: {src.crs} vs EPSG:{crs}")
        win = from_bounds(*bounds, transform=src.transform)
        data = src.read(window=win)
        h, w = data.shape[-2:]
        transform = from_origin(bounds[0], bounds[3], target_res, target_res)
        profile = src.profile
        profile.update({"height": h, "width": w, "transform": transform, "crs": src.crs})
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)
```

- [ ] **Step 2: 配置高分辨率数据源**

`scripts/data/preprocess_highres_harbin.json`：

```json
{
  "region": "harbin",
  "lowres_patch_size": 128,
  "lowres_resolution": 10,
  "sources": {
    "highres_optical": {
      "inputs": [
        "raw/harbin/highres_optical/_mosaic/dom_202512.tif",
        "raw/harbin/highres_optical/_mosaic/dom_202605.tif"
      ],
      "month": [202512, 202605]
    }
  }
}
```

- [ ] **Step 3: 运行高分辨率切 patch**

```bash
python scripts/data/preprocess_highres.py --config scripts/data/preprocess_highres_harbin.json
python scripts/data/preprocess_highres.py --config scripts/data/preprocess_highres_haidian.json
```

- [ ] **Step 4: 验证输出**

```bash
find processed/harbin/patches/highres_optical -name '*.tif' | wc -l
find processed/haidian/patches/highres_optical -name '*.tif' | wc -l
```

- [ ] **Step 5: Commit & push**

```bash
git add scripts/data/preprocess_highres.py scripts/data/preprocess_highres_*.json
git commit -m "data: add native-resolution highres patch cropping"
git push origin feat/data-and-model-rework
```

---

## Task 3: 多源逐月 Manifest

**Files:**
- Modify: `scripts/data/generate_manifest.py`
- Modify: `src/xuannv_embedding/utils/manifest.py`（如有需要）
- Test: `tests/test_manifest.py`

- [ ] **Step 1: 扩展 manifest 条目格式**

每个 patch 记录：

```python
{
    "patch_id": str,
    "region": str,
    "crs": str,
    "bounds": [xmin, ymin, xmax, ymax],
    "sources": {
        source_name: {"months": List[int], "count": int}
    }
}
```

- [ ] **Step 2: 修改 generate_manifest.py 扫描所有 source**

```python
SOURCES = ["s2", "s1", "landsat", "highres_optical", "highres_sar"]

for patch_id in all_patch_ids:
    entry = {"patch_id": patch_id, ...}
    for source in SOURCES:
        months = extract_months_from_patch_dir(source, patch_id)
        entry["sources"][source] = {"months": sorted(months), "count": len(months)}
```

- [ ] **Step 3: 重新生成 manifest**

```bash
python scripts/data/generate_manifest.py --regions harbin,haidian
```

- [ ] **Step 4: 运行测试**

```bash
pytest tests/test_manifest.py -v
```

Expected: PASS

- [ ] **Step 5: Commit & push**

```bash
git add scripts/data/generate_manifest.py src/xuannv_embedding/utils/manifest.py tests/test_manifest.py
git commit -m "data: multi-source monthly manifest generation"
git push origin feat/data-and-model-rework
```

---

## Task 4: 无 L2 归一化 Bottleneck

**Files:**
- Modify: `src/xuannv_embedding/models/bottleneck.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: 编写 LearnedBottleneck**

```python
import torch
import torch.nn as nn


class LearnedBottleneck(nn.Module):
    """无单位球面约束的 learned bottleneck。

    使用 1x1 卷积投影到目标维度，并通过 GroupNorm 稳定训练。
    不调用 F.normalize，允许模型学习 embedding 的幅度。
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        num_groups = 8 if out_dim % 8 == 0 else out_dim
        self.norm = nn.GroupNorm(num_groups, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))
```

- [ ] **Step 2: 删除 VMFBottleneck 导入并替换**

在 `src/xuannv_embedding/models/model.py` 中：

```python
from xuannv_embedding.models.bottleneck import LearnedBottleneck
```

- [ ] **Step 3: 测试输出不再为单位范数**

```python
def test_learned_bottleneck_no_unit_norm():
    import torch
    from xuannv_embedding.models.bottleneck import LearnedBottleneck
    m = LearnedBottleneck(16, 8)
    x = torch.randn(2, 16, 4, 4)
    y = m(x)
    norms = y.norm(dim=1).mean()
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)
```

Run: `pytest tests/test_model.py::test_learned_bottleneck_no_unit_norm -v`
Expected: PASS

- [ ] **Step 4: Commit & push**

```bash
git add src/xuannv_embedding/models/bottleneck.py tests/test_model.py
git commit -m "feat: replace VMF bottleneck with unconstrained LearnedBottleneck"
git push origin feat/data-and-model-rework
```

---

## Task 5: 逐月 Embedding 模块

**Files:**
- Modify: `src/xuannv_embedding/models/blocks.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: 编写 MonthlyEmbeddingModule**

```python
import torch
import torch.nn as nn


class MonthlyEmbeddingModule(nn.Module):
    """将 STP 编码器的逐观测特征聚合到逐月 embedding 网格。

    输入: (B, T_obs, H, W, C), timestamps (B, T_obs), mask (B, T_obs)
    输出: (B, T_month, H, W, embed_dim), monthly_mask (B, T_month)
    """

    def __init__(self, feature_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.month_proj = nn.Linear(feature_dim, embed_dim, bias=False)
        self.missing_token = nn.Parameter(torch.zeros(1, 1, 1, 1, embed_dim))

    def forward(
        self,
        feats: torch.Tensor,
        timestamps: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, H, W, C = feats.shape
        if mask is None:
            mask = torch.ones(B, T, device=feats.device, dtype=torch.float32)

        # timestamps 为整数月份，例如 202501
        t_min = int(timestamps[mask.bool()].min().item())
        t_max = int(timestamps[mask.bool()].max().item())
        months = self._month_range(t_min, t_max)
        T_month = len(months)

        monthly = torch.zeros(B, T_month, H, W, C, device=feats.device, dtype=feats.dtype)
        counts = torch.zeros(B, T_month, device=feats.device, dtype=torch.float32)

        for b in range(B):
            for t_idx, ts in enumerate(timestamps[b]):
                if mask[b, t_idx] == 0:
                    continue
                m_idx = months.index(int(ts.item()))
                monthly[b, m_idx] += feats[b, t_idx]
                counts[b, m_idx] += 1.0

        counts = counts.clamp(min=1.0)
        monthly = monthly / counts.view(B, T_month, 1, 1, 1)

        monthly_mask = (counts > 0).float()
        missing = self.missing_token.expand(B, T_month, H, W, self.embed_dim)
        monthly = torch.where(monthly_mask.view(B, T_month, 1, 1, 1) > 0, monthly, missing)

        monthly = self.month_proj(monthly)
        return monthly, monthly_mask

    @staticmethod
    def _month_range(start: int, end: int) -> list[int]:
        months = []
        y, m = start // 100, start % 100
        while y * 100 + m <= end:
            months.append(y * 100 + m)
            m += 1
            if m > 12:
                m = 1
                y += 1
        return months
```

> 注意：上述逐样本 Python 循环仅用于最小验证。后续可优化为向量化实现。

- [ ] **Step 2: 运行形状测试**

```python
def test_monthly_embedding_shape():
    from xuannv_embedding.models.blocks import MonthlyEmbeddingModule
    import torch
    m = MonthlyEmbeddingModule(feature_dim=64, embed_dim=32)
    feats = torch.randn(2, 5, 4, 4, 64)
    timestamps = torch.tensor([[202501, 202502, 202503, 202504, 202505],
                               [202501, 202503, 202505, 202507, 202509]])
    mask = torch.ones(2, 5)
    out, m_mask = m(feats, timestamps, mask)
    assert out.shape == (2, 9, 4, 4, 32)
    assert m_mask.shape == (2, 9)
```

Run: `pytest tests/test_model.py::test_monthly_embedding_shape -v`
Expected: PASS

- [ ] **Step 3: Commit & push**

```bash
git add src/xuannv_embedding/models/blocks.py tests/test_model.py
git commit -m "feat: add MonthlyEmbeddingModule for monthly embedding grid"
git push origin feat/data-and-model-rework
```

---

## Task 6: 高分辨率原生分辨率编码器

**Files:**
- Modify: `src/xuannv_embedding/models/sensor_encoders.py`
- Modify: `src/xuannv_embedding/models/highres_fusion.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: 新增 NativeResolutionHighResEncoder**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class NativeResolutionHighResEncoder(nn.Module):
    """将任意尺寸的高分辨率影像编码为固定 base 网格的特征。

    输入: (B, C, H_native, W_native)
    输出: (B, out_dim, H_base, W_base)
    """

    def __init__(self, in_channels: int, out_dim: int, base_size: int = 128) -> None:
        super().__init__()
        self.base_size = base_size
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(8, 64)
        self.conv3 = nn.Conv2d(64, out_dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm3 = nn.GroupNorm(min(8, out_dim), out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.norm1(self.conv1(x)))
        x = F.gelu(self.norm2(self.conv2(x)))
        x = F.gelu(self.norm3(self.conv3(x)))
        x = F.adaptive_avg_pool2d(x, (self.base_size, self.base_size))
        return x
```

- [ ] **Step 2: 修改 SensorEncoderBank 支持 highres 编码器类型**

在 `SensorEncoderBank.__init__` 中根据 source 名称选择 encoder：

```python
if source.startswith("highres"):
    encoder = NativeResolutionHighResEncoder(in_channels, out_channels, base_size=128)
else:
    encoder = SensorEncoder(in_channels, out_channels, spatial_stride)
```

- [ ] **Step 3: 测试高分辨率编码器输出尺寸**

```python
def test_native_resolution_highres_encoder():
    from xuannv_embedding.models.sensor_encoders import NativeResolutionHighResEncoder
    import torch
    m = NativeResolutionHighResEncoder(3, 64)
    x = torch.randn(2, 3, 2560, 2560)
    y = m(x)
    assert y.shape == (2, 64, 128, 128)
```

Run: `pytest tests/test_model.py::test_native_resolution_highres_encoder -v`
Expected: PASS

- [ ] **Step 4: Commit & push**

```bash
git add src/xuannv_embedding/models/sensor_encoders.py tests/test_model.py
git commit -m "feat: native-resolution highres encoder with adaptive pooling to base grid"
git push origin feat/data-and-model-rework
```

---

## Task 7: 逐月解码器头

**Files:**
- Modify: `src/xuannv_embedding/models/decoders.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: 修改 ContinuousDecoder 支持时间维度**

```python
class ContinuousDecoder(nn.Module):
    def __init__(self, in_dim: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:  # (B, T, C, H, W)
            B, T, C, H, W = x.shape
            x = x.reshape(B * T, C, H, W)
            x = self.conv(x)
            return x.reshape(B, T, -1, H, W)
        # fallback (B, C, H, W)
        return self.conv(x)
```

`CategoricalDecoder` 类似，最后输出通道数为类别数。

- [ ] **Step 2: 测试逐月解码器输出形状**

```python
def test_temporal_continuous_decoder():
    from xuannv_embedding.models.decoders import ContinuousDecoder
    import torch
    m = ContinuousDecoder(64, 13)
    x = torch.randn(2, 12, 64, 16, 16)
    y = m(x)
    assert y.shape == (2, 12, 13, 16, 16)
```

Run: `pytest tests/test_model.py::test_temporal_continuous_decoder -v`
Expected: PASS

- [ ] **Step 3: Commit & push**

```bash
git add src/xuannv_embedding/models/decoders.py tests/test_model.py
git commit -m "feat: make decoders temporal-aware for monthly outputs"
git push origin feat/data-and-model-rework
```

---

## Task 8: AEFModel 整合

**Files:**
- Modify: `src/xuannv_embedding/models/model.py`
- Test: `tests/test_model.py`

- [ ] **Step 1: 替换 TemporalSummarizer 为 MonthlyEmbeddingModule**

```python
from xuannv_embedding.models.blocks import (
    EmbeddingUpsampleHead,
    MonthlyEmbeddingModule,
    STPEncoder,
)
from xuannv_embedding.models.bottleneck import LearnedBottleneck
```

```python
self.monthly_embed = MonthlyEmbeddingModule(
    feature_dim=stp_cfg["precision_dim"],
    embed_dim=embed_dim,
)
```

- [ ] **Step 2: 修改 forward 流程**

```python
# STP 编码器
feats, input_size = self.stp_encoder(temporal_input, timestamps, mask=combined_mask)
# feats: (B, T_obs, H/2, W/2, precision_dim)

# 逐月 embedding
monthly_emb, monthly_mask = self.monthly_embed(feats, timestamps, mask=combined_mask)
# monthly_emb: (B, T_month, H/2, W/2, embed_dim)

# 逐月上采样到原始分辨率
B, T_month, Hh, Wh, _ = monthly_emb.shape
monthly_emb_flat = monthly_emb.reshape(B * T_month, Hh, Wh, embed_dim).permute(0, 3, 1, 2)
monthly_emb_up = self.upsample_head(monthly_emb_flat, target_size=input_size)
monthly_emb_up = monthly_emb_up.permute(0, 2, 3, 1).reshape(B, T_month, input_size[0], input_size[1], embed_dim)
base_feat = monthly_emb_up.permute(0, 1, 4, 2, 3)  # (B, T_month, embed_dim, H, W)

# 高分辨率融合（单帧，复制到 T_month）
if highres_frames:
    for source, highres_frame in highres_frames.items():
        highres_feat = self.highres_encoder_bank(highres_frame, source)  # (B, embed_dim, H, W)
        highres_mask = highres_masks[source]
        fused = []
        for t in range(T_month):
            fused_t = self.highres_fusion(base_feat[:, t], highres_feat, highres_mask)
            fused.append(fused_t)
        base_feat = torch.stack(fused, dim=1)  # (B, T_month, embed_dim, H, W)

# bottleneck
emb_map = self.bottleneck(base_feat.reshape(B * T_month, embed_dim, *input_size))
emb_map = emb_map.reshape(B, T_month, embed_dim, *input_size)

# 全局场景 embedding：逐月空间平均
emb = emb_map.mean(dim=[3, 4])  # (B, T_month, embed_dim)

# 解码
reconstructions = {}
for name, decoder in self.decoders.items():
    reconstructions[name] = decoder(emb_map)

return AEFOutput(embedding_map=emb_map, embedding=emb, reconstructions=reconstructions)
```

- [ ] **Step 3: 更新 AEFOutput dataclass 注释**

```python
@dataclass
class AEFOutput:
    embedding_map: torch.Tensor       # (B, T_month, D, H, W)
    embedding: torch.Tensor           # (B, T_month, D)
    reconstructions: dict[str, torch.Tensor]
```

- [ ] **Step 4: 测试 AEFModel 输出形状**

```python
def test_aef_model_monthly_output():
    from xuannv_embedding.models.model import AEFModel
    import torch
    model = AEFModel(
        sensor_channels={"s2": 13, "s1": 2, "landsat": 11},
        embed_dim=64,
        target_heads={"s2_recon": ("continuous", 13), "lulc": ("categorical", 10)},
    )
    source_frames = {
        "s2": torch.randn(2, 5, 13, 32, 32),
        "s1": torch.randn(2, 5, 2, 32, 32),
    }
    source_masks = {k: torch.ones(2, 5) for k in source_frames}
    timestamps = torch.tensor([[202501, 202502, 202503, 202504, 202505],
                               [202501, 202502, 202503, 202504, 202505]])
    out = model(source_frames, source_masks, timestamps)
    assert out.embedding_map.shape == (2, 5, 64, 32, 32)
    assert out.reconstructions["s2_recon"].shape == (2, 5, 13, 32, 32)
```

Run: `pytest tests/test_model.py::test_aef_model_monthly_output -v`
Expected: PASS

- [ ] **Step 5: Commit & push**

```bash
git add src/xuannv_embedding/models/model.py tests/test_model.py
git commit -m "feat: integrate monthly embeddings and native highres into AEFModel"
git push origin feat/data-and-model-rework
```

---

## Task 9: 训练 Batch 准备与逐月 Loss

**Files:**
- Modify: `src/xuannv_embedding/training/batch_preparation.py`
- Modify: `src/xuannv_embedding/training/loss.py`
- Test: `tests/test_batch_preparation.py`

- [ ] **Step 1: 修改 prepare_batch 生成月度 target**

对于每个 target head，不再调用 `_weighted_temporal_mean`，而是保留时序维度：

```python
if source_name in source_frames and source_frames[source_name].shape[1] > 0:
    frames = source_frames[source_name]  # (B, T, C, H, W)
    masks = source_masks[source_name]    # (B, T)
    if loss_type == "continuous":
        target = frames  # (B, T, C, H, W)
        target_mask = masks[:, :, None, None, None].expand_as(target)
    else:
        if frames.shape[2] == 1:
            target = frames[:, :, 0].long()  # (B, T, H, W)
        else:
            target = frames.argmax(dim=2).long()
        target_mask = (target != 0).float()
    targets[head_name] = target
    target_masks[head_name] = target_mask
```

- [ ] **Step 2: 高分辨率处理保留单帧**

高分辨率 source 仍聚合为单帧（因其本身时间稀疏），但记录其原生尺寸：

```python
highres_frame = _weighted_temporal_mean(hr_frames, hr_masks)
avail = (hr_masks.sum(dim=1) > 0).float()
highres_frames[source] = highres_frame
highres_masks[source] = avail[:, None, None, None].expand(-1, 1, *highres_frame.shape[-2:])
```

- [ ] **Step 3: 修改 loss.py 支持逐月 masked loss**

```python
def masked_mse(pred, target, mask):
    diff = (pred - target) ** 2
    return (diff * mask).sum() / mask.sum().clamp(min=1.0)


def masked_cross_entropy(pred, target, mask):
    # pred: (B, T, C, H, W), target: (B, T, H, W)
    B, T, C, H, W = pred.shape
    pred = pred.reshape(B * T, C, H * W).transpose(1, 2)
    target = target.reshape(B * T, H * W)
    loss = F.cross_entropy(pred, target, reduction="none")
    mask = mask.reshape(B * T, H * W)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)
```

- [ ] **Step 4: 测试 batch_preparation**

```python
def test_prepare_batch_temporal_targets():
    from xuannv_embedding.training.batch_preparation import prepare_batch
    import torch
    batch = {
        "patch_ids": ["p1", "p2"],
        "source_frames": {"s2": torch.randn(2, 5, 13, 32, 32)},
        "source_masks": {"s2": torch.ones(2, 5)},
        "timestamps": {"s2": torch.tensor([[202501]*5, [202501]*5])},
    }
    target_heads = {"s2_recon": {"loss_type": "continuous", "channels": 13}}
    out = prepare_batch(batch, target_heads)
    assert out["targets"]["s2_recon"].shape == (2, 5, 13, 32, 32)
```

Run: `pytest tests/test_batch_preparation.py -v`
Expected: PASS

- [ ] **Step 5: Commit & push**

```bash
git add src/xuannv_embedding/training/batch_preparation.py \
        src/xuannv_embedding/training/loss.py \
        tests/test_batch_preparation.py
git commit -m "feat: monthly targets and masked losses"
git push origin feat/data-and-model-rework
```

---

## Task 10: 数据集月度窗口采样

**Files:**
- Modify: `src/xuannv_embedding/data/dataset.py`
- Test: `tests/test_dataset.py`

- [ ] **Step 1: 在 dataset 中实现月度窗口采样**

```python
class AEFDataset(Dataset):
    def __init__(self, manifest, sources, window_months=12, ...):
        self.manifest = manifest
        self.sources = sources
        self.window_months = window_months

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        all_months = sorted({m for s in self.sources for m in entry["sources"].get(s, {}).get("months", [])})
        if len(all_months) <= self.window_months:
            start_month = all_months[0]
        else:
            start_month = random.choice(all_months[:-self.window_months + 1])
        window = self._month_range(start_month, self.window_months)
        # 加载各 source 在窗口内的帧
        ...
```

- [ ] **Step 2: 配置最小验证窗口为 17 个月**

在 `configs/base.yaml` 中：

```yaml
model:
  temporal_window_months: 12  # 默认训练窗口

training:
  use_full_sequence: true  # 最小验证时禁用窗口，使用全部月份
```

- [ ] **Step 3: Commit & push**

```bash
git add src/xuannv_embedding/data/dataset.py tests/test_dataset.py configs/base.yaml
git commit -m "feat: monthly window sampling in dataset"
git push origin feat/data-and-model-rework
```

---

## Task 11: 多 NPU 训练优化

**Files:**
- Modify: `src/xuannv_embedding/training/trainer.py`
- Modify: `src/xuannv_embedding/config.py`
- Modify: `scripts/train/train.py`
- Test: `tests/test_train_entry.py`

- [ ] **Step 1: 在 config 中新增训练参数**

```python
@dataclass
class TrainConfig:
    ...
    use_amp: bool = True
    gradient_checkpointing: bool = True
    accumulate_grad_batches: int = 1
    num_workers: int = 4
    pin_memory: bool = True
```

- [ ] **Step 2: 在 model 中支持 gradient checkpointing**

在 `STPEncoder` 的 `forward` 中：

```python
from torch.utils.checkpoint import checkpoint

for block in self.blocks:
    if self.training and self.use_checkpoint:
        space_features, time_features, precision_features = checkpoint(
            block, space_features, time_features, precision_features, timestamps, mask
        )
    else:
        space_features, time_features, precision_features = block(...)
```

- [ ] **Step 3: 在 trainer 中使用 AMP + DDP**

```python
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_npu.amp import autocast, GradScaler

with autocast(enabled=self.use_amp):
    outputs = self.model(inputs)
    loss = self.compute_loss(outputs, targets)

self.scaler.scale(loss).backward()
self.scaler.step(self.optimizer)
self.scaler.update()
```

- [ ] **Step 4: 启动脚本支持 torchrun**

`scripts/train/train.py` 增加：

```python
import os
import torch.distributed as dist

if __name__ == "__main__":
    dist.init_process_group(backend="hccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    ...
```

启动命令：

```bash
torchrun --nproc_per_node=6 scripts/train/train.py --config configs/base.yaml
```

- [ ] **Step 5: 运行训练入口测试**

```bash
pytest tests/test_train_entry.py -v
```

Expected: PASS

- [ ] **Step 6: Commit & push**

```bash
git add src/xuannv_embedding/training/trainer.py \
        src/xuannv_embedding/config.py \
        scripts/train/train.py \
        tests/test_train_entry.py
git commit -m "feat: multi-NPU training with AMP, DDP, and gradient checkpointing"
git push origin feat/data-and-model-rework
```

---

## Task 12: 配置更新

**Files:**
- Modify: `configs/base.yaml`, `configs/harbin_128.yaml`, `configs/haidian_128.yaml`

- [ ] **Step 1: 更新 base.yaml**

```yaml
model:
  embed_dim: 64
  stem_dim: 32
  sensor_channels:
    s2: 13
    s1: 2
    landsat: 11
    highres_optical: 3
    highres_sar: 1
  target_heads:
    s2_recon:
      loss_type: continuous
      channels: 13
    s1_recon:
      loss_type: continuous
      channels: 2
    landsat_recon:
      loss_type: continuous
      channels: 11
    lulc:
      loss_type: categorical
      channels: 10
  temporal_window_months: 12
  use_amp: true
  gradient_checkpointing: true

training:
  batch_size: 4
  accumulate_grad_batches: 2
  num_workers: 4
  pin_memory: true
  epochs: 10
  use_full_sequence: true  # 最小验证：用全部 17 个月
```

- [ ] **Step 2: 更新 harbin_128.yaml / haidian_128.yaml**

仅覆盖 `data.region` 与 `data.manifest_path`：

```yaml
base: configs/base.yaml
data:
  region: harbin
  manifest_path: processed/harbin/manifest.json
```

- [ ] **Step 3: Commit & push**

```bash
git add configs/base.yaml configs/harbin_128.yaml configs/haidian_128.yaml
git commit -m "config: update model/training configs for monthly multi-source training"
git push origin feat/data-and-model-rework
```

---

## Task 13: 最终集成测试与最小规模训练

**Files:**
- All above
- Test: `pytest`

- [ ] **Step 1: 全量测试**

```bash
pytest tests/ -v
```

Expected: all PASS

- [ ] **Step 2: 生成最终数据清单**

```bash
python scripts/data/update_inventory.py
```

- [ ] **Step 3: 启动最小规模训练**

```bash
torchrun --nproc_per_node=6 scripts/train/train.py --config configs/harbin_128.yaml
```

观察前 100 step：
- 无 OOM
- loss 下降
- 日志中同时出现 harbin 与 haidian 样本

- [ ] **Step 4: Commit & push**

```bash
git add docs/data_inventory.md
git commit -m "docs: update data inventory after full preprocessing"
git push origin feat/data-and-model-rework
```

---

## Self-Review Checklist

- [ ] Spec coverage：每个设计点都有对应任务。
- [ ] Placeholder scan：无 TBD/TODO。
- [ ] Type consistency：`MonthlyEmbeddingModule` 输出 `(B, T_month, H, W, embed_dim)`，`AEFModel` 与 decoder 均按此时序维度处理。
- [ ] Data/代码分离：所有数据操作脚本进 git，实际数据与输出不进 git。
