# 下游任务框架实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `downstreams/` 目录下实现一个可扩展的下游任务评测框架，首个任务为海淀建筑工地像素级分割（5-fold CV + 可选 10/25/50/100% 标签比例），并预留变化检测接口。

**Architecture:** 预训练 AEFModel 离线生成 `embedding_map` 与 `scene_embedding`；下游任务冻结 backbone，在 embedding 上训练轻量任务头（linear probe / UNet / UperNet）；训练、评测、可视化全部通过 CLI 脚本驱动，结果与元数据完整溯源。

**Tech Stack:** PyTorch, torch_npu, rasterio, shapely, scikit-learn, scikit-image, matplotlib, seaborn.

---

## 任务总览

| 任务 | 内容 | 产出 |
|---|---|---|
| 1 | 项目脚手架与依赖 | `downstreams/` 目录、`pyproject.toml` 更新、README |
| 2 | 标注准备 | `prepare_downstream.py`：rar 解压、labelme → mask、label_meta.json |
| 3 | 数据划分 | `split.py` + `create_split.py`：分层 5-fold + 标签比例 |
| 4 | embedding 预生成 | `inference.py` + `precompute_embeddings.py` |
| 5 | EmbeddingDataset | `embedding_dataset.py`：加载 .pt + mask，增强 |
| 6 | 任务头 | base / linear_probe / segmentation_head / classification / change_detection |
| 7 | 评测指标 | `segmentation.py` + `visualization.py`：IoU、F1、AP、可视化 |
| 8 | 任务训练器 | `BaseTask` + `ConstructionSegmentationTask` + 配置 |
| 9 | CLI 与端到端 | `train_task.py` + `visualize_results.py` + smoke test |
| 10 | 文档与提交 | 更新 README、data_layout、CHANGELOG，最终 commit/push |

---

## Task 1: 项目脚手架与依赖

**Files:**
- Create: `downstreams/README.md`
- Create: `downstreams/downstreams/__init__.py`
- Modify: `pyproject.toml`
- Test: `downstreams/tests/test_import.py`

- [ ] **Step 1: 创建目录结构**

```bash
mkdir -p downstreams/{downstreams/{data,heads,tasks,metrics,utils},scripts,configs,tests}
touch downstreams/downstreams/__init__.py
touch downstreams/downstreams/data/__init__.py
touch downstreams/downstreams/heads/__init__.py
touch downstreams/downstreams/tasks/__init__.py
touch downstreams/downstreams/metrics/__init__.py
touch downstreams/downstreams/utils/__init__.py
```

- [ ] **Step 2: 写入 README.md**

```markdown
# xuannv_embedding downstream tasks

本目录存放预训练 `AEFModel` 的下游任务评测框架。

## 安装

```bash
pip install -e ".[downstream]"
```

## 数据准备

```bash
python downstreams/scripts/prepare_downstream.py \
  --task construction_site \
  --region haidian \
  --labelme-rar /data/xuannv_embedding/raw/haidian/labels/6511215751_/haidianlabel.rar \
  --patch-dir /data/xuannv_embedding/processed/haidian/patches/s2 \
  --out-dir /data/xuannv_embedding/downstream/labels/haidian/construction_site
```

## 预生成 embedding

阶段二训练完成后：

```bash
python downstreams/scripts/precompute_embeddings.py \
  --config configs/harbin_128_stage2.yaml \
  --checkpoint /data/xuannv_embedding/outputs/harbin_128_stage2_v1/best.pt \
  --regions haidian harbin \
  --output-root /data/xuannv_embedding/embeddings
```

## 训练 + 评测

```bash
python downstreams/scripts/train_task.py \
  --task construction_segmentation \
  --config downstreams/configs/construction_segmentation.yaml \
  --embedding-root /data/xuannv_embedding/embeddings/20260621_harbin_128_stage2_v1_a1b2c3d \
  --label-root /data/xuannv_embedding/downstream/labels/haidian/construction_site \
  --output-root /data/xuannv_embedding/outputs/downstream/construction_site_haidian_20260621_stage2
```

## 可视化

```bash
python downstreams/scripts/visualize_results.py \
  --output-root /data/xuannv_embedding/outputs/downstream/construction_site_haidian_20260621_stage2 \
  --label-root /data/xuannv_embedding/downstream/labels/haidian/construction_site \
  --rgb-source /data/xuannv_embedding/processed/haidian/patches/s2
```
```

- [ ] **Step 3: 修改 pyproject.toml**

在 `[project]` 段末尾追加 optional dependency 组，并修改 `packages.find`：

```toml
[project.optional-dependencies]
downstream = [
    "matplotlib>=3.7",
    "seaborn>=0.12",
    "scikit-image>=0.21",
    "shapely>=2.0",
]

[tool.setuptools.packages.find]
where = ["src", "downstreams"]
```

- [ ] **Step 4: 验证安装**

```bash
cd /root/workspace/xuannv/.worktrees/feat-data-and-model-rework
pip install -e ".[downstream]"
python -c "import downstreams; print('ok')"
```

- [ ] **Step 5: 写入并运行 import 测试**

```python
# downstreams/tests/test_import.py
def test_import_downstreams():
    import downstreams
    assert downstreams.__file__
```

```bash
python -m pytest downstreams/tests/test_import.py -v
```

- [ ] **Step 6: Commit**

```bash
git add downstreams/ pyproject.toml
git commit -m "chore(downstream): scaffold downstream package, deps and README"
git push origin feat/data-and-model-rework-wt
```

---

## Task 2: 标注准备脚本

**Files:**
- Create: `downstreams/downstreams/data/label_loaders.py`
- Create: `downstreams/scripts/prepare_downstream.py`
- Test: `downstreams/tests/test_label_loaders.py`

- [ ] **Step 1: 实现 label_loaders.py**

```python
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.features import rasterize
from shapely.geometry import Polygon, shape

logger = logging.getLogger(__name__)


def parse_patch_id_from_labelme_name(path: Path) -> str:
    """从 labelme 文件名解析 patch id。

    例：patch_000002_20260430_rgb_uint8.json -> patch_000002
    """
    name = path.stem
    parts = name.split("_")
    if len(parts) < 2 or not parts[0].startswith("patch"):
        raise ValueError(f"无法解析 patch id: {path}")
    return f"{parts[0]}_{parts[1]}"


def load_labelme_shapes(label_path: Path) -> list[dict[str, Any]]:
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("shapes", [])


def rasterize_labelme(
    label_path: Path,
    out_shape: tuple[int, int],
    class_map: dict[str, int] | None = None,
) -> np.ndarray:
    """把 labelme 多边形栅格化为 (H, W) uint8 mask。

    像素值：0=背景，class_map[label]=前景。
    多边形洞/重叠通过 shapely 处理为有效几何后栅格化。
    """
    if class_map is None:
        class_map = {"jiazhudongdi": 1}

    shapes: list[tuple[Any, int]] = []
    for s in load_labelme_shapes(label_path):
        label = s.get("label")
        if label not in class_map:
            continue
        cls = class_map[label]
        points = s.get("points", [])
        if len(points) < 3:
            continue
        geom = Polygon(points)
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty:
            continue
        shapes.append((geom, cls))

    mask = rasterize(
        shapes,
        out_shape=out_shape,
        fill=0,
        default_value=0,
        dtype=np.uint8,
        transform=Affine.identity(),
    )
    return mask


def get_reference_patch_path(patch_dir: Path, patch_id: str) -> Path | None:
    """在 patch_dir 中查找匹配 patch_id 的参考影像。"""
    candidates = sorted(patch_dir.glob(f"*_{patch_id}.tif"))
    if not candidates:
        return None
    return candidates[0]
```

- [ ] **Step 2: 实现 prepare_downstream.py**

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from downstreams.data.label_loaders import (
    get_reference_patch_path,
    parse_patch_id_from_labelme_name,
    rasterize_labelme,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="解压 labelme 标注并生成对齐 mask")
    p.add_argument("--task", required=True)
    p.add_argument("--region", required=True)
    p.add_argument("--labelme-rar", type=Path, required=True)
    p.add_argument("--patch-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--class-map", type=str, default='{"jiazhudongdi": 1}')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    class_map = json.loads(args.class_map)

    out_dir = args.out_dir
    raw_dir = out_dir / "labelme_raw"
    mask_dir = out_dir / "masks"
    raw_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    crs: Any = None
    transform: list[float] | None = None

    # 解压 rar
    logger.info("解压 %s -> %s", args.labelme_rar, raw_dir)
    shutil.rmtree(raw_dir, ignore_errors=True)
    subprocess.run(
        ["bsdtar", "xf", str(args.labelme_rar), "-C", str(raw_dir)],
        check=True,
    )

    label_files = sorted(raw_dir.rglob("*.json"))
    logger.info("找到 %d 个 labelme json", len(label_files))

    processed = 0
    for label_path in label_files:
        try:
            patch_id = parse_patch_id_from_labelme_name(label_path)
        except ValueError as exc:
            logger.warning("跳过 %s: %s", label_path, exc)
            continue

        ref_path = get_reference_patch_path(args.patch_dir, patch_id)
        if ref_path is None:
            logger.warning("找不到参考影像: %s", patch_id)
            continue

        with rasterio.open(ref_path) as src:
            height, width = src.height, src.width
            crs = src.crs.to_string() if src.crs else None
            transform = list(src.transform)

        mask = rasterize_labelme(label_path, (height, width), class_map)
        out_mask = mask_dir / f"{patch_id}.tif"
        with rasterio.open(
            out_mask,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=mask.dtype,
            crs=src.crs,
            transform=src.transform,
        ) as dst:
            dst.write(mask, 1)
        processed += 1

    label_meta = {
        "task": args.task,
        "region": args.region,
        "labelme_rar": str(args.labelme_rar),
        "patch_dir": str(args.patch_dir),
        "class_map": class_map,
        "num_patches": processed,
        "mask_dir": str(mask_dir),
        "crs": crs,
        "transform": transform,
    }
    with open(out_dir / "label_meta.json", "w", encoding="utf-8") as f:
        json.dump(label_meta, f, ensure_ascii=False, indent=2)

    logger.info("完成：处理 %d 张 mask，元数据写入 %s", processed, out_dir / "label_meta.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 测试 label_loaders**

```python
# downstreams/tests/test_label_loaders.py
from pathlib import Path

import numpy as np

from downstreams.data.label_loaders import parse_patch_id_from_labelme_name


def test_parse_patch_id() -> None:
    assert parse_patch_id_from_labelme_name(Path("patch_000002_20260430_rgb_uint8.json")) == "patch_000002"
```

- [ ] **Step 4: 运行测试**

```bash
python -m pytest downstreams/tests/test_label_loaders.py -v
```

- [ ] **Step 5: Commit**

```bash
git add downstreams/
git commit -m "feat(downstream): add labelme-to-mask preparation script"
git push origin feat/data-and-model-rework-wt
```

---

## Task 3: 分层 5-fold 与有限标签划分

**Files:**
- Create: `downstreams/downstreams/data/split.py`
- Create: `downstreams/scripts/create_split.py`
- Test: `downstreams/tests/test_split.py`

- [ ] **Step 1: 实现 split.py**

```python
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


def _positive_ratio(mask_path: Path) -> float:
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    total = mask.size
    if total == 0:
        return 0.0
    return float((mask > 0).sum() / total)


def create_stratified_folds(
    mask_dir: Path,
    n_folds: int = 5,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    """按 mask 正像素比例分层，生成 5-fold。

    每 fold：4 folds 训练，1 fold 测试；训练集内部再切 val_ratio 做验证。
    """
    mask_paths = sorted(mask_dir.glob("*.tif"))
    patch_ids = [p.stem for p in mask_paths]
    ratios = np.array([_positive_ratio(p) for p in mask_paths])

    # 分层：按正像素比例分桶
    bins = np.percentile(ratios, [0, 25, 50, 75, 100])
    strata = np.digitize(ratios, bins[:-1]) - 1

    # 若样本太少或比例过于集中导致唯一 strata 不足 n_folds，则退化为排序后循环分组
    if len(np.unique(strata)) < n_folds:
        order = np.argsort(ratios)
        strata = np.empty_like(order)
        for i, idx in enumerate(order):
            strata[idx] = i % n_folds

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(patch_ids, strata)):
        train_val_ids = [patch_ids[i] for i in train_val_idx]
        test_ids = [patch_ids[i] for i in test_idx]

        np.random.seed(seed + fold_idx)
        n_val = max(1, int(len(train_val_ids) * val_ratio))
        val_indices = np.random.choice(len(train_val_ids), size=n_val, replace=False)
        train_indices = np.setdiff1d(np.arange(len(train_val_ids)), val_indices)

        train_ids = [train_val_ids[i] for i in train_indices]
        val_ids = [train_val_ids[i] for i in val_indices]

        folds.append({"fold": fold_idx, "train": train_ids, "val": val_ids, "test": test_ids})

    # 生成 10/25/50/100% 标签比例子集
    fractions = {"0.1": {}, "0.25": {}, "0.5": {}, "1.0": {}}
    for fold in folds:
        train_ids = fold["train"]
        train_ratios = np.array([ratios[patch_ids.index(pid)] for pid in train_ids])
        train_strata = np.digitize(train_ratios, np.percentile(train_ratios, [0, 50, 100])[:-1]) - 1
        for frac_str, frac in [("0.1", 0.1), ("0.25", 0.25), ("0.5", 0.5), ("1.0", 1.0)]:
            n = max(1, int(len(train_ids) * frac))
            selected = _stratified_sample(train_ids, train_strata, n, seed)
            fractions[frac_str][f"fold_{fold['fold']}"] = selected

    return {
        "seed": seed,
        "n_folds": n_folds,
        "val_ratio": val_ratio,
        "stratify_by": "positive_pixel_ratio",
        "folds": folds,
        "fractions": fractions,
    }


def _stratified_sample(ids: list[str], strata: np.ndarray, n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    selected: list[str] = []
    unique_strata = np.unique(strata)
    per_stratum = max(1, n // len(unique_strata))
    for s in unique_strata:
        candidates = [ids[i] for i in np.where(strata == s)[0]]
        k = min(per_stratum, len(candidates))
        selected.extend(rng.choice(candidates, size=k, replace=False).tolist())
    # 补足到 n
    if len(selected) < n:
        remaining = [i for i in ids if i not in selected]
        k = min(n - len(selected), len(remaining))
        selected.extend(rng.choice(remaining, size=k, replace=False).tolist())
    return selected
```

- [ ] **Step 2: 实现 create_split.py**

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from downstreams.data.split import create_stratified_folds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mask-dir", type=Path, required=True)
    p.add_argument("--out-path", type=Path, required=True)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    split = create_stratified_folds(
        mask_dir=args.mask_dir,
        n_folds=args.n_folds,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)
    logger.info("split 已保存到 %s", args.out_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 测试 split**

```python
# downstreams/tests/test_split.py
from pathlib import Path

import numpy as np
import rasterio

from downstreams.data.split import create_stratified_folds


def test_create_folds(tmp_path: Path) -> None:
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    for i in range(10):
        mask = np.zeros((16, 16), dtype=np.uint8)
        if i % 2 == 0:
            mask[:4, :4] = 1
        with rasterio.open(
            mask_dir / f"patch_{i:06d}.tif",
            "w",
            driver="GTiff",
            height=16,
            width=16,
            count=1,
            dtype=mask.dtype,
            crs=None,
            transform=rasterio.Affine.identity(),
        ) as dst:
            dst.write(mask, 1)

    split = create_stratified_folds(mask_dir, n_folds=5, seed=42)
    assert len(split["folds"]) == 5
    for fold in split["folds"]:
        assert len(fold["train"]) > 0
        assert len(fold["test"]) > 0
        assert set(fold["train"]) & set(fold["test"]) == set()
```

- [ ] **Step 4: 运行测试并 commit**

```bash
python -m pytest downstreams/tests/test_split.py -v
git add downstreams/
git commit -m "feat(downstream): stratified 5-fold split with limited-label fractions"
git push origin feat/data-and-model-rework-wt
```

---

## Task 4: embedding 预生成

**Files:**
- Create: `downstreams/downstreams/inference.py`
- Create: `downstreams/scripts/precompute_embeddings.py`
- Test: `downstreams/tests/test_inference.py`（使用 dummy model）

- [ ] **Step 1: 实现 inference.py**

```python
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from xuannv_embedding.config import Config
from xuannv_embedding.data.collate import collate_fn
from xuannv_embedding.data.dataset import MonthlyEmbeddingDataset
from xuannv_embedding.models.model import AEFModel
from xuannv_embedding.training.batch_preparation import prepare_batch
from xuannv_embedding.utils.device import get_device

logger = logging.getLogger(__name__)


def load_model_for_inference(
    config_path: str | Path,
    checkpoint_path: str | Path | None,
    random_init: bool = False,
) -> tuple[AEFModel, Config, torch.device]:
    cfg = Config.from_yaml(config_path)
    device = get_device()

    aef_target_heads = {
        name: (head_cfg["loss_type"], head_cfg["channels"])
        for name, head_cfg in cfg.model.target_heads.items()
    }
    model = AEFModel(
        sensor_channels=cfg.model.sensor_channels,
        embed_dim=cfg.model.embed_dim,
        target_heads=aef_target_heads,
        stem_dim=cfg.model.stem_dim,
        num_months=cfg.model.num_months,
        stp=cfg.model.stp,
        gradient_checkpointing=False,  # 推理关闭
    )
    model = model.to(device)

    if random_init:
        logger.info("使用随机初始化 AEFModel（无预训练权重）")
    elif checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        logger.info("加载模型: %s", checkpoint_path)
    else:
        raise ValueError("checkpoint_path 与 random_init 不能同时为空")
    model.eval()
    return model, cfg, device


def build_inference_loader(cfg: Config, region: str, split: str = "all") -> DataLoader:
    manifest_path = Path(cfg.data.root).parent / region / "manifest.json"
    statistics_dir = Path(cfg.data.root).parent / "statistics" / region
    dataset = MonthlyEmbeddingDataset(
        manifest_path=manifest_path,
        statistics_dir=statistics_dir,
        sources=cfg.data.sources,
        patch_size=cfg.data.patch_size,
        num_months=cfg.model.num_months,
    )

    target_heads = cfg.model.target_heads

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return prepare_batch(collate_fn(batch), target_heads)

    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        pin_memory=False,
    )


def precompute_embeddings(
    model: AEFModel,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for batch in loader:
            patch_ids = batch["patch_ids"]
            source_frames = {
                k: v.to(device, non_blocking=True) for k, v in batch["source_frames"].items()
            }
            source_masks = {
                k: v.to(device, non_blocking=True) for k, v in batch["source_masks"].items()
            }
            timestamps = batch["timestamps"].to(device)
            highres_frames = None
            highres_masks = None
            if batch.get("highres_frames"):
                highres_frames = {
                    k: v.to(device, non_blocking=True)
                    for k, v in batch["highres_frames"].items()
                }
                highres_masks = {
                    k: v.to(device, non_blocking=True)
                    for k, v in batch["highres_masks"].items()
                }

            output = model(
                source_frames=source_frames,
                source_masks=source_masks,
                timestamps=timestamps,
                highres_frames=highres_frames,
                highres_masks=highres_masks,
            )

            # output.embedding_map: (B, T_month, D, H, W)
            # output.embedding:     (B, T_month, D)
            emb_map = output.embedding_map.cpu()
            scene_emb = output.embedding.cpu()
            ts = batch["timestamps"].cpu()  # (B, T_month)

            for b, patch_id in enumerate(patch_ids):
                patch_dir = output_dir / patch_id
                patch_dir.mkdir(parents=True, exist_ok=True)
                for m in range(emb_map.shape[1]):
                    month_int = int(ts[b, m].item())
                    torch.save(emb_map[b, m], patch_dir / f"{month_int}_embedding_map.pt")
                    torch.save(scene_emb[b, m], patch_dir / f"{month_int}_scene_embedding.pt")


def write_meta_json(
    output_dir: Path,
    checkpoint_path: Path,
    config_path: Path,
    command_line: str,
) -> None:
    import subprocess
    import sys
    from datetime import datetime, timezone

    sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()[:16]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
    ).stdout.strip() != ""

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command_line": command_line,
        "git_commit": commit,
        "git_dirty": dirty,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha,
        "config_path": str(config_path),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch_version": torch.__version__,
        "month_format": "YYYYMM",
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
```

- [ ] **Step 2: 实现 precompute_embeddings.py**

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from downstreams.inference import (
    build_inference_loader,
    load_model_for_inference,
    precompute_embeddings,
    write_meta_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--regions", nargs="+", required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--split", default="all")
    p.add_argument("--suffix", default="")
    p.add_argument("--random-init", action="store_true", help="随机初始化 backbone，生成 random-init 基线 embedding")
    args = p.parse_args()

    if not args.random_init and args.checkpoint is None:
        p.error("--checkpoint 或 --random-init 至少指定一个")

    model, cfg, device = load_model_for_inference(
        args.config, args.checkpoint, random_init=args.random_init
    )

    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    exp_name = cfg.experiment.name
    sha = args.checkpoint.stem[:8] if args.checkpoint else "random"
    init_tag = "_random_init" if args.random_init else ""
    suffix = f"_{args.suffix}" if args.suffix else ""
    out_root = args.output_root / f"{date_str}_{exp_name}_{sha}{init_tag}{suffix}"

    for region in args.regions:
        logger.info("生成 %s embedding", region)
        loader = build_inference_loader(cfg, region, split=args.split)
        region_dir = out_root / region
        precompute_embeddings(model, loader, device, region_dir)

    write_meta_json(out_root, args.checkpoint, args.config, " ".join(sys.argv))
    logger.info("embedding 保存至 %s", out_root)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行 inference 测试**

由于依赖完整 AEFModel 与真实数据，此测试可在阶段二完成后用 smoke 配置实际运行验证；单元测试先跳过 NPU 相关部分。

```bash
# 阶段二完成后执行 smoke 验证：
python downstreams/scripts/precompute_embeddings.py \
  --config configs/harbin_128_stage2_quick.yaml \
  --checkpoint /data/xuannv_embedding/outputs/harbin_128_v1/best.pt \
  --regions haidian \
  --output-root /tmp/test_embeddings \
  --suffix smoke

# random-init 基线 embedding
python downstreams/scripts/precompute_embeddings.py \
  --config configs/harbin_128_stage2_quick.yaml \
  --random-init \
  --regions haidian \
  --output-root /tmp/test_embeddings \
  --suffix random_init
```

- [ ] **Step 4: Commit**

```bash
git add downstreams/
git commit -m "feat(downstream): add embedding precomputation module and script"
git push origin feat/data-and-model-rework-wt
```

---

## Task 5: EmbeddingDataset

**Files:**
- Create: `downstreams/downstreams/data/embedding_dataset.py`
- Test: `downstreams/tests/test_embedding_dataset.py`

- [ ] **Step 1: 实现 embedding_dataset.py**

```python
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import rasterio
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class EmbeddingDataset(Dataset):
    """从预生成 embedding 与 mask 加载下游任务数据。"""

    def __init__(
        self,
        embedding_root: Path,
        label_root: Path,
        patch_ids: list[str],
        month: int | str = 202604,
        augment: bool = False,
    ) -> None:
        self.embedding_root = Path(embedding_root)
        self.label_root = Path(label_root)
        self.patch_ids = patch_ids
        self.month = str(month)
        self.augment = augment
        self.mask_dir = self.label_root / "masks"

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        patch_id = self.patch_ids[idx]
        emb_path = self.embedding_root / patch_id / f"{self.month}_embedding_map.pt"
        mask_path = self.mask_dir / f"{patch_id}.tif"

        if not emb_path.exists():
            raise FileNotFoundError(f"embedding 不存在: {emb_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"mask 不存在: {mask_path}")

        emb = torch.load(emb_path, map_location="cpu", weights_only=True)  # (D, H, W)
        with rasterio.open(mask_path) as src:
            mask = torch.from_numpy(src.read(1)).long()  # (H, W)

        if emb.shape[-2:] != mask.shape:
            raise ValueError(
                f"{patch_id} embedding {emb.shape[-2:]} 与 mask {mask.shape} 尺寸不一致"
            )

        if self.augment:
            emb, mask = self._apply_augment(emb, mask)

        return {
            "embedding_map": emb,
            "mask": mask,
            "patch_id": patch_id,
        }

    def _apply_augment(self, emb: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # 同步水平翻转
        if torch.rand(1).item() > 0.5:
            emb = torch.flip(emb, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        # 同步垂直翻转
        if torch.rand(1).item() > 0.5:
            emb = torch.flip(emb, dims=[-2])
            mask = torch.flip(mask, dims=[-2])
        return emb, mask


def collate_embeddings(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "embedding_map": torch.stack([b["embedding_map"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "patch_ids": [b["patch_id"] for b in batch],
    }
```

- [ ] **Step 2: 测试 EmbeddingDataset**

```python
# downstreams/tests/test_embedding_dataset.py
from pathlib import Path

import numpy as np
import rasterio
import torch

from downstreams.data.embedding_dataset import EmbeddingDataset, collate_embeddings


def test_embedding_dataset(tmp_path: Path) -> None:
    emb_root = tmp_path / "embeddings"
    label_root = tmp_path / "labels"
    mask_dir = label_root / "masks"
    mask_dir.mkdir(parents=True)

    patch_id = "patch_000000"
    (emb_root / patch_id).mkdir(parents=True)
    emb = torch.randn(64, 16, 16)
    torch.save(emb, emb_root / patch_id / "202604_embedding_map.pt")

    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:8, 4:8] = 1
    with rasterio.open(
        mask_dir / f"{patch_id}.tif",
        "w",
        driver="GTiff",
        height=16,
        width=16,
        count=1,
        dtype=mask.dtype,
        crs=None,
        transform=rasterio.Affine.identity(),
    ) as dst:
        dst.write(mask, 1)

    ds = EmbeddingDataset(emb_root, label_root, [patch_id], month=202604)
    sample = ds[0]
    assert sample["embedding_map"].shape == (64, 16, 16)
    assert sample["mask"].shape == (16, 16)

    batch = collate_embeddings([sample, sample])
    assert batch["embedding_map"].shape == (2, 64, 16, 16)
```

- [ ] **Step 3: 运行测试并 commit**

```bash
python -m pytest downstreams/tests/test_embedding_dataset.py -v
git add downstreams/
git commit -m "feat(downstream): add EmbeddingDataset for precomputed embeddings"
git push origin feat/data-and-model-rework-wt
```

---

## Task 6: 任务头

**Files:**
- Create: `downstreams/downstreams/heads/base.py`
- Create: `downstreams/downstreams/heads/linear_probe.py`
- Create: `downstreams/downstreams/heads/segmentation_head.py`
- Create: `downstreams/downstreams/heads/classification_head.py`
- Create: `downstreams/downstreams/heads/change_detection_head.py`
- Test: `downstreams/tests/test_heads.py`

- [ ] **Step 1: 实现 base.py**

```python
from __future__ import annotations

from abc import abstractmethod

import torch
from torch import nn


class TaskHead(nn.Module):
    @abstractmethod
    def forward(self, embedding_map: torch.Tensor, scene_emb: torch.Tensor | None = None) -> torch.Tensor:
        ...
```

- [ ] **Step 2: 实现 linear_probe.py**

```python
from __future__ import annotations

import torch
from torch import nn

from downstreams.heads.base import TaskHead


class LinearProbeHead(TaskHead):
    """严格线性探测：单 1x1 conv，无激活/无 BN。"""

    def __init__(self, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, embedding_map: torch.Tensor, scene_emb: torch.Tensor | None = None) -> torch.Tensor:
        return self.conv(embedding_map)
```

- [ ] **Step 3: 实现 segmentation_head.py**

```python
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from downstreams.heads.base import TaskHead


class FCNHead(TaskHead):
    def __init__(self, embed_dim: int, num_classes: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, scene_emb: torch.Tensor | None = None) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        return self.conv2(x)


class UNetHead(TaskHead):
    """轻量 UNet decoder，只有两层，skip 来自输入本身。"""

    def __init__(self, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.up1 = nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(embed_dim // 2, embed_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(embed_dim // 4, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, scene_emb: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, D, H, W)
        x1 = self.up1(x)  # (B, D/2, 2H, 2W)
        x1 = self.conv1(torch.cat([x1, F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)], dim=1))
        x2 = self.up2(x1)  # (B, D/4, 4H, 4W)
        x2 = self.conv2(torch.cat([x2, F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)], dim=1))
        out = self.final(x2)  # (B, C, 4H, 4W)
        # 下采样回原始尺寸
        return F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)


class UperNetHead(TaskHead):
    """简化版 UperNet：PSP module + fusion conv。"""

    def __init__(self, embed_dim: int, num_classes: int, pool_scales: tuple[int, ...] = (1, 2, 3, 6)) -> None:
        super().__init__()
        self.pool_scales = pool_scales
        self.psp_modules = nn.ModuleList()
        for scale in pool_scales:
            self.psp_modules.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=1, bias=False),
                    nn.BatchNorm2d(embed_dim // 4),
                    nn.ReLU(inplace=True),
                )
            )
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, scene_emb: torch.Tensor | None = None) -> torch.Tensor:
        feats = [x]
        for module in self.psp_modules:
            pooled = module(x)
            pooled = F.interpolate(pooled, size=x.shape[-2:], mode="bilinear", align_corners=False)
            feats.append(pooled)
        fused = torch.cat(feats, dim=1)
        fused = self.fusion(fused)
        return self.classifier(fused)


def build_segmentation_head(head_type: str, embed_dim: int, num_classes: int) -> TaskHead:
    head_type = head_type.lower()
    if head_type == "linear" or head_type == "linear_probe":
        return LinearProbeHead(embed_dim, num_classes)
    if head_type == "fcn":
        return FCNHead(embed_dim, num_classes)
    if head_type == "unet":
        return UNetHead(embed_dim, num_classes)
    if head_type == "upernet":
        return UperNetHead(embed_dim, num_classes)
    raise ValueError(f"未知 head 类型: {head_type}")
```

- [ ] **Step 4: 实现 classification_head.py**

```python
from __future__ import annotations

import torch
from torch import nn

from downstreams.heads.base import TaskHead


class ClassificationHead(TaskHead):
    def __init__(self, embed_dim: int, num_classes: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, embedding_map: torch.Tensor, scene_emb: torch.Tensor) -> torch.Tensor:
        if scene_emb is None:
            raise ValueError("ClassificationHead 需要 scene_emb")
        return self.mlp(scene_emb)
```

- [ ] **Step 5: 实现 change_detection_head.py**

```python
from __future__ import annotations

import torch
from torch import nn

from downstreams.heads.base import TaskHead


class ChangeDetectionHead(TaskHead):
    """双时相变化检测头（预留接口，当前仅实现 concat + 1x1 conv）。"""

    def __init__(self, embed_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.fusion = nn.Conv2d(embed_dim * 2, hidden_dim, kernel_size=1)
        self.classifier = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward_two(self, emb_t1: torch.Tensor, emb_t2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([emb_t1, emb_t2], dim=1)
        x = torch.relu(self.fusion(x))
        return self.classifier(x)

    def forward(self, embedding_map: torch.Tensor, scene_emb: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError("ChangeDetectionHead 请使用 forward_two(t1, t2)")
```

- [ ] **Step 6: 测试 heads**

```python
# downstreams/tests/test_heads.py
import torch

from downstreams.heads.segmentation_head import build_segmentation_head


def test_linear_probe_head() -> None:
    head = build_segmentation_head("linear", embed_dim=64, num_classes=2)
    x = torch.randn(2, 64, 16, 16)
    out = head(x)
    assert out.shape == (2, 2, 16, 16)


def test_upernet_head() -> None:
    head = build_segmentation_head("upernet", embed_dim=64, num_classes=2)
    x = torch.randn(2, 64, 16, 16)
    out = head(x)
    assert out.shape == (2, 2, 16, 16)
```

- [ ] **Step 7: 运行测试并 commit**

```bash
python -m pytest downstreams/tests/test_heads.py -v
git add downstreams/
git commit -m "feat(downstream): add task heads (linear, FCN, UNet, UperNet, classification, change-detection stub)"
git push origin feat/data-and-model-rework-wt
```

---

## Task 7: 评测指标与可视化

**Files:**
- Create: `downstreams/downstreams/metrics/segmentation.py`
- Create: `downstreams/downstreams/metrics/visualization.py`
- Test: `downstreams/tests/test_metrics.py`

- [ ] **Step 1: 实现 segmentation.py**

```python
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve


def _ensure_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def compute_segmentation_metrics(
    pred_logits: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    ignore_index: int = -1,
    return_curve: bool = False,
) -> dict[str, Any]:
    """pred_logits: (N, H, W) 或 (H, W)；target: 同 shape 的 int 标签。"""
    pred_logits = _ensure_numpy(pred_logits)
    target = _ensure_numpy(target)

    if pred_logits.ndim == 2:
        pred_logits = pred_logits[None]
        target = target[None]

    valid = target != ignore_index
    probs = 1.0 / (1.0 + np.exp(-pred_logits))
    preds_05 = (probs > 0.5).astype(np.uint8)

    tp = int(((preds_05 == 1) & (target == 1) & valid).sum())
    fp = int(((preds_05 == 1) & (target == 0) & valid).sum())
    fn = int(((preds_05 == 0) & (target == 1) & valid).sum())
    tn = int(((preds_05 == 0) & (target == 0) & valid).sum())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    intersection = tp
    union = tp + fp + fn
    iou = intersection / (union + 1e-8)

    # PR 曲线 / AP / AUPRC
    y_true = (target[valid] == 1).astype(np.int32)
    y_score = probs[valid]
    p_arr = np.array([])
    r_arr = np.array([])
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        ap = auprc = 0.0
        best_f1 = 0.0
    else:
        p_arr, r_arr, _ = precision_recall_curve(y_true, y_score)
        auprc = auc(r_arr, p_arr)
        # AP via sklearn average_precision_score 更稳
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(y_true, y_score)
        f1s = 2 * p_arr * r_arr / (p_arr + r_arr + 1e-8)
        best_f1 = float(f1s.max())

    result: dict[str, Any] = {
        "miou": float(iou),
        "f1_0.5": float(f1),
        "f1_best": best_f1,
        "precision": float(precision),
        "recall": float(recall),
        "ap": float(ap),
        "auprc": float(auprc),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
    if return_curve:
        result["precision_curve"] = p_arr
        result["recall_curve"] = r_arr
    return result
```

- [ ] **Step 2: 实现可视化**

```python
# downstreams/downstreams/metrics/visualization.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def overlay_prediction(
    rgb: np.ndarray,
    pred: np.ndarray,
    target: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """rgb: (H, W, 3) [0,1]；pred/target: (H, W) bool/uint8。"""
    overlay = rgb.copy()
    overlay[pred > 0] = overlay[pred > 0] * (1 - alpha) + np.array([1.0, 0.0, 0.0]) * alpha
    overlay[target > 0] = overlay[target > 0] * (1 - alpha) + np.array([0.0, 1.0, 0.0]) * alpha
    return np.clip(overlay, 0, 1)


def plot_pr_curve(precision: np.ndarray, recall: np.ndarray, ap: float, out_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"AP={ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path, dpi=150)
    plt.close()
```

- [ ] **Step 3: 测试 metrics**

```python
# downstreams/tests/test_metrics.py
import numpy as np
import torch

from downstreams.metrics.segmentation import compute_segmentation_metrics


def test_perfect_prediction() -> None:
    target = np.array([[0, 1], [1, 0]], dtype=np.int64)
    logits = np.array([[0, 10], [10, 0]], dtype=np.float32)
    m = compute_segmentation_metrics(logits, target)
    assert m["miou"] == 1.0
    assert m["f1_0.5"] == 1.0
```

- [ ] **Step 4: 运行测试并 commit**

```bash
python -m pytest downstreams/tests/test_metrics.py -v
git add downstreams/
git commit -m "feat(downstream): add segmentation metrics and visualization helpers"
git push origin feat/data-and-model-rework-wt
```

---

## Task 8: 任务训练器与配置

**Files:**
- Create: `downstreams/downstreams/tasks/base.py`
- Create: `downstreams/downstreams/tasks/construction_segmentation.py`
- Create: `downstreams/configs/_base_.yaml`
- Create: `downstreams/configs/construction_segmentation.yaml`
- Test: `downstreams/tests/test_task.py`

- [ ] **Step 1: 实现 base.py**

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


class BaseTask(ABC):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def build_head(self) -> nn.Module:
        ...

    @abstractmethod
    def build_loss(self) -> nn.Module:
        ...

    @abstractmethod
    def train_one_epoch(self, model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
        ...

    @abstractmethod
    def evaluate(self, model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
        ...
```

- [ ] **Step 2: 实现 construction_segmentation.py**

```python
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from downstreams.heads.segmentation_head import build_segmentation_head
from downstreams.metrics.segmentation import compute_segmentation_metrics
from downstreams.tasks.base import BaseTask

logger = logging.getLogger(__name__)


def _dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1.0 - (2.0 * intersection + smooth) / (union + smooth)


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha: float = 0.8, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, target.float())
        probs = torch.sigmoid(logits)
        pt = probs * target + (1 - probs) * (1 - target)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean() + _dice_loss(logits, target)


class ConstructionSegmentationTask(BaseTask):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._loss: nn.Module | None = None

    def build_head(self) -> nn.Module:
        return build_segmentation_head(
            self.config["head_type"],
            self.config["embed_dim"],
            self.config["num_classes"],
        )

    def build_loss(self) -> nn.Module:
        if self._loss is None:
            loss_name = self.config.get("loss", "focal_dice").lower()
            if loss_name == "focal_dice":
                self._loss = FocalDiceLoss()
            elif loss_name == "bce":
                pos_weight = torch.tensor(self.config.get("pos_weight", 1.0))
                self._loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                raise ValueError(f"未知 loss: {loss_name}")
        return self._loss

    def train_one_epoch(self, model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
        model.train()
        loss_fn = self.build_loss()
        total_loss = 0.0
        for batch in loader:
            emb = batch["embedding_map"].to(device)
            mask = batch["mask"].to(device)  # (B, H, W)
            optimizer.zero_grad()
            logits = model(emb)[:, 1]  # 二分类只取前景通道
            loss = loss_fn(logits, mask.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def evaluate(self, model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
        model.eval()
        all_logits: list[torch.Tensor] = []
        all_masks: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                emb = batch["embedding_map"].to(device)
                mask = batch["mask"].to(device)
                logits = model(emb)[:, 1]
                all_logits.append(logits.cpu())
                all_masks.append(mask.cpu())
        logits = torch.cat([x.flatten() for x in all_logits])
        targets = torch.cat([x.flatten() for x in all_masks])
        return compute_segmentation_metrics(logits, targets)
```

- [ ] **Step 3: 配置 YAML**

```yaml
# downstreams/configs/_base_.yaml
experiment:
  name: downstream_construction_segmentation
  seed: 42
  device: auto

training:
  epochs: 100
  lr: 1.0e-4
  weight_decay: 1.0e-2
  batch_size: 8
  num_workers: 0
  early_stop_patience: 10
  head_type: upernet
  loss: focal_dice
  pos_weight: 10.0
  label_fraction: 1.0
  month: 202604

data:
  embed_dim: 64
  num_classes: 2
```

```yaml
# downstreams/configs/construction_segmentation.yaml
_base_: _base_.yaml

experiment:
  name: construction_site_haidian

training:
  head_type: upernet
  loss: focal_dice
  lr: 1.0e-4
```

- [ ] **Step 4: 测试 task**

```python
# downstreams/tests/test_task.py
from downstreams.tasks.construction_segmentation import ConstructionSegmentationTask


def test_task_build() -> None:
    cfg = {"head_type": "linear", "embed_dim": 64, "num_classes": 2, "loss": "focal_dice"}
    task = ConstructionSegmentationTask(cfg)
    head = task.build_head()
    assert head is not None
```

- [ ] **Step 5: 运行测试并 commit**

```bash
python -m pytest downstreams/tests/test_task.py -v
git add downstreams/
git commit -m "feat(downstream): add BaseTask and ConstructionSegmentationTask with Focal+Dice loss"
git push origin feat/data-and-model-rework-wt
```

---

## Task 9: 训练与可视化 CLI

**Files:**
- Create: `downstreams/scripts/train_task.py`
- Create: `downstreams/scripts/visualize_results.py`
- Test: end-to-end smoke test

- [ ] **Step 1: 实现 train_task.py**

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from downstreams.data.embedding_dataset import EmbeddingDataset, collate_embeddings
from downstreams.data.split import create_stratified_folds
from downstreams.tasks.construction_segmentation import ConstructionSegmentationTask
from downstreams.utils.device import get_downstream_device
from downstreams.utils.reproducibility import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    base = raw.pop("_base_", None)
    if base:
        base_path = path.parent / base
        with open(base_path, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f)
        base_cfg.update(raw)
        return base_cfg
    return raw


def save_test_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    pred_dir: Path,
    mask_dir: Path,
) -> None:
    """将测试集概率图保存为 GeoTIFF，便于后续可视化与溯源。"""
    model.eval()
    pred_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for batch in loader:
            emb = batch["embedding_map"].to(device)
            patch_ids = batch["patch_ids"]
            logits = model(emb)[:, 1]
            probs = torch.sigmoid(logits).cpu().numpy()
            for b, patch_id in enumerate(patch_ids):
                mask_path = mask_dir / f"{patch_id}.tif"
                with rasterio.open(mask_path) as src:
                    profile = src.profile.copy()
                profile.update(
                    dtype=rasterio.float32,
                    count=1,
                    compress="lzw",
                    nodata=None,
                )
                out_path = pred_dir / f"{patch_id}_prob.tif"
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(probs[b].astype(np.float32), 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="construction_segmentation")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--embedding-root", type=Path, required=True)
    p.add_argument("--label-root", type=Path, required=True)
    p.add_argument(
        "--region",
        type=str,
        default=None,
        help="embedding 子目录名；默认从 label-root 父目录推断",
    )
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--fold", type=int, default=None, help="只跑单个 fold 调试")
    p.add_argument("--fraction", type=float, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["experiment"]["seed"])
    device = get_downstream_device(cfg["experiment"].get("device", "auto"))

    region = args.region if args.region else args.label_root.parent.name
    emb_region_root = args.embedding_root / region
    mask_dir = args.label_root / "masks"
    split_path = args.label_root / "split_5fold.json"
    if not split_path.exists():
        logger.info("split_5fold.json 不存在，自动生成")
        split = create_stratified_folds(mask_dir, seed=cfg["experiment"]["seed"])
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=False, indent=2)
    else:
        with open(split_path, "r", encoding="utf-8") as f:
            split = json.load(f)

    task = ConstructionSegmentationTask(cfg["training"])
    folds = [split["folds"][args.fold]] if args.fold is not None else split["folds"]

    summary = []
    for fold_info in folds:
        fold_idx = fold_info["fold"]
        logger.info("===== Fold %d =====", fold_idx)
        out_dir = args.output_root / f"fold_{fold_idx}"
        out_dir.mkdir(parents=True, exist_ok=True)

        train_ids = fold_info["train"]
        if args.fraction is not None:
            frac_str = str(args.fraction)
            train_ids = split["fractions"][frac_str][f"fold_{fold_idx}"]

        train_ds = EmbeddingDataset(
            emb_region_root,
            args.label_root,
            train_ids,
            month=cfg["training"]["month"],
            augment=True,
        )
        val_ds = EmbeddingDataset(
            emb_region_root,
            args.label_root,
            fold_info["val"],
            month=cfg["training"]["month"],
        )
        test_ds = EmbeddingDataset(
            emb_region_root,
            args.label_root,
            fold_info["test"],
            month=cfg["training"]["month"],
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
            num_workers=cfg["training"].get("num_workers", 0),
            collate_fn=collate_embeddings,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=cfg["training"].get("num_workers", 0),
            collate_fn=collate_embeddings,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=cfg["training"].get("num_workers", 0),
            collate_fn=collate_embeddings,
        )

        model = task.build_head().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["training"]["epochs"]
        )
        loss_fn = task.build_loss()

        best_miou = -1.0
        patience_counter = 0
        best_state: dict[str, torch.Tensor] | None = None
        for epoch in range(cfg["training"]["epochs"]):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                emb = batch["embedding_map"].to(device)
                mask = batch["mask"].to(device)
                optimizer.zero_grad()
                logits = model(emb)[:, 1]
                loss = loss_fn(logits, mask.float())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            scheduler.step()
            train_loss /= len(train_loader)

            val_metrics = task.evaluate(model, val_loader, device)
            logger.info(
                "Epoch %d train_loss=%.4f val_miou=%.4f",
                epoch,
                train_loss,
                val_metrics["miou"],
            )

            if val_metrics["miou"] > best_miou:
                best_miou = val_metrics["miou"]
                patience_counter = 0
                best_state = model.state_dict()
                (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
                torch.save(best_state, out_dir / "checkpoints" / "best.pt")
            else:
                patience_counter += 1
                if patience_counter >= cfg["training"]["early_stop_patience"]:
                    logger.info("早停于 epoch %d", epoch)
                    break

        # 测试
        assert best_state is not None
        model.load_state_dict(best_state)
        test_metrics = task.evaluate(model, test_loader, device)
        test_metrics["fold"] = fold_idx
        test_metrics["best_epoch"] = epoch - patience_counter
        test_metrics["region"] = region
        test_metrics["fraction"] = args.fraction
        summary.append(test_metrics)
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, ensure_ascii=False, indent=2)

        # 保存测试集概率图
        save_test_predictions(
            model, test_loader, device, out_dir / "predictions", mask_dir
        )

    # 汇总
    with open(args.output_root / "summary_5fold.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("5-fold 汇总：%s", args.output_root / "summary_5fold.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 实现 visualize_results.py**

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from downstreams.metrics.segmentation import compute_segmentation_metrics
from downstreams.metrics.visualization import overlay_prediction, plot_pr_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _find_rgb_path(rgb_source: Path, patch_id: str) -> Path | None:
    candidates = sorted(rgb_source.rglob(f"*{patch_id}*.tif"))
    if not candidates:
        candidates = sorted(rgb_source.glob(f"*{patch_id}*.tif"))
    return candidates[0] if candidates else None


def _read_rgb(path: Path) -> np.ndarray | None:
    with rasterio.open(path) as src:
        if src.count < 3:
            return None
        arr = src.read([1, 2, 3])
    arr = np.transpose(arr, (1, 2, 0)).astype(np.float32)
    for i in range(3):
        band = arr[..., i]
        mn, mx = np.percentile(band, [2, 98])
        if mx > mn:
            arr[..., i] = np.clip((band - mn) / (mx - mn), 0, 1)
        else:
            arr[..., i] = np.zeros_like(band)
    return arr


def _collect_fold_probs(
    pred_dir: Path, mask_dir: Path
) -> tuple[np.ndarray, np.ndarray] | None:
    probs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for prob_path in sorted(pred_dir.glob("*_prob.tif")):
        patch_id = prob_path.stem.replace("_prob", "")
        mask_path = mask_dir / f"{patch_id}.tif"
        if not mask_path.exists():
            continue
        with rasterio.open(prob_path) as src:
            prob = src.read(1)
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        probs.append(prob.ravel())
        masks.append(mask.ravel())
    if not probs:
        return None
    return np.concatenate(probs), np.concatenate(masks)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--label-root", type=Path, required=True)
    p.add_argument("--rgb-source", type=Path, default=None)
    p.add_argument("--n-samples", type=int, default=10)
    args = p.parse_args()

    mask_dir = args.label_root / "masks"

    for fold_dir in sorted(args.output_root.glob("fold_*")):
        metrics_path = fold_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        logger.info(
            "%s: miou=%.3f f1=%.3f", fold_dir.name, metrics["miou"], metrics["f1_0.5"]
        )

        pred_dir = fold_dir / "predictions"
        vis_dir = fold_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # 汇总 PR 曲线
        collected = _collect_fold_probs(pred_dir, mask_dir)
        if collected is not None:
            probs, masks = collected
            fold_metrics = compute_segmentation_metrics(
                probs, masks, return_curve=True
            )
            plot_pr_curve(
                fold_metrics["precision_curve"],
                fold_metrics["recall_curve"],
                fold_metrics["ap"],
                vis_dir / "pr_curve.png",
            )
            logger.info(
                "%s fold-level AP=%.3f AUPRC=%.3f",
                fold_dir.name,
                fold_metrics["ap"],
                fold_metrics["auprc"],
            )

        # 样本叠加图
        if not pred_dir.exists():
            continue
        for prob_path in sorted(pred_dir.glob("*_prob.tif"))[: args.n_samples]:
            patch_id = prob_path.stem.replace("_prob", "")
            mask_path = mask_dir / f"{patch_id}.tif"
            if not mask_path.exists():
                continue
            with rasterio.open(prob_path) as src:
                prob = src.read(1)
            with rasterio.open(mask_path) as src:
                mask = src.read(1)
            pred = (prob > 0.5).astype(np.uint8)

            rgb: np.ndarray | None = None
            if args.rgb_source:
                rgb_path = _find_rgb_path(args.rgb_source, patch_id)
                if rgb_path:
                    rgb = _read_rgb(rgb_path)
            if rgb is None or rgb.shape[:2] != pred.shape:
                # 退化为概率灰度图
                gray = np.stack([prob] * 3, axis=-1)
                mn, mx = gray[..., 0].min(), gray[..., 0].max()
                if mx > mn:
                    rgb = (gray - mn) / (mx - mn)
                else:
                    rgb = np.zeros_like(gray)

            overlay = overlay_prediction(rgb, pred, mask)
            out_png = vis_dir / f"{patch_id}_overlay.png"
            plt.imsave(out_png, overlay)
            logger.info("保存可视化 %s", out_png)

    logger.info("可视化完成")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 实现 utils 模块**

```python
# downstreams/downstreams/utils/device.py
from __future__ import annotations

import torch

from xuannv_embedding.utils.device import get_device


def get_downstream_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        return get_device()
    return torch.device(preference)
```

```python
# downstreams/downstreams/utils/reproducibility.py
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
```

- [ ] **Step 4: 运行 smoke test**

由于需要真实 embedding，阶段二完成前可手工构造 dummy embedding 跑通 pipeline：

```bash
# 生成 dummy embedding
python - <<'PY'
from pathlib import Path
import torch, numpy as np, rasterio
root = Path('/tmp/ds_smoke/embeddings/haidian')
mask_root = Path('/data/xuannv_embedding/downstream/labels/haidian/construction_site/masks')
for p in mask_root.glob('*.tif'):
    (root / p.stem).mkdir(parents=True, exist_ok=True)
    torch.randn(64, 128, 128).save(root / p.stem / '202604_embedding_map.pt')
PY

# 单 fold smoke
python downstreams/scripts/train_task.py \
  --config downstreams/configs/construction_segmentation.yaml \
  --embedding-root /tmp/ds_smoke/embeddings \
  --label-root /data/xuannv_embedding/downstream/labels/haidian/construction_site \
  --output-root /tmp/ds_smoke/outputs \
  --fold 0 \
  --fraction 0.1
```

- [ ] **Step 5: Commit**

```bash
git add downstreams/
git commit -m "feat(downstream): add train_task and visualize_results CLI"
git push origin feat/data-and-model-rework-wt
```

---

## Task 10: 文档、数据布局与最终提交

**Files:**
- Modify: `downstreams/README.md`
- Modify: `docs/data_layout.md`（若存在，否则创建）
- Modify: `CHANGELOG.md`（若存在，否则创建）

- [ ] **Step 1: 在 `downstreams/README.md` 追加以下内容**

```markdown
## 目录结构

```
downstreams/
├── downstreams/          # Python 包
│   ├── data/             # label loaders / split / EmbeddingDataset
│   ├── heads/            # 任务头（linear / FCN / UNet / UperNet / classification / change-detection stub）
│   ├── metrics/          # 指标与可视化
│   ├── tasks/            # BaseTask / ConstructionSegmentationTask
│   └── utils/            # device / reproducibility
├── scripts/              # CLI
├── configs/              # 下游任务配置
└── tests/                # 单元测试
```

## Mask 规范

- 格式：单波段 `uint8` GeoTIFF，与参考影像同尺寸、同 CRS、同 Affine。
- 像素值：`0`=背景，`1`=建筑工地（`jiazhudongdi`）。
- 命名：`{patch_id}.tif`，例如 `patch_000002.tif`。

## Random-init 基线

```bash
python downstreams/scripts/precompute_embeddings.py \
  --config configs/harbin_128_stage2.yaml \
  --random-init \
  --regions haidian \
  --output-root /data/xuannv_embedding/embeddings

python downstreams/scripts/train_task.py \
  --config downstreams/configs/construction_segmentation.yaml \
  --embedding-root /data/xuannv_embedding/embeddings/YYYYMMDD_xxx_random \
  --label-root /data/xuannv_embedding/downstream/labels/haidian/construction_site \
  --output-root /data/xuannv_embedding/outputs/downstream/construction_site_haidian_random
```

## 常见问题

| 问题 | 原因 | 解决 |
|---|---|---|
| `embedding 不存在` | 月份不匹配 | 确认 `--month` 与预生成目录名一致 |
| `split_5fold.json` 不存在 | 首次运行 | 脚本会自动按正像素比例分层生成 |
| mIoU 接近 0 | 类别极度不平衡 | 检查 `pos_weight` 与 Focal loss 配置 |
```

- [ ] **Step 2: 在 `docs/data_layout.md` 末尾追加**

```markdown
## 下游任务与 Embedding 产物

- **`downstream/labels/<region>/<task>/`**：labelme 原始标注、对齐后的 mask、`split_5fold.json`、`label_meta.json`。
- **`embeddings/YYYYMMDD_<exp>_<sha>/`**：预生成 embedding，含 per-patch `{month}_embedding_map.pt`、`{month}_scene_embedding.pt` 与 `meta.json`。
- **`outputs/downstream/<task>_<region>_<date>_<suffix>/`**：下游训练输出，含 `fold_*/{metrics.json,predictions/,checkpoints/,visualizations/}` 与 `summary_5fold.json`。
```

- [ ] **Step 3: 创建或更新 `CHANGELOG.md`**

若文件不存在，写入：

```markdown
# Changelog

## 2026-06-21

### Added
- 新增 `downstreams/` 下游任务评测框架。
- 支持海淀建筑工地像素级分割（5-fold CV、10/25/50/100% 标签比例）。
- 支持 embedding 离线预生成、元数据溯源与 random-init 基线。
- 提供 linear / FCN / UNet / UperNet 任务头及 Focal + Dice 默认损失。
- 评测指标：mIoU、F1、AP、AUPRC；支持 PR 曲线与预测叠加图可视化。
```

若文件已存在，在顶部追加 `## 2026-06-21` 段落。

- [ ] **Step 4: 最终 commit/push**

```bash
git add downstreams/ docs/ CHANGELOG.md
git commit -m "docs(downstream): complete README, data layout and CHANGELOG"
git push origin feat/data-and-model-rework-wt
```

---

## Spec 覆盖检查

| Spec 要求 | 对应任务 |
|---|---|
| 冻结 backbone + 轻量任务头 | Task 6, 8 |
| 离线生成并保存 embedding | Task 4 |
| 海淀建筑工地分割 | Task 2, 8, 9 |
| 5-fold CV + 10/25/50/100% 标签比例 | Task 3, 9 |
| Focal + Dice 默认损失 | Task 8 |
| IoU/F1/AP/AUPRC/可视化 | Task 7, 9 |
| 随机初始化基线 | Task 4（`precompute_embeddings.py --random-init`）+ Task 10 文档 |
| 变化检测预留接口 | Task 6 |
| 完整路径/依赖/版本溯源 | Task 1, 4, 10 |

**已知缺口**：无。所有设计 spec 要求均已对应到具体任务与代码。
