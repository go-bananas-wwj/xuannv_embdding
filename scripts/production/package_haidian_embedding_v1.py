#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_RELEASE = "haidian-embedding-v1"
DEFAULT_DATASET = "WeijieWu/xuannv_haidian_embdding"
DEFAULT_STAGE = Path("/data/xuannv_embedding/modelscope_upload/haidian_embedding_v1_p10c_epoch800")
DEFAULT_CHECKPOINT = Path(
    "/data/xuannv_embedding/outputs/v2_p10c_haidian_202512_202605_osm_semantic_hardneg_20260704/epoch_800.pt"
)
DEFAULT_VIS = Path(
    "/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/p10_epoch800_full_domain_visuals_20260705"
)
DEFAULT_BENCH = Path(
    "/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/p10_periodic/p10c_epoch800_202604_fold0"
)
HEADS = {
    "building": DEFAULT_BENCH / "building/fold_0/checkpoints/best.pt",
    "road": DEFAULT_BENCH / "road/fold_0/checkpoints/best.pt",
    "water": DEFAULT_BENCH / "water/fold_0/checkpoints/best.pt",
}
MONTHS = ["202512", "202601", "202602", "202603", "202604", "202605"]

QUICK_EVAL_202604 = {
    "building": {
        "f1_best": 0.48324818606032377,
        "miou": 0.31806746235200134,
        "precision": 0.40830820565411047,
        "recall": 0.5900203485800231,
        "ap": 0.43919582866632767,
        "auc_roc": 0.8955988741701253,
    },
    "road": {
        "f1_best": 0.5231966158695528,
        "miou": 0.35396746971284015,
        "precision": 0.5077363571934798,
        "recall": 0.5389114893002431,
        "ap": 0.5726360111352338,
        "auc_roc": 0.8362612886360627,
    },
    "water": {
        "f1_best": 0.6305895707468961,
        "miou": 0.4599946241376221,
        "precision": 0.774914154182861,
        "recall": 0.530934098606479,
        "ap": 0.6092140720948176,
        "auc_roc": 0.9012394930670107,
    },
}

FEWSHOT_BASELINE = [
    ("建筑 Building", 5, 0.459, 0.418, "+9.8%", 0.413, 0.354, 0.879, 0.858),
    ("建筑 Building", 10, 0.454, 0.439, "+3.5%", 0.405, 0.375, 0.883, 0.874),
    ("建筑 Building", 50, 0.490, 0.467, "+5.0%", 0.448, 0.409, 0.898, 0.888),
    ("道路 Road", 5, 0.474, 0.401, "+18.3%", 0.507, 0.329, 0.795, 0.740),
    ("道路 Road", 10, 0.487, 0.435, "+11.8%", 0.511, 0.374, 0.803, 0.768),
    ("道路 Road", 50, 0.517, 0.501, "+3.3%", 0.561, 0.523, 0.828, 0.820),
    ("水体 Water", 5, 0.613, 0.433, "+41.5%", 0.599, 0.468, 0.878, 0.816),
    ("水体 Water", 10, 0.612, 0.487, "+25.6%", 0.583, 0.488, 0.874, 0.810),
    ("水体 Water", 50, 0.631, 0.631, "基本持平", 0.613, 0.625, 0.894, 0.878),
]

FULL_LABEL_BASELINE = [
    ("建筑 Building", 0.484, 0.503, "Raw image + UNet"),
    ("道路 Road", 0.520, 0.580, "Raw image + UNet"),
    ("水体 Water", 0.623, 0.667, "Raw image + DeepLab-lite"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: Path, dst: Path) -> dict[str, object]:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"bytes": dst.stat().st_size, "sha256": sha256_file(dst)}


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def make_tar(src_dir: Path, tar_path: Path) -> dict[str, object]:
    if not src_dir.exists():
        raise FileNotFoundError(src_dir)
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(src_dir, arcname=src_dir.name)
    return {"bytes": tar_path.stat().st_size, "sha256": sha256_file(tar_path)}


def write_readme(stage: Path, manifest: dict[str, object]) -> None:
    metrics = manifest["quick_eval_fold0_202604"]
    artifact_root = f"artifacts/{manifest['release']}"
    embedding_sha = manifest["artifacts"]["monthly_embeddings"].get("sha256") or "pending"
    embedding_bytes = manifest["artifacts"]["monthly_embeddings"].get("bytes") or 0
    checkpoint_sha = manifest["artifacts"]["embedding_model"]["sha256"]
    checkpoint_bytes = manifest["artifacts"]["embedding_model"]["bytes"]
    official_branch = "haidian-embedding-v1-production"
    quick_rows = "\n".join(
        "| {name} | {f1:.4f} | {ap:.4f} | {auc:.4f} | {miou:.4f} | {precision:.4f} | {recall:.4f} |".format(
            name={"building": "建筑 Building", "road": "道路 Road", "water": "水体 Water"}[task],
            f1=values["f1_best"],
            ap=values["ap"],
            auc=values["auc_roc"],
            miou=values["miou"],
            precision=values["precision"],
            recall=values["recall"],
        )
        for task, values in QUICK_EVAL_202604.items()
    )
    fewshot_rows = "\n".join(
        f"| {task} | {shot} | {xf1:.3f} | {rf1:.3f} | {gain} | {xap:.3f} | {rap:.3f} | {xauc:.3f} | {rauc:.3f} |"
        for task, shot, xf1, rf1, gain, xap, rap, xauc, rauc in FEWSHOT_BASELINE
    )
    full_label_rows = "\n".join(
        f"| {task} | {xuannv:.3f} | {raw:.3f} | {raw_model} |"
        for task, xuannv, raw, raw_model in FULL_LABEL_BASELINE
    )
    text = f"""---
license: apache-2.0
tags:
  - remote-sensing
  - geospatial-embedding
  - earth-observation
  - haidian
  - xuannv
  - pytorch
  - torch-npu
task_categories:
  - feature-extraction
  - image-segmentation
---

# 玄女海淀地理嵌入 V1

`haidian-embedding-v1` 是面向北京市海淀区的月度地理 embedding 生产版。它把 Sentinel-2、Sentinel-1、Landsat、高分辨率光学、高分辨率 SAR 和 OSM 弱语义信息融合到统一的 64 维空间表征中，输出每个 1280m x 1280m patch 的 `64 x 128 x 128` dense embedding map。

这个数据集用于发布生产 artifact：模型权重、2025-12 至 2026-05 的月度 embedding、下游轻量头、全域可视化和 manifest。代码、配置和评测脚本在 GitHub 分支 `{official_branch}`。

## 适用场景

- 少量标注快速制图：建筑、道路、水体等类别只标少量样本后训练轻量头。
- 多任务复用：同一份 embedding 支持多个下游任务，不需要每个任务重新处理多源遥感输入。
- 区域语义检索：用一个局部区域的 embedding 检索海淀区内相似地物。
- 月度地表表达：对 2025-12 至 2026-05 的海淀区 patch 生成统一表征。

## 不适用场景

- 该版本是海淀区专用生产版，不承诺跨城市、跨传感器域直接泛化。
- 随包下游头是 fold-0 quick eval 结果，不等价于完整业务验收。
- OSM 是弱标签，存在漏标、错标和时间滞后；下游结果应结合人工抽检。

## 目录

```text
{artifact_root}/
  checkpoints/haidian_embedding_v1_p10c_epoch800.pt
  downstream_heads/
    building_mlp_fold0_best.pt
    road_mlp_fold0_best.pt
    water_mlp_fold0_best.pt
  embeddings/haidian_202512_202605_p10c_epoch800.tar.gz
  visualizations/
  manifests/haidian_artifacts_manifest.json
```

## 文件说明

| 路径 | 内容 | 大小 | SHA256 |
| --- | --- | ---: | --- |
| `{artifact_root}/checkpoints/haidian_embedding_v1_p10c_epoch800.pt` | P10C epoch800 embedding 主模型权重 | {checkpoint_bytes:,} bytes | `{checkpoint_sha}` |
| `{artifact_root}/embeddings/haidian_202512_202605_p10c_epoch800.tar.gz` | 六个月海淀区 320 patch embedding 压缩包 | {embedding_bytes:,} bytes | `{embedding_sha}` |
| `{artifact_root}/downstream_heads/building_mlp_fold0_best.pt` | 建筑提取 MLP 下游头 | - | 见 manifest |
| `{artifact_root}/downstream_heads/road_mlp_fold0_best.pt` | 道路提取 MLP 下游头 | - | 见 manifest |
| `{artifact_root}/downstream_heads/water_mlp_fold0_best.pt` | 水体提取 MLP 下游头 | - | 见 manifest |
| `{artifact_root}/visualizations/` | 320 patch 全域 PCA 和下游可视化 | - | - |
| `{artifact_root}/manifests/haidian_artifacts_manifest.json` | 完整路径、大小、校验和、指标清单 | - | - |

## 模型与数据

- 生产权重：P10C epoch800
- 区域：北京市海淀区 320 个 patch
- 月份：2025-12 至 2026-05
- 输出：每个 patch 每个月 `64 x 128 x 128` embedding map，以及一个 scene-level embedding
- 训练信号：多源重建、高分辨率重建、OSM 弱语义、困难重建

## 快速指标：2026-04 核心任务

2026-04 fold-0 quick eval，当前随包发布的是轻量 MLP pixel probe。阈值在验证集选择，再报告测试集结果。

| 任务 | F1_best | AP | AUC-ROC | mIoU | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{quick_rows}

说明：

- **F1_best**：验证集选阈值后，在测试集上的最佳 F1，综合考虑 precision 和 recall。
- **AP**：Average Precision，反映排序质量，适合类别不均衡任务。
- **AUC-ROC**：阈值无关的二分类区分能力。
- **mIoU**：预测区域和真实区域的交并比。

## 与 baseline 的对比

### 少样本制图：玄女 embedding vs 原始影像强基线

这个场景是玄女 embedding 的核心价值：只标很少的 patch，快速训练下游头，然后在海淀区 320 个 patch 全域制图。

| 任务 | Shot | 玄女最佳 F1 | Raw 最佳 F1 | 相对提升 | 玄女 AP | Raw AP | 玄女 AUC | Raw AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{fewshot_rows}

### 完整监督：玄女轻量头 vs 原始影像强模型

当每个任务有较充分标签，并允许原始影像训练 UNet/DeepLab-lite 这类强模型时，raw image strong baseline 仍然很强。这也是当前版本的已知边界。

| 任务 | 玄女 P10C conv3x3 F1 | 原始影像最佳 F1 | 原始影像最佳模型 |
| --- | ---: | ---: | --- |
{full_label_rows}

说明：该指标用于生产包快速验收。更完整的 few-shot、raw image baseline 和强下游模型对比见 GitHub 分支中的 `docs/production/`。

## 可视化结果

### Embedding PCA

下图是 2026-04 海淀区 320 个 patch 的 embedding PCA 全域可视化。颜色来自 embedding 主成分，不是人工标签；相近颜色通常表示 embedding 认为语义/纹理更相近。

![P10C embedding PCA](artifacts/haidian-embedding-v1/visualizations/P10C_embedding_pca_202604_geo.png)

P10A/P10B/P10C 三个候选版本 PCA 对比：

![P10 embedding PCA comparison](artifacts/haidian-embedding-v1/visualizations/p10_embedding_pca_202604_compare.png)

### 建筑提取 Building

![Building prediction comparison](artifacts/haidian-embedding-v1/visualizations/building/building_gt_p10a_p10b_p10c_prediction_compare.png)

### 道路提取 Road

![Road prediction comparison](artifacts/haidian-embedding-v1/visualizations/road/road_gt_p10a_p10b_p10c_prediction_compare.png)

### 水体提取 Water

![Water prediction comparison](artifacts/haidian-embedding-v1/visualizations/water/water_gt_p10a_p10b_p10c_prediction_compare.png)

## 快速开始

### 1. 下载 artifact

可以直接在 ModelScope 页面下载，也可以使用 ModelScope SDK 下载数据集快照。

```bash
pip install modelscope

python - <<'PY'
from modelscope.hub.snapshot_download import snapshot_download

snapshot_download(
    repo_id="WeijieWu/xuannv_haidian_embdding",
    repo_type="dataset",
    local_dir="/data/xuannv_embedding/modelscope_download/xuannv_haidian_embdding",
)
PY
```

### 2. 解压月度 embedding

```bash
cd /data/xuannv_embedding/modelscope_download/xuannv_haidian_embdding
tar -xzf {artifact_root}/embeddings/haidian_202512_202605_p10c_epoch800.tar.gz \\
  -C /data/xuannv_embedding/embeddings/production
```

解压后每个 patch 的结构类似：

```text
haidian/patch_000000/
  202512_embedding_map.pt
  202512_scene_embedding.pt
  ...
  202605_embedding_map.pt
  202605_scene_embedding.pt
```

### 3. 使用 GitHub 代码重新导出 embedding

```bash
git clone git@github.com:go-bananas-wwj/xuannv_embdding.git
cd xuannv_embdding
git checkout {official_branch}

bash scripts/production/export_haidian_v1_embeddings.sh
```

### 4. 训练或评估下游头

```bash
python scripts/production/run_haidian_downstream_probe.py \\
  --embedding-root /data/xuannv_embedding/embeddings/production/<export_dir> \\
  --output-root /data/xuannv_embedding/experiments/production/haidian_v1_downstream \\
  --month 202604 \\
  --tasks building road water \\
  --device npu:0 \\
  --head mlp \\
  --save-predictions
```

## 复现与版本

- GitHub 分支：`{official_branch}`
- 源实验：`v2_p10c_haidian_202512_202605_osm_semantic_hardneg_20260704`
- 选定权重：`epoch_800.pt`
- 生产包 manifest：`{artifact_root}/manifests/haidian_artifacts_manifest.json`

## 引用

如果在汇报或实验中使用该版本，请注明：

```text
Xuannv Haidian Embedding V1, P10C epoch800, 2025-12 to 2026-05 monthly geospatial embeddings.
ModelScope dataset: WeijieWu/xuannv_haidian_embdding.
```
"""
    (stage / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Package Haidian Embedding V1 artifacts for ModelScope.")
    p.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE)
    p.add_argument("--embedding-root", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--visualization-root", type=Path, default=DEFAULT_VIS)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--release", default=DEFAULT_RELEASE)
    p.add_argument("--clean", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    stage = args.stage_dir
    if args.clean and stage.exists():
        shutil.rmtree(stage)
    base = stage / "artifacts" / args.release
    base.mkdir(parents=True, exist_ok=True)

    checkpoint_info = copy_file(
        args.checkpoint,
        base / "checkpoints" / "haidian_embedding_v1_p10c_epoch800.pt",
    )
    head_infos = {}
    for name, src in HEADS.items():
        head_infos[name] = copy_file(src, base / "downstream_heads" / f"{name}_mlp_fold0_best.pt")

    copy_tree(args.visualization_root, base / "visualizations")

    embedding_info = None
    if args.embedding_root is not None:
        embedding_info = make_tar(
            args.embedding_root,
            base / "embeddings" / "haidian_202512_202605_p10c_epoch800.tar.gz",
        )

    manifest = {
        "release": args.release,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "modelscope_dataset": args.dataset,
        "github_branch": "haidian-embedding-v1-production",
        "source_experiment": "v2_p10c_haidian_202512_202605_osm_semantic_hardneg_20260704",
        "selected_checkpoint": "epoch_800.pt",
        "month_range": MONTHS,
        "artifacts": {
            "embedding_model": {
                "modelscope_path": f"artifacts/{args.release}/checkpoints/haidian_embedding_v1_p10c_epoch800.pt",
                "local_source": str(args.checkpoint),
                "bytes": checkpoint_info["bytes"],
                "sha256": checkpoint_info["sha256"],
            },
            "downstream_heads": {
                k: {
                    "modelscope_path": f"artifacts/{args.release}/downstream_heads/{k}_mlp_fold0_best.pt",
                    "local_source": str(HEADS[k]),
                    "bytes": v["bytes"],
                    "sha256": v["sha256"],
                }
                for k, v in head_infos.items()
            },
            "monthly_embeddings": {
                "modelscope_path": f"artifacts/{args.release}/embeddings/haidian_202512_202605_p10c_epoch800.tar.gz",
                "local_source": str(args.embedding_root) if args.embedding_root else None,
                "bytes": embedding_info["bytes"] if embedding_info else None,
                "sha256": embedding_info["sha256"] if embedding_info else None,
            },
            "visualizations": {
                "modelscope_path": f"artifacts/{args.release}/visualizations/",
                "local_source": str(args.visualization_root),
            },
        },
        "quick_eval_fold0_202604": {
            "building_f1_best": 0.48324818606032377,
            "road_f1_best": 0.5231966158695528,
            "water_f1_best": 0.6305895707468961,
            "macro_f1_best": 0.5456781242255909,
        },
    }
    manifest_dir = base / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "haidian_artifacts_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_readme(stage, manifest)
    print(stage)


if __name__ == "__main__":
    main()
