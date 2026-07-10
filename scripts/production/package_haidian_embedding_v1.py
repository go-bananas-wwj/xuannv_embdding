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

这个数据集用于发布生产 artifact：模型权重、2025-12 至 2026-05 的月度 embedding、下游轻量头、全域可视化和 manifest。代码、配置和评测脚本在 GitHub 分支 `codex/haidian-embedding-v1-production`。

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

## 快速指标

2026-04 fold-0 quick eval，当前随包发布的是轻量 MLP pixel probe：

| 任务 | F1_best |
| --- | ---: |
| 建筑 | {metrics['building_f1_best']:.4f} |
| 道路 | {metrics['road_f1_best']:.4f} |
| 水体 | {metrics['water_f1_best']:.4f} |
| 平均 | {metrics['macro_f1_best']:.4f} |

说明：该指标用于生产包快速验收。更完整的 few-shot、raw image baseline 和强下游模型对比见 GitHub 分支中的 `docs/production/`。

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
git checkout codex/haidian-embedding-v1-production

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

## 可视化

全域 320 patch 可视化保存在 `{artifact_root}/visualizations/`：

- `p10_embedding_pca_202604_compare.png`：P10A/P10B/P10C embedding PCA 对比。
- `P10C_embedding_pca_202604_geo.png`：P10C 海淀区全域 PCA。
- `building/`、`road/`、`water/`：真实标签、概率图和预测图。

## 复现与版本

- GitHub 分支：`codex/haidian-embedding-v1-production`
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
        "github_branch": "codex/haidian-embedding-v1-production",
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
