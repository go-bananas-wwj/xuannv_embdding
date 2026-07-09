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
    text = f"""# 玄女海淀地理嵌入 V1

这是 `haidian-embedding-v1` 的 ModelScope artifact 包。代码、配置和说明在 GitHub 分支
`codex/haidian-embedding-v1-production`；大文件保存在本数据集。

## 目录

```text
artifacts/haidian-embedding-v1/
  checkpoints/haidian_embedding_v1_p10c_epoch800.pt
  downstream_heads/
    building_mlp_fold0_best.pt
    road_mlp_fold0_best.pt
    water_mlp_fold0_best.pt
  embeddings/haidian_202512_202605_p10c_epoch800.tar.gz
  visualizations/
  manifests/haidian_artifacts_manifest.json
```

## 模型

- 生产权重：P10C epoch800
- 区域：北京市海淀区 320 个 patch
- 月份：2025-12 至 2026-05
- 输出：每个 patch 每个月 `64 x 128 x 128` embedding map，以及 scene embedding

## 快速指标

2026-04 fold-0 quick eval：

| 任务 | F1_best |
| --- | ---: |
| 建筑 | {metrics['building_f1_best']:.4f} |
| 道路 | {metrics['road_f1_best']:.4f} |
| 水体 | {metrics['water_f1_best']:.4f} |
| 平均 | {metrics['macro_f1_best']:.4f} |

完整校验信息见 `artifacts/haidian-embedding-v1/manifests/haidian_artifacts_manifest.json`。
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
