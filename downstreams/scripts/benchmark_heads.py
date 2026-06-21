#!/usr/bin/env python3
"""下游分割头对比实验（single-fold, limited-epoch）。

为每个候选 head 训练单个 fold，收集测试集 AUC-ROC、F1-best 与 mIoU，
输出汇总 JSON 与 markdown 表格。
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from downstreams.data.split import (
    create_combined_stratified_folds,
    create_stratified_folds,
)
from downstreams.utils.device import get_downstream_device
from downstreams.utils.reproducibility import set_seed

from downstreams.scripts.train_task import run_fold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _build_config(head: str, task: str, epochs: int, batch_size: int, lr: float) -> dict[str, Any]:
    """构造单个 head 的训练配置。

    bitemporal 输入通道数为 ``64 * 3 = 192``，由 ``ConstructionSegmentationTask.build_head()``
    根据 ``training.months`` 自动计算。
    """
    return {
        "experiment": {
            "name": f"{task}_{head}_benchmark",
            "seed": 42,
            "device": "auto",
        },
        "training": {
            "head_type": head,
            "loss": "bce_dice_tversky",
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "pos_weight": 200.0,
            "tversky_beta": 0.7,
            "tversky_weight": 1.5,
            "early_stop_metric": "f1_best",
            "early_stop_patience": 10,
            "use_weighted_sampler": True,
            "use_threshold_tuning": True,
            "pos_prior": 0.015,
            "num_workers": 0,
        },
        "data": {
            "embed_dim": 64,
            "num_classes": 2,
        },
    }


def _ensure_split(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    mask_dir: Path,
    mask_dirs: dict[str, Path] | None,
) -> dict[str, Any]:
    """确保 split json 存在，若不存在则生成。"""
    if args.regions:
        split_dir = args.label_root / args.task
        split_path = split_dir / "split_joint_5fold.json"
    else:
        # 非联合模式下 --label-root 已经是任务级目录，如 .../{region}/labels/{task}
        split_dir = args.label_root
        split_path = split_dir / "split_5fold.json"

    if not split_path.exists():
        logger.info("split 文件不存在，自动生成: %s", split_path)
        if args.regions:
            split = create_combined_stratified_folds(
                mask_dirs, seed=cfg["experiment"]["seed"]
            )
        else:
            split = create_stratified_folds(mask_dir, seed=cfg["experiment"]["seed"])
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=False, indent=2)

    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _make_run_args(
    args: argparse.Namespace, head: str, label_root: Path
) -> SimpleNamespace:
    """构造传给 ``run_fold`` 的伪 args 对象。

    ``run_fold`` 在非联合模式下会在内部再次拼接 ``--task``，
    因此这里传入的 ``label_root`` 应为任务目录的父目录（即 .../labels）。
    """
    return SimpleNamespace(
        task=args.task,
        embedding_root=args.embedding_root,
        label_root=label_root,
        months=args.months,
        regions=args.regions,
        output_root=args.output_root / head,
        fold=args.fold,
        fraction=None,
    )


def _print_markdown_table(results: list[dict[str, Any]]) -> None:
    header = "| head | auc_roc | f1_best | miou |"
    sep = "|------|---------|---------|------|"
    lines = [header, sep]
    for r in results:
        lines.append(
            f"| {r['head']} | {r['auc_roc']:.4f} | {r['f1_best']:.4f} | {r['miou']:.4f} |"
        )
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(
        epilog=(
            "示例（非联合模式）:\n"
            "  python -m downstreams.scripts.benchmark_heads --task construction \\\n"
            "    --label-root /data/xuannv_embedding/processed/harbin/labels/construction \\\n"
            "    --region harbin \\\n"
            "    --embedding-root /data/xuannv_embedding/embeddings/20260621_harbin_128_stage2_v1_best \\\n"
            "    --output-root /tmp/bench_test --fold 0 --epochs 1 --batch-size 2 --heads linear\n\n"
            "示例（联合多区域模式）:\n"
            "  python -m downstreams.scripts.benchmark_heads --task construction \\\n"
            "    --label-root /data/xuannv_embedding/processed \\\n"
            "    --regions haidian harbin \\\n"
            "    --embedding-root /data/xuannv_embedding/embeddings \\\n"
            "    --output-root /tmp/bench_test_joint --fold 0 --epochs 1 --batch-size 2 --heads linear"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--task", type=str, required=True)
    p.add_argument(
        "--region",
        type=str,
        default=None,
        help="embedding 子目录名；非联合模式下默认从 --label-root 推断",
    )
    p.add_argument(
        "--regions",
        type=str,
        nargs="+",
        default=None,
        help="联合训练的区域列表，例如 haidian harbin",
    )
    p.add_argument("--months", type=str, nargs="+", default=["202605"])
    p.add_argument(
        "--heads",
        type=str,
        nargs="+",
        default=["linear", "mlp", "fcn", "unet", "upernet", "diff_unet"],
    )
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--embedding-root", type=Path, required=True)
    p.add_argument(
        "--label-root",
        type=Path,
        required=True,
        help=(
            "label 根目录。两种约定："
            "1) 非联合模式：传入任务级目录，如 /processed/{region}/labels/{task}；"
            "2) 联合模式：传入 /processed 并配合 --regions region1 region2"
        ),
    )
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-5)
    args = p.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    bitemporal = len(args.months) == 2

    if args.regions:
        # 联合模式：--label-root 指向 /processed，内部按 region/labels/task 展开
        label_root = args.label_root / args.task
        mask_dir = label_root / "masks"  # 联合模式下仅作为 split 路径占位
        emb_region_root = args.embedding_root
        mask_dirs = {
            r: args.label_root / r / "labels" / args.task / "masks"
            for r in args.regions
        }
        region = "joint"
        label_root_for_run = args.label_root
    else:
        # 非联合模式：--label-root 已经是任务级目录，如 .../{region}/labels/{task}
        label_root = args.label_root
        mask_dir = label_root / "masks"
        region = args.region if args.region else args.label_root.parent.parent.name
        emb_region_root = args.embedding_root / region
        mask_dirs = None
        label_root_for_run = args.label_root.parent

    results: list[dict[str, Any]] = []
    for head in args.heads:
        logger.info("===== Benchmark head: %s =====", head)
        cfg = _build_config(head, args.task, args.epochs, args.batch_size, args.lr)
        split = _ensure_split(args, cfg, mask_dir, mask_dirs)
        fold_info = split["folds"][args.fold]
        run_args = _make_run_args(args, head, label_root_for_run)

        set_seed(cfg["experiment"]["seed"])
        # 预热设备，确保 NPU 可用
        _ = get_downstream_device(cfg["experiment"].get("device", "auto"))

        try:
            test_metrics = run_fold(
                cfg,
                run_args,
                fold_info,
                emb_region_root=emb_region_root,
                mask_dir=mask_dir,
                mask_dirs=mask_dirs,
                region=region,
                bitemporal=bitemporal,
                fractions=split.get("fractions"),
                region_of=split.get("region_of"),
                rank=0,
                world_size=1,
            )
        except Exception as exc:
            logger.exception("Head %s failed: %s", head, exc)
            results.append(
                {
                    "head": head,
                    "auc_roc": None,
                    "f1_best": None,
                    "miou": None,
                    "error": str(exc),
                }
            )
            continue

        if test_metrics is None:
            logger.warning("Head %s produced no metrics", head)
            results.append(
                {
                    "head": head,
                    "auc_roc": None,
                    "f1_best": None,
                    "miou": None,
                    "error": "no metrics",
                }
            )
            continue

        results.append(
            {
                "head": head,
                "auc_roc": test_metrics.get("auc_roc"),
                "f1_best": test_metrics.get("f1_best"),
                "miou": test_metrics.get("miou"),
            }
        )

    valid_results = [r for r in results if r["auc_roc"] is not None]
    best_by_auc = (
        max(valid_results, key=lambda r: r["auc_roc"])["head"]
        if valid_results
        else None
    )
    best_by_f1 = (
        max(valid_results, key=lambda r: r["f1_best"])["head"]
        if valid_results
        else None
    )

    summary = {
        "task": args.task,
        "regions": args.regions,
        "months": args.months,
        "fold": args.fold,
        "epochs": args.epochs,
        "results": results,
        "best_by_auc": best_by_auc,
        "best_by_f1": best_by_f1,
    }

    summary_path = args.output_root / "benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Benchmark summary written to %s", summary_path)

    _print_markdown_table(valid_results)


if __name__ == "__main__":
    main()
