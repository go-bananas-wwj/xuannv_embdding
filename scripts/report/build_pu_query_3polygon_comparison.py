#!/usr/bin/env python3
"""Build figures and a report for the fair three-polygon PU + Query comparison."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image

from scripts.eval import run_pu_query_sparse_eval as evaluation
from scripts.report.visualize_osm_downstream_outputs import load_highres

CJK_FONT = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"
if Path(CJK_FONT).exists():
    font_manager.fontManager.addfont(CJK_FONT)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


DEFAULT_RESULTS = Path(
    "/data/xuannv_embedding/experiments/production/"
    "haidian_pu_query_3polygon_compare_20260726/results.json"
)
DEFAULT_ASSETS = Path(
    "/root/workspace/xuannv/docs/production/assets/"
    "haidian_pu_query_3polygon_compare_20260726"
)
DEFAULT_REPORT = Path(
    "/root/workspace/xuannv/docs/production/"
    "haidian_pu_query_3polygon_comparison_20260726.md"
)
PROCESSED_ROOT = Path("/data/xuannv_embedding/processed")
TASK_ORDER = ("building", "road", "water")
FEATURE_ORDER = ("xuannv", "aef", "traditional")
FEATURE_LABELS = {
    "xuannv": "玄女 P10C\n2026-04 月度嵌入",
    "aef": "AEF\n2025 年度嵌入",
    "traditional": "传统 2026-04\n多源特征",
}
TASK_LABELS = {"building": "建筑物", "road": "道路", "water": "水体"}
COLORS = {"xuannv": "#d62728", "aef": "#3572b0", "traditional": "#7f8c8d"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--assets", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--skip-task-visuals", action="store_true")
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_payload(payload: dict[str, Any]) -> None:
    protocol = payload.get("protocol", {})
    if protocol.get("polygon_count") != 3 or not protocol.get("shared_supports"):
        raise ValueError("comparison must use exactly three shared support polygons")
    rows = payload.get("rows", [])
    expected = {(task, feature) for task in TASK_ORDER for feature in FEATURE_ORDER}
    actual = {(row.get("task"), row.get("feature")) for row in rows}
    if actual != expected or len(rows) != 9:
        raise ValueError("comparison must contain nine task-feature rows")
    for task in TASK_ORDER:
        task_rows = [row for row in rows if row["task"] == task]
        supports = {tuple(row["support_patch_ids"]) for row in task_rows}
        tests = {tuple(row["test_patch_ids"]) for row in task_rows}
        if len(supports) != 1:
            raise ValueError(f"{task}: shared support polygons differ across features")
        if len(tests) != 1:
            raise ValueError(f"{task}: shared test patches differ across features")


def row_map(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["task"], row["feature"]): row for row in payload["rows"]}


def red_binary(mask: np.ndarray) -> np.ndarray:
    output = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    output[np.asarray(mask, dtype=bool)] = (226, 42, 42)
    return output


def red_score(score: np.ndarray) -> np.ndarray:
    flat = np.asarray(score, dtype=np.float32).reshape(-1)
    order = np.argsort(flat, kind="stable")
    normalized = np.empty_like(flat)
    normalized[order] = np.linspace(0, 1, len(flat), dtype=np.float32)
    normalized = normalized.reshape(score.shape)
    output = np.ones((*score.shape, 3), dtype=np.float32)
    output[..., 1] = 1.0 - 0.84 * normalized
    output[..., 2] = 1.0 - 0.84 * normalized
    return output


def metric_figure(payload: dict[str, Any], output: Path) -> None:
    rows = row_map(payload)
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    x = np.arange(len(TASK_ORDER))
    width = 0.24
    for axis, metric, title in zip(axes, ("f1", "auc"), ("F1", "AUC")):
        for index, feature in enumerate(FEATURE_ORDER):
            values = [rows[(task, feature)]["metrics"][metric] for task in TASK_ORDER]
            bars = axis.bar(
                x + (index - 1) * width,
                values,
                width,
                label=FEATURE_LABELS[feature].replace("\n", " "),
                color=COLORS[feature],
            )
            axis.bar_label(bars, fmt="%.3f", fontsize=9, padding=2)
        axis.set_title(title, fontsize=15, fontweight="bold")
        axis.set_xticks(x, [TASK_LABELS[task] for task in TASK_ORDER])
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("指标值（越高越好）")
    axes[1].legend(loc="upper center", bbox_to_anchor=(-0.08, -0.15), ncol=3, frameon=False)
    figure.suptitle("同一批 3 个标注多边形、同一 PU+Query 流程、64 个独立测试 patch")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def markdown_report(payload: dict[str, Any], asset_dir_name: str) -> str:
    rows = row_map(payload)
    lines = [
        "# 海淀区 3 多边形 PU+Query 公平对比",
        "",
        "## 结论",
        "",
        "只标注 3 个目标多边形时，玄女在道路和水体的 F1、AUC、AP 上均高于 "
        "AEF 与传统多源特征；建筑任务由 AEF 领先，说明玄女仍存在建筑假正例偏多的问题。",
        "",
        f"![三类任务指标对比](assets/{asset_dir_name}/pu_query_3polygon_metrics.png)",
        "",
        "## 评测方法",
        "",
        "1. 每个任务只从训练区随机选择 3 个目标多边形，模拟用户少量圈选。",
        "2. 三种特征使用完全相同的标注、多边形顺序、64 个测试 patch、PU 背景挖掘、"
        "Query 自适应和阈值选择。",
        "3. 玄女使用 P10C epoch800 的 2026-04 月度嵌入；AEF 使用官方 2025 年度嵌入；"
        "传统方法使用 2026-04 多源观测与指数构成的 42 维特征。",
        "4. F1 衡量最终二值制图的准确性与完整性平衡；AUC 衡量排序能力；"
        "AP 更关注稀少目标在高置信区域中的检出质量。",
        "",
        "## 指标",
        "",
        "| 任务 | 特征 | F1 | AUC | AP | Precision | Recall |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for task in TASK_ORDER:
        for feature in FEATURE_ORDER:
            metric = rows[(task, feature)]["metrics"]
            lines.append(
                f"| {TASK_LABELS[task]} | {FEATURE_LABELS[feature].replace(chr(10), ' ')} | "
                f"{metric['f1']:.3f} | {metric['auc']:.3f} | {metric['ap']:.3f} | "
                f"{metric.get('precision', float('nan')):.3f} | "
                f"{metric.get('recall', float('nan')):.3f} |"
            )
    lines.extend(
        [
            "",
            "## 可视化",
            "",
            "每行左侧展示 3 个用户标注多边形，中间展示同一代表性测试 patch 的原始影像和"
            "三种方法的候选得分，右侧为真实标签。候选图仅做单 patch 百分位显示归一化，"
            "白色表示低分、红色表示高分，不改变汇总指标或二值阈值。为保证图像可读性，"
            "示例选择三种方法单 patch 平均 F1 的 75% 分位案例；该选择不参与汇总指标计算。",
            "",
        ]
    )
    for task in TASK_ORDER:
        lines.extend(
            [
                f"### {TASK_LABELS[task]}",
                "",
                f"![{TASK_LABELS[task]} 3多边形对比]"
                f"(assets/{asset_dir_name}/{task}_3polygon_compare.png)",
                "",
            ]
        )
    lines.extend(
        [
            "## 结果边界",
            "",
            "本实验是单个 fold、单次随机标注的严格稀疏标注对比，适合展示交互式快速制图能力；"
            "论文级结论仍应补充多 fold、多随机种子的均值与标准差。水体 F1 仍较低，当前更适合"
            "候选区域发现，而不是直接替代人工精确制图。",
            "",
        ]
    )
    return "\n".join(lines)


def build_metric_outputs(results: Path, assets: Path, report: Path) -> dict[str, Any]:
    payload = load_payload(results)
    validate_payload(payload)
    assets.mkdir(parents=True, exist_ok=True)
    metric_figure(payload, assets / "pu_query_3polygon_metrics.png")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(markdown_report(payload, assets.name), encoding="utf-8")
    return payload


def support_montage(label_root: Path, supports: list[evaluation.PolygonSupport]) -> np.ndarray:
    canvas = np.ones((256, 256, 3), dtype=np.float32)
    for index, support in enumerate(supports):
        image = load_highres(PROCESSED_ROOT, "haidian", support.patch_id, "202604")
        if image is None:
            image = np.ones((128, 128, 3), dtype=np.float32)
        image = np.asarray(image, dtype=np.float32)
        mask = support.mask
        if image.shape[:2] != mask.shape:
            image = np.asarray(
                Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255)).resize(
                    (mask.shape[1], mask.shape[0]),
                    Image.Resampling.BILINEAR,
                ),
                dtype=np.float32,
            ) / 255.0
        marked = image.copy()
        marked[mask] = 0.45 * marked[mask] + 0.55 * np.array([1.0, 0.0, 0.0])
        y = 0 if index < 2 else 128
        x = 128 * (index % 2)
        canvas[y : y + 128, x : x + 128] = marked
    return canvas


def visual_patch(
    label_root: Path,
    test_ids: list[str],
    specs: dict[str, evaluation.FeatureSpec],
    models: dict[str, dict[str, np.ndarray | float | int]],
) -> str:
    readable = []
    for patch_id in test_ids:
        gt = evaluation.load_mask(label_root, patch_id)
        if not gt.any():
            continue
        patch_f1 = []
        for feature in FEATURE_ORDER:
            score, _ = evaluation.score_pu_query(
                evaluation.load_feature(specs[feature], patch_id),
                models[feature],
            )
            patch_f1.append(
                evaluation.metrics(
                    score,
                    gt,
                    float(models[feature]["threshold"]),
                )["f1"]
            )
        readable.append((float(np.mean(patch_f1)), patch_id))
    if not readable:
        return test_ids[0]
    readable.sort()
    target = int(round(0.75 * (len(readable) - 1)))
    return readable[target][1]


def build_task_visuals(payload: dict[str, Any], assets: Path) -> None:
    rows = row_map(payload)
    args = SimpleNamespace(
        embedding_root=Path(
            "/data/xuannv_embedding/embeddings/production/p10c_epoch800_202604/"
            "artifacts/haidian-embedding-v1/embeddings/"
            "haidian_202512_202605_p10c_epoch800"
        ),
        aef_embedding_root=Path(
            "/data/xuannv_embedding/embeddings/aef_official_2025_annual"
        ),
        data_root=Path("/data/xuannv_embedding/processed/haidian"),
        manifest=Path(
            "/data/xuannv_embedding/processed/haidian/"
            "manifest_p6a_202512_202605_pixelmask_clean_osm_landcover.json"
        ),
        fold=0,
        seed=42,
    )
    specs = evaluation.feature_specs(args)
    for task in TASK_ORDER:
        _, label_root = evaluation.TASKS[task]
        split = evaluation.split_for(label_root, args.fold)
        candidates = evaluation.collect_components(label_root, split["train"])
        supports = evaluation.pick_supports(
            candidates,
            3,
            args.seed + sum(map(ord, task)),
        )
        expected_ids = rows[(task, "xuannv")]["support_patch_ids"]
        if [support.patch_id for support in supports] != expected_ids:
            raise ValueError(f"{task}: reconstructed supports do not match saved results")
        models = {
            feature: evaluation.train_pu_query(specs[feature], supports, args.seed)
            for feature in FEATURE_ORDER
        }
        patch_id = visual_patch(label_root, split["test"], specs, models)
        gt = evaluation.load_mask(label_root, patch_id)
        optical = load_highres(PROCESSED_ROOT, "haidian", patch_id, "202604")
        panels: list[tuple[np.ndarray, str]] = [
            (support_montage(label_root, supports), "用户圈选的 3 个多边形"),
            (
                optical if optical is not None else np.ones((*gt.shape, 3), dtype=np.float32),
                f"代表性测试影像\n{patch_id}",
            ),
        ]
        for feature in FEATURE_ORDER:
            score, _ = evaluation.score_pu_query(
                evaluation.load_feature(specs[feature], patch_id),
                models[feature],
            )
            panels.append(
                (
                    red_score(score),
                    FEATURE_LABELS[feature] + "\n候选得分",
                )
            )
        panels.append((red_binary(gt), "真实标签 GT"))
        figure, axes = plt.subplots(
            1,
            len(panels),
            figsize=(16, 3.2),
            gridspec_kw={"width_ratios": [1.12, 1, 1, 1, 1, 1]},
        )
        for axis, (image, title) in zip(axes, panels):
            axis.imshow(image)
            axis.set_title(title, fontsize=10, fontweight="bold")
            axis.axis("off")
        figure.suptitle(
            f"{TASK_LABELS[task]}：同一组 3 个用户标注、同一 PU+Query 流程",
            fontsize=14,
            fontweight="bold",
        )
        figure.tight_layout()
        figure.savefig(
            assets / f"{task}_3polygon_compare.png",
            dpi=190,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(figure)


def sync_to_experiment(assets: Path, results: Path) -> None:
    destination = results.parent / "visualizations"
    destination.mkdir(parents=True, exist_ok=True)
    for path in assets.glob("*.png"):
        shutil.copy2(path, destination / path.name)


def main() -> None:
    args = parse_args()
    payload = build_metric_outputs(args.results, args.assets, args.report)
    if not args.skip_task_visuals:
        build_task_visuals(payload, args.assets)
        sync_to_experiment(args.assets, args.results)
    print(args.report)


if __name__ == "__main__":
    main()
