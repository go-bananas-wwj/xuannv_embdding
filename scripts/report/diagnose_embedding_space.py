#!/usr/bin/env python3
"""Diagnose embedding collapse, uniformity, and semantic separability."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score


TASKS = {
    "building": "building_osm",
    "road": "road_osm",
    "water": "osm_water",
    "park_green": "merged_park_green",
    "education": "merged_education",
    "sports_pitch": "merged_sports_pitch",
}

TASK_LABELS = {
    "building": "Building",
    "road": "Road",
    "water": "Water",
    "park_green": "Park/Green",
    "education": "Education",
    "sports_pitch": "Sports/Pitch",
    "background": "Background",
}


@dataclass(frozen=True)
class EmbeddingSpec:
    name: str
    root: Path
    region: str
    month: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xuannv-root",
        type=Path,
        default=Path("/data/xuannv_embedding/embeddings/production/p10c_epoch800_202604"),
    )
    parser.add_argument(
        "--aef-root",
        type=Path,
        default=Path("/data/xuannv_embedding/embeddings/aef_official_2025_annual"),
    )
    parser.add_argument("--label-root", type=Path, default=Path("/data/xuannv_embedding/processed/haidian/labels"))
    parser.add_argument("--region", default="haidian")
    parser.add_argument("--xuannv-month", default="202604")
    parser.add_argument("--aef-month", default="202512")
    parser.add_argument("--samples-per-class", type=int, default=40000)
    parser.add_argument("--background-samples", type=int, default=40000)
    parser.add_argument("--max-patches", type=int, default=320)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/xuannv_embedding/experiments/production/haidian_v1_embedding_space_diagnostics_20260706"),
    )
    return parser.parse_args()


def load_mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1) > 0


def index_embeddings(root: Path, region: str, month: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in root.glob(f"**/{region}/patch_*/{month}_embedding_map.pt"):
        patch_id = path.parent.name
        out.setdefault(patch_id, path)
    return out


def load_embedding(path: Path) -> np.ndarray:
    emb = torch.load(path, map_location="cpu", weights_only=True).float()
    if emb.ndim != 3 or emb.shape[0] != 64:
        raise ValueError(f"Expected (64,H,W) embedding, got {tuple(emb.shape)} from {path}")
    return emb.numpy()


def available_patch_ids(label_root: Path, embedding_indices: list[dict[str, Path]]) -> list[str]:
    mask_dir = label_root / TASKS["building"] / "masks"
    patch_ids = sorted(path.stem for path in mask_dir.glob("patch_*.tif"))
    shared = []
    for patch_id in patch_ids:
        if all(patch_id in index for index in embedding_indices):
            shared.append(patch_id)
    return shared


def collect_pixel_locations(
    label_root: Path,
    patch_ids: list[str],
    samples_per_class: int,
    background_samples: int,
    rng: np.random.Generator,
) -> dict[str, list[tuple[str, np.ndarray, np.ndarray]]]:
    locations: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {task: [] for task in TASKS}
    locations["background"] = []
    positive_total = {task: 0 for task in TASKS}
    background_total = 0

    shuffled = list(patch_ids)
    rng.shuffle(shuffled)
    for patch_id in shuffled:
        masks = {}
        union = None
        for task, label_dir in TASKS.items():
            mask_path = label_root / label_dir / "masks" / f"{patch_id}.tif"
            if not mask_path.exists():
                continue
            mask = load_mask(mask_path)
            masks[task] = mask
            union = mask.copy() if union is None else (union | mask)

        for task, mask in masks.items():
            if positive_total[task] >= samples_per_class:
                continue
            ys, xs = np.nonzero(mask)
            if ys.size == 0:
                continue
            take = min(ys.size, max(256, samples_per_class // max(len(patch_ids), 1)))
            if positive_total[task] + take > samples_per_class:
                take = samples_per_class - positive_total[task]
            choice = rng.choice(ys.size, size=take, replace=False)
            locations[task].append((patch_id, ys[choice], xs[choice]))
            positive_total[task] += take

        if union is not None and background_total < background_samples:
            ys, xs = np.nonzero(~union)
            if ys.size > 0:
                take = min(ys.size, max(256, background_samples // max(len(patch_ids), 1)))
                if background_total + take > background_samples:
                    take = background_samples - background_total
                choice = rng.choice(ys.size, size=take, replace=False)
                locations["background"].append((patch_id, ys[choice], xs[choice]))
                background_total += take

        if all(positive_total[task] >= samples_per_class for task in TASKS) and background_total >= background_samples:
            break
    return locations


def gather_features(embedding_index: dict[str, Path], locations: dict[str, list[tuple[str, np.ndarray, np.ndarray]]]) -> tuple[np.ndarray, np.ndarray]:
    cache: dict[str, np.ndarray] = {}
    feats = []
    labels = []
    for label_idx, task in enumerate([*TASKS.keys(), "background"]):
        for patch_id, ys, xs in locations[task]:
            if patch_id not in cache:
                cache[patch_id] = load_embedding(embedding_index[patch_id])
            emb = cache[patch_id]
            sampled = emb[:, ys, xs].T
            feats.append(sampled)
            labels.append(np.full(sampled.shape[0], label_idx, dtype=np.int64))
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def covariance_stats(z: np.ndarray) -> dict[str, float]:
    centered = z - z.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eig = np.linalg.eigvalsh(cov).clip(min=0.0)[::-1]
    total = float(eig.sum())
    probs = eig / max(total, 1e-12)
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    return {
        "effective_rank_entropy": float(math.exp(entropy)),
        "participation_ratio": float(total * total / max(float((eig * eig).sum()), 1e-12)),
        "top1_var_ratio": float(probs[0]),
        "top5_var_ratio": float(probs[:5].sum()),
        "top10_var_ratio": float(probs[:10].sum()),
        "dead_dim_count_std_lt_1e-3": int((z.std(axis=0) < 1e-3).sum()),
    }


def uniformity_score(z_norm: np.ndarray, rng: np.random.Generator, pairs: int = 120000) -> dict[str, float]:
    n = z_norm.shape[0]
    left = rng.integers(0, n, size=pairs)
    right = rng.integers(0, n, size=pairs)
    cos = (z_norm[left] * z_norm[right]).sum(axis=1)
    sq_dist = 2.0 - 2.0 * cos
    return {
        "pair_cos_mean": float(cos.mean()),
        "pair_cos_p95": float(np.percentile(cos, 95)),
        "uniformity_log_mean_exp_t2": float(np.log(np.exp(-2.0 * sq_dist).mean() + 1e-12)),
    }


def class_separation(z_norm: np.ndarray, y: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    label_names = [*TASKS.keys(), "background"]
    centroids = []
    within_rows = []
    for idx, name in enumerate(label_names):
        pts = z_norm[y == idx]
        centroid = l2_normalize(pts.mean(axis=0, keepdims=True))[0]
        centroids.append(centroid)
        dist = 1.0 - pts @ centroid
        within_rows.append(
            {
                "class": name,
                "class_label": TASK_LABELS[name],
                "sample_count": int(pts.shape[0]),
                "within_cos_dist_mean": float(dist.mean()),
                "within_cos_dist_p90": float(np.percentile(dist, 90)),
            }
        )
    cent = np.stack(centroids, axis=0)
    sim = cent @ cent.T
    dist = 1.0 - sim
    matrix = pd.DataFrame(dist, index=label_names, columns=label_names)
    offdiag = dist[~np.eye(dist.shape[0], dtype=bool)]
    within_mean = float(pd.DataFrame(within_rows)["within_cos_dist_mean"].mean())
    summary = {
        "centroid_between_cos_dist_mean": float(offdiag.mean()),
        "centroid_between_cos_dist_min": float(offdiag.min()),
        "within_cos_dist_mean": within_mean,
        "between_within_ratio": float(offdiag.mean() / max(within_mean, 1e-12)),
    }
    return pd.DataFrame(within_rows), matrix, summary


def centroid_auc(z_norm: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    label_names = [*TASKS.keys(), "background"]
    rows = []
    for idx, name in enumerate(label_names[:-1]):
        pos = z_norm[y == idx]
        neg = z_norm[y != idx]
        centroid = l2_normalize(pos.mean(axis=0, keepdims=True))[0]
        scores = np.concatenate([pos @ centroid, neg @ centroid])
        labels = np.concatenate([np.ones(pos.shape[0], dtype=np.int64), np.zeros(neg.shape[0], dtype=np.int64)])
        auc = roc_auc_score(labels, scores)
        neg_scores = neg @ centroid
        pos_scores = pos @ centroid
        threshold = np.percentile(pos_scores, 10)
        rows.append(
            {
                "class": name,
                "class_label": TASK_LABELS[name],
                "centroid_similarity_auc": float(auc),
                "pos_similarity_mean": float(pos_scores.mean()),
                "neg_similarity_mean": float(neg_scores.mean()),
                "neg_above_pos_p10_ratio": float((neg_scores >= threshold).mean()),
            }
        )
    return pd.DataFrame(rows)


def pca_scatter(z_norm: np.ndarray, y: np.ndarray, model_name: str, out_path: Path, rng: np.random.Generator) -> None:
    max_points = min(25000, z_norm.shape[0])
    idx = rng.choice(z_norm.shape[0], size=max_points, replace=False)
    coords = PCA(n_components=2, random_state=0).fit_transform(z_norm[idx])
    labels = y[idx]
    label_names = [*TASKS.keys(), "background"]
    palette = sns.color_palette("tab10", n_colors=len(label_names))
    fig, ax = plt.subplots(figsize=(8.2, 6.8), dpi=220)
    for class_idx, name in enumerate(label_names):
        mask = labels == class_idx
        ax.scatter(coords[mask, 0], coords[mask, 1], s=4, alpha=0.45, label=TASK_LABELS[name], color=palette[class_idx])
    ax.set_title(f"{model_name}: PCA of normalized embedding samples")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(markerscale=3, fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def heatmap(matrix: pd.DataFrame, title: str, out_path: Path) -> None:
    labels = [TASK_LABELS.get(item, item) for item in matrix.index]
    fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=220)
    sns.heatmap(matrix, xticklabels=labels, yticklabels=labels, annot=True, fmt=".3f", cmap="mako", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Centroid")
    ax.set_ylabel("Centroid")
    plt.xticks(rotation=35, ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def diagnose(spec: EmbeddingSpec, locations: dict[str, list[tuple[str, np.ndarray, np.ndarray]]], rng: np.random.Generator, out_dir: Path) -> dict[str, Any]:
    index = index_embeddings(spec.root, spec.region, spec.month)
    z, y = gather_features(index, locations)
    z_norm = l2_normalize(z)
    norm = np.linalg.norm(z, axis=1)

    cov = covariance_stats(z_norm)
    uniform = uniformity_score(z_norm, rng)
    within, centroid_dist, sep = class_separation(z_norm, y)
    auc = centroid_auc(z_norm, y)

    within.insert(0, "model", spec.name)
    auc.insert(0, "model", spec.name)
    centroid_dist.to_csv(out_dir / f"{spec.name}_centroid_cosine_distance.csv")
    within.to_csv(out_dir / f"{spec.name}_within_class.csv", index=False)
    auc.to_csv(out_dir / f"{spec.name}_centroid_auc.csv", index=False)

    pca_scatter(z_norm, y, spec.name, out_dir / "figures" / f"{spec.name}_pca_scatter.png", rng)
    heatmap(centroid_dist, f"{spec.name}: centroid cosine distance", out_dir / "figures" / f"{spec.name}_centroid_distance.png")

    return {
        "model": spec.name,
        "sample_count": int(z.shape[0]),
        "norm_mean": float(norm.mean()),
        "norm_std": float(norm.std()),
        "norm_p05": float(np.percentile(norm, 5)),
        "norm_p95": float(np.percentile(norm, 95)),
        **cov,
        **uniform,
        **sep,
        "mean_centroid_auc": float(auc["centroid_similarity_auc"].mean()),
        "mean_neg_above_pos_p10_ratio": float(auc["neg_above_pos_p10_ratio"].mean()),
    }


def plot_summary(summary: pd.DataFrame, out_path: Path) -> None:
    metrics = [
        "effective_rank_entropy",
        "participation_ratio",
        "top10_var_ratio",
        "between_within_ratio",
        "mean_centroid_auc",
        "mean_neg_above_pos_p10_ratio",
    ]
    plot_df = summary.melt(id_vars=["model"], value_vars=metrics, var_name="metric", value_name="value")
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), dpi=220)
    for ax, metric in zip(axes.reshape(-1), metrics, strict=True):
        g = plot_df[plot_df["metric"] == metric]
        sns.barplot(g, x="model", y="value", hue="model", ax=ax, palette=["#D1495B", "#4C78A8"], legend=False)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Embedding-space health metrics", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, out_root: Path) -> None:
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            f"| {row['model']} | {row['effective_rank_entropy']:.1f} | {row['top10_var_ratio']:.3f} | "
            f"{row['between_within_ratio']:.2f} | {row['mean_centroid_auc']:.3f} | "
            f"{row['mean_neg_above_pos_p10_ratio']:.3f} |"
        )
    text = "\n".join(
        [
            "# 海淀 V1 Embedding 空间诊断",
            "",
            "本报告检查 64 维 embedding 是否出现整体坍缩、维度未被充分使用、不同 OSM 类别中心距离过近，以及负样本靠近正类中心导致假正例风险的问题。",
            "",
            "## 核心指标",
            "",
            "| 模型 | 有效维度 | Top10 方差占比 | 类间/类内距离比 | 类中心 AUC | 负样本靠近正类比例 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *rows,
            "",
            "解释：有效维度越高越不容易坍缩；Top10 方差占比越高表示少数维度主导越强；类间/类内距离比越高表示不同类别分得越开；类中心 AUC 越高表示用类别中心做最近邻越容易区分；负样本靠近正类比例越高，越容易产生假正例。",
            "",
            "## 可视化",
            "",
            "![summary](figures/summary_metrics.png)",
            "",
            "![xuannv pca](figures/xuannv_pca_scatter.png)",
            "",
            "![aef pca](figures/aef_pca_scatter.png)",
            "",
            "![xuannv centroid](figures/xuannv_centroid_distance.png)",
            "",
            "![aef centroid](figures/aef_centroid_distance.png)",
            "",
        ]
    )
    (out_root / "embedding_space_diagnostics.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    xuannv_index = index_embeddings(args.xuannv_root, args.region, args.xuannv_month)
    aef_index = index_embeddings(args.aef_root, args.region, args.aef_month)
    patch_ids = available_patch_ids(args.label_root, [xuannv_index, aef_index])[: args.max_patches]
    if not patch_ids:
        raise RuntimeError("No shared patch ids found between labels and embeddings.")

    locations = collect_pixel_locations(
        args.label_root,
        patch_ids,
        args.samples_per_class,
        args.background_samples,
        rng,
    )
    sample_counts = {task: int(sum(len(ys) for _, ys, _ in items)) for task, items in locations.items()}
    (args.output_root / "sample_counts.json").write_text(json.dumps(sample_counts, indent=2), encoding="utf-8")

    specs = [
        EmbeddingSpec("xuannv", args.xuannv_root, args.region, args.xuannv_month),
        EmbeddingSpec("aef", args.aef_root, args.region, args.aef_month),
    ]
    summaries = [diagnose(spec, locations, rng, args.output_root) for spec in specs]
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(args.output_root / "embedding_space_summary.csv", index=False)
    plot_summary(summary_df, args.output_root / "figures" / "summary_metrics.png")
    write_report(summary_df, args.output_root)
    print(args.output_root / "embedding_space_diagnostics.md")


if __name__ == "__main__":
    main()
