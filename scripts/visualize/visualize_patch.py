"""可视化单个 patch 在所有可用数据源上的影像。

用法示例
--------
python -m scripts.visualize.visualize_patch \
    --patch-id 20250103_p013_r000 \
    --processed-dir /data/xuannv_embedding/processed/haidian \
    --manifest /data/xuannv_embedding/processed/haidian/manifest.json \
    --output-dir scripts/visualize/outputs
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap


# ESA WorldCover v100 官方颜色（0 为背景/无效）
WORLDCOVER_COLORS = {
    0: (0, 0, 0, 0),
    10: (0, 100, 0),
    20: (255, 187, 34),
    30: (255, 255, 76),
    40: (240, 150, 255),
    50: (250, 0, 0),
    60: (180, 180, 180),
    70: (240, 240, 240),
    80: (0, 100, 200),
    90: (0, 150, 160),
    95: (0, 207, 117),
    100: (250, 230, 160),
}


# 为不在 ESA 官方映射中的类别准备的备用颜色
_FALLBACK_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
]


def _worldcover_cmap(classes: set[int]) -> ListedColormap:
    """根据实际出现的类别生成颜色映射；0 为背景/无效，显示为白色。"""
    max_class = max(classes | {0})
    colors = [(1, 1, 1, 1)] * (max_class + 1)  # 默认白色（含 0）
    fallback_idx = 0
    for c in range(1, max_class + 1):
        if c in WORLDCOVER_COLORS:
            rgb = WORLDCOVER_COLORS[c]
        else:
            rgb = _FALLBACK_COLORS[fallback_idx % len(_FALLBACK_COLORS)]
            fallback_idx += 1
        colors[c] = (rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, 1.0)
    return ListedColormap(colors)


def _normalize_rgb(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    percentile: tuple[float, float] = (2, 98),
    gamma: float = 1.0,
) -> np.ndarray:
    """把三个单波段归一化到 [0,1] 并合成 RGB。"""
    stack = np.stack([r, g, b], axis=-1).astype(np.float32)
    for i in range(3):
        band = stack[..., i]
        valid = np.isfinite(band) & (band > 0)
        if valid.any():
            vmin, vmax = np.percentile(band[valid], percentile)
            band = np.clip((band - vmin) / max(vmax - vmin, 1e-6), 0, 1)
        else:
            band = np.zeros_like(band)
        stack[..., i] = band ** gamma
    return np.nan_to_num(stack, nan=0.0)


def _read_s2_true_color(path: Path) -> np.ndarray:
    """读取 S2 patch 并合成真彩色 (B04, B03, B02)。"""
    with rasterio.open(path) as src:
        desc = [d.upper() if d else "" for d in src.descriptions]
        try:
            r_idx = desc.index("B04") + 1
            g_idx = desc.index("B03") + 1
            b_idx = desc.index("B02") + 1
        except ValueError:
            # 没有 band description 时回退到前 3 个波段
            r_idx, g_idx, b_idx = 3, 2, 1
        r = src.read(r_idx).astype(np.float32)
        g = src.read(g_idx).astype(np.float32)
        b = src.read(b_idx).astype(np.float32)
    return _normalize_rgb(r, g, b)


def _read_highres_optical_rgb(path: Path) -> np.ndarray:
    """高分辨率光学：按 BGR -> RGB 或前 3 波段显示。"""
    with rasterio.open(path) as src:
        count = src.count
        if count >= 3:
            # 常见 PlanetScope 顺序为 BGRN，取 band 3/2/1 作为 RGB
            r = src.read(3).astype(np.float32)
            g = src.read(2).astype(np.float32)
            b = src.read(1).astype(np.float32)
        else:
            r = g = b = src.read(1).astype(np.float32)
    return _normalize_rgb(r, g, b, percentile=(1, 99))


def _read_s1_vv(path: Path) -> np.ndarray:
    """读取 S1 VV 波段，用对数拉伸增强可读性。"""
    with rasterio.open(path) as src:
        vv = src.read(1).astype(np.float32)
    vv = np.where(vv > 0, vv, np.nan)
    vv_db = 10.0 * np.log10(vv)
    vmin, vmax = np.nanpercentile(vv_db, (2, 98))
    vv_db = np.clip((vv_db - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    return np.nan_to_num(vv_db, nan=0.0)


def _read_highres_sar(path: Path) -> np.ndarray:
    """高分辨率 SAR 已归一化到 [0,1]，直接显示。"""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
    return np.clip(arr, 0, 1)


def _read_worldcover(path: Path) -> tuple[np.ndarray, ListedColormap, set[int]]:
    """读取 WorldCover 标签并返回数组、颜色映射、实际类别。"""
    with rasterio.open(path) as src:
        arr = src.read(1)
    classes = set(int(x) for x in np.unique(arr) if x >= 0)
    return arr, _worldcover_cmap(classes), classes


def _extract_date(stem: str) -> str | None:
    """从文件名中提取第一个 YYYYMMDD 日期。"""
    for part in stem.split("_"):
        if part.isdigit() and len(part) == 8:
            return part
    return None


def _pick_nearest_date(paths: list[str], target_date: str) -> Path:
    """从文件路径中挑选日期最接近 target_date 的一个。"""
    target = int(target_date.replace("-", ""))
    best: Path | None = None
    best_diff = math.inf
    for p in paths:
        date = _extract_date(Path(p).stem)
        if date is None:
            continue
        diff = abs(int(date) - target)
        if diff < best_diff:
            best_diff = diff
            best = Path(p)
    return best or Path(paths[0])


def _make_title(source: str, path: Path) -> str:
    """根据文件名生成子图标题。"""
    date = _extract_date(path.stem)
    if date is not None:
        return f"{source}\n{date[:4]}-{date[4:6]}-{date[6:]}"
    return source


def _collect_subplots(
    record: dict[str, Any],
    processed_dir: Path,
    target_date: str,
) -> list[tuple[np.ndarray, str, Any]]:
    """按可用数据源读取影像并返回 (array, title, cmap_or_none) 列表。"""
    subplots: list[tuple[np.ndarray, str, Any]] = []

    if record.get("s2"):
        path = _pick_nearest_date(record["s2"], target_date)
        arr = _read_s2_true_color(processed_dir / path)
        subplots.append((arr, _make_title("S2 RGB", processed_dir / path), None))

    if record.get("s1"):
        path = _pick_nearest_date(record["s1"], target_date)
        arr = _read_s1_vv(processed_dir / path)
        subplots.append((arr, _make_title("S1 VV log", processed_dir / path), "gray"))

    if record.get("landsat"):
        path = _pick_nearest_date(record["landsat"], target_date)
        with rasterio.open(processed_dir / path) as src:
            # 尽量用真彩色 (B4,B3,B2)；若只有 6 波段则取 3/2/1
            count = src.count
            r = src.read(min(4, count)).astype(np.float32)
            g = src.read(min(3, count)).astype(np.float32)
            b = src.read(min(2, count)).astype(np.float32)
        arr = _normalize_rgb(r, g, b, percentile=(2, 98))
        subplots.append((arr, _make_title("Landsat RGB", processed_dir / path), None))

    if record.get("worldcover"):
        path = Path(record["worldcover"][0])
        arr, cmap, _ = _read_worldcover(processed_dir / path)
        subplots.append((arr, _make_title("WorldCover", processed_dir / path), cmap))

    if record.get("highres_optical"):
        path = _pick_nearest_date(record["highres_optical"], target_date)
        arr = _read_highres_optical_rgb(processed_dir / path)
        subplots.append((arr, _make_title("High-res Optical", processed_dir / path), None))

    if record.get("highres_sar"):
        path = _pick_nearest_date(record["highres_sar"], target_date)
        arr = _read_highres_sar(processed_dir / path)
        subplots.append((arr, _make_title("High-res SAR", processed_dir / path), "gray"))

    return subplots


def visualize_patch(
    patch_id: str,
    processed_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    target_date: str | None = None,
    dpi: int = 150,
) -> Path:
    """为指定 patch 生成一张多数据源对比图。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path) as f:
        manifest = json.load(f)

    record = next((r for r in manifest if r.get("patch_id") == patch_id), None)
    if record is None:
        raise ValueError(f"manifest 中找不到 patch_id={patch_id}")

    if target_date is None and record.get("s2"):
        # 默认用 S2 影像的日期作为基准
        stem = Path(record["s2"][0]).stem
        parts = stem.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            target_date = parts[1]
    target_date = target_date or "20250101"

    subplots = _collect_subplots(record, processed_dir, target_date)
    if not subplots:
        raise ValueError(f"patch {patch_id} 没有任何可用的数据源")

    n = len(subplots)
    ncols = 3
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False
    )

    for ax, (arr, title, cmap) in zip(axes.flat, subplots):
        if cmap is not None:
            ax.imshow(arr, cmap=cmap, interpolation="nearest")
        else:
            ax.imshow(arr, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # 隐藏多余的子图
    for ax in axes.flat[n:]:
        ax.axis("off")

    fig.suptitle(f"Patch: {patch_id}", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = output_dir / f"patch_{patch_id}.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="可视化单个 patch 的多源数据")
    parser.add_argument("--patch-id", required=True, help="patch ID，例如 20250103_p013_r000")
    parser.add_argument(
        "--processed-dir",
        default="/data/xuannv_embedding/processed/haidian",
        help="processed 目录根路径",
    )
    parser.add_argument(
        "--manifest",
        default="/data/xuannv_embedding/processed/haidian/manifest.json",
        help="manifest.json 路径",
    )
    parser.add_argument(
        "--output-dir",
        default="scripts/visualize/outputs",
        help="输出图片目录",
    )
    parser.add_argument(
        "--target-date",
        default=None,
        help="基准日期 (YYYYMMDD)，用于挑选各源最接近的时相",
    )
    parser.add_argument("--dpi", type=int, default=150, help="输出 DPI")
    args = parser.parse_args()

    out_path = visualize_patch(
        patch_id=args.patch_id,
        processed_dir=Path(args.processed_dir),
        manifest_path=Path(args.manifest),
        output_dir=Path(args.output_dir),
        target_date=args.target_date,
        dpi=args.dpi,
    )
    print(f"已保存可视化结果: {out_path}")


if __name__ == "__main__":
    main()
