#!/usr/bin/env python3
"""使用 v3 frac1.0 最佳预测重新生成综合 patch 可视化图。"""
from pathlib import Path
import subprocess

import numpy as np
import torch

emb_root = Path("/data/xuannv_embedding/embeddings/haidian")
data_root = Path("/data/xuannv_embedding/processed/haidian")
mask_dir = Path("/data/xuannv_embedding/processed/haidian/labels/construction/masks")
out_dir = Path("/root/workspace/report/assets")
out_dir.mkdir(parents=True, exist_ok=True)

# 仅使用有标注的 patch，选取 GT 面积最大的前 3 个示例
patch_fold_map = {
    "patch_000198": "fold_1",
    "patch_000090": "fold_1",
    "patch_000209": "fold_2",
}


def _stripe_score(emb: np.ndarray) -> float:
    """衡量 embedding 中水平条纹（行方向高频噪声）的强度，越小越平滑。"""
    row_mean = emb.mean(axis=(0, 2))
    spec = np.abs(np.fft.rfft(row_mean))
    low = spec[:5].sum()
    high = spec[5:].sum()
    return high / (low + 1e-8)


def _best_pca_month(patch_id: str) -> str:
    patch_dir = emb_root / patch_id
    months = [p.stem[:6] for p in patch_dir.glob("*_embedding_map.pt")]
    best_month, best_score = None, float("inf")
    for month in months:
        emb = torch.load(
            patch_dir / f"{month}_embedding_map.pt",
            map_location="cpu",
            weights_only=True,
        ).numpy()
        score = _stripe_score(emb)
        if score < best_score:
            best_score, best_month = score, month
    return best_month if best_month else "202601"


for patch_id, fold in patch_fold_map.items():
    pred_dir = Path(
        f"/data/xuannv_embedding/outputs/downstream/"
        f"stage2_haidian_construction_v3_frac1.0_20260620_1814/{fold}/predictions"
    )
    month = _best_pca_month(patch_id)
    print(f"Generating visualization for {patch_id} using {fold}, PCA month {month}...")
    cmd = [
        "python", "downstreams/scripts/visualize_patches.py",
        "--emb-root", str(emb_root),
        "--data-root", str(data_root),
        "--pred-dir", str(pred_dir),
        "--mask-dir", str(mask_dir),
        "--patch-ids", patch_id,
        "--out-dir", str(out_dir),
        "--month", month,
        "--pca-smooth-sigma", "1.0",
    ]
    subprocess.run(cmd, check=True)

    # visualize_patches.py 输出为 {patch_id}_visualization.png，重命名为 comprehensive_patch_NNN.png
    src = out_dir / f"{patch_id}_visualization.png"
    dst = out_dir / f"comprehensive_{patch_id}.png"
    if src.exists():
        # 避免文件名冲突时覆盖失败
        if dst.exists():
            dst.unlink()
        src.rename(dst)
        print(f"Renamed to {dst}")

print("Done. Output files:")
for p in sorted(out_dir.glob("comprehensive_patch_*.png")):
    print(p)
