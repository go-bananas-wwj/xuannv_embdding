#!/usr/bin/env python3
"""使用 v3 frac1.0 最佳预测重新生成综合 patch 可视化图。"""
from pathlib import Path
import subprocess

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

for patch_id, fold in patch_fold_map.items():
    pred_dir = Path(
        f"/data/xuannv_embedding/outputs/downstream/"
        f"stage2_haidian_construction_v3_frac1.0_20260620_1814/{fold}/predictions"
    )
    cmd = [
        "python", "downstreams/scripts/visualize_patches.py",
        "--emb-root", str(emb_root),
        "--data-root", str(data_root),
        "--pred-dir", str(pred_dir),
        "--mask-dir", str(mask_dir),
        "--patch-ids", patch_id,
        "--out-dir", str(out_dir),
        "--month", "202604",
    ]
    print(f"Generating visualization for {patch_id} using {fold}...")
    subprocess.run(cmd, check=True)

    # visualize_patches.py 输出为 {patch_id}_visualization.png，重命名为 comprehensive_patch_NNN.png
    src = out_dir / f"{patch_id}_visualization.png"
    dst = out_dir / f"comprehensive_{patch_id}.png"
    if src.exists():
        src.rename(dst)
        print(f"Renamed to {dst}")

print("Done. Output files:")
for p in sorted(out_dir.glob("comprehensive_patch_*.png")):
    print(p)
