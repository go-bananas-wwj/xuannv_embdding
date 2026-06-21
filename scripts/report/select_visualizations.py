from pathlib import Path
import shutil

src_root = Path("/data/xuannv_embedding/experiments/v1.0/visualizations")
dst = Path("/root/workspace/report/assets")
dst.mkdir(parents=True, exist_ok=True)

tasks = ["construction", "building_change", "farm_change", "rubbish", "construction_joint"]
selected = {}

for task in tasks:
    task_dir = src_root / task
    if not task_dir.exists():
        print(f"Task dir not found: {task_dir}")
        continue
    # Pick first fold and first overlay patch
    overlays = sorted(task_dir.glob("fold_*/patch_*_overlay.png"))
    pr_curves = sorted(task_dir.glob("fold_*/pr_curve.png"))
    if overlays:
        first = overlays[0]
        out = dst / f"{task}_overlay_example.png"
        shutil.copy(first, out)
        selected[f"{task}_overlay"] = str(out)
    else:
        print(f"No overlay found for {task}")
    if pr_curves:
        first = pr_curves[0]
        out = dst / f"{task}_pr_curve_example.png"
        shutil.copy(first, out)
        selected[f"{task}_pr_curve"] = str(out)
    else:
        print(f"No PR curve found for {task}")

for k, v in selected.items():
    print(f"{k}: {v}")
