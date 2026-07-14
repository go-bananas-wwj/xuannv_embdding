#!/usr/bin/env python3
"""从 ESRI LULC 2023 构建独立评测标签层（论文 E4 泄露控制）。

ESRI 类值 → 二值任务层：
  1 water → esri_water；2 trees → esri_trees；5 crops → esri_crops；
  7 built → esri_built；8 bare → esri_bare；11 rangeland → esri_range。
与 OSM 标签完全独立的标注源，用于回应"OSM 既进训练又做评测"的泄露质疑。
输出结构与既有标签层一致（masks/*.tif + metadata.json + split_5fold.json）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

REPO_ROOT = Path(__file__).resolve().parents[2]
import importlib.util


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_weak = _load("osm_weak_labels", REPO_ROOT / "scripts/data/build_osm_weak_semantic_labels.py")
_split = _load("ds_split", REPO_ROOT / "downstreams/downstreams/data/split.py")

TASKS = {
    "esri_water": (1,),
    "esri_trees": (2,),
    "esri_crops": (5,),
    "esri_built": (7,),
    "esri_bare": (8,),
    "esri_range": (11,),
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--region", default="haidian")
    p.add_argument("--esri", type=Path,
                   default=Path("/data/xuannv_embedding/raw/haidian/esri_lulc_2023/esri_lulc_2023_haidian.tif"))
    p.add_argument("--processed-root", type=Path, default=Path("/data/xuannv_embedding/processed"))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    refs = _weak.choose_patch_refs(args.processed_root / args.region / "patches" / "s2")
    print(f"patch refs: {len(refs)}")

    with rasterio.open(args.esri) as src:
        esri = src.read(1)
        esri_transform, esri_crs = src.transform, src.crs

    for task, class_vals in TASKS.items():
        out_root = args.processed_root / args.region / "labels" / task
        meta_path = out_root / "metadata.json"
        if meta_path.exists() and not args.overwrite:
            print(f"skip {task}")
            continue
        binary = np.isin(esri, class_vals).astype(np.uint8)
        records = []
        for ref in refs:
            dst = np.zeros(ref.shape, dtype=np.uint8)
            reproject(
                source=binary, destination=dst,
                src_transform=esri_transform, src_crs=esri_crs,
                dst_transform=ref.transform, dst_crs=ref.crs,
                resampling=Resampling.nearest,
            )
            _weak.write_mask(out_root / "masks" / f"{ref.patch_id}.tif", ref, dst)
            pos = int(dst.sum())
            records.append({"patch_id": ref.patch_id, "positive_pixels": pos,
                            "positive_ratio": pos / float(dst.size)})
        meta = {"region": args.region, "task": task, "label_kind": "esri_lulc_2023_independent",
                "source": str(args.esri), "class_values": list(class_vals),
                "summary": _weak.summarize(records), "records": records}
        out_root.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        split = _split.create_stratified_folds(out_root / "masks", n_folds=5, val_ratio=0.1, seed=42)
        (out_root / "split_5fold.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
        print(task, meta["summary"])


if __name__ == "__main__":
    main()
