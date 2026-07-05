#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import yaml


CONTINUOUS_SOURCES = {
    "s2",
    "s1",
    "landsat",
    "highres_optical_haidian",
    "highres_sar_haidian",
}


def sha256_file(path: Path, chunk_size: int = 32 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return src.stat().st_size


def collect_manifest_files(processed_root: Path, manifest_path: Path) -> tuple[set[Path], dict[str, int]]:
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    files: set[Path] = set()
    source_counts: dict[str, int] = {}
    for entry in entries:
        for source, values in entry.items():
            if source == "patch_id" or not isinstance(values, list):
                continue
            source_counts[source] = source_counts.get(source, 0) + len(values)
            for rel in values:
                path = processed_root / rel
                files.add(path)
                if source in CONTINUOUS_SOURCES and rel.endswith(".tif") and not rel.endswith("_mask.tif"):
                    files.add(path.with_name(f"{path.stem}_mask.tif"))
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"{len(missing)} files referenced by manifest are missing:\n{preview}")
    return files, source_counts


def collect_label_files(label_roots: dict[str, str]) -> set[Path]:
    files: set[Path] = set()
    for root in label_roots.values():
        label_root = Path(root)
        if not label_root.exists():
            raise FileNotFoundError(label_root)
        for path in label_root.rglob("*"):
            if path.is_file():
                files.add(path)
    return files


def relative_to_any(path: Path, roots: list[tuple[str, Path]]) -> Path:
    for prefix, root in roots:
        try:
            return Path(prefix) / path.relative_to(root)
        except ValueError:
            continue
    raise ValueError(f"{path} is not under any configured root")


def write_readme(path: Path, metadata: dict) -> None:
    text = f"""# 玄女海淀 V1 P10C 筛选后训练数据包

本目录用于复现 `haidian-embedding-v1` 的 P10C 生产版训练。数据已经按训练 manifest 做过筛选，只包含本次训练实际引用的图块、有效像素 mask、OSM 弱语义标签、统计文件和配置文件。

## 基本信息

| 项目 | 内容 |
|---|---|
| 区域 | 北京市海淀区 |
| 图块数量 | {metadata['num_entries']} |
| 时间范围 | 2025-12 至 2026-05 |
| 模型版本 | {metadata['release']} |
| 来源实验 | {metadata['source_experiment']} |
| 训练配置 | `configs/v2_p10c_haidian_202512_202605_osm_semantic_hardneg_20260704.yaml` |
| 训练 manifest | `processed/haidian/manifest_p6a_202512_202605_pixelmask_clean_osm_landcover.json` |

## 目录结构

```text
{metadata['package_name']}/
  README.md
  manifest.json
  filelist.sha256
  configs/
    v2_p10c_haidian_202512_202605_osm_semantic_hardneg_20260704.yaml
    haidian_embedding_v1.yaml
  processed/haidian/
    manifest_p6a_202512_202605_pixelmask_clean_osm_landcover.json
    manifest_p6a_202512_202605_pixelmask_clean.quality_meta.json
    manifest_p6a_202512_202605_pixelmask_clean.quality_records.json
    manifest_p6a_202512_202605_pixelmask.meta.json
    manifest_p5h_random20_val_seed42.json
    patches/
      s2/
      s1/
      landsat/
      highres_optical/
      highres_sar/
    labels/
      osm_building/
      osm_major_road/
      osm_minor_road/
      osm_rail/
      osm_water/
      osm_green/
      osm_agriculture/
      osm_residential/
      osm_commercial/
      osm_industrial/
      osm_construction/
      osm_path_walk/
      osm_playground/
      osm_landcover/
  statistics/haidian/
```

## 数据内容

- `patches/s2/`：Sentinel-2 筛选后观测及对应 `_mask.tif` 有效像素掩码。
- `patches/s1/`：Sentinel-1 SAR 观测及对应 `_mask.tif`。
- `patches/landsat/`：Landsat 筛选后观测及对应 `_mask.tif`。
- `patches/highres_optical/`：高分辨率光学图块及对应 `_mask.tif`。
- `patches/highres_sar/`：高分辨率 SAR 图块及对应 `_mask.tif`。
- `labels/osm_*`：OSM 弱语义标签，包括建筑、道路、水体、绿地、农业、居住、商业、工业、施工、步行路径、操场和合并土地覆盖标签。
- `statistics/haidian/`：训练归一化使用的各源统计量。

## 筛选口径

训练使用 `manifest_p6a_202512_202605_pixelmask_clean_osm_landcover.json`。该 manifest 基于 pixel valid mask 和质量评分筛选 S2/Landsat 观测，并补充 OSM landcover 弱语义标签。

质量筛选摘要：

- 输入 manifest：`manifest_p6a_202512_202605_pixelmask.json`
- 输出 manifest：`manifest_p6a_202512_202605_pixelmask_clean.json`
- 参与评分源：S2、Landsat
- 每个源/月份优先保留质量靠前观测
- 评分观测数：{metadata['quality_meta'].get('num_scored_observations', 'N/A')}
- 保留观测数：{metadata['quality_meta'].get('num_kept_observations', 'N/A')}

## 文件统计

| 类型 | 数量 |
|---|---:|
| manifest 引用和 mask 文件 | {metadata['manifest_file_count']} |
| OSM 弱语义标签文件 | {metadata['label_file_count']} |
| 统计文件 | {metadata['statistics_file_count']} |
| 包内总文件 | {metadata['total_file_count']} |
| 包内未压缩大小 | {metadata['total_size_gb']:.2f} GB |

## 使用方式

解压：

```bash
tar --use-compress-program=zstd -xf haidian_embedding_v1_p10c_filtered_training_data_202512_202605.tar.zst
```

如果需要在原项目中复现训练，可将解压后的 `processed/haidian` 与 `statistics/haidian` 对应到 `/data/xuannv_embedding/processed/haidian` 和 `/data/xuannv_embedding/statistics/haidian`，或修改配置文件中的 `data.root`、`manifest_path`、`statistics_dir`。

## 校验

压缩包同目录提供：

- `haidian_embedding_v1_p10c_filtered_training_data_202512_202605.tar.zst.sha256`
- 包内 `filelist.sha256`

可以用：

```bash
sha256sum -c haidian_embedding_v1_p10c_filtered_training_data_202512_202605.tar.zst.sha256
```

## 注意

- OSM 标签是弱语义标签，存在漏标、错标和时效误差。
- 该包服务于海淀区 P10C 生产版本复现，不声明跨城市泛化。
- 权重和 embedding 另存于 ModelScope 的 `checkpoints/` 与 `embeddings/` 目录。
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package filtered Haidian P10C training data.")
    parser.add_argument("--config", type=Path, default=Path("configs/v2_p10c_haidian_202512_202605_osm_semantic_hardneg_20260704.yaml"))
    parser.add_argument("--production-config", type=Path, default=Path("configs/production/haidian_embedding_v1.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("/data/xuannv_embedding/releases/haidian_embedding_v1_p10c_training_data"))
    parser.add_argument("--skip-archive", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_cfg = config["data"]
    processed_root = Path(data_cfg["root"])
    manifest_path = Path(data_cfg["manifest_path"])
    statistics_root = Path(data_cfg["statistics_dir"])

    package_name = "haidian_embedding_v1_p10c_filtered_training_data_202512_202605"
    staging = args.output_root / package_name
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    manifest_files, source_counts = collect_manifest_files(processed_root, manifest_path)
    label_files = collect_label_files(data_cfg["supervised_label_roots"])
    statistics_files = {path for path in statistics_root.rglob("*") if path.is_file()}

    extra_files = {
        manifest_path,
        Path(data_cfg["train_manifest_path"]),
        Path(data_cfg["val_manifest_path"]),
        processed_root / "manifest_p6a_202512_202605_pixelmask_clean.quality_meta.json",
        processed_root / "manifest_p6a_202512_202605_pixelmask_clean.quality_records.json",
        processed_root / "manifest_p6a_202512_202605_pixelmask.meta.json",
    }
    extra_files = {path for path in extra_files if path.exists()}

    all_data_files = manifest_files | label_files | statistics_files | extra_files
    roots = [
        ("processed/haidian", processed_root),
        ("statistics/haidian", statistics_root),
    ]

    total_size = 0
    copied: list[tuple[Path, Path]] = []
    for src in sorted(all_data_files):
        rel = relative_to_any(src, roots)
        dst = staging / rel
        total_size += copy_file(src, dst)
        copied.append((src, rel))

    config_dir = staging / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    copy_file(args.config, config_dir / args.config.name)
    if args.production_config.exists():
        copy_file(args.production_config, config_dir / args.production_config.name)

    quality_meta_path = processed_root / "manifest_p6a_202512_202605_pixelmask_clean.quality_meta.json"
    quality_meta = json.loads(quality_meta_path.read_text(encoding="utf-8")) if quality_meta_path.exists() else {}
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = {
        "release": "haidian-embedding-v1",
        "package_name": package_name,
        "source_experiment": config["experiment"]["name"],
        "num_entries": len(entries),
        "manifest_path": str(manifest_path),
        "source_counts": source_counts,
        "quality_meta": quality_meta,
        "manifest_file_count": len(manifest_files),
        "label_file_count": len(label_files),
        "statistics_file_count": len(statistics_files),
        "total_file_count": len(copied) + 4,
        "total_size_bytes": total_size,
        "total_size_gb": total_size / 1024**3,
    }
    (staging / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(staging / "README.md", metadata)

    filelist_path = staging / "filelist.sha256"
    with filelist_path.open("w", encoding="utf-8") as handle:
        for path in sorted(p for p in staging.rglob("*") if p.is_file()):
            rel = path.relative_to(staging)
            if rel == Path("filelist.sha256"):
                continue
            handle.write(f"{sha256_file(path)}  {rel.as_posix()}\n")

    if args.skip_archive:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        print(f"staging={staging}")
        return

    archive = args.output_root / f"{package_name}.tar.zst"
    if archive.exists():
        archive.unlink()
    subprocess.run(
        [
            "tar",
            "--use-compress-program=zstd -T0 -6",
            "-cf",
            str(archive),
            "-C",
            str(args.output_root),
            package_name,
        ],
        check=True,
    )
    sha_path = archive.with_suffix(archive.suffix + ".sha256")
    sha_path.write_text(f"{sha256_file(archive)}  {archive.name}\n", encoding="utf-8")
    metadata["archive_path"] = str(archive)
    metadata["archive_sha256"] = sha_path.read_text(encoding="utf-8").split()[0]
    metadata["archive_bytes"] = archive.stat().st_size
    (staging / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"archive={archive}")
    print(f"sha256={sha_path}")


if __name__ == "__main__":
    main()
