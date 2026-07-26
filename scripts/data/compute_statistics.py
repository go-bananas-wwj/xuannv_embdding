"""遍历 processed patches 计算各数据源各波段的均值/标准差，输出 JSON。"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 默认输出根目录，保持与项目数据目录约定一致
DEFAULT_OUTPUT_ROOT = Path("/data/xuannv_embedding/statistics")


def _valid_mask(values: np.ndarray, nodata: float | None) -> npt.NDArray[np.bool_]:
    """生成有效像素掩码：排除 nan、inf 以及 nodata 值。"""
    mask = np.isfinite(values)
    if nodata is not None and not math.isnan(nodata):
        mask &= values != nodata
    return mask


class _WelfordAccumulator:
    """Welford 在线算法累加器，增量计算单变量流的均值与方差。"""

    def __init__(self) -> None:
        self.count: int = 0
        self.mean: float = 0.0
        self.m2: float = 0.0

    def update(self, values: np.ndarray) -> None:
        """接入一个新的有效值数组（已过滤 nan/inf/nodata）。"""
        if values.size == 0:
            return
        # 使用 float64 精度计算，避免大数求和误差
        values = values.astype(np.float64, copy=False)
        n = values.size
        self.count += n

        # 批量 Welford 更新：先得到当前批次均值，再合并到全局统计量
        batch_mean = float(values.mean())
        delta = batch_mean - self.mean
        self.mean += delta * n / self.count

        # 合并平方差：M2_new = M2_old + M2_batch + delta^2 * n_old * n_batch / n_new
        m2_batch = float(((values - batch_mean) ** 2).sum())
        self.m2 += m2_batch + delta * delta * (self.count - n) * n / self.count

    def std(self) -> float:
        """返回总体标准差（除以 n）。"""
        if self.count == 0:
            return float("nan")
        return math.sqrt(self.m2 / self.count)

    def to_dict(self) -> dict[str, float | int]:
        """导出为可序列化的字典。"""
        return {
            "mean": float(self.mean),
            "std": self.std(),
            "count": self.count,
        }


def _collect_tif_files(source_dir: Path, max_patches: int | None) -> list[Path]:
    """收集 source 目录下的所有 .tif 文件（排除 *_mask.tif），支持数量上限。"""
    if not source_dir.exists():
        logger.warning("目录不存在，跳过：%s", source_dir)
        return []

    files = sorted(
        p for p in source_dir.glob("*.tif") if p.is_file() and not p.stem.endswith("_mask")
    )
    if not files:
        logger.warning("未找到 .tif 文件：%s", source_dir)
        return []

    if max_patches is not None and max_patches > 0:
        files = files[:max_patches]
    return files


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_fold_statistics_registry(
    split_path: Path,
    manifest_audits: list[Path],
    statistics_root: Path,
    protocol_name: str = "rse_v5_registered_20260726",
) -> dict[str, object]:
    """Declare fold-only normalization inputs without scanning production rasters."""
    split_hash = sha256(split_path)
    folds: dict[str, dict[str, object]] = {}
    for audit_path in manifest_audits:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        fold_id = str(audit["fold"])
        if audit.get("spatial_split_sha256") != split_hash:
            raise ValueError("Manifest audit source split hash does not match statistics split")
        if audit.get("protocol_name") not in (None, protocol_name):
            raise ValueError("Manifest audit protocol name does not match statistics protocol")
        train_pool = audit.get("manifests", {}).get("train_pool")
        if not isinstance(train_pool, dict):
            raise ValueError(f"Manifest audit has no full train pool for fold {fold_id}")
        source_manifest = Path(str(train_pool["path"]))
        if train_pool.get("sha256") != sha256(source_manifest):
            raise ValueError(f"Manifest audit train pool hash mismatch for fold {fold_id}")
        statistics_dir = statistics_root / f"fold{fold_id}"
        folds[fold_id] = {
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": sha256(source_manifest),
            "manifest_audit": str(audit_path),
            "manifest_audit_sha256": sha256(audit_path),
            "statistics_dir": str(statistics_dir),
            "statistics_audit": str(statistics_dir / "fold_statistics_audit.json"),
            "status": "registered_pending_materialization",
        }
    return {
        "schema_version": 1,
        "protocol_name": protocol_name,
        "source_split": str(split_path),
        "source_split_sha256": split_hash,
        "normalization_scope": "full_upstream_training_manifest_only",
        "folds": folds,
    }


def validate_fold_statistics_registry(
    registry: dict[str, object],
    fold_id: int,
    manifest_audit_path: Path,
    split_path: Path,
) -> dict[str, object]:
    """Verify the fold-only statistics declaration against immutable provenance."""
    if registry.get("source_split_sha256") != sha256(split_path):
        raise ValueError("Fold statistics registry source split hash mismatch")
    folds = registry.get("folds")
    if not isinstance(folds, dict) or str(fold_id) not in folds:
        raise ValueError(f"Fold statistics registry has no fold {fold_id}")
    entry = folds[str(fold_id)]
    if not isinstance(entry, dict):
        raise ValueError(f"Fold statistics registry entry is invalid for fold {fold_id}")
    if entry.get("manifest_audit_sha256") != sha256(manifest_audit_path):
        raise ValueError("Fold statistics registry manifest audit hash mismatch")
    audit = json.loads(manifest_audit_path.read_text(encoding="utf-8"))
    train_pool = audit.get("manifests", {}).get("train_pool", {})
    if entry.get("source_manifest") != train_pool.get("path"):
        raise ValueError("Fold statistics registry source manifest path mismatch")
    if entry.get("source_manifest_sha256") != train_pool.get("sha256"):
        raise ValueError("Fold statistics registry source manifest hash mismatch")
    return entry


def mark_fold_statistics_materialized(
    registry: dict[str, object], fold_id: int, statistics_audit_path: Path
) -> None:
    """Seal a completed fold's statistics audit into its immutable registry entry."""
    folds = registry.get("folds")
    if not isinstance(folds, dict) or str(fold_id) not in folds:
        raise ValueError(f"Fold statistics registry has no fold {fold_id}")
    entry = folds[str(fold_id)]
    if not isinstance(entry, dict):
        raise ValueError(f"Fold statistics registry entry is invalid for fold {fold_id}")
    audit = json.loads(statistics_audit_path.read_text(encoding="utf-8"))
    if audit.get("source_manifest") != entry.get("source_manifest") or audit.get(
        "source_manifest_sha256"
    ) != entry.get("source_manifest_sha256"):
        raise ValueError("Fold statistics audit source manifest does not match registry")
    statistics_files = audit.get("statistics_files")
    if not isinstance(statistics_files, dict) or not statistics_files:
        raise ValueError("Fold statistics audit has no statistics files")
    sources = sorted(statistics_files)
    source_key = "__".join(sources)
    statistics_audits = entry.setdefault("statistics_audits", {})
    if not isinstance(statistics_audits, dict):
        raise ValueError("Fold statistics registry audits entry is invalid")
    audit_hash = sha256(statistics_audit_path)
    existing = statistics_audits.get(source_key)
    if existing is not None and existing.get("sha256") != audit_hash:
        raise ValueError("Fold statistics audit variant is already sealed with another hash")
    statistics_audits[source_key] = {
        "path": str(statistics_audit_path),
        "sha256": audit_hash,
        "sources": sources,
    }
    entry["status"] = "materialized"
    if Path(str(entry.get("statistics_audit", ""))) == statistics_audit_path:
        entry["statistics_audit_sha256"] = audit_hash


def compute_statistics(
    processed_dir: Path,
    source: str,
    max_patches: int | None = None,
    source_dirs: dict[str, str] | None = None,
    files: list[Path] | None = None,
) -> dict[str, list[float] | list[int] | int | str]:
    """计算单个数据源所有波段的统计量。

    参数
    ----------
    processed_dir:
        processed 数据根目录，预期子目录结构为 processed_dir / patches / source / *.tif。
    source:
        数据源名称，例如 "s2"、"s1"、"landsat"。
    max_patches:
        用于快速测试的最大 patch 数量，None 表示处理全部。
    source_dirs:
        source 到相对子目录的显式映射；未提供时使用 ``patches/<source>``，
        不存在则回退到 ``<source>``。

    返回
    -------
    dict[str, list[float] | list[int] | int | str]:
        包含以下字段的统计字典：
        - mean: 各波段有效像素均值列表。
        - std: 各波段有效像素标准差列表。
        - count: 所有波段有效像素总数。
        - band_counts: 各波段有效像素数量列表。
        - num_files: 成功处理的文件数量。
        - source: 数据源名称。
    """
    if source_dirs and source in source_dirs:
        rel_dir = source_dirs[source]
    else:
        rel_dir = f"patches/{source}"
        if not (processed_dir / rel_dir).exists():
            rel_dir = source
    source_dir = processed_dir / rel_dir
    if files is None:
        files = _collect_tif_files(source_dir, max_patches)
    if not files:
        return {
            "mean": [],
            "std": [],
            "count": 0,
            "num_files": 0,
            "source": source,
        }

    accumulators: list[_WelfordAccumulator] | None = None
    num_success = 0

    for idx, tif_path in enumerate(files, start=1):
        try:
            with rasterio.open(tif_path) as src:
                nodata = src.nodata
                # 读取所有波段，形状为 (bands, height, width)
                data = src.read(out_dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            logger.warning("读取文件失败，跳过 %s: %s", tif_path, exc)
            continue

        if accumulators is None:
            num_bands = data.shape[0]
            accumulators = [_WelfordAccumulator() for _ in range(num_bands)]
        else:
            num_bands = data.shape[0]
            if num_bands != len(accumulators):
                logger.warning(
                    "波段数量不一致：%s 有 %d 个波段，期望 %d，跳过该文件",
                    tif_path,
                    num_bands,
                    len(accumulators),
                )
                continue

        for band_idx in range(num_bands):
            band_data = data[band_idx]
            mask = _valid_mask(band_data, nodata)
            accumulators[band_idx].update(band_data[mask])

        num_success += 1
        if idx % 100 == 0 or idx == len(files):
            logger.info("%s: 已处理 %d/%d 个文件", source, idx, len(files))

    if accumulators is None:
        logger.warning("%s: 没有成功读取任何文件", source)
        return {
            "mean": [],
            "std": [],
            "count": 0,
            "num_files": 0,
            "source": source,
        }

    means = [acc.mean for acc in accumulators]
    stds = [acc.std() for acc in accumulators]
    band_counts = [acc.count for acc in accumulators]
    total_count = sum(band_counts)

    stats = {
        "mean": means,
        "std": stds,
        "count": total_count,
        "band_counts": band_counts,
        "num_files": num_success,
        "source": source,
    }

    # 记录每个 band 的简要统计结果
    logger.info(
        "%s: bands=%d, total_valid_pixels=%d, files=%d/%d",
        source,
        len(means),
        total_count,
        num_success,
        len(files),
    )
    for band_idx, (m, s, c) in enumerate(zip(means, stds, band_counts)):
        logger.info("  band %d: mean=%.6f, std=%.6f, count=%d", band_idx, m, s, c)

    return stats


def manifest_source_files(processed_dir: Path, manifest_path: Path, source: str) -> list[Path]:
    """Resolve only a fold's source rasters from its full upstream train manifest."""
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = [
        processed_dir / relative_path
        for record in records
        for relative_path in (record.get(source) or [])
        if isinstance(relative_path, str) and relative_path
    ]
    return sorted(set(files))


def write_fold_statistics_audit(
    audit_path: Path,
    source_manifest_path: Path,
    statistics_paths: dict[str, Path],
    protocol_name: str,
) -> None:
    """Seal fold-only normalization outputs to their complete upstream train manifest."""
    payload = {
        "schema_version": 1,
        "protocol_name": protocol_name,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256(source_manifest_path),
        "statistics_files": {
            source: {"path": str(path), "sha256": sha256(path)}
            for source, path in statistics_paths.items()
        },
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="计算 processed patches 各数据源各波段的 mean/std 统计量。",
    )
    parser.add_argument(
        "--processed-dir",
        required=True,
        type=Path,
        help="processed 数据根目录，例如 /data/xuannv_embedding/processed/harbin/scenes",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="统计量输出目录，例如 /data/xuannv_embedding/statistics/harbin",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["s2", "s1", "landsat"],
        help="需要统计的数据源名称列表",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="每个 source 最多处理的 patch 数量，用于快速测试",
    )
    parser.add_argument(
        "--source-dir",
        action="append",
        default=[],
        help="source 到子目录的映射，格式 source=relative_dir；例如 s2=patches/s2。"
        "未指定的 source 默认使用 patches/<source>。",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="仅使用该训练 manifest 中列出的栅格计算统计量。",
    )
    parser.add_argument(
        "--fold-manifest-audit",
        action="append",
        type=Path,
        default=[],
        help="构建 fold-only 统计量注册表时使用的 manifest audit，可重复。",
    )
    parser.add_argument(
        "--statistics-registry",
        type=Path,
        help="写入 fold-only 统计量注册表而不扫描栅格。",
    )
    parser.add_argument(
        "--spatial-split",
        type=Path,
        help="fold-only 统计量注册表所绑定的 spatial split。",
    )
    parser.add_argument(
        "--protocol-name",
        default="rse_v5_registered_20260726",
    )
    parser.add_argument(
        "--statistics-audit",
        type=Path,
        help="由 --manifest 产生的 fold-only 统计量 provenance sidecar。",
    )
    parser.add_argument("--fold", type=int, help="写入 materialized statistics registry 的 fold。")
    args = parser.parse_args()

    if args.fold_manifest_audit:
        if args.statistics_registry is None or args.spatial_split is None:
            raise ValueError(
                "构建 fold statistics registry 需要 --statistics-registry 和 --spatial-split"
            )
        registry = build_fold_statistics_registry(
            args.spatial_split,
            args.fold_manifest_audit,
            args.output_dir,
            args.protocol_name,
        )
        args.statistics_registry.parent.mkdir(parents=True, exist_ok=True)
        args.statistics_registry.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return

    source_dirs: dict[str, str] = {}
    for mapping in args.source_dir:
        if "=" not in mapping:
            raise ValueError(f"--source-dir 格式错误，应为 source=relative_dir: {mapping}")
        source, rel_dir = mapping.split("=", 1)
        source_dirs[source] = rel_dir

    args.output_dir.mkdir(parents=True, exist_ok=True)

    statistics_paths: dict[str, Path] = {}
    for source in args.sources:
        files = (
            manifest_source_files(args.processed_dir, args.manifest, source)
            if args.manifest is not None
            else None
        )
        stats = compute_statistics(
            args.processed_dir,
            source,
            max_patches=args.max_patches,
            source_dirs=source_dirs,
            files=files,
        )
        out_path = args.output_dir / f"{source}_stats.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info("已保存统计文件：%s", out_path)
        statistics_paths[source] = out_path

    if args.statistics_audit is not None:
        if args.manifest is None:
            raise ValueError("--statistics-audit 需要 --manifest")
        write_fold_statistics_audit(
            args.statistics_audit,
            args.manifest,
            statistics_paths,
            args.protocol_name,
        )
        if args.statistics_registry is not None:
            if args.fold is None:
                raise ValueError("更新 materialized statistics registry 需要 --fold")
            registry = json.loads(args.statistics_registry.read_text(encoding="utf-8"))
            mark_fold_statistics_materialized(registry, args.fold, args.statistics_audit)
            args.statistics_registry.write_text(
                json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )


if __name__ == "__main__":
    main()
