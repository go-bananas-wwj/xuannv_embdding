from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 支持的 patch 影像扩展名
PATCH_EXTENSIONS: set[str] = {".tif", ".tiff"}


def _extract_patch_id(stem: str, source: str) -> str:
    """从文件 stem 中提取 patch_id。

    约定文件名格式为 ``{source}_{patch_id}``，例如 ``s2_20250129_p000_r000``。
    返回第一个下划线之后的部分作为 patch_id；若无法识别则返回完整 stem。
    """
    prefix = f"{source}_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


def _find_patches_for_patch_id(
    source_dir: Path,
    patch_id: str,
    source: str,
) -> list[Path] | None:
    """在指定 source 目录中查找与 patch_id 匹配的 patch 文件。

    优先精确匹配 ``{source}_{patch_id}``；当不同 sensor 的观测日期不一致时，
    退而匹配 patch_id 末尾的网格编号 ``pXXX_rXXX``，以便将同一空间格网但不同
    日期的影像归到同一条目。
    """
    if not source_dir.exists():
        return None

    candidates: list[Path] = []
    expected_stem = f"{source}_{patch_id}"
    grid_match = re.search(r"p\d{3}_r\d{3}$", patch_id)
    grid_suffix = f"_{grid_match.group()}" if grid_match else None

    for path in source_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in PATCH_EXTENSIONS:
            continue
        stem = path.stem
        # 完全同名，或以 ``_{patch_id}`` 结尾（日期也相同）
        if stem == expected_stem or stem == patch_id or stem.endswith(f"_{patch_id}"):
            candidates.append(path)
        elif grid_suffix and stem.endswith(grid_suffix):
            candidates.append(path)

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.name)
    return candidates


def _index_source_dir(
    processed_dir: Path,
    source: str,
    rel_dir: str,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """扫描 source 子目录，建立两个索引。

    返回 ``(patch_id_index, grid_index)``：
    - ``patch_id_index`` 以完整 ``{date}_{grid}`` 为键；
    - ``grid_index`` 仅以末尾 ``pXXX_rXXX`` 网格编号为键，用于跨源日期不一致时匹配。
    跳过 *_mask.tif 掩膜文件，只保留数据影像。
    """
    patch_index: dict[str, list[Path]] = {}
    grid_index: dict[str, list[Path]] = {}
    source_dir = processed_dir / rel_dir
    if not source_dir.exists():
        return patch_index, grid_index

    grid_match = re.compile(r"p\d{3}_r\d{3}$")

    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in PATCH_EXTENSIONS:
            continue
        if path.stem.endswith("_mask"):
            continue
        patch_id = _extract_patch_id(path.stem, source)
        if not patch_id:
            continue
        patch_index.setdefault(patch_id, []).append(path.relative_to(processed_dir))
        grid = grid_match.search(patch_id)
        if grid:
            grid_index.setdefault(grid.group(), []).append(path.relative_to(processed_dir))

    for idx in (patch_index, grid_index):
        for key in idx:
            idx[key].sort(key=lambda p: p.name)

    return patch_index, grid_index


def generate_manifest(
    processed_dir: Path,
    sources: list[str],
    source_dirs: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """以第一个 source 为基准生成 patch manifest。

    Args:
        processed_dir: 预处理数据根目录（region 根目录）。
        sources: 数据源名称列表，第一个 source 决定 patch_id 集合。
        source_dirs: source 到子目录的映射。例如 ``{"s2": "patches/s2", "worldcover": "labels/worldcover"}``。
            为 ``None`` 时默认子目录名与 source 名相同（向后兼容）。

    Returns:
        manifest 列表，每个元素包含 ``patch_id`` 与各 source 对应的相对路径列表；
        缺失 source 对应字段为 ``None``。
    """
    if not sources:
        return []

    if source_dirs is None:
        source_dirs = {s: s for s in sources}

    base_source = sources[0]
    if base_source not in source_dirs:
        logger.warning("基准 source 缺少目录映射: %s", base_source)
        return []

    # 为每个 source 建立 patch_id 索引与 grid 索引，避免 O(N^2) 查找
    patch_indices: dict[str, dict[str, list[Path]]] = {}
    grid_indices: dict[str, dict[str, list[Path]]] = {}
    for source in sources:
        pidx, gidx = _index_source_dir(processed_dir, source, source_dirs[source])
        patch_indices[source] = pidx
        grid_indices[source] = gidx

    if not patch_indices[base_source]:
        logger.warning("基准 source 目录为空或不存在: %s", source_dirs[base_source])
        return []

    patch_ids = sorted(patch_indices[base_source].keys())
    grid_match = re.compile(r"p\d{3}_r\d{3}$")

    manifest: list[dict[str, Any]] = []
    for patch_id in patch_ids:
        entry: dict[str, Any] = {"patch_id": patch_id}
        grid = grid_match.search(patch_id)
        grid_key = grid.group() if grid else None
        for source in sources:
            paths = patch_indices[source].get(patch_id)
            if paths is None and grid_key and source != base_source:
                paths = grid_indices[source].get(grid_key)
            entry[source] = paths if paths else None
        manifest.append(entry)

    return manifest


def _convert_paths_to_strings(obj: Any) -> Any:
    """递归将对象中的 Path 转换为字符串，便于 JSON 序列化。"""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, list):
        return [_convert_paths_to_strings(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _convert_paths_to_strings(value) for key, value in obj.items()}
    return obj


def save_manifest(manifest: list[dict[str, Any]], output_path: Path) -> None:
    """将 manifest 保存为 JSON 文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _convert_paths_to_strings(manifest)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    logger.info("manifest 已保存: %s，共 %d 条记录", output_path, len(manifest))


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """读取 JSON manifest 并将路径字符串反序列化为 Path。"""
    with path.open("r", encoding="utf-8") as f:
        manifest: list[dict[str, Any]] = json.load(f)

    for entry in manifest:
        for key, value in entry.items():
            if key == "patch_id":
                continue
            if isinstance(value, list):
                entry[key] = [Path(p) for p in value]
            elif value is None:
                entry[key] = None
            else:
                entry[key] = Path(value)

    return manifest
