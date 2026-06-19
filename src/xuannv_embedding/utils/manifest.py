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
    base_dir = processed_dir / source_dirs[base_source]

    if not base_dir.exists():
        logger.warning("基准 source 目录不存在: %s", base_dir)
        return []

    # 按字母顺序遍历基准 source，提取 patch_id 并去重
    seen: set[str] = set()
    patch_ids: list[str] = []
    for path in sorted(base_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in PATCH_EXTENSIONS:
            continue
        patch_id = _extract_patch_id(path.stem, base_source)
        if patch_id and patch_id not in seen:
            seen.add(patch_id)
            patch_ids.append(patch_id)

    manifest: list[dict[str, Any]] = []
    for patch_id in patch_ids:
        entry: dict[str, Any] = {"patch_id": patch_id}
        for source in sources:
            source_dir = processed_dir / source_dirs[source]
            patches = _find_patches_for_patch_id(source_dir, patch_id, source)
            if patches is None:
                entry[source] = None
            else:
                entry[source] = [p.relative_to(processed_dir) for p in patches]
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
