from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 支持的 patch 影像扩展名
PATCH_EXTENSIONS: set[str] = {".tif", ".tiff"}

# 文件名约定：
#   - 时序数据：{source}_{YYYYMMDD}_{patch_id}.tif
#   - 静态数据：{source}_{patch_id}.tif
# patch_id 可以是参考项目格式 patch_000000，或旧格式 pXXX_rXXX。
_DATE_PREFIX_RE = re.compile(r"^(\d{8})_(.+)$")


def _extract_patch_id(stem: str, source: str) -> str | None:
    """从文件 stem 中提取 patch_id。

    约定文件名格式为 ``{source}_{patch_id}`` 或 ``{source}_{YYYYMMDD}_{patch_id}``。
    返回 patch_id；无法识别或掩膜文件返回 ``None``。
    """
    prefix = f"{source}_"
    if stem.endswith("_mask") or not stem.startswith(prefix):
        return None
    body = stem[len(prefix) :]
    m = _DATE_PREFIX_RE.match(body)
    if m:
        return m.group(2)
    return body


def _index_source_dir(
    processed_dir: Path,
    source: str,
    rel_dir: str,
) -> dict[str, list[Path]]:
    """扫描 source 子目录，建立 patch_id -> 相对路径列表索引。

    跳过 *_mask.tif 掩膜文件，只保留数据影像。同一 patch_id 的多个日期文件
    会按文件名排序。
    """
    patch_index: dict[str, list[Path]] = {}
    source_dir = processed_dir / rel_dir
    if not source_dir.exists():
        return patch_index

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

    for key in patch_index:
        patch_index[key].sort(key=lambda p: p.name)

    return patch_index


def generate_manifest(
    processed_dir: Path,
    sources: list[str],
    source_dirs: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """生成以 patch_id 为键的 manifest。

    Args:
        processed_dir: 预处理数据根目录（region 根目录）。
        sources: 数据源名称列表。所有 source 中的 patch_id 取并集。
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

    patch_indices: dict[str, dict[str, list[Path]]] = {}
    for source in sources:
        patch_indices[source] = _index_source_dir(processed_dir, source, source_dirs[source])

    if not patch_indices[base_source]:
        logger.warning("基准 source 目录为空或不存在: %s", source_dirs[base_source])
        return []

    all_patch_ids: set[str] = set()
    for idx in patch_indices.values():
        all_patch_ids.update(idx.keys())

    def _sort_key(pid: str) -> tuple[int, str]:
        # patch_000000 按数字排序；pXXX_rXXX 按旧格式元组排序；其余按字符串排序
        m = re.match(r"patch_(\d+)$", pid)
        if m:
            return (0, int(m.group(1)))
        m = re.match(r"p(\d{3})_r(\d{3})$", pid)
        if m:
            return (1, int(m.group(1)), int(m.group(2)))
        return (2, pid)

    patch_ids = sorted(all_patch_ids, key=_sort_key)

    manifest: list[dict[str, Any]] = []
    for patch_id in patch_ids:
        entry: dict[str, Any] = {"patch_id": patch_id}
        for source in sources:
            paths = patch_indices[source].get(patch_id)
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
