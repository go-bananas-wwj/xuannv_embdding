"""生成 processed patches 的 manifest JSON。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from xuannv_embedding.utils.manifest import generate_manifest, save_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 patch manifest")
    parser.add_argument(
        "--processed-dir",
        required=True,
        type=Path,
        help="预处理数据根目录（region 根目录）",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="输出 manifest JSON 路径",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="数据源名称列表，第一个作为 patch_id 基准",
    )
    parser.add_argument(
        "--source-dir",
        action="append",
        default=[],
        help="source 到子目录的映射，格式 source=relative_dir；例如 s2=patches/s2。"
             "未指定的 source 默认使用同名子目录。",
    )
    args = parser.parse_args()

    source_dirs: dict[str, str] = {}
    for mapping in args.source_dir:
        if "=" not in mapping:
            raise ValueError(f"--source-dir 格式错误，应为 source=relative_dir: {mapping}")
        source, rel_dir = mapping.split("=", 1)
        source_dirs[source] = rel_dir

    logger.info(
        "开始生成 manifest: processed_dir=%s, sources=%s, source_dirs=%s",
        args.processed_dir,
        args.sources,
        source_dirs,
    )
    manifest = generate_manifest(args.processed_dir, args.sources, source_dirs=source_dirs)
    logger.info("共发现 %d 个 patches", len(manifest))
    save_manifest(manifest, args.output)


if __name__ == "__main__":
    main()
