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
        help="预处理数据根目录，子目录按 source 组织",
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
    args = parser.parse_args()

    logger.info(
        "开始生成 manifest: processed_dir=%s, sources=%s",
        args.processed_dir,
        args.sources,
    )
    manifest = generate_manifest(args.processed_dir, args.sources)
    logger.info("共发现 %d 个 patches", len(manifest))
    save_manifest(manifest, args.output)


if __name__ == "__main__":
    main()
