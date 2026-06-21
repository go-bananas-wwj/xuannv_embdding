#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from xuannv_embedding.config import Config
from xuannv_embedding.data.dataset import MonthlyEmbeddingDataset

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预生成训练数据缓存")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件")
    parser.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="最多预处理多少个 patch（用于测试）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    if not cfg.data.cache_dir:
        raise ValueError("配置中未设置 data.cache_dir")

    dataset = MonthlyEmbeddingDataset(
        manifest_path=cfg.data.manifest_path,
        statistics_dir=cfg.data.statistics_dir,
        sources=cfg.data.sources,
        patch_size=cfg.data.patch_size,
        max_patches=args.max_patches,
        num_months=cfg.model.num_months,
        teacher_embedding_root=cfg.data.teacher_embedding_root,
        region=cfg.data.region,
        cache_dir=cfg.data.cache_dir,
    )

    logger.info("预生成 %d 个 patch 的缓存到 %s", len(dataset), cfg.data.cache_dir)
    for i in tqdm(range(len(dataset))):
        _ = dataset[i]
    logger.info("完成")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main()
