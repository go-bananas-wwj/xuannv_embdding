"""双时相 embedding 变化检测 AUC。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。

    参数:
        argv: 可选参数列表，用于测试；为 ``None`` 时解析 ``sys.argv``。
    """
    parser = argparse.ArgumentParser(description="双时相 embedding 变化检测 AUC")
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--label", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def _main(argv: list[str] | None = None) -> None:
    """变化检测评估主函数（可被测试直接调用）。"""
    args = parse_args(argv)

    before = np.load(args.before)["embeddings"]
    after = np.load(args.after)["embeddings"]
    label = np.load(args.label)["labels"]

    diff = np.linalg.norm(before - after, axis=-1)
    auc = roc_auc_score(label, diff)
    logger.info("AUC: %.4f", auc)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, auc=auc, diff=diff, labels=label)
    logger.info("saved metrics to %s", args.output)


def main() -> None:
    """命令行入口。"""
    _main()


if __name__ == "__main__":
    main()
