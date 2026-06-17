"""双时相 embedding 变化检测 AUC。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

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


def _check_input_files(*paths: Path) -> None:
    """校验所有输入文件均存在，缺失时抛出清晰的 FileNotFoundError。"""
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"输入文件不存在: {path}")


def _check_binary_labels(label: np.ndarray) -> None:
    """校验标签同时包含 0 和 1，不满足时抛出 ValueError。"""
    unique = set(np.unique(label).tolist())
    if not ({0, 1} <= unique):
        raise ValueError("变化检测标签必须同时包含 0（未变化）和 1（变化），无法计算 AUC")


def _main(argv: list[str] | None = None) -> None:
    """变化检测评估主函数（可被测试直接调用）。"""
    args = parse_args(argv)

    _check_input_files(args.before, args.after, args.label)

    before = np.load(args.before)["embeddings"]
    after = np.load(args.after)["embeddings"]
    label = np.load(args.label)["labels"]

    _check_binary_labels(label)

    diff = np.linalg.norm(before - after, axis=-1)
    auc = roc_auc_score(label, diff)
    logger.info("AUC: %.4f", auc)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, auc=auc, diff=diff, labels=label)
    logger.info("saved metrics to %s", args.output)


def main() -> None:
    """命令行入口。"""
    logging.basicConfig(level=logging.INFO)
    _main()


if __name__ == "__main__":
    main()
