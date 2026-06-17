"""KNN 下游评估（WorldCover / JRC Water / Dynamic World）。"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。

    参数:
        argv: 可选参数列表，用于测试；为 ``None`` 时解析 ``sys.argv``。
    """
    parser = argparse.ArgumentParser(description="KNN 下游评估")
    parser.add_argument("--embedding-train", required=True, type=Path)
    parser.add_argument("--label-train", required=True, type=Path)
    parser.add_argument("--embedding-test", required=True, type=Path)
    parser.add_argument("--label-test", required=True, type=Path)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def _main(argv: list[str] | None = None) -> None:
    """KNN 评估主函数（可被测试直接调用）。"""
    args = parse_args(argv)

    train = np.load(args.embedding_train)
    test = np.load(args.embedding_test)
    train_labels = np.load(args.label_train)["labels"]
    test_labels = np.load(args.label_test)["labels"]

    clf = KNeighborsClassifier(n_neighbors=args.k)
    clf.fit(train["embeddings"], train_labels)
    pred = clf.predict(test["embeddings"])

    acc = accuracy_score(test_labels, pred)
    f1 = f1_score(test_labels, pred, average="macro", zero_division=0)
    logger.info("accuracy: %.4f", acc)
    logger.info("f1_macro: %.4f", f1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, accuracy=acc, f1_macro=f1, predictions=pred, labels=test_labels)
    logger.info("saved metrics to %s", args.output)


def main() -> None:
    """命令行入口。"""
    _main()


if __name__ == "__main__":
    main()
