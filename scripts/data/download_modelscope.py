"""从 ModelScope 下载天仪高分辨率 SAR 数据集。

使用方式::

    export MODELSCOPE_TOKEN="your-token"
    python scripts/data/download_modelscope.py \
        --dataset WeijieWu/haidian_sar_2025 \
        --output /data/xuannv_embedding/raw/haidian/highres_sar \
        --cache-dir /tmp/modelscope_cache
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ENV_TOKEN = "MODELSCOPE_TOKEN"
DEFAULT_DATASET = "WeijieWu/haidian_sar_2025"


def _get_token() -> str | None:
    """从环境变量读取 ModelScope token，允许空值表示未提供。"""
    token = os.environ.get(ENV_TOKEN, "").strip()
    return token if token else None


def login(api: HubApi, token: str | None) -> None:
    """使用 token 登录 ModelScope；若 token 为空则跳过登录（仅公开数据集可下载）。"""
    if token:
        logger.info("使用环境变量 %s 登录 ModelScope", ENV_TOKEN)
        try:
            api.login(access_token=token)
            logger.info("ModelScope 登录成功")
        except Exception as exc:
            logger.error("ModelScope 登录失败: %s", exc)
            raise
    else:
        logger.warning(
            "未设置环境变量 %s，尝试以未登录状态下载（仅公开数据集）",
            ENV_TOKEN,
        )


def download_dataset(
    dataset_id: str,
    output_dir: Path,
    cache_dir: Path | None = None,
    api: HubApi | None = None,
    token: str | None = None,
) -> Path:
    """下载指定 ModelScope 数据集到本地目录。

    参数:
        dataset_id: ModelScope 数据集 ID，例如 ``WeijieWu/haidian_sar_2025``。
        output_dir: 数据保存目录，不存在时自动创建。
        cache_dir: ModelScope 下载缓存目录，None 则使用默认缓存。
        api: 可选的已初始化 HubApi 实例；为空时内部创建。
        token: 可选的 access token；为空时尝试从环境变量读取。

    返回:
        数据集在本地的下载目录路径。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if api is None:
        api = HubApi()
        if token is None:
            token = _get_token()
        login(api, token)

    logger.info(
        "开始下载数据集: dataset_id=%s, output=%s, cache_dir=%s",
        dataset_id,
        output_dir,
        cache_dir,
    )

    kwargs: dict[str, Any] = {
        "repo_id": dataset_id,
        "repo_type": "dataset",
        "local_dir": str(output_dir),
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)

    try:
        local_path = snapshot_download(**kwargs)
    except Exception as exc:
        logger.exception("ModelScope 下载失败: %s", exc)
        raise RuntimeError(f"ModelScope 下载失败: {exc}") from exc

    local_path_obj = Path(local_path)
    logger.info("数据集下载完成: %s", local_path_obj)
    return local_path_obj


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        description="从 ModelScope 下载天仪高分辨率 SAR 数据集",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"ModelScope 数据集 ID（默认 {DEFAULT_DATASET}）",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="数据保存目录",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="ModelScope 下载缓存目录",
    )
    args = parser.parse_args(argv)

    token = _get_token()
    if token is None:
        logger.warning(
            "建议设置环境变量 %s 以访问私有/受限数据集",
            ENV_TOKEN,
        )

    try:
        api = HubApi()
        login(api, token)
        download_dataset(
            dataset_id=args.dataset,
            output_dir=args.output,
            cache_dir=args.cache_dir,
            api=api,
            token=token,
        )
    except Exception as exc:
        logger.error("下载失败: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
