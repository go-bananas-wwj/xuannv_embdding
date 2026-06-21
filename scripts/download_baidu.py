#!/usr/bin/env python3
"""使用 BDUSS 从百度网盘下载文件（支持断点续传）"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def download_with_resume(
    bduss: str,
    remote_path: str,
    output_path: Path,
    chunk_size: int = 1024 * 1024,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers_base = {
        "User-Agent": "netdisk;pan.baidu.com",
        "Referer": "https://pan.baidu.com/disk/home",
    }
    session = requests.Session()
    session.headers.update(headers_base)
    session.cookies.set("BDUSS", bduss)

    start_byte = output_path.stat().st_size if output_path.exists() else 0
    headers = dict(headers_base)
    if start_byte:
        headers["Range"] = f"bytes={start_byte}-"
        logger.info("从 %d 字节处续传", start_byte)

    url = "https://d.pcs.baidu.com/rest/2.0/pcs/file"
    params = {"method": "download", "app_id": "250528", "path": remote_path}
    r = session.get(url, params=params, headers=headers, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("Content-Length", 0)) + start_byte
    mode = "ab" if start_byte else "wb"
    downloaded = start_byte
    logger.info("开始下载 %s -> %s, 总大小 %.2f GB", remote_path, output_path, total / 1e9)
    with open(output_path, mode) as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded / total * 100
                if downloaded % (50 * chunk_size) < chunk_size:
                    logger.info("已下载 %.2f GB / %.2f GB (%.1f%%)", downloaded / 1e9, total / 1e9, percent)
    logger.info("下载完成: %s", output_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bduss", required=True)
    p.add_argument("--remote-path", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    download_with_resume(args.bduss, args.remote_path, args.output)


if __name__ == "__main__":
    main()
