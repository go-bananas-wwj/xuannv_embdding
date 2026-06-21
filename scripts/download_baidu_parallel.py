#!/usr/bin/env python3
"""多线程断点续传下载百度网盘文件（使用 BDUSS + PCS API）"""
from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _total_size(bduss: str, remote_path: str) -> int:
    headers = {"User-Agent": "netdisk;pan.baidu.com", "Referer": "https://pan.baidu.com/disk/home"}
    s = requests.Session()
    s.headers.update(headers)
    s.cookies.set("BDUSS", bduss)
    url = "https://d.pcs.baidu.com/rest/2.0/pcs/file"
    params = {"method": "download", "app_id": "250528", "path": remote_path}
    r = s.get(url, params=params, stream=True, timeout=30)
    r.raise_for_status()
    total = int(r.headers.get("Content-Length", 0))
    r.close()
    return total


def _download_part(
    bduss: str,
    remote_path: str,
    part_path: Path,
    start: int,
    end: int,
    chunk_size: int = 1024 * 1024,
    max_retries: int = 5,
) -> int:
    expected = end - start + 1
    part_path.parent.mkdir(parents=True, exist_ok=True)
    # 如果已完整则跳过
    if part_path.exists() and part_path.stat().st_size == expected:
        return expected

    url = "https://d.pcs.baidu.com/rest/2.0/pcs/file"
    params = {"method": "download", "app_id": "250528", "path": remote_path}

    for attempt in range(max_retries):
        headers = {
            "User-Agent": "netdisk;pan.baidu.com",
            "Referer": "https://pan.baidu.com/disk/home",
        }
        existing = part_path.stat().st_size if part_path.exists() else 0
        downloaded = existing
        headers["Range"] = f"bytes={start + existing}-{end}"
        mode = "ab" if existing else "wb"
        try:
            s = requests.Session()
            s.headers.update(headers)
            s.cookies.set("BDUSS", bduss)
            with s.get(url, params=params, headers=headers, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(part_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
            if downloaded == expected:
                return downloaded
            logger.warning("part %s 大小不足: %d/%d, 将重试", part_path.name, downloaded, expected)
        except Exception as e:
            logger.warning("part %s 第 %d 次下载失败: %s", part_path.name, attempt + 1, e)
    raise RuntimeError(f"part {part_path.name} 下载失败，重试 {max_retries} 次后仍未完成")


def download_parallel(
    bduss: str,
    remote_path: str,
    output_path: Path,
    num_threads: int = 4,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = _total_size(bduss, remote_path)
    logger.info("总大小 %.2f GB, 使用 %d 线程下载", total / 1e9, num_threads)

    part_dir = output_path.parent / f"{output_path.name}.parts"
    part_dir.mkdir(exist_ok=True)

    chunk_len = total // num_threads
    tasks = []
    for i in range(num_threads):
        start = i * chunk_len
        end = start + chunk_len - 1 if i < num_threads - 1 else total - 1
        part_path = part_dir / f"part_{i:03d}"
        tasks.append((i, start, end, part_path))

    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        futures = {
            ex.submit(_download_part, bduss, remote_path, t[3], t[1], t[2]): t
            for t in tasks
        }
        for future in as_completed(futures):
            i, start, end, part_path = futures[future]
            try:
                n = future.result()
                logger.info("part %d 完成 %.2f MB", i, n / 1e6)
            except Exception as e:
                logger.error("part %d 失败: %s", i, e)

    # concatenate
    logger.info("合并分块...")
    with open(output_path, "wb") as out:
        for i, _, _, part_path in tasks:
            with open(part_path, "rb") as f:
                while True:
                    data = f.read(1024 * 1024)
                    if not data:
                        break
                    out.write(data)
    logger.info("下载完成: %s", output_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bduss", required=True)
    p.add_argument("--remote-path", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    download_parallel(args.bduss, args.remote_path, args.output, args.threads)


if __name__ == "__main__":
    main()
