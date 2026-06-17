"""调用 BaiduPCS-Go 批量下载百度网盘文件/目录。

使用方式::

    export BAIDUPCS_GO_CONFIG_DIR=/root/workspace/xuannv/tools/baidupcs
    python scripts/data/download_baidu.py \
        --links-file scripts/data/baidu_links.example.txt \
        --output /data/xuannv_embedding/raw/beijing/highres_optical

links 文件每行一个远程路径，以 ``#`` 开头的行为注释。
若本地已存在同名文件/目录，默认跳过；使用 ``--overwrite`` 可强制重新下载。
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_BAIDUPCS_CONFIG_DIR = Path("/root/workspace/xuannv/tools/baidupcs")
DEFAULT_BAIDUPCS_BINARY = DEFAULT_BAIDUPCS_CONFIG_DIR / "BaiduPCS-Go"
ENV_BAIDUPCS_CONFIG_DIR = "BAIDUPCS_GO_CONFIG_DIR"


class BaiduPCSAccessError(RuntimeError):
    """BaiduPCS-Go 未登录或无法访问网盘时抛出。"""


def _get_baidupcs_env(config_dir: Path) -> dict[str, str]:
    """构造包含 BAIDUPCS_GO_CONFIG_DIR 的子进程环境变量。"""
    env = os.environ.copy()
    env[ENV_BAIDUPCS_CONFIG_DIR] = str(config_dir)
    return env


def _check_access(binary: Path, config_dir: Path) -> None:
    """检查 BaiduPCS-Go 是否已登录并可访问网盘。

    通过执行 ``who`` 命令验证 uid，若失败则抛出 ``BaiduPCSAccessError``。
    """
    cmd = [str(binary), "who"]
    env = _get_baidupcs_env(config_dir)
    logger.info("检查 BaiduPCS-Go 登录状态: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except FileNotFoundError as exc:
        raise BaiduPCSAccessError(f"找不到 BaiduPCS-Go 可执行文件: {binary}") from exc
    except subprocess.TimeoutExpired as exc:
        raise BaiduPCSAccessError("检查登录状态超时") from exc

    if result.returncode != 0 or "uid" not in result.stdout:
        err_msg = (result.stderr or result.stdout or "未知错误").strip()
        raise BaiduPCSAccessError(f"BaiduPCS-Go 未登录或无法访问网盘: {err_msg}")

    # 解析 uid，uid 为 0 表示未登录
    uid_match = re.search(r"uid:\s*(\d+)", result.stdout)
    if uid_match is None:
        raise BaiduPCSAccessError("无法解析 BaiduPCS-Go 登录 uid")
    uid = int(uid_match.group(1))
    if uid == 0:
        raise BaiduPCSAccessError("BaiduPCS-Go 未登录（uid=0）")

    logger.info("BaiduPCS-Go 登录状态正常: uid=%d", uid)


def _local_name(remote_path: str) -> str:
    """从远程路径提取本地保存使用的文件/目录名。"""
    # 去除尾部斜杠，避免 basename 返回空字符串
    return Path(remote_path.rstrip("/")).name


def _already_exists(local_dir: Path, remote_path: str) -> bool:
    """判断本地目标是否已存在。"""
    local_target = local_dir / _local_name(remote_path)
    return local_target.exists()


def _remove_existing(local_target: Path) -> None:
    """删除已存在的本地文件或目录，为重新下载做准备。"""
    if local_target.is_dir():
        shutil.rmtree(local_target)
    elif local_target.is_file():
        local_target.unlink()
    else:
        # 其他类型（如符号链接）直接删除
        local_target.unlink(missing_ok=True)


def download_link(
    binary: Path,
    config_dir: Path,
    remote_path: str,
    local_dir: Path,
    overwrite: bool = False,
) -> bool:
    """下载单个远程路径到本地目录。

    参数:
        binary: BaiduPCS-Go 可执行文件路径。
        config_dir: BaiduPCS-Go 配置目录（含登录 session）。
        remote_path: 百度网盘远程路径，例如 ``/玄女科技/haidian_aoi_boundary.shp``。
        local_dir: 本地保存目录。
        overwrite: 是否覆盖已存在文件/目录。

    返回:
        下载成功返回 True，跳过或失败返回 False。
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    local_name = _local_name(remote_path)
    local_target = local_dir / local_name

    if not overwrite and local_target.exists():
        logger.info("本地已存在，跳过: %s -> %s", remote_path, local_target)
        return False

    if overwrite and local_target.exists():
        logger.warning("覆盖模式下删除旧文件/目录: %s", local_target)
        _remove_existing(local_target)

    cmd = [str(binary), "download", "--saveto", str(local_dir), remote_path]
    logger.info("开始下载: %s -> %s", remote_path, local_dir)
    logger.debug("执行命令: %s", " ".join(cmd))

    env = _get_baidupcs_env(config_dir)
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=3600,
        )
    except subprocess.TimeoutExpired:
        logger.error("下载超时: %s", remote_path)
        return False

    # 将 BaiduPCS-Go 的输出按行记录，便于排查问题
    for line in (result.stdout or "").splitlines():
        if line.strip():
            logger.info("[BaiduPCS-Go] %s", line.strip())
    for line in (result.stderr or "").splitlines():
        if line.strip():
            logger.warning("[BaiduPCS-Go] %s", line.strip())

    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        logger.error("下载失败: %s\n%s", remote_path, err)
        return False

    # BaiduPCS-Go 对不存在的路径也可能返回 0，需结合输出判断是否真的加入了队列
    if "下载完成" not in (result.stdout or "") and "加入下载队列" not in (result.stdout or ""):
        logger.error("下载失败（远程路径不存在或为空）: %s", remote_path)
        return False

    logger.info("下载完成: %s", remote_path)
    return True


def read_links(links_file: Path) -> list[str]:
    """读取 links 文件，返回去重后的远程路径列表。"""
    links: list[str] = []
    seen: set[str] = set()
    with links_file.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # 去重，保持顺序
            if stripped not in seen:
                seen.add(stripped)
                links.append(stripped)
    return links


def main(argv: list[str] | None = None) -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        description="调用 BaiduPCS-Go 批量下载百度网盘文件/目录",
    )
    parser.add_argument(
        "--links-file",
        required=True,
        type=Path,
        help="每行一个远程路径的文本文件，``#`` 开头为注释",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="本地保存目录，不存在时自动创建",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="强制重新下载并覆盖已存在文件/目录",
    )
    parser.add_argument(
        "--baidupcs-config-dir",
        type=Path,
        default=DEFAULT_BAIDUPCS_CONFIG_DIR,
        help=f"BaiduPCS-Go 配置目录（默认 {DEFAULT_BAIDUPCS_CONFIG_DIR}）",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=DEFAULT_BAIDUPCS_BINARY,
        help=f"BaiduPCS-Go 可执行文件路径（默认 {DEFAULT_BAIDUPCS_BINARY}）",
    )
    args = parser.parse_args(argv)

    binary = Path(args.binary)
    config_dir = Path(args.baidupcs_config_dir)
    output_dir = Path(args.output)

    if not args.links_file.is_file():
        logger.error("links 文件不存在: %s", args.links_file)
        return 1

    # 优先使用命令行指定的配置目录，同时写入当前进程环境变量以影响子进程
    os.environ[ENV_BAIDUPCS_CONFIG_DIR] = str(config_dir)

    try:
        _check_access(binary, config_dir)
    except BaiduPCSAccessError as exc:
        logger.error("%s", exc)
        logger.error("状态: BLOCKED")
        return 1

    links = read_links(args.links_file)
    if not links:
        logger.warning("links 文件为空或仅包含注释: %s", args.links_file)
        return 0

    logger.info("共读取 %d 个下载任务，输出目录: %s", len(links), output_dir)

    success_count = 0
    skip_count = 0
    fail_count = 0
    for remote_path in links:
        ok = download_link(
            binary=binary,
            config_dir=config_dir,
            remote_path=remote_path,
            local_dir=output_dir,
            overwrite=args.overwrite,
        )
        if ok:
            success_count += 1
        elif (output_dir / _local_name(remote_path)).exists() and not args.overwrite:
            skip_count += 1
        else:
            fail_count += 1

    logger.info(
        "批量下载结束: 成功 %d, 跳过 %d, 失败 %d",
        success_count,
        skip_count,
        fail_count,
    )
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
