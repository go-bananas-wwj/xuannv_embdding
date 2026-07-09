#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from modelscope.hub.api import HubApi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload Xuannv Haidian production artifacts to ModelScope.")
    p.add_argument("--stage-dir", type=Path, required=True)
    p.add_argument("--repo-id", default="WeijieWu/xuannv_haidian_embdding")
    p.add_argument("--repo-type", default="dataset")
    p.add_argument("--revision", default="master")
    p.add_argument("--path-in-repo", default="")
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--commit-message", default="release: update haidian embedding v1 production artifacts")
    p.add_argument("--token-env", default="MODELSCOPE_SDK_TOKEN")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.stage_dir.exists():
        raise FileNotFoundError(args.stage_dir)
    token = os.environ.get(args.token_env)
    if not token:
        raise RuntimeError(f"Missing ModelScope token in environment variable {args.token_env}")

    api = HubApi()
    if not api.repo_exists(args.repo_id, repo_type=args.repo_type, token=token):
        api.create_repo(
            args.repo_id,
            repo_type=args.repo_type,
            token=token,
            visibility="public",
            exist_ok=True,
        )

    info = api.upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=str(args.stage_dir),
        path_in_repo=args.path_in_repo,
        token=token,
        revision=args.revision,
        max_workers=args.max_workers,
        commit_message=args.commit_message,
    )
    print(info)


if __name__ == "__main__":
    main()
