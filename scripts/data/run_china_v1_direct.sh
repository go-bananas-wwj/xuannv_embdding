#!/usr/bin/env bash
# Run a China V1 acquisition command without inheriting local proxy settings.
set -euo pipefail

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export NO_PROXY='*'
export no_proxy='*'
export PYTHONPATH="/root/workspace/xuannv/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <command> [args...]" >&2
  exit 2
fi

echo "[china-v1] direct network mode: proxy variables cleared" >&2
exec "$@"
