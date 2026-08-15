#!/usr/bin/env bash
set -euo pipefail

WORKTREE=/root/workspace/xuannv/.worktrees/codex-china-v1-fusion-smoke
cd "${WORKTREE}"
PYTHONPATH="${WORKTREE}/src:${WORKTREE}/downstreams:${WORKTREE}" \
  python -m experiments.china_v1_fusion_smoke.bootstrap
