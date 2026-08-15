#!/usr/bin/env bash
set -euo pipefail

SANDBOX=/data/xuannv_embedding/sandboxes/china_v1_fusion_smoke_20260815
mkdir -p "${SANDBOX}"
touch "${SANDBOX}/.xuannv_isolated_smoke"
python -m venv --system-site-packages "${SANDBOX}/env"
mkdir -p "${SANDBOX}"/{cache,synthetic/aef,synthetic/highres_2m,outputs,checkpoints,logs,manifests,tmp}

printf '%s\n' \
  "${SANDBOX}" \
  "${SANDBOX}/env" \
  "${SANDBOX}/cache" \
  "${SANDBOX}/synthetic/aef" \
  "${SANDBOX}/synthetic/highres_2m" \
  "${SANDBOX}/outputs" \
  "${SANDBOX}/checkpoints" \
  "${SANDBOX}/logs" \
  "${SANDBOX}/manifests" \
  "${SANDBOX}/tmp"
