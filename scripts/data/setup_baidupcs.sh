#!/usr/bin/env bash
# 自动下载并安装 BaiduPCS-Go 到项目 tools/baidupcs/ 目录。
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TOOLS_DIR="${PROJECT_ROOT}/tools/baidupcs"
mkdir -p "${TOOLS_DIR}"
cd "${TOOLS_DIR}"

# 将 session / 配置文件保留在项目 tools/baidupcs/ 目录下，避免提交到 git。
export BAIDUPCS_GO_CONFIG_DIR="${TOOLS_DIR}"

echo "[setup_baidupcs] 项目根目录: ${PROJECT_ROOT}"
echo "[setup_baidupcs] 工具目录: ${TOOLS_DIR}"
echo "[setup_baidupcs] 配置文件目录: ${BAIDUPCS_GO_CONFIG_DIR}"

BIN="./BaiduPCS-Go"
if [ -f "${BIN}" ]; then
    echo "[setup_baidupcs] 二进制已存在，跳过下载。"
    "${BIN}" -v
    exit 0
fi

echo "[setup_baidupcs] 从 GitHub API 获取最新 release tag..."
LATEST="$(curl -sL https://api.github.com/repos/qjfoidnh/BaiduPCS-Go/releases/latest | python3 -c 'import sys, json; print(json.load(sys.stdin)["tag_name"])')"
if [ -z "${LATEST}" ]; then
    echo "[setup_baidupcs] 无法获取最新 release tag，请检查网络。" >&2
    exit 1
fi
echo "[setup_baidupcs] 最新 tag: ${LATEST}"

ZIP="BaiduPCS-Go-${LATEST}-linux-amd64.zip"
DOWNLOAD_URL="https://github.com/qjfoidnh/BaiduPCS-Go/releases/download/${LATEST}/${ZIP}"

echo "[setup_baidupcs] 下载: ${DOWNLOAD_URL}"
if command -v curl >/dev/null 2>&1; then
    curl -L -o "${ZIP}" "${DOWNLOAD_URL}"
elif command -v wget >/dev/null 2>&1; then
    wget -O "${ZIP}" "${DOWNLOAD_URL}"
else
    echo "[setup_baidupcs] 需要 curl 或 wget。" >&2
    exit 1
fi

echo "[setup_baidupcs] 解压..."
unzip -o "${ZIP}"
mv "BaiduPCS-Go-${LATEST}-linux-amd64/BaiduPCS-Go" ./
rm -rf "BaiduPCS-Go-${LATEST}-linux-amd64"
chmod +x "${BIN}"

echo "[setup_baidupcs] 安装完成。"
"${BIN}" -v

echo ""
echo "[setup_baidupcs] 登录方式（选择其一）："
echo "  1) 用户名密码登录: ${BIN} login"
echo "  2) BDUSS 登录:    ${BIN} login -bduss=<BDUSS> -stoken=<STOKEN>"
echo "  3) Cookies 登录:  ${BIN} login -cookies=\"BDUSS=xxx; STOKEN=yyy; ...\""
echo "登录后请执行: ${BIN} who"
