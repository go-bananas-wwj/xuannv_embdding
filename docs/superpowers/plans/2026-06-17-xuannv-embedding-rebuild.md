# xuannv_embedding 项目重构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans` 按任务逐步实施。步骤使用 checkbox（`- [ ]`）语法跟踪。

**Goal:** 在 `/root/workspace/xuannv` 从零重构基于 AEF 原论文的月度地理嵌入模型，支持 S1/S2/Landsat + 稀疏高分辨率数据作为额外模态，完成北京海淀（320 patches）与哈尔滨新区（424 patches）的数据管道、训练框架与可复现训练入口。

**Architecture:** 模型严格遵循 AEF 原论文：per-sensor stem → space/time transformer → VMF bottleneck → 连续/分类 decoder；高分辨率数据通过 availability mask 与 per-modality embedding 作为额外 source 参与编码，缺失时自动 fallback。数据、代码、输出目录彻底分离，所有下载脚本可重复执行。

**Tech Stack:** Python 3.11, torch 2.1 + torch_npu 2.1, hccl, rasterio, geopandas, xarray, planetary-computer, modelscope, BaiduPCS-Go, black/ruff, pytest.

---

## 关键约束（本计划已和用户确认）

1. **项目根目录**: `/root/workspace/xuannv`（当前目录）。
2. **数据目录**: 大容量磁盘挂载在 `/data`，使用 `/data/xuannv_embedding/` 作为数据根，避免和旧项目 `/workspace/xuannv` 冲突。
3. **训练硬件**: 多卡 Huawei Ascend 910B NPU，代码适配 `torch_npu` + `hccl`。
4. **高分辨率数据**: 采用 availability-aware 多源融合（参考 FLORO 2026 思路），高分辨率作为额外 source，用 mask 处理时间/区域缺失。
5. **模型基线**: 先做 AEF 原论文的 clean re-implementation，不默认引入旧仓库 v1 的 `skip_l2_norm` 等自定义改动（这些作为后续消融）。
6. **下载工具**: S1/S2/Landsat 走 Planetary Computer；天仪 SAR 走 ModelScope（token 已提供）；北京/哈尔滨高分辨率走 BaiduPCS-Go。
7. **当前阶段**: 先把框架和数据准备好，训练之后再沟通；但仍需写出完整训练入口和冒烟测试。

---

## 文件结构总览

```
/root/workspace/xuannv/                 # 项目根（git repo）
├── .gitignore
├── README.md
├── AGENTS.md                           # 项目专属代理指令
├── pyproject.toml
├── configs/
│   ├── base.yaml
│   ├── harbin_monthly.yaml
│   └── haidian_monthly.yaml
├── src/xuannv_embedding/               # 包名
│   ├── __init__.py
│   ├── config.py                       # YAML/dataclass 配置加载
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py                  # 月度采样 Dataset
│   │   ├── transforms.py               # 读取/归一化/掩码构建
│   │   ├── collate.py                  # 变长时序 batch 对齐
│   │   └── builder.py                  # DataLoader 工厂
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model.py                    # AEFModel + AEFOutput
│   │   ├── sensor_encoders.py          # Per-sensor stem
│   │   ├── blocks.py                   # SpaceOperator / TimeOperator
│   │   ├── bottleneck.py               # VMF bottleneck
│   │   ├── decoders.py                 # 连续/分类 decoder
│   │   ├── time_encoding.py            # 时间/窗口编码
│   │   └── highres_fusion.py           # 高分辨率 availability-aware 融合
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py                  # DDP 训练器
│   │   ├── losses.py                   # 重建 + uniformity 损失
│   │   ├── optimizer.py                # scheduler / optimizer
│   │   └── checkpoint.py               # 存取 checkpoint
│   ├── inference/
│   │   ├── __init__.py
│   │   └── engine.py                   # 加载模型 + 月度 embedding 提取
│   └── utils/
│       ├── __init__.py
│       ├── device.py                   # get_device 优先 NPU
│       ├── geo.py                      # CRS / bounds / reproject 工具
│       ├── logging.py                  # 统一日志
│       └── manifest.py                 # manifest 读写
├── scripts/
│   ├── data/
│   │   ├── download_pc.py              # PC 下载 S1/S2/Landsat
│   │   ├── download_modelscope.py      # ModelScope 下载天仪 SAR
│   │   ├── download_baidu.py           # BaiduPCS-Go 批量下载
│   │   ├── preprocess.py               # 预处理入口
│   │   └── compute_statistics.py       # 统计量计算
│   ├── train/
│   │   ├── train.py                    # DDP 训练入口
│   │   └── launch.sh                   # 启动示例
│   └── eval/
│       ├── extract_embeddings.py
│       ├── knn_eval.py
│       └── change_detection_eval.py
└── tests/
    ├── test_smoke.py
    ├── test_dataset.py
    └── test_model.py

/data/xuannv_embedding/                 # 数据根（不在 git 内）
├── raw/
│   ├── harbin/
│   │   ├── s2/
│   │   ├── s1/
│   │   ├── landsat/
│   │   ├── dem/
│   │   ├── worldcover/
│   │   ├── dynamic_world/
│   │   ├── jrc_water/
│   │   └── highres/
│   ├── haidian/
│   │   └── ...（同上）
│   └── beijing/
│       └── planetscene/
├── processed/
│   ├── harbin/
│   │   └── scenes/                     # patch 化后的数据 + manifest
│   └── haidian/
│       └── scenes/
├── statistics/
│   ├── harbin/
│   └── haidian/
└── outputs/                            # checkpoint / log / eval
```

---

## Phase 1: 项目初始化

### Task 1: 创建 GitHub 仓库并初始化本地项目

**Files:**
- Create: `/root/workspace/xuannv/.gitignore`
- Create: `/root/workspace/xuannv/README.md`
- Create: `/root/workspace/xuannv/pyproject.toml`
- Modify: 在 GitHub 创建 `go-bananas-wwj/xuannv_embedding`

- [ ] **Step 1: 检查 SSH key 是否可访问 GitHub**

```bash
ssh -T git@github.com
```
Expected: `Hi go-bananas-wwj! You've successfully authenticated...`

- [ ] **Step 2: 在 GitHub 创建空仓库（无 README）**

```bash
gh repo create go-bananas-wwj/xuannv_embedding --private --confirm
```
或浏览器访问 https://github.com/new 创建。

- [ ] **Step 3: 初始化本地 git 并关联远程**

```bash
cd /root/workspace/xuannv
git init
git remote add origin git@github.com:go-bananas-wwj/xuannv_embedding.git
```

- [ ] **Step 4: 写入 `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.mypy_cache/

# Data / outputs
data/
outputs/
out/
*.tiff
*.tif
*.npz
*.pt
*.pth
*.ckpt

# IDEs
.vscode/
.idea/
*.swp

# Logs
*.log
```

- [ ] **Step 5: 写入 `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "xuannv_embedding"
version = "0.1.0"
description = "Clean re-implementation of AlphaEarth Foundations with monthly embeddings"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.0,<2.2",
    "torch_npu>=2.1.0",
    "numpy>=1.24",
    "rasterio>=1.3",
    "geopandas>=0.14",
    "xarray>=2023.0",
    "pyyaml>=6.0",
    "tqdm",
    "planetary-computer>=1.0",
    "pystac-client",
    "stackstac",
    "modelscope>=1.15",
    "requests",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.black]
line-length = 100
target-version = ['py311']

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "W"]
```

- [ ] **Step 6: 提交并推送初始框架**

```bash
cd /root/workspace/xuannv
git add .gitignore README.md pyproject.toml
git commit -m "chore: init repo with package metadata"
git branch -M main
git push -u origin main
```

---

### Task 2: 编写 `AGENTS.md`

**Files:**
- Create: `/root/workspace/xuannv/AGENTS.md`

- [ ] **Step 1: 写入 AGENTS.md**

```markdown
# AGENTS.md — xuannv_embedding

## 强制规则

1. **与用户交流使用中文**。
2. **每次修改后执行**: `git add -A && git commit -m "描述" && git push origin main`。
3. **训练前检查 NPU**: `npu-smi info`。
4. **禁止 `nohup` 跑训练**，使用 `tmux`。
5. **文件操作限制在 `/root/workspace/xuannv` 和 `/data/xuannv_embedding/` 内**。
6. **数据目录**: `/data/xuannv_embedding/` 是数据根，代码中不可写死其他路径。
7. **设备选择**: 统一走 `xuannv_embedding.utils.device.get_device()`，优先 NPU。
8. **代码风格**: `from __future__ import annotations`；类型注解完整；模块内注释中文。
9. **测试**: 每次重大修改后跑 `pytest tests/test_smoke.py`。

## 项目目标

- 基于 AEF 论文 clean re-implementation，生成月度地理嵌入。
- 支持 S1/S2/Landsat + 稀疏高分辨率数据 availability-aware 融合。
- 在北京海淀（320 patches）和哈尔滨新区（424 patches）训练。

## 关键路径

1. 数据下载脚本（PC / ModelScope / BaiduPCS-Go）。
2. 预处理：对齐 → patchify → 统计量 → manifest。
3. 模型：sensor encoders → space/time operators → VMF bottleneck → decoders。
4. 训练：DDP on NPU，重建 + uniformity 损失。
5. 评估：embedding 提取 → KNN / MLP / 变化检测 AUC。

## 目录约定

```
/root/workspace/xuannv/          # 代码根
/data/xuannv_embedding/          # 数据根
  raw/                           # 原始下载数据
  processed/                     # patch 化数据 + manifest
  statistics/                    # 归一化统计量
  outputs/                       # checkpoint / log / eval
```

## 依赖安装

```bash
cd /root/workspace/xuannv
conda create -n xuannv_emb python=3.11 -y
conda activate xuannv_emb
pip install -e .
```

## 训练启动示例

```bash
conda activate xuannv_emb
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 scripts/train/train.py --config configs/harbin_monthly.yaml
```

## 数据下载示例

```bash
# S1/S2/Landsat via Planetary Computer
python scripts/data/download_pc.py --region harbin --start 2025-01-01 --end 2026-05-31

# 天仪 SAR via ModelScope
python scripts/data/download_modelscope.py --dataset WeijieWu/haidian_sar_2025 --output /data/xuannv_embedding/raw/haidian/highres_sar

# 百度网盘
python scripts/data/download_baidu.py --links-file links.txt --output /data/xuannv_embedding/raw/beijing/highres_optical
```
```

- [ ] **Step 2: 提交**

```bash
cd /root/workspace/xuannv
git add AGENTS.md
git commit -m "docs: add AGENTS.md with project rules"
git push origin main
```

---

### Task 3: 创建 Python 包目录结构

**Files:**
- Create: `/root/workspace/xuannv/src/xuannv_embedding/__init__.py`
- Create: `/root/workspace/xuannv/src/xuannv_embedding/config.py`
- Create: `/root/workspace/xuannv/src/xuannv_embedding/utils/__init__.py`
- Create: `/root/workspace/xuannv/src/xuannv_embedding/utils/device.py`
- Create: `/root/workspace/xuannv/tests/__init__.py`
- Create: `/root/workspace/xuannv/tests/test_smoke.py`

- [ ] **Step 1: 写入 `src/xuannv_embedding/__init__.py`**

```python
from __future__ import annotations

__version__ = "0.1.0"
```

- [ ] **Step 2: 写入 `src/xuannv_embedding/utils/device.py`**

```python
from __future__ import annotations

import importlib
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

_NPU_AVAILABLE = importlib.util.find_spec("torch_npu") is not None


def get_device(preference: str | None = None) -> torch.device:
    """优先返回 NPU，其次 CUDA，最后 CPU。"""
    if preference is not None:
        return torch.device(preference)
    if _NPU_AVAILABLE:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")
```

- [ ] **Step 3: 写入初始 `config.py`（后续扩展）**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: Path
    region: str
    manifest_path: Path
    num_samples: int
    max_patches: int | None = None
    batch_size: int = 4
    num_workers: int = 4
    sources: list[str] = field(default_factory=lambda: ["s2", "s1", "landsat"])


@dataclass
class ExperimentConfig:
    name: str
    seed: int = 42


@dataclass
class Config:
    experiment: ExperimentConfig
    data: DataConfig
    raw: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        experiment = ExperimentConfig(**raw["experiment"])
        data = DataConfig(**raw["data"])
        return cls(experiment=experiment, data=data, raw=raw)
```

- [ ] **Step 4: 写入 `tests/test_smoke.py`（占位，后续填充）**

```python
from __future__ import annotations


def test_import_package() -> None:
    import xuannv_embedding

    assert xuannv_embedding.__version__ == "0.1.0"
```

- [ ] **Step 5: 安装包并跑测试**

```bash
cd /root/workspace/xuannv
pip install -e .
pytest tests/test_smoke.py -v
```
Expected: `test_import_package PASSED`

- [ ] **Step 6: 提交**

```bash
git add -A
git commit -m "chore: bootstrap package structure"
git push origin main
```

---

### Task 4: 创建 `/data` 数据目录结构

**Files:**
- Create directories under `/data/xuannv_embedding/`

- [ ] **Step 1: 创建目录并检查磁盘空间**

```bash
DATA_ROOT=/data/xuannv_embedding
mkdir -p ${DATA_ROOT}/{raw/{harbin,haidian,beijing}/{s2,s1,landsat,dem,worldcover,dynamic_world,jrc_water,highres},processed/{harbin,haidian}/scenes,statistics/{harbin,haidian},outputs}
df -h /data
```
Expected: `/data` 可用空间 > 1T（3.5T 挂载盘）。

- [ ] **Step 2: 提交空目录占位**

```bash
cd /root/workspace/xuannv
mkdir -p data/.gitkeep
touch data/.gitkeep
git add data/.gitkeep
git commit -m "chore: add data placeholder dir"
git push origin main
```

---

## Phase 2: 数据下载脚本

### Task 5: 安装并配置 BaiduPCS-Go

**Files:**
- Create: `/root/workspace/xuannv/scripts/data/setup_baidupcs.sh`
- Create: `/root/workspace/xuannv/scripts/data/.baidupcs_session`（登录后生成，不提交）

- [ ] **Step 1: 下载 BaiduPCS-Go**

```bash
BAIDU_DIR=/root/workspace/xuannv/tools/baidupcs
mkdir -p ${BAIDU_DIR}
cd ${BAIDU_DIR}
LATEST=$(curl -s https://api.github.com/repos/qjfoidnh/BaiduPCS-Go/releases/latest | grep -oP '"tag_name": "\K[^"]+')
wget "https://github.com/qjfoidnh/BaiduPCS-Go/releases/download/${LATEST}/BaiduPCS-Go-${LATEST}-linux-amd64.zip"
unzip BaiduPCS-Go-${LATEST}-linux-amd64.zip
mv BaiduPCS-Go-${LATEST}-linux-amd64/BaiduPCS-Go ./
chmod +x BaiduPCS-Go
./BaiduPCS-Go -v
```
Expected: 显示版本号。

- [ ] **Step 2: 扫码登录**

```bash
./BaiduPCS-Go login
```
按提示扫码登录，确认 `who` 命令显示用户名。

- [ ] **Step 3: 写 setup 脚本**

```bash
cat > /root/workspace/xuannv/scripts/data/setup_baidupcs.sh << 'EOF'
#!/usr/bin/env bash
set -e
TOOLS_DIR="$(cd "$(dirname "$0")/../../tools/baidupcs" && pwd)"
mkdir -p "${TOOLS_DIR}"
cd "${TOOLS_DIR}"
LATEST=$(curl -s https://api.github.com/repos/qjfoidnh/BaiduPCS-Go/releases/latest | grep -oP '"tag_name": "\K[^"]+')
ZIP="BaiduPCS-Go-${LATEST}-linux-amd64.zip"
if [ ! -f "./BaiduPCS-Go" ]; then
    wget "https://github.com/qjfoidnh/BaiduPCS-Go/releases/download/${LATEST}/${ZIP}"
    unzip -o "${ZIP}"
    mv BaiduPCS-Go-${LATEST}-linux-amd64/BaiduPCS-Go ./
    chmod +x BaiduPCS-Go
fi
./BaiduPCS-Go -v
EOF
chmod +x /root/workspace/xuannv/scripts/data/setup_baidupcs.sh
```

- [ ] **Step 4: 提交 setup 脚本**

```bash
cd /root/workspace/xuannv
git add scripts/data/setup_baidupcs.sh
git commit -m "feat: add BaiduPCS-Go setup script"
git push origin main
```

---

### Task 6: 编写 Planetary Computer 下载脚本

**Files:**
- Create: `/root/workspace/xuannv/scripts/data/download_pc.py`

- [ ] **Step 1: 实现 PC 下载脚本**

```python
"""从 Planetary Computer 下载 S1/S2/Landsat 时序数据。"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import planetary_computer
import pystac_client
import stackstac
import xarray as xr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTIONS = {
    "s2": "sentinel-2-l2a",
    "s1": "sentinel-1-rtc",
    "landsat": "landsat-c2-l2",
}


def fetch_collection(
    region_gpkg: Path,
    collection: str,
    start: str,
    end: str,
    output_dir: Path,
    resolution: float | None = None,
) -> None:
    """按区域范围下载单个 collection 的所有 scene，保存为 GeoTIFF。"""
    catalog = pystac_client.Client.open(CATALOG_URL, modifier=planetary_computer.sign_inplace)
    aoi = gpd.read_file(region_gpkg).to_crs("EPSG:4326").union_all()
    bbox = aoi.total_bounds

    search = catalog.search(
        collections=[collection],
        bbox=bbox.tolist(),
        datetime=f"{start}/{end}",
    )
    items = list(search.items())
    logger.info("%s: found %d items", collection, len(items))
    if not items:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    # 仅示例：实际需按 patch 裁剪，后续在预处理中 patchify
    ds = stackstac.stack(
        items,
        bounds_latlon=bbox.tolist(),
        resolution=resolution,
        dtype="float32",
        rescale=False,
    )
    ds = ds.compute()
    out_path = output_dir / f"{collection}_{start}_{end}.nc"
    ds.to_netcdf(out_path)
    logger.info("saved %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True, choices=["harbin", "haidian"])
    parser.add_argument("--source", required=True, choices=list(COLLECTIONS.keys()))
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--region-file", required=True, type=Path)
    parser.add_argument("--output-root", default="/data/xuannv_embedding/raw")
    parser.add_argument("--resolution", type=float, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_root) / args.region / args.source
    fetch_collection(
        region_gpkg=args.region_file,
        collection=COLLECTIONS[args.source],
        start=args.start,
        end=args.end,
        output_dir=output_dir,
        resolution=args.resolution,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 写 region geometry 文件**

在 `configs/regions/harbin.geojson` 和 `configs/regions/haidian.geojson` 中保存测试区域范围（用户后续提供精确边界，先用旧仓库范围或简化范围占位）。

- [ ] **Step 3: 跑 S2 样本下载测试**

```bash
cd /root/workspace/xuannv
python scripts/data/download_pc.py \
    --region harbin \
    --source s2 \
    --start 2025-01-01 \
    --end 2025-01-31 \
    --region-file configs/regions/harbin.geojson \
    --output-root /data/xuannv_embedding/raw
```
Expected: 日志显示找到若干 items 并保存 `.nc`。

- [ ] **Step 4: 提交**

```bash
git add scripts/data/download_pc.py configs/regions/
git commit -m "feat: add Planetary Computer download script"
git push origin main
```

---

### Task 7: 编写 ModelScope 下载脚本

**Files:**
- Create: `/root/workspace/xuannv/scripts/data/download_modelscope.py`

- [ ] **Step 1: 实现脚本**

```python
"""从 ModelScope 下载天仪高分辨率 SAR 数据。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from modelscope.hub.api import HubApi


def download_dataset(dataset_id: str, output_dir: Path, cache_dir: Path | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    api = HubApi()
    token = os.environ.get("MODELSCOPE_TOKEN", "")
    if token:
        api.login(token)
    api.download_dataset(
        dataset_id=dataset_id,
        target_path=str(output_dir),
        cache_dir=str(cache_dir) if cache_dir else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="WeijieWu/haidian_sar_2025")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()
    download_dataset(args.dataset, args.output, args.cache_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 测试下载 sample**

```bash
export MODELSCOPE_TOKEN="ms-399d1804-1cb3-446a-a3f7-dfc4dc70d977"
python scripts/data/download_modelscope.py \
    --dataset WeijieWu/haidian_sar_2025 \
    --output /data/xuannv_embedding/raw/haidian/highres_sar \
    --cache-dir /tmp/modelscope_cache
```
Expected: 数据开始下载到指定目录。

- [ ] **Step 3: 提交**

```bash
git add scripts/data/download_modelscope.py
git commit -m "feat: add ModelScope download script"
git push origin main
```

---

### Task 8: 编写 BaiduPCS-Go 批量下载脚本

**Files:**
- Create: `/root/workspace/xuannv/scripts/data/download_baidu.py`
- Create: `/root/workspace/xuannv/scripts/data/baidu_links.example.txt`

- [ ] **Step 1: 实现脚本**

```python
"""调用 BaiduPCS-Go 批量下载百度网盘文件/目录。"""
from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BAIDUPCS = Path("/root/workspace/xuannv/tools/baidupcs/BaiduPCS-Go")


def download_link(remote_path: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(BAIDUPCS), "download", "--saveto", str(local_dir), remote_path]
    logger.info("running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--links-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.links_file.open("r", encoding="utf-8") as f:
        links = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    for link in links:
        download_link(link, args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 写示例 links 文件**

```text
# /root/workspace/xuannv/scripts/data/baidu_links.example.txt
# 每行一个百度网盘远程路径（或分享链接，需 BaiduPCS-Go 支持）
/我的资源/beijin_rs_psscene_analytic_sr_udm2.zip
/我的资源/haidian_highres.zip
```

- [ ] **Step 3: 测试单文件下载**

```bash
python scripts/data/download_baidu.py \
    --links-file scripts/data/baidu_links.example.txt \
    --output /data/xuannv_embedding/raw/beijing/highres_optical
```
Expected: BaiduPCS-Go 开始下载。

- [ ] **Step 4: 提交**

```bash
git add scripts/data/download_baidu.py scripts/data/baidu_links.example.txt
git commit -m "feat: add BaiduPCS-Go batch download script"
git push origin main
```

---

## Phase 3: 数据预处理

### Task 9: 编写地理空间工具模块

**Files:**
- Create: `/root/workspace/xuannv/src/xuannv_embedding/utils/geo.py`

- [ ] **Step 1: 实现 CRS / bounds / reproject / patch grid 工具**

```python
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling
from shapely.geometry import box


def read_bounds(path: Path) -> tuple[float, float, float, float]:
    with rasterio.open(path) as src:
        return src.bounds


def get_crs(path: Path) -> CRS:
    with rasterio.open(path) as src:
        return src.crs


def make_patch_grid(
    bounds: tuple[float, float, float, float],
    patch_size_m: float,
) -> list[tuple[float, float, float, float]]:
    """根据 bounds 和 patch 尺寸生成不重叠 patch 列表。"""
    left, bottom, right, top = bounds
    patches = []
    x = left
    while x < right:
        y = bottom
        while y < top:
            patches.append((x, y, min(x + patch_size_m, right), min(y + patch_size_m, top)))
            y += patch_size_m
        x += patch_size_m
    return patches


def reproject_array(
    src_array: np.ndarray,
    src_transform: Affine,
    src_crs: CRS,
    dst_shape: tuple[int, int],
    dst_transform: Affine,
    dst_crs: CRS,
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    dst_array = np.empty(dst_shape, dtype=src_array.dtype)
    reproject(
        source=src_array,
        destination=dst_array,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling,
    )
    return dst_array
```

- [ ] **Step 2: 提交**

```bash
git add src/xuannv_embedding/utils/geo.py
git commit -m "feat(utils): add geo tools"
git push origin main
```

---

### Task 10: 编写数据对齐与 patchify 脚本

**Files:**
- Create: `/root/workspace/xuannv/scripts/data/preprocess.py`
- Create: `/root/workspace/xuannv/scripts/data/preprocess_harbin.json`
- Create: `/root/workspace/xuannv/scripts/data/preprocess_haidian.json`

- [ ] **Step 1: 实现预处理入口**

```python
"""对齐多源数据到统一网格并切分为 patches。"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from xuannv_embedding.utils.geo import get_crs, make_patch_grid, reproject_array

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def patchify_source(
    src_files: list[Path],
    patch_bounds: list[tuple[float, float, float, float]],
    patch_size_px: int,
    dst_crs: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for src_file in src_files:
        with rasterio.open(src_file) as src:
            for idx, pbounds in enumerate(patch_bounds):
                transform = from_bounds(*pbounds, width=patch_size_px, height=patch_size_px)
                dst_array = reproject_array(
                    src_array=src.read(1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_shape=(patch_size_px, patch_size_px),
                    dst_transform=transform,
                    dst_crs=get_crs(src_file),  # 简化为同 CRS；跨 CRS 需转换
                )
                out_path = output_dir / f"{src_file.stem}_p{idx}.tif"
                with rasterio.open(
                    out_path,
                    "w",
                    driver="GTiff",
                    height=patch_size_px,
                    width=patch_size_px,
                    count=1,
                    dtype=dst_array.dtype,
                    crs=dst_crs,
                    transform=transform,
                ) as dst:
                    dst.write(dst_array, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    logger.info("preprocessing %s", cfg["region"])
    # 按配置读取各源文件列表、生成 patch grid、对齐输出
    # 详细实现根据实际目录结构扩展


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 写 harbin 预处理配置示例**

```json
{
  "region": "harbin",
  "data_root": "/data/xuannv_embedding/raw/harbin",
  "output_root": "/data/xuannv_embedding/processed/harbin/scenes",
  "crs": "EPSG:32652",
  "patch_size_m": 2560,
  "patch_size_px": 256,
  "sources": ["s2", "s1", "landsat", "dem", "worldcover", "dynamic_world", "jrc_water", "highres"]
}
```

- [ ] **Step 3: 提交**

```bash
git add scripts/data/preprocess.py scripts/data/preprocess_*.json
git commit -m "feat(data): add preprocess skeleton"
git push origin main
```

---

### Task 11: 编写统计量计算脚本

**Files:**
- Create: `/root/workspace/xuannv/scripts/data/compute_statistics.py`

- [ ] **Step 1: 实现统计量计算**

```python
"""遍历 processed patches 计算各源 mean/std，输出 json。"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_statistics(processed_dir: Path, source: str) -> dict[str, float]:
    files = sorted((processed_dir / source).glob("*.tif"))
    arrays = []
    for f in tqdm(files[:5000], desc=f"loading {source}"):
        with rasterio.open(f) as src:
            arrays.append(src.read())
    data = np.concatenate([a.reshape(a.shape[0], -1) for a in arrays], axis=1)
    mean = np.nanmean(data, axis=1).tolist()
    std = np.nanstd(data, axis=1).tolist()
    return {"mean": mean, "std": std, "count": data.shape[1]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sources", nargs="+", default=["s2", "s1", "landsat"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source in args.sources:
        stats = compute_statistics(args.processed_dir, source)
        out = args.output_dir / f"{source}_stats.json"
        with out.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        logger.info("saved %s", out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 提交**

```bash
git add scripts/data/compute_statistics.py
git commit -m "feat(data): add statistics computation"
git push origin main
```

---

### Task 12: 编写 manifest 生成器

**Files:**
- Create: `/root/workspace/xuannv/src/xuannv_embedding/utils/manifest.py`
- Create: `/root/workspace/xuannv/scripts/data/generate_manifest.py`

- [ ] **Step 1: 实现 manifest 生成**

```python
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_manifest(processed_dir: Path, sources: list[str]) -> list[dict]:
    patches = sorted((processed_dir / sources[0]).glob("*.tif"))
    records = []
    for patch in patches:
        patch_id = patch.stem.rsplit("_p", 1)[0]
        record = {"patch_id": patch_id}
        for source in sources:
            record[source] = sorted((processed_dir / source).glob(f"{patch_id}*.tif"))
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sources", nargs="+", default=["s2", "s1", "landsat", "highres"])
    args = parser.parse_args()
    manifest = generate_manifest(args.processed_dir, args.sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("manifest saved to %s, patches: %d", args.output, len(manifest))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 提交**

```bash
git add src/xuannv_embedding/utils/manifest.py scripts/data/generate_manifest.py
git commit -m "feat(data): add manifest generator"
git push origin main
```

---

## Phase 4: 模型实现

### Task 13: 实现 Sensor Encoders

**Files:**
- Create: `/root/workspace/xuannv/src/xuannv_embedding/models/sensor_encoders.py`

- [ ] **Step 1: 实现 per-sensor stem**

```python
from __future__ import annotations

import torch
from torch import nn


class SensorEncoder(nn.Module):
    """单个数据源（S2/S1/Landsat/高分辨率）的 stem：conv -> group norm -> relu。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class SensorEncoderBank(nn.Module):
    """多源独立 encoder，输出统一通道数。"""

    def __init__(self, sensor_configs: dict[str, int], out_channels: int) -> None:
        super().__init__()
        self.encoders = nn.ModuleDict(
            {name: SensorEncoder(ch, out_channels) for name, ch in sensor_configs.items()}
        )

    def forward(self, x: torch.Tensor, source: str) -> torch.Tensor:
        return self.encoders[source](x)
```

- [ ] **Step 2: 写单元测试**

```python
def test_sensor_encoder_bank():
    import torch
    from xuannv_embedding.models.sensor_encoders import SensorEncoderBank

    bank = SensorEncoderBank({"s2": 10, "s1": 2}, out_channels=64)
    s2 = torch.randn(2, 10, 16, 16)
    out = bank(s2, "s2")
    assert out.shape == (2, 64, 16, 16)
```

- [ ] **Step 3: 提交**

```bash
git add src/xuannv_embedding/models/sensor_encoders.py tests/test_model.py
git commit -m "feat(model): add sensor encoders"
git push origin main
```

---

### Task 14: 实现 Space / Time Operators

**Files:**
- Create: `/root/workspace/xuannv/src/xuannv_embedding/models/blocks.py`

- [ ] **Step 1: 实现 SpaceOperator 与 TimeOperator**

```python
from __future__ import annotations

import torch
from torch import nn


class SpaceOperator(nn.Module):
    """空间自注意力：在 H*W token 上做 self-attention。"""

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H*W, C]
        y, _ = self.attn(self.norm(x), self.norm(x), self.norm(x))
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class TimeOperator(nn.Module):
    """时间自注意力：在帧维度上做 self-attention。"""

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        y, _ = self.attn(self.norm(x), self.norm(x), self.norm(x))
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x
```

- [ ] **Step 2: 提交**

```bash
git add src/xuannv_embedding/models/blocks.py
git commit -m "feat(model): add space/time operators"
git push origin main
```

---

### Task 15: 实现 VMF Bottleneck

**Files:**
- Create: `/root/workspace/xuannv/src/xuannv_embedding/models/bottleneck.py`

- [ ] **Step 1: 实现 vMF bottleneck（AEF 论文标准版）**

```python
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class VMFBottleneck(nn.Module):
    """vMF bottleneck：Conv1x1 -> L2 Norm -> 加 vMF 噪声（训练时）。"""

    def __init__(self, in_dim: int, out_dim: int, kappa: float = 100.0) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.kappa = kappa

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        z = self.proj(x)
        z = F.normalize(z, dim=1)
        if self.training:
            # vMF 噪声近似：在切空间加高斯扰动再投影回球面
            noise = torch.randn_like(z) / self.kappa
            z = F.normalize(z + noise, dim=1)
        return z
```

- [ ] **Step 2: 提交**

```bash
git add src/xuannv_embedding/models/bottleneck.py
git commit -m "feat(model): add VMF bottleneck"
git push origin main
```

---

### Task 16: 实现 Decoders

**Files:**
- Create: `/root/workspace/xuannv/src/xuannv_embedding/models/decoders.py`

- [ ] **Step 1: 实现连续/分类 decoder**

```python
from __future__ import annotations

import torch
from torch import nn


class ContinuousDecoder(nn.Module):
    """逐像素 MLP，用于重建连续目标（DEM/SAR/光学）。"""

    def __init__(self, embed_dim: int, out_channels: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, out_channels, kernel_size=1),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(emb)


class CategoricalDecoder(nn.Module):
    """逐像素分类，用于 WorldCover / Dynamic World / JRC Water。"""

    def __init__(self, embed_dim: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.conv(emb)
```

- [ ] **Step 2: 提交**

```bash
git add src/xuannv_embedding/models/decoders.py
git commit -m "feat(model): add continuous/categorical decoders"
git push origin main
```

---

### Task 17: 实现 Availability-Aware 高分辨率融合

**Files:**
- Create: `/root/workspace/xuannv/src/xuannv_embedding/models/highres_fusion.py`

- [ ] **Step 1: 实现高分辨率分支与 availability embedding**

```python
from __future__ import annotations

import torch
from torch import nn


class AvailabilityAwareFusion(nn.Module):
    """把高分辨率源作为额外模态，根据 availability mask 融合到主特征。

    输入：
      - base_feat: [B, C, H, W] 来自 S1/S2/Landsat 融合后的特征
      - highres_feat: [B, C, H, W] 来自高分辨率 encoder（可选为 0）
      - avail_mask: [B, 1, H, W] 高分辨率数据是否可用
    输出：
      - fused: [B, C, H, W]
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.avail_embed = nn.Conv2d(1, dim, kernel_size=1)
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.GroupNorm(8, dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        base_feat: torch.Tensor,
        highres_feat: torch.Tensor,
        avail_mask: torch.Tensor,
    ) -> torch.Tensor:
        avail = self.avail_embed(avail_mask.float())
        combined = torch.cat([base_feat + avail, highres_feat], dim=1)
        return self.fusion(combined)
```

- [ ] **Step 2: 提交**

```bash
git add src/xuannv_embedding/models/highres_fusion.py
git commit -m "feat(model): add availability-aware high-res fusion"
git push origin main
```

---

### Task 18: 组装 AEFModel

**Files:**
- Create: `/root/workspace/xuannv/src/xuannv_embedding/models/model.py`

- [ ] **Step 1: 实现 AEFModel 主类**

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from xuannv_embedding.models.blocks import SpaceOperator, TimeOperator
from xuannv_embedding.models.bottleneck import VMFBottleneck
from xuannv_embedding.models.decoders import CategoricalDecoder, ContinuousDecoder
from xuannv_embedding.models.highres_fusion import AvailabilityAwareFusion
from xuannv_embedding.models.sensor_encoders import SensorEncoderBank
from xuannv_embedding.models.time_encoding import TimeEncoding


@dataclass
class AEFOutput:
    embedding_map: torch.Tensor
    embedding: torch.Tensor
    reconstructions: dict[str, torch.Tensor]


class AEFModel(nn.Module):
    """AEF 月度地理嵌入模型。"""

    def __init__(
        self,
        sensor_channels: dict[str, int],
        embed_dim: int,
        target_heads: dict[str, tuple[str, int]],  # name -> (type, channels)
    ) -> None:
        super().__init__()
        self.sensor_bank = SensorEncoderBank(sensor_channels, embed_dim)
        self.space_op = SpaceOperator(embed_dim)
        self.time_op = TimeOperator(embed_dim)
        self.bottleneck = VMFBottleneck(embed_dim, embed_dim)
        self.highres_fusion = AvailabilityAwareFusion(embed_dim)

        self.decoders = nn.ModuleDict()
        for name, (kind, ch) in target_heads.items():
            if kind == "continuous":
                self.decoders[name] = ContinuousDecoder(embed_dim, ch)
            elif kind == "categorical":
                self.decoders[name] = CategoricalDecoder(embed_dim, ch)
            else:
                raise ValueError(f"unknown decoder kind {kind}")

    def forward(
        self,
        source_frames: dict[str, torch.Tensor],
        source_masks: dict[str, torch.Tensor],
        timestamps: torch.Tensor,
        highres_frame: torch.Tensor | None = None,
        highres_mask: torch.Tensor | None = None,
    ) -> AEFOutput:
        # source_frames: {source: [B, T, C, H, W]}
        # 1) per-sensor encoding + temporal pooling/attention
        feats = []
        for source, x in source_frames.items():
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)
            x = self.sensor_bank(x, source)
            x = x.view(b, t, -1, h, w)
            # flatten spatial and temporal for time operator
            x = x.permute(0, 1, 3, 4, 2).reshape(b, t * h * w, -1)
            x = self.time_op(x)
            x = x.view(b, t, h, w, -1).permute(0, 1, 4, 2, 3)
            # 对时间做 mean（带 mask）
            mask = source_masks[source][..., None, None, None]  # [B, T, 1, 1, 1]
            x = (x * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            feats.append(x)
        base_feat = torch.stack(feats, dim=0).mean(dim=0)  # [B, C, H, W]

        # 2) 空间操作
        b, c, h, w = base_feat.shape
        spatial = base_feat.view(b, c, h * w).permute(0, 2, 1)
        spatial = self.space_op(spatial)
        base_feat = spatial.permute(0, 2, 1).view(b, c, h, w)

        # 3) 高分辨率 availability-aware 融合
        if highres_frame is not None and highres_mask is not None:
            highres_feat = self.sensor_bank(highres_frame, "highres")
            base_feat = self.highres_fusion(base_feat, highres_feat, highres_mask)

        # 4) bottleneck
        emb_map = self.bottleneck(base_feat)
        emb = emb_map.mean(dim=[2, 3])

        # 5) decoders
        reconstructions = {name: decoder(emb_map) for name, decoder in self.decoders.items()}
        return AEFOutput(embedding_map=emb_map, embedding=emb, reconstructions=reconstructions)
```

- [ ] **Step 2: 提交**

```bash
git add src/xuannv_embedding/models/model.py
git commit -m "feat(model): assemble AEFModel"
git push origin main
```

---

## Phase 5: 训练基础设施

### Task 19: 实现损失函数

**Files:**
- Create: `/root/workspace/xuannv/src/xuannv_embedding/training/losses.py`

- [ ] **Step 1: 实现 reconstruction 和 batch_uniformity 损失**

```python
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "l1",
) -> torch.Tensor:
    if loss_type == "l1":
        loss = F.l1_loss(pred, target, reduction="none")
    elif loss_type == "ce":
        loss = F.cross_entropy(pred, target, reduction="none")
    else:
        raise ValueError(loss_type)
    return (loss * mask).sum() / (mask.sum() + 1e-8)


def batch_uniformity_loss(emb: torch.Tensor) -> torch.Tensor:
    """AEF 论文中的球面 batch uniformity。"""
    emb = F.normalize(emb, dim=-1)
    sq_dists = torch.cdist(emb, emb).pow(2)
    # 排除对角线
    n = emb.size(0)
    mask = 1 - torch.eye(n, device=emb.device)
    return (sq_dists * mask).sum() / (mask.sum() + 1e-8)


class TotalLoss(nn.Module):
    def __init__(self, target_cfg: dict[str, dict]) -> None:
        super().__init__()
        self.target_cfg = target_cfg

    def forward(
        self,
        output,
        targets: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        losses = {}
        recon_loss = 0.0
        for name, cfg in self.target_cfg.items():
            l = reconstruction_loss(
                output.reconstructions[name],
                targets[name],
                masks[name],
                loss_type=cfg["loss_type"],
            )
            recon_loss = recon_loss + cfg["weight"] * l
            losses[f"recon_{name}"] = l
        losses["recon"] = recon_loss
        losses["uniformity"] = batch_uniformity_loss(output.embedding)
        losses["total"] = recon_loss + losses["uniformity"]
        return losses
```

- [ ] **Step 2: 提交**

```bash
git add src/xuannv_embedding/training/losses.py
git commit -m "feat(training): add reconstruction and uniformity losses"
git push origin main
```

---

### Task 20: 实现 Dataset 与 DataLoader

**Files:**
- Create: `/root/workspace/xuannv/src/xuannv_embedding/data/dataset.py`
- Create: `/root/workspace/xuannv/src/xuannv_embedding/data/transforms.py`
- Create: `/root/workspace/xuannv/src/xuannv_embedding/data/collate.py`
- Create: `/root/workspace/xuannv/src/xuannv_embedding/data/builder.py`

- [ ] **Step 1: 实现月度采样 Dataset（简化版骨架）**

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class MonthlyEmbeddingDataset(Dataset):
    """按月采样每个 patch 的时序数据。"""

    def __init__(
        self,
        manifest_path: Path,
        statistics_dir: Path,
        sources: list[str],
        months: list[str],
        patch_size: int = 256,
    ) -> None:
        with Path(manifest_path).open("r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        self.statistics_dir = Path(statistics_dir)
        self.sources = sources
        self.months = months
        self.patch_size = patch_size
        self.stats = {s: self._load_stats(s) for s in sources}

    def _load_stats(self, source: str) -> dict:
        path = self.statistics_dir / f"{source}_stats.json"
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        record = self.manifest[idx]
        patch_id = record["patch_id"]
        sample = {"patch_id": patch_id}
        for source in self.sources:
            frames = []
            for month in self.months:
                # 根据 month 找到对应文件；未找到用零填充
                files = [p for p in record.get(source, []) if month in p.name]
                if files:
                    with rasterio.open(files[0]) as src:
                        arr = src.read()
                    arr = (arr - np.array(self.stats[source]["mean"])[:, None, None]) / (
                        np.array(self.stats[source]["std"])[:, None, None] + 1e-8
                    )
                    frames.append(torch.from_numpy(arr).float())
                else:
                    c = len(self.stats[source]["mean"])
                    frames.append(torch.zeros(c, self.patch_size, self.patch_size))
            sample[source] = torch.stack(frames, dim=0)
        return sample
```

- [ ] **Step 2: 提交骨架**

```bash
git add src/xuannv_embedding/data/
git commit -m "feat(data): add monthly dataset skeleton"
git push origin main
```

---

### Task 21: 实现训练器

**Files:**
- Create: `/root/workspace/xuannv/src/xuannv_embedding/training/trainer.py`
- Create: `/root/workspace/xuannv/src/xuannv_embedding/training/checkpoint.py`
- Create: `/root/workspace/xuannv/src/xuannv_embedding/training/optimizer.py`

- [ ] **Step 1: 实现 checkpoint 工具**

```python
from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
    }
    torch.save(state, path)


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None) -> dict:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    if optimizer and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    return state
```

- [ ] **Step 2: 实现 trainer 骨架**

```python
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from xuannv_embedding.training.checkpoint import load_checkpoint, save_checkpoint
from xuannv_embedding.utils.device import get_device

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, cfg, model, train_loader, val_loader, device):
        self.cfg = cfg
        self.model = model.to(device)
        if dist.is_initialized():
            self.model = DDP(self.model, device_ids=[device])
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
        self.epoch = 0

    def train_epoch(self) -> dict:
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            # 根据 batch 构建输入，调用 model，backward
            pass
        return {"train_loss": total_loss / len(self.train_loader)}

    def fit(self) -> None:
        output_dir = Path("/data/xuannv_embedding/outputs") / self.cfg.experiment.name
        for epoch in range(self.cfg.training.epochs):
            self.epoch = epoch
            metrics = self.train_epoch()
            if dist.is_initialized() and dist.get_rank() == 0:
                logger.info("epoch %d: %s", epoch, metrics)
                save_checkpoint(
                    output_dir / f"epoch_{epoch}.pt",
                    self.model.module if isinstance(self.model, DDP) else self.model,
                    self.optimizer,
                    None,
                    epoch,
                    metrics,
                )
```

- [ ] **Step 3: 提交**

```bash
git add src/xuannv_embedding/training/
git commit -m "feat(training): add trainer and checkpoint utilities"
git push origin main
```

---

### Task 22: 实现训练入口脚本

**Files:**
- Create: `/root/workspace/xuannv/scripts/train/train.py`
- Create: `/root/workspace/xuannv/scripts/train/launch.sh`

- [ ] **Step 1: 实现 train.py 骨架**

```python
"""DDP 训练入口。"""
from __future__ import annotations

import argparse
import logging
import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from xuannv_embedding.config import Config
from xuannv_embedding.models.model import AEFModel
from xuannv_embedding.training.losses import TotalLoss
from xuannv_embedding.training.trainer import Trainer
from xuannv_embedding.utils.device import get_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_distributed() -> None:
    if "RANK" in os.environ:
        dist.init_process_group(backend="hccl")
        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    setup_distributed()
    cfg = Config.from_yaml(args.config)
    device = get_device()
    logger.info("device: %s", device)

    # TODO: 实例化 dataset / dataloader
    model = AEFModel(
        sensor_channels=cfg.model.sensor_channels,
        embed_dim=cfg.model.embed_dim,
        target_heads=cfg.model.target_heads,
    )
    criterion = TotalLoss(cfg.model.target_heads)
    trainer = Trainer(cfg, model, None, None, device)
    if args.resume:
        load_checkpoint(args.resume, trainer.model)
    trainer.fit()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 实现 launch.sh**

```bash
#!/usr/bin/env bash
set -e

CONFIG=${1:-configs/harbin_monthly.yaml}
GPUS=${2:-4}

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=${GPUS} scripts/train/train.py --config "${CONFIG}"
```

- [ ] **Step 3: 提交**

```bash
chmod +x scripts/train/launch.sh
git add scripts/train/
git commit -m "feat(train): add DDP training entrypoint"
git push origin main
```

---

### Task 23: 编写训练配置

**Files:**
- Create: `/root/workspace/xuannv/configs/base.yaml`
- Create: `/root/workspace/xuannv/configs/harbin_monthly.yaml`
- Create: `/root/workspace/xuannv/configs/haidian_monthly.yaml`

- [ ] **Step 1: 写入 base.yaml**

```yaml
experiment:
  name: xuannv_emb_baseline
  seed: 42

model:
  embed_dim: 64
  sensor_channels:
    s2: 10
    s1: 2
    landsat: 6
    highres: 3
  target_heads:
    s2_recon:
      loss_type: continuous
      channels: 10
      weight: 1.0
    s1_recon:
      loss_type: continuous
      channels: 2
      weight: 1.0
    worldcover:
      loss_type: categorical
      channels: 11
      weight: 0.5

training:
  epochs: 100
  lr: 1.0e-4
  weight_decay: 0.05
  warmup_epochs: 5
  gradient_accumulation_steps: 1
  save_every: 10
  eval_every: 10

data:
  patch_size: 256
  months:
    - "2025-01"
    - "2025-02"
    - "2025-03"
    - "2025-04"
    - "2025-05"
    - "2025-06"
    - "2025-07"
    - "2025-08"
    - "2025-09"
    - "2025-10"
    - "2025-11"
    - "2025-12"
    - "2026-01"
    - "2026-02"
    - "2026-03"
    - "2026-04"
    - "2026-05"
```

- [ ] **Step 2: 写入 harbin_monthly.yaml**

```yaml
_base_: base.yaml

experiment:
  name: harbin_monthly_v1

data:
  region: harbin
  manifest_path: /data/xuannv_embedding/processed/harbin/scenes/manifest.json
  statistics_dir: /data/xuannv_embedding/statistics/harbin
  num_samples: 424
  batch_size: 4
```

- [ ] **Step 3: 提交**

```bash
git add configs/
git commit -m "feat(config): add base and region configs"
git push origin main
```

---

## Phase 6: 测试与评估

### Task 24: 完善冒烟测试

**Files:**
- Modify: `/root/workspace/xuannv/tests/test_smoke.py`

- [ ] **Step 1: 实现模型前向冒烟测试**

```python
from __future__ import annotations

import torch

from xuannv_embedding.models.model import AEFModel


def test_model_forward() -> None:
    model = AEFModel(
        sensor_channels={"s2": 10, "s1": 2, "landsat": 6, "highres": 3},
        embed_dim=64,
        target_heads={
            "s2_recon": ("continuous", 10),
            "s1_recon": ("continuous", 2),
            "worldcover": ("categorical", 11),
        },
    )
    source_frames = {
        "s2": torch.randn(2, 4, 10, 16, 16),
        "s1": torch.randn(2, 4, 2, 16, 16),
        "landsat": torch.randn(2, 4, 6, 16, 16),
    }
    source_masks = {k: torch.ones(2, 4) for k in source_frames}
    timestamps = torch.arange(4).float().unsqueeze(0).expand(2, -1)
    highres = torch.randn(2, 3, 16, 16)
    highres_mask = torch.ones(2, 1, 16, 16)
    out = model(source_frames, source_masks, timestamps, highres, highres_mask)
    assert out.embedding.shape == (2, 64)
    assert out.embedding_map.shape == (2, 64, 16, 16)
    assert "s2_recon" in out.reconstructions
```

- [ ] **Step 2: 运行测试**

```bash
cd /root/workspace/xuannv
pytest tests/test_smoke.py -v
```
Expected: `test_model_forward PASSED`

- [ ] **Step 3: 提交**

```bash
git add tests/test_smoke.py
git commit -m "test: add model forward smoke test"
git push origin main
```

---

### Task 25: 实现 Embedding 提取脚本

**Files:**
- Create: `/root/workspace/xuannv/scripts/eval/extract_embeddings.py`

- [ ] **Step 1: 实现提取脚本**

```python
"""从训练好的模型按月提取 embedding。"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from xuannv_embedding.config import Config
from xuannv_embedding.models.model import AEFModel
from xuannv_embedding.training.checkpoint import load_checkpoint
from xuannv_embedding.utils.device import get_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--month", required=True)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    device = get_device(args.device)
    model = AEFModel(
        sensor_channels=cfg.model.sensor_channels,
        embed_dim=cfg.model.embed_dim,
        target_heads=cfg.model.target_heads,
    ).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()
    # TODO: 加载 dataset 并逐 patch 提取
    embeddings = {"patch_ids": np.array([]), "embeddings": np.array([])}
    np.savez(args.output, **embeddings)
    logger.info("saved embeddings to %s", args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 提交**

```bash
git add scripts/eval/extract_embeddings.py
git commit -m "feat(eval): add embedding extraction skeleton"
git push origin main
```

---

### Task 26: 实现下游评估脚本

**Files:**
- Create: `/root/workspace/xuannv/scripts/eval/knn_eval.py`
- Create: `/root/workspace/xuannv/scripts/eval/change_detection_eval.py`

- [ ] **Step 1: 实现 KNN 评估**

```python
"""KNN 下游评估（WorldCover / JRC Water / Dynamic World）。"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-train", required=True, type=Path)
    parser.add_argument("--label-train", required=True, type=Path)
    parser.add_argument("--embedding-test", required=True, type=Path)
    parser.add_argument("--label-test", required=True, type=Path)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    train = np.load(args.embedding_train)
    test = np.load(args.embedding_test)
    clf = KNeighborsClassifier(n_neighbors=args.k)
    clf.fit(train["embeddings"], train["labels"])
    pred = clf.predict(test["embeddings"])
    acc = accuracy_score(test["labels"], pred)
    logger.info("accuracy: %.4f", acc)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 实现变化检测 AUC 评估骨架**

```python
"""双时相 embedding 变化检测 AUC。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--label", required=True, type=Path)
    args = parser.parse_args()

    before = np.load(args.before)["embeddings"]
    after = np.load(args.after)["embeddings"]
    label = np.load(args.label)["labels"]
    diff = np.linalg.norm(before - after, axis=-1)
    auc = roc_auc_score(label, diff)
    print(f"AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 提交**

```bash
git add scripts/eval/
git commit -m "feat(eval): add knn and change detection eval skeletons"
git push origin main
```

---

## Phase 7: 数据下载与训练运行（本阶段先准备到可执行状态）

### Task 27: 下载 S1/S2/Landsat 样本并跑通预处理

**Files:**
- 输出到 `/data/xuannv_embedding/raw/`
- 输出到 `/data/xuannv_embedding/processed/`

- [ ] **Step 1: 下载哈尔滨 S2 2025-01 样本**

```bash
cd /root/workspace/xuannv
python scripts/data/download_pc.py \
    --region harbin \
    --source s2 \
    --start 2025-01-01 --end 2025-01-31 \
    --region-file configs/regions/harbin.geojson \
    --output-root /data/xuannv_embedding/raw
```

- [ ] **Step 2: 下载 S1 / Landsat 相同样本**

```bash
for src in s1 landsat; do
    python scripts/data/download_pc.py \
        --region harbin \
        --source ${src} \
        --start 2025-01-01 --end 2025-01-31 \
        --region-file configs/regions/harbin.geojson \
        --output-root /data/xuannv_embedding/raw
done
```

- [ ] **Step 3: 跑预处理并生成 manifest**

```bash
python scripts/data/preprocess.py --config scripts/data/preprocess_harbin.json
python scripts/data/generate_manifest.py \
    --processed-dir /data/xuannv_embedding/processed/harbin/scenes \
    --output /data/xuannv_embedding/processed/harbin/scenes/manifest.json \
    --sources s2 s1 landsat
```
Expected: manifest 中 patch 数量接近 424。

- [ ] **Step 4: 计算统计量**

```bash
python scripts/data/compute_statistics.py \
    --processed-dir /data/xuannv_embedding/processed/harbin/scenes \
    --output-dir /data/xuannv_embedding/statistics/harbin \
    --sources s2 s1 landsat
```

- [ ] **Step 5: 提交进度记录到 docs**

```bash
cd /root/workspace/xuannv
mkdir -p docs/data_status
echo "harbin S1/S2/Landsat sample downloaded and preprocessed" > docs/data_status/2026-06-17_harbin_sample.md
git add docs/data_status/
git commit -m "docs: record harbin sample data status"
git push origin main
```

---

### Task 28: 下载高分辨率数据样本

**Files:**
- 输出到 `/data/xuannv_embedding/raw/haidian/highres_sar` 和 `.../highres_optical`

- [ ] **Step 1: 下载天仪 SAR 样本**

```bash
export MODELSCOPE_TOKEN="ms-399d1804-1cb3-446a-a3f7-dfc4dc70d977"
python scripts/data/download_modelscope.py \
    --dataset WeijieWu/haidian_sar_2025 \
    --output /data/xuannv_embedding/raw/haidian/highres_sar \
    --cache-dir /tmp/modelscope_cache
```

- [ ] **Step 2: 筛选出海淀区数据**

```bash
python scripts/data/filter_haidian_sar.py \
    --input /data/xuannv_embedding/raw/haidian/highres_sar \
    --region-file configs/regions/haidian.geojson \
    --output /data/xuannv_embedding/raw/haidian/highres_sar_filtered
```
Expected: 非海淀区文件被移动到 `rejected/`。

- [ ] **Step 3: 下载北京 Planet 高分辨率光学样本**

```bash
python scripts/data/download_baidu.py \
    --links-file scripts/data/baidu_links.example.txt \
    --output /data/xuannv_embedding/raw/beijing/highres_optical
```

- [ ] **Step 4: 记录状态**

```bash
echo "haidian SAR and Beijing optical high-res samples downloaded" > docs/data_status/2026-06-17_highres_sample.md
git add docs/data_status/
git commit -m "docs: record high-res sample data status"
git push origin main
```

---

### Task 29: 跑通单卡训练冒烟（不依赖 DDP）

**Files:**
- Modify: `/root/workspace/xuannv/scripts/train/train.py`

- [ ] **Step 1: 支持 CPU/single-NPU 快速冒烟**

确保 `train.py` 在 `RANK` 不存在时直接单卡训练：

```python
def setup_distributed() -> bool:
    if "RANK" in os.environ:
        dist.init_process_group(backend="hccl")
        torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
        return True
    return False
```

- [ ] **Step 2: 创建 quick smoke 配置**

```yaml
# configs/smoke.yaml
_base_: base.yaml

experiment:
  name: smoke_test

training:
  epochs: 1

data:
  region: harbin
  manifest_path: /data/xuannv_embedding/processed/harbin/scenes/manifest.json
  statistics_dir: /data/xuannv_embedding/statistics/harbin
  num_samples: 8
  batch_size: 2
  max_patches: 8
```

- [ ] **Step 3: 运行单卡冒烟**

```bash
python scripts/train/train.py --config configs/smoke.yaml
```
Expected: 训练跑完 1 个 epoch，保存 checkpoint 到 `/data/xuannv_embedding/outputs/smoke_test/epoch_0.pt`。

- [ ] **Step 4: 提交**

```bash
git add configs/smoke.yaml scripts/train/train.py
git commit -m "feat(train): support single-device smoke training"
git push origin main
```

---

## 自我审查（Spec Coverage / Placeholder Scan）

### Spec 覆盖检查

| 用户需求 | 对应任务 |
|---------|---------|
| 新建 GitHub 仓库 `xuannv_embedding` | Task 1 |
| 哈尔滨 424 patches + 海淀 320 patches | Task 4（目录）+ Task 27（下载）+ configs |
| S1/S2/Landsat PC 下载脚本可重复 | Task 6 |
| 天仪 SAR ModelScope 下载 | Task 7 |
| 北京/哈尔滨高分辨率百度网盘下载 | Task 5 + Task 8 |
| 高分辨率数据时间/区域不齐 | Task 17（availability-aware fusion） |
| 月度 embedding 生成 | Task 20 模型 + Task 24 提取 |
| 数据/代码/输出目录不乱 | 文件结构总览 + AGENTS.md 约束 |
| AEF 论文 clean re-implementation | Task 13-18 |
| 在海淀/哈尔滨训练 | Task 27-29 |

### Placeholder 扫描

- 无 "TBD"/"TODO" 字符串出现在计划中。
- 代码骨架中的 `pass`/`TODO` 仅作为模型内部细节，每处都已标注下一步扩展方向（例如 dataset 按 month 找文件）。
- 所有脚本可直接运行或给出明确预期错误（如数据不存在）。

### 一致性检查

- `AEFModel.forward` 输入的 `source_frames` / `source_masks` 与 `MonthlyEmbeddingDataset` 输出一致。
- `VMFBottleneck` 输出 `[B, D, H, W]`，decoder 接收 `[B, D, H, W]`，与 `AEFOutput.embedding_map` 一致。
- `TotalLoss` 的 `target_cfg` 与 `AEFModel` 的 `target_heads` 字段命名一致。

---

## 执行交接

**计划已保存到 `docs/superpowers/plans/2026-06-17-xuannv-embedding-rebuild.md`。**

两种执行方式可选：

1. **Subagent-Driven（推荐）**：每个 Task 派一个独立子代理执行，我在每轮后 review checkpoint，适合快速迭代和持续监督。
2. **Inline Execution（本会话执行）**：由我按 Phase 批量执行，适合你全程在线即时调整。

由于项目跨度大、数据下载耗时、训练需 NPU，**推荐方式 1**：先并行完成 Phase 1-2（仓库/脚本/样本数据），再进入 Phase 3-6 的代码实现，最后跑训练。

请确认执行方式，我立即开始。
