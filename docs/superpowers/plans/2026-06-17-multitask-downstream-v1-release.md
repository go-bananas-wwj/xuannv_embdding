# V1.0 多任务下游分割版本发布实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将当前多任务下游分割实验状态固定为 V1.0，包括本地归档、可视化、Git tag、GitHub Release 和 ModelScope 权重发布。

**Architecture:** 在本地创建 `/data/xuannv_embedding/experiments/v1.0/` 阶段目录，通过符号链接聚合 5 任务实验结果；编写 manifest 记录版本元数据；使用现有可视化脚本生成 patch 叠加图；最后通过 git tag 和 ModelScope SDK 发布代码与权重。

**Tech Stack:** Python, PyTorch, Git, GitHub CLI (optional), ModelScope Python SDK, matplotlib/rasterio

---

### Task 1: 创建 V1.0 本地目录结构

**Files:**
- Create: `/data/xuannv_embedding/experiments/v1.0/` (directory tree)

- [ ] **Step 1: 创建目录**

Run:
```bash
mkdir -p /data/xuannv_embedding/experiments/v1.0/{heads,encoder,visualizations,configs}
```

Expected: 4 个子目录创建成功。

- [ ] **Step 2: 提交目录占位说明**

Create: `/data/xuannv_embedding/experiments/v1.0/README.md`

```markdown
# V1.0 Multitask Downstream Release

- `heads/`: best fold segmentation head weights
- `encoder/`: pre-trained AEF encoder checkpoint
- `visualizations/`: downstream task overlay images
- `configs/`: YAML configs used for training
- `v1.0_manifest.json`: version metadata
```

Run:
```bash
cd /root/workspace/xuannv && git add docs/superpowers/plans/2026-06-17-multitask-downstream-v1-release.md
git commit -m "docs(plan): add v1.0 release plan"
```

---

### Task 2: 复制/链接 5 任务实验结果到 V1.0

**Files:**
- Create symlinks: `/data/xuannv_embedding/experiments/v1.0/<task>/`
- Source dirs:
  - `construction_upernet_harbin_5fold_v3/`
  - `building_change_diff_unet_harbin_5fold/`
  - `farm_change_diff_unet_harbin_5fold/`
  - `rubbish_diff_unet_harbin_5fold/`
  - `construction_upernet_joint_5fold_v3/`

- [ ] **Step 1: 链接实验目录**

Run:
```bash
cd /data/xuannv_embedding/experiments/v1.0
ln -s /data/xuannv_embedding/experiments/construction_upernet_harbin_5fold_v3 construction
ln -s /data/xuannv_embedding/experiments/building_change_diff_unet_harbin_5fold building_change
ln -s /data/xuannv_embedding/experiments/farm_change_diff_unet_harbin_5fold farm_change
ln -s /data/xuannv_embedding/experiments/rubbish_diff_unet_harbin_5fold rubbish
ln -s /data/xuannv_embedding/experiments/construction_upernet_joint_5fold_v3 construction_joint
```

Expected: `ls -l` 显示 5 个符号链接。

- [ ] **Step 2: 复制汇总报告**

Run:
```bash
cp /data/xuannv_embedding/experiments/FINAL_REPORT.md /data/xuannv_embedding/experiments/v1.0/
cp /data/xuannv_embedding/experiments/all_tasks_summary_final.json /data/xuannv_embedding/experiments/v1.0/
```

---

### Task 3: 生成下游任务可视化图片

**Files:**
- Modify: `downstreams/scripts/visualize_results.py` (if needed to support multi-task output dir)
- Create: `/data/xuannv_embedding/experiments/v1.0/visualizations/`

- [ ] **Step 1: 为每个任务生成 PR 曲线和叠加图**

Run for each task:
```bash
cd /root/workspace/xuannv/.worktrees/feat-multitask-downstream
python downstreams/scripts/visualize_results.py \
  --output-root /data/xuannv_embedding/experiments/v1.0/<task> \
  --label-root /data/xuannv_embedding/processed/<region>/masks \
  --rgb-source /data/xuannv_embedding/processed/<region>/s2 \
  --out-dir /data/xuannv_embedding/experiments/v1.0/visualizations/<task>
```

Where `<task>` and `<region>` mapping:
- construction → harbin
- building_change → harbin
- farm_change → harbin
- rubbish → harbin
- construction_joint → haidian+harbin (use harbin RGB if haidian not available)

Expected: 每个任务目录下出现 `visualizations/` 子目录，包含 PR 曲线和若干 patch 叠加图。

- [ ] **Step 2: 生成 5 任务指标汇总柱状图**

Create: `/root/workspace/xuannv/downstreams/scripts/plot_task_summary.py`

```python
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

summary_path = "/data/xuannv_embedding/experiments/v1.0/all_tasks_summary_final.json"
out_path = "/data/xuannv_embedding/experiments/v1.0/visualizations/task_metrics_comparison.png"

with open(summary_path) as f:
    data = json.load(f)

tasks = list(data.keys())
auc = [data[t]["auc_roc_mean"] for t in tasks]
f1 = [data[t]["f1_best_mean"] for t in tasks]

x = np.arange(len(tasks))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - width/2, auc, width, label="AUC-ROC")
ax.bar(x + width/2, f1, width, label="F1_best")
ax.axhline(0.8, color="r", linestyle="--", label="AUC target")
ax.axhline(0.6, color="g", linestyle="--", label="F1 target")
ax.set_xticks(x)
ax.set_xticklabels(tasks, rotation=15, ha="right")
ax.set_ylim(0, 1)
ax.legend()
ax.set_title("V1.0 Multitask Downstream Metrics")
fig.tight_layout()
fig.savefig(out_path, dpi=300)
print(f"Saved summary plot to {out_path}")
```

Run:
```bash
python downstreams/scripts/plot_task_summary.py
```

Expected: `/data/xuannv_embedding/experiments/v1.0/visualizations/task_metrics_comparison.png` 生成。

- [ ] **Step 3: 提交可视化脚本**

Run:
```bash
cd /root/workspace/xuannv/.worktrees/feat-multitask-downstream
git add downstreams/scripts/plot_task_summary.py
git commit -m "feat(viz): add multi-task metrics summary plotter"
```

---

### Task 4: 收集 best fold head 权重和 configs

**Files:**
- Source: each task experiment dir `fold_<i>/checkpoints/best.pt`
- Source: `downstreams/configs/construction_segmentation_harbin.yaml` etc.
- Target: `/data/xuannv_embedding/experiments/v1.0/heads/` and `/data/xuannv_embedding/experiments/v1.0/configs/`

- [ ] **Step 1: 编写收集脚本**

Create: `/root/workspace/xuannv/downstreams/scripts/collect_v1.0_artifacts.py`

```python
import json
import shutil
from pathlib import Path

summary = json.load(open("/data/xuannv_embedding/experiments/v1.0/all_tasks_summary_final.json"))
v1_root = Path("/data/xuannv_embedding/experiments/v1.0")
heads_dir = v1_root / "heads"
configs_dir = v1_root / "configs"
heads_dir.mkdir(exist_ok=True)
configs_dir.mkdir(exist_ok=True)

for task, meta in summary.items():
    best_fold = meta.get("best_fold", 0)
    src_head = v1_root / task / f"fold_{best_fold}" / "checkpoints" / "best.pt"
    dst_head = heads_dir / f"{task}_fold{best_fold}_best.pt"
    if src_head.exists():
        shutil.copy2(src_head, dst_head)
        print(f"Copied {src_head} -> {dst_head}")
    else:
        print(f"Missing {src_head}")

# Copy configs used
config_map = {
    "construction": "downstreams/configs/construction_segmentation_harbin.yaml",
    "building_change": "downstreams/configs/building_change_segmentation_harbin.yaml",
    "farm_change": "downstreams/configs/farm_change_segmentation_harbin.yaml",
    "rubbish": "downstreams/configs/rubbish_segmentation_harbin.yaml",
    "construction_joint": "downstreams/configs/construction_segmentation_joint.yaml",
}
for task, cfg in config_map.items():
    src = Path("/root/workspace/xuannv/.worktrees/feat-multitask-downstream") / cfg
    if src.exists():
        shutil.copy2(src, configs_dir / f"{task}.yaml")
```

Run:
```bash
python downstreams/scripts/collect_v1.0_artifacts.py
```

Expected: `heads/` 下有 5 个 `.pt` 文件，`configs/` 下有 5 个 YAML。

- [ ] **Step 2: 复制 encoder checkpoint**

Find encoder checkpoint path from current experiments (e.g. `20260621_harbin_128_stage2_v1_best`):

Run:
```bash
ls /data/xuannv_embedding/embeddings/20260621_harbin_128_stage2_v1_best/
# Identify the model checkpoint file
```

Then copy the latest encoder checkpoint:
```bash
cp /data/xuannv_embedding/outputs/<encoder_checkpoint_dir>/best.pt /data/xuannv_embedding/experiments/v1.0/encoder/
cp /data/xuannv_embedding/outputs/<encoder_checkpoint_dir>/meta.json /data/xuannv_embedding/experiments/v1.0/encoder/ 2>/dev/null || true
```

Replace `<encoder_checkpoint_dir>` with the actual path (inspect `/data/xuannv_embedding/outputs/`).

- [ ] **Step 3: 提交收集脚本**

Run:
```bash
git add downstreams/scripts/collect_v1.0_artifacts.py
git commit -m "feat(release): add v1.0 artifact collection script"
```

---

### Task 5: 生成 V1.0 manifest

**Files:**
- Create: `/data/xuannv_embedding/experiments/v1.0/v1.0_manifest.json`
- Create: `/root/workspace/xuannv/downstreams/scripts/generate_v1.0_manifest.py`

- [ ] **Step 1: 编写 manifest 生成脚本**

```python
import json
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime, timezone

v1_root = Path("/data/xuannv_embedding/experiments/v1.0")
summary = json.load(open(v1_root / "all_tasks_summary_final.json"))

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd="/root/workspace/xuannv/.worktrees/feat-multitask-downstream"
).decode().strip()

manifest = {
    "version": "1.0",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "git_branch": "feat/multitask-downstream",
    "git_commit": commit,
    "github_release_tag": "v1.0-multitask-downstream",
    "modelscope_repo": "WeijieWu/xuannv_train_data",
    "modelscope_tag": "v1.0",
    "tasks": {}
}

for task, meta in summary.items():
    best_fold = meta.get("best_fold", 0)
    head_path = v1_root / "heads" / f"{task}_fold{best_fold}_best.pt"
    cfg_path = v1_root / "configs" / f"{task}.yaml"
    manifest["tasks"][task] = {
        "auc_roc_mean": meta["auc_roc_mean"],
        "f1_best_mean": meta["f1_best_mean"],
        "best_fold": best_fold,
        "head_file": str(head_path.relative_to(v1_root)),
        "head_sha256": sha256_file(head_path) if head_path.exists() else None,
        "config_file": str(cfg_path.relative_to(v1_root)) if cfg_path.exists() else None,
        "experiment_dir": str(v1_root / task)
    }

encoder_dir = v1_root / "encoder"
encoder_files = [str(p.relative_to(v1_root)) for p in encoder_dir.rglob("*") if p.is_file()]
manifest["encoder"] = {
    "files": encoder_files,
    "sha256": {str(p.relative_to(v1_root)): sha256_file(p) for p in encoder_dir.rglob("*") if p.is_file()}
}

out_path = v1_root / "v1.0_manifest.json"
with open(out_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Manifest written to {out_path}")
```

Run:
```bash
python downstreams/scripts/generate_v1.0_manifest.py
```

Expected: `/data/xuannv_embedding/experiments/v1.0/v1.0_manifest.json` 生成。

- [ ] **Step 2: 提交 manifest 脚本**

Run:
```bash
git add downstreams/scripts/generate_v1.0_manifest.py
git commit -m "feat(release): add v1.0 manifest generator"
```

---

### Task 6: Git tag 与 GitHub Release

**Files:**
- Git tag: `v1.0-multitask-downstream`

- [ ] **Step 1: 确保代码已 push**

Run:
```bash
cd /root/workspace/xuannv/.worktrees/feat-multitask-downstream
git push origin feat/multitask-downstream
```

Expected: push 成功。

- [ ] **Step 2: 打 tag 并 push**

Run:
```bash
git tag -a v1.0-multitask-downstream -m "V1.0 multitask downstream segmentation release"
git push origin v1.0-multitask-downstream
```

Expected: tag 出现在 GitHub 上。

- [ ] **Step 3: 创建 GitHub Release（命令行或网页）**

Using GitHub CLI if available:
```bash
gh release create v1.0-multitask-downstream \
  --title "V1.0 Multitask Downstream" \
  --notes-file /data/xuannv_embedding/experiments/v1.0/FINAL_REPORT.md \
  /data/xuannv_embedding/experiments/v1.0/all_tasks_summary_final.json \
  /data/xuannv_embedding/experiments/v1.0/visualizations/task_metrics_comparison.png
```

If `gh` not available, create release manually via GitHub web UI and upload the same files.

---

### Task 7: ModelScope 发布 V1.0

**Files:**
- Target: `WeijieWu/xuannv_train_data` model repo
- Local staging: `/tmp/xuannv_modelscope_v1.0/`

- [ ] **Step 1: 安装/确认 modelscope SDK**

Run:
```bash
pip install modelscope
```

Expected: `python -c "import modelscope; print(modelscope.__version__)"` 成功。

- [ ] **Step 2: 编写 ModelScope 上传脚本**

Create: `/root/workspace/xuannv/downstreams/scripts/upload_v1.0_to_modelscope.py`

```python
import os
import shutil
from pathlib import Path
from modelscope.hub.api import HubApi
from modelscope.hub.repository import Repository

MODELSCOPE_SDK_TOKEN = os.environ["MODELSCOPE_SDK_TOKEN"]
REPO_ID = "WeijieWu/xuannv_train_data"
LOCAL_DIR = "/tmp/xuannv_modelscope_v1.0"
SOURCE_DIR = "/data/xuannv_embedding/experiments/v1.0"

# Prepare local model dir
if Path(LOCAL_DIR).exists():
    shutil.rmtree(LOCAL_DIR)
Path(LOCAL_DIR).mkdir(parents=True)

# Copy artifacts
shutil.copytree(SOURCE_DIR, LOCAL_DIR, dirs_exist_ok=True)

# Login
api = HubApi()
api.login(MODELSCOPE_SDK_TOKEN)

# Clone or create repo
repo = Repository(LOCAL_DIR, REPO_ID)
try:
    repo.push("feat/multitask-downstream-v1.0")
except Exception as e:
    print(f"Push failed: {e}")
    raise

print("Push complete. Now create tag v1.0 via git.")
```

Note: The exact Repository API may need adjustment based on modelscope version.

- [ ] **Step 3: 运行上传脚本**

Run:
```bash
export MODELSCOPE_SDK_TOKEN=ms-399d1804-1cb3-446a-a3f7-dfc4dc70d977
python downstreams/scripts/upload_v1.0_to_modelscope.py
```

Expected: 文件上传成功，命令行输出 `Push complete`。

- [ ] **Step 4: 在本地 modelscope 仓库打 tag v1.0**

Run:
```bash
cd /tmp/xuannv_modelscope_v1.0
git tag -a v1.0 -m "V1.0 release: encoder + 5 downstream heads"
git push origin v1.0
```

Expected: ModelScope 网页上 `WeijieWu/xuannv_train_data` 出现 `v1.0` tag。

- [ ] **Step 5: 提交上传脚本**

Run:
```bash
cd /root/workspace/xuannv/.worktrees/feat-multitask-downstream
git add downstreams/scripts/upload_v1.0_to_modelscope.py
git commit -m "feat(release): add ModelScope v1.0 upload script"
git push origin feat/multitask-downstream
```

---

### Task 8: 验证 V1.0 发布

- [ ] **Step 1: 检查本地目录完整性**

Run:
```bash
ls /data/xuannv_embedding/experiments/v1.0/
# Expected: README.md, FINAL_REPORT.md, all_tasks_summary_final.json, v1.0_manifest.json, heads/, encoder/, configs/, visualizations/, 5 task symlinks
```

- [ ] **Step 2: 验证 GitHub tag 存在**

Run:
```bash
git ls-remote --tags origin | grep v1.0-multitask-downstream
```

Expected: 输出包含 tag。

- [ ] **Step 3: 验证 ModelScope tag**

Open browser or use ModelScope API:
```bash
python -c "from modelscope.hub.api import HubApi; api=HubApi(); print(api.get_model_version('WeijieWu/xuannv_train_data'))"
```

Expected: 返回包含 `v1.0`。

---

### Self-Review Checklist

- [ ] Spec coverage: Git tag, GitHub Release, ModelScope, local snapshot, visualizations all have tasks.
- [ ] Placeholder scan: no TBD/TODO/fill in details.
- [ ] Type consistency: file names match between scripts and manifest.
- [ ] Security: ModelScope token only via env var.
