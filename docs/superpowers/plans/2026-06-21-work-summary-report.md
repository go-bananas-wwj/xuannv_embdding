> I'm using the writing-plans skill to create the implementation plan.

# xuannv_embedding 工作总结报告实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 整理 xuannv_embedding 项目阶段成果，生成 `/root/workspace/report/index.html` 汇报报告，并同步整理原始材料到 `assets/`、`data/`、`sources/`。

**Architecture:** 直接基于现有实验结果（`/data/xuannv_embedding/experiments/v1.0/` 与 `aef_benchmark/`）生成静态 HTML 报告，图片以相对路径引用 `assets/` 中的文件，指标数据以 JSON 放入 `data/`。

**Tech Stack:** Python 3.11, standard library (`pathlib`, `json`, `shutil`), HTML/CSS, matplotlib（如需补充图表）

---

## 文件结构

```
/root/workspace/report/
├── index.html              # 完整 HTML 报告
├── assets/                 # 图片素材
│   ├── aef_vs_v1.0.png
│   ├── architecture_diagram.png
│   ├── task_metrics_comparison.png
│   ├── loss_curve.png          # 可选
│   ├── construction_overlay_*.png
│   ├── building_change_overlay_*.png
│   ├── farm_change_overlay_*.png
│   ├── rubbish_overlay_*.png
│   └── construction_joint_overlay_*.png
├── data/
│   ├── v1.0_summary.json
│   ├── aef_benchmark_summary.json
│   └── README.md
└── sources/
    └── sources.md
```

---

### Task 1: 创建报告目录结构

**Files:**
- Create: `/root/workspace/report/`
- Create: `/root/workspace/report/assets/`
- Create: `/root/workspace/report/data/`
- Create: `/root/workspace/report/sources/`

- [ ] **Step 1: 创建目录**

Run:
```bash
mkdir -p /root/workspace/report/{assets,data,sources}
ls -la /root/workspace/report/
```

Expected: 四个目录均存在。

- [ ] **Step 2: 创建 sources 说明文件**

Create: `/root/workspace/report/sources/sources.md`

```markdown
# 数据来源说明

- 项目代码与文档：`/root/workspace/xuannv/`
- V1.0 下游实验结果：`/data/xuannv_embedding/experiments/v1.0/`
- AEF benchmark 结果：`/data/xuannv_embedding/experiments/aef_benchmark/`
- V1.0 可视化：`/data/xuannv_embedding/experiments/v1.0/visualizations/`
- 设计文档：`docs/superpowers/specs/2026-06-21-work-summary-report-design.md`
```

- [ ] **Step 3: Commit**

```bash
cd /root/workspace/xuannv
git add docs/superpowers/plans/2026-06-21-work-summary-report.md
git commit -m "docs(plan): add work summary report implementation plan"
git push origin main
```

---

### Task 2: 汇总指标数据

**Files:**
- Read: `/data/xuannv_embedding/experiments/v1.0/all_tasks_summary_final.json`
- Read: `/data/xuannv_embedding/experiments/aef_benchmark/AEF_BENCHMARK_REPORT.md`
- Create: `/root/workspace/report/data/v1.0_summary.json`
- Create: `/root/workspace/report/data/aef_benchmark_summary.json`
- Create: `/root/workspace/report/data/README.md`

- [ ] **Step 1: 复制 V1.0 汇总 JSON**

Run:
```bash
cp /data/xuannv_embedding/experiments/v1.0/all_tasks_summary_final.json /root/workspace/report/data/v1.0_summary.json
```

- [ ] **Step 2: 解析并保存 AEF benchmark 指标**

Create and run: `/root/workspace/xuannv/scripts/report/extract_aef_metrics.py`

```python
import json
import re
from pathlib import Path

src = Path("/data/xuannv_embedding/experiments/aef_benchmark/AEF_BENCHMARK_REPORT.md")
out = Path("/root/workspace/report/data/aef_benchmark_summary.json")
out.parent.mkdir(parents=True, exist_ok=True)

# Parse the markdown table into structured JSON
lines = [l.strip() for l in src.read_text().splitlines() if l.strip()]
rows = []
for line in lines:
    if line.startswith("|") and "Task" not in line and "---" not in line:
        parts = [p.strip() for p in line.split("|")][1:-1]
        if len(parts) >= 7:
            rows.append({
                "task": parts[0],
                "version": parts[1],
                "auc_roc": float(parts[2]),
                "f1_best": float(parts[3]),
                "f1_at_0_5": float(parts[4]),
                "miou": float(parts[5]),
            })

out.write_text(json.dumps({"tasks": rows}, indent=2, ensure_ascii=False))
print(f"Wrote {out}")
```

Run:
```bash
python /root/workspace/xuannv/scripts/report/extract_aef_metrics.py
```

Expected: `/root/workspace/report/data/aef_benchmark_summary.json` 生成，包含 V1.0 与 AEF 各任务指标。

- [ ] **Step 3: 创建 data 目录 README**

Create: `/root/workspace/report/data/README.md`

```markdown
# 报告数据文件

- `v1.0_summary.json`：自研 V1.0 embedding 在 5 个下游任务上的 5-fold 汇总指标。
- `aef_benchmark_summary.json`：AEF 官方 2025 年度 embedding 与 V1.0 的对比指标。
```

- [ ] **Step 4: Commit**

```bash
cd /root/workspace/xuannv
git add scripts/report/extract_aef_metrics.py
git commit -m "feat(report): extract AEF benchmark metrics for report"
git push origin main
```

---

### Task 3: 收集可视化素材

**Files:**
- Copy from: `/data/xuannv_embedding/experiments/aef_benchmark/aef_vs_v1.0.png`
- Copy from: `/data/xuannv_embedding/experiments/v1.0/visualizations/task_metrics_comparison.png`
- Copy from: `/data/xuannv_embedding/experiments/v1.0/visualizations/*/fold_*/patch_*_overlay.png`
- Copy from: `/data/xuannv_embedding/experiments/v1.0/visualizations/*/fold_*/pr_curve.png`

- [ ] **Step 1: 复制核心对比图**

Run:
```bash
cp /data/xuannv_embedding/experiments/aef_benchmark/aef_vs_v1.0.png /root/workspace/report/assets/
cp /data/xuannv_embedding/experiments/v1.0/visualizations/task_metrics_comparison.png /root/workspace/report/assets/
ls -la /root/workspace/report/assets/
```

- [ ] **Step 2: 为每个任务挑选代表性 overlay 图**

Create and run: `/root/workspace/xuannv/scripts/report/select_visualizations.py`

```python
from pathlib import Path
import shutil

src_root = Path("/data/xuannv_embedding/experiments/v1.0/visualizations")
dst = Path("/root/workspace/report/assets")
dst.mkdir(parents=True, exist_ok=True)

tasks = ["construction", "building_change", "farm_change", "rubbish", "construction_joint"]
selected = {}

for task in tasks:
    task_dir = src_root / task
    if not task_dir.exists():
        continue
    # Pick first fold and first overlay patch
    overlays = sorted(task_dir.glob("fold_*/patch_*_overlay.png"))
    pr_curves = sorted(task_dir.glob("fold_*/pr_curve.png"))
    if overlays:
        first = overlays[0]
        out = dst / f"{task}_overlay_example.png"
        shutil.copy(first, out)
        selected[task] = str(out)
    if pr_curves:
        first = pr_curves[0]
        out = dst / f"{task}_pr_curve_example.png"
        shutil.copy(first, out)

for task, path in selected.items():
    print(f"{task}: {path}")
```

Run:
```bash
python /root/workspace/xuannv/scripts/report/select_visualizations.py
```

Expected: `assets/` 下出现 5 个任务的 overlay 和 pr_curve 示例图。

- [ ] **Step 3: 生成架构示意图（如不存在）**

如果项目中没有现成的架构图，使用 Python 绘制一张简洁的流程图。

Create and run: `/root/workspace/xuannv/scripts/report/draw_architecture.py`

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

fig, ax = plt.subplots(figsize=(14, 4))
ax.set_xlim(0, 14)
ax.set_ylim(0, 4)
ax.axis("off")

boxes = [
    (0.5, 1.5, 2.0, 1.0, "多源时序输入\nS2/S1/Landsat"),
    (3.0, 1.5, 2.0, 1.0, "SensorEncoderBank"),
    (5.5, 1.5, 2.0, 1.0, "STP Encoder\n多分辨率时空编码"),
    (8.0, 1.5, 2.0, 1.0, "MonthlyEmbedding\n月度 binning"),
    (10.5, 1.5, 2.0, 1.0, "vMF Bottleneck\n+ Decoder Heads"),
]

for x, y, w, h, text in boxes:
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", ec="black", fc="lightblue")
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=9)

for i in range(len(boxes) - 1):
    ax.annotate("", xy=(boxes[i+1][0], boxes[i+1][1] + boxes[i+1][3]/2),
                xytext=(boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]/2),
                arrowprops=dict(arrowstyle="->", lw=2))

out = Path("/root/workspace/report/assets/architecture_diagram.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
```

Run:
```bash
python /root/workspace/xuannv/scripts/report/draw_architecture.py
```

Expected: `/root/workspace/report/assets/architecture_diagram.png` 生成。

- [ ] **Step 4: Commit**

```bash
cd /root/workspace/xuannv
git add scripts/report/
git commit -m "feat(report): select visualizations and draw architecture diagram"
git push origin main
```

---

### Task 4: 生成 HTML 报告

**Files:**
- Create: `/root/workspace/report/index.html`

- [ ] **Step 1: 编写 HTML 报告**

Create: `/root/workspace/report/index.html`

内容需覆盖设计文档中的 11 个章节，并引用 `assets/` 中的图片和 `data/` 中的 JSON。HTML 应包含内联 CSS，页面清晰、响应式。

关键内容：
- 标题与项目目标
- 里程碑卡片
- 数据准备说明
- 架构图
- 多源数据运用说明
- 训练进展简述
- AEF 对比表格（使用具体数字）
- 模型升级方向（强调数据规模扩大）
- 5 个任务典型案例可视化
- 附录与数据来源

- [ ] **Step 2: 验证 HTML 文件存在且图片路径正确**

Run:
```bash
ls -la /root/workspace/report/index.html
python -c "
from pathlib import Path
import re
html = Path('/root/workspace/report/index.html').read_text()
imgs = re.findall(r'src=\"(assets/[^\"]+)\"', html)
missing = [i for i in imgs if not (Path('/root/workspace/report') / i).exists()]
print('Referenced images:', imgs)
print('Missing images:', missing)
assert not missing, 'Missing images!'
"
```

Expected: 无 missing images。

- [ ] **Step 3: Commit**

```bash
cd /root/workspace/xuannv
# index.html is outside the git repo; only commit the generator if used, or note in chat
# For this deliverable, we commit any generator script but the report itself lives in /root/workspace/report
git status
```

Note: `/root/workspace/report/` is outside the xuannv git repo, so `index.html` is not committed to the project repo. If a generator script is created in the repo, commit that instead.

---

### Task 5: 验证报告完整性

**Files:**
- Read: `/root/workspace/report/index.html`
- Read: `/root/workspace/report/data/aef_benchmark_summary.json`

- [ ] **Step 1: 检查 HTML 关键章节是否齐全**

Run:
```bash
python -c "
from pathlib import Path
html = Path('/root/workspace/report/index.html').read_text()
required = ['项目目标', '里程碑', '数据准备', '模型架构', '多源数据', '训练进展', 'AEF 对比', '模型升级', '典型案例', '附录']
missing = [r for r in required if r not in html]
print('Missing sections:', missing)
assert not missing
"
```

Expected: 无 missing sections。

- [ ] **Step 2: 用浏览器/无头方式检查 HTML 可渲染**

Run:
```bash
python -c "
from pathlib import Path
html = Path('/root/workspace/report/index.html').read_text()
assert '<html' in html and '<body' in html
print('HTML structure OK')
"
```

Expected: HTML structure OK。

- [ ] **Step 3: 确认不提及“蒸馏”**

Run:
```bash
python -c "
from pathlib import Path
html = Path('/root/workspace/report/index.html').read_text()
assert '蒸馏' not in html and 'distill' not in html.lower()
print('No distillation mention')
"
```

Expected: No distillation mention。

---

## 自我审查

1. **Spec coverage:** 设计文档中的 11 个章节、AEF 对比数字、模型升级方向、典型案例可视化均已对应到任务。
2. **Placeholder scan:** 无 TBD/TODO，所有步骤包含具体命令或代码。
3. **Type consistency:** JSON 字段与 HTML 表格一致；图片路径使用相对路径 `assets/...`。

---

## 执行方式

Plan complete and saved to `docs/superpowers/plans/2026-06-21-work-summary-report.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
