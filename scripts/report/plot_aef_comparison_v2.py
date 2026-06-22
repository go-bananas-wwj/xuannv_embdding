import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load AEF benchmark summary
data = json.loads(Path("/root/workspace/report/data/aef_benchmark_summary.json").read_text(encoding="utf-8"))

# Extract per-task metrics
tasks = ["construction", "building_change", "farm_change", "rubbish"]
metrics = ["auc_roc", "f1_best", "f1_at_0_5", "miou"]
metric_labels = ["AUC-ROC", "F1-best", "F1@0.5", "mIoU"]

v1_values = {m: [] for m in metrics}
aef_values = {m: [] for m in metrics}

for task in tasks:
    for row in data["tasks"]:
        if row["task"] == task:
            if row["version"] == "V1.0":
                for m in metrics:
                    v1_values[m].append(row[m]["value"])
            elif row["version"] == "AEF":
                for m in metrics:
                    aef_values[m].append(row[m]["value"])

# Create 1x4 subplots, each with its own y-axis
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
x = np.arange(len(tasks))
width = 0.35

for ax, metric, label in zip(axes, metrics, metric_labels):
    v1 = v1_values[metric]
    aef = aef_values[metric]
    ax.bar(x - width/2, v1, width, label="V1.0", color="#3498db")
    ax.bar(x + width/2, aef, width, label="AEF", color="#e74c3c")
    ax.set_ylabel(label, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=20, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    # Annotate max value for readability
    ax.set_ylim(0, max(max(v1), max(aef)) * 1.2)

fig.suptitle("V1.0 vs AEF 2025 Benchmark (per-metric axes)", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.95])

out = Path("/root/workspace/report/assets/aef_vs_v1.0_v2.png")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
