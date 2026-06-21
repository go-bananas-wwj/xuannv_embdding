import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

fig, ax = plt.subplots(figsize=(16, 5))
ax.set_xlim(0, 16)
ax.set_ylim(0, 5)
ax.axis("off")

boxes = [
    (0.3, 1.8, 2.2, 1.2, "Multi-source Input\nS2 / S1 / Landsat"),
    (3.0, 1.8, 2.2, 1.2, "SensorEncoderBank\nper-source stems"),
    (5.7, 1.8, 2.2, 1.2, "STP Encoder\nspatio-temporal"),
    (8.4, 1.8, 2.2, 1.2, "MonthlyEmbedding\nmonthly binning"),
    (11.1, 1.8, 2.2, 1.2, "vMF Bottleneck\n+ Decoder Heads"),
    (13.8, 1.8, 1.8, 1.2, "Monthly Embedding\n128×128×D"),
]

for x, y, w, h, text in boxes:
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.08",
        ec="#2c3e50", fc="#ecf0f1", linewidth=2
    )
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=10, weight="bold")

for i in range(len(boxes) - 1):
    ax.annotate(
        "", xy=(boxes[i+1][0], boxes[i+1][1] + boxes[i+1][3]/2),
        xytext=(boxes[i][0] + boxes[i][2], boxes[i][1] + boxes[i][3]/2),
        arrowprops=dict(arrowstyle="->", lw=2.5, color="#3498db")
    )

# High-res branch annotation
ax.annotate(
    "High-res Optical / SAR\nindependent branch",
    xy=(8.4 + 2.2/2, 1.8),
    xytext=(8.4 + 2.2/2, 0.6),
    ha="center", fontsize=9, color="#e74c3c",
    arrowprops=dict(arrowstyle="->", lw=1.5, color="#e74c3c")
)

out = Path("/root/workspace/report/assets/architecture_diagram.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
