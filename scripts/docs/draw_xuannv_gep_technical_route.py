from __future__ import annotations

"""Draw the four-stage technical route used by the Xuannv GEP application."""

from pathlib import Path

import matplotlib.font_manager as font_manager
import matplotlib.patches as patches
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs/projects/assets/xuannv_gep_transferable_20260729/technical_route_20260729.png"


def cjk_font() -> str:
    known_path = Path("/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc")
    if known_path.exists():
        return str(known_path)
    for candidate in ("Noto Sans CJK SC", "WenQuanYi Zen Hei"):
        try:
            path = font_manager.findfont(candidate, fallback_to_default=False)
        except ValueError:
            continue
        if path:
            return path
    return font_manager.findfont("DejaVu Sans")


FONT = font_manager.FontProperties(fname=cjk_font())


def label(ax, x: float, y: float, text: str, size: float = 12, weight: str = "normal") -> None:
    ax.text(x, y, text, ha="center", va="center", fontsize=size, fontproperties=FONT, fontweight=weight, color="#1F2933")


def box(ax, x: float, y: float, width: float, height: float, title: str, body: str, fill: str) -> None:
    patch = patches.FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.015,rounding_size=0.018",
        linewidth=1.2, edgecolor="#40515E", facecolor=fill,
    )
    ax.add_patch(patch)
    label(ax, x + width / 2, y + height * 0.67, title, size=13, weight="bold")
    label(ax, x + width / 2, y + height * 0.34, body, size=10)


def arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops={"arrowstyle": "-|>", "lw": 1.5, "color": "#40515E", "shrinkA": 6, "shrinkB": 6},
    )


def stage(ax, y: float, title: str, subtitle: str, items: list[tuple[str, str]], fill: str) -> None:
    ax.add_patch(patches.Rectangle((0.035, y), 0.93, 0.19, linewidth=1.2, edgecolor="#40515E", facecolor="#FFFFFF"))
    ax.add_patch(patches.Rectangle((0.035, y + 0.145), 0.93, 0.045, linewidth=1.2, edgecolor="#40515E", facecolor=fill))
    label(ax, 0.50, y + 0.167, title, size=14, weight="bold")
    label(ax, 0.50, y + 0.126, subtitle, size=10)
    count = len(items)
    width = 0.20 if count == 4 else 0.24
    gap = (0.90 - count * width) / (count - 1)
    x = 0.05
    for index, (item_title, item_body) in enumerate(items):
        box(ax, x, y + 0.025, width, 0.078, item_title, item_body, "#FAFBFC")
        if index < count - 1:
            arrow(ax, (x + width, y + 0.064), (x + width + gap, y + 0.064))
        x += width + gap


def draw() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(12, 15), dpi=220, facecolor="white")
    ax = figure.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    label(ax, 0.5, 0.975, "自研嵌入场范式模型支撑GEP智能核算技术路线", size=18, weight="bold")
    label(ax, 0.5, 0.948, "以多源地球观测、生态过程模型和独立监测为共同约束", size=11)

    stage(
        ax, 0.735, "第一阶段：自主数据底座与样本库构建", "形成可追溯的月度多源数据和服务样本库",
        [
            ("自主多源数据", "GF / ZY / HJ / FY\n及可用多源遥感"),
            ("时空谱协同", "辐射归一、配准\n质量控制、月度合成"),
            ("地面观测与过程模型", "通量塔、水文站、样地\nInVEST / CASA 等"),
            ("GEP样本与参数台账", "生态状态、实物量\n价值参数、版本记录"),
        ], "#E8F1F5",
    )
    stage(
        ax, 0.505, "第二阶段：自研嵌入场范式模型预训练", "学习可复用、时空连续的多模态地表状态表征",
        [
            ("多模态输入", "光学、SAR、高分辨率\n气象、地形等辅助变量"),
            ("自监督预训练", "时相、空间、模态\n掩码重建与一致性学习"),
            ("物理与知识先验", "地形、生态规律\n质量信息软约束"),
            ("自研嵌入场范式模型", "约1亿参数\n64维月度密集嵌入"),
        ], "#EAF3E6",
    )
    stage(
        ax, 0.275, "第三阶段：下游专题、服务核算与独立验证", "嵌入先生成专题和过程变量，再进入服务专用模型与价值核算",
        [
            ("生态状态专题", "冻结嵌入 + 轻量头\n土地覆盖、绿地、水体"),
            ("服务专题与过程变量", "NPP等基础变量\n水源涵养、土壤保持等"),
            ("服务专用过程模型", "三类服务价值试点\n完整GEP满足条件后汇总"),
            ("独立验证与审计", "样地、站点资料\n不确定性、版本复算"),
        ], "#FFF3DE",
    )
    stage(
        ax, 0.045, "第四阶段：知识图谱约束的可信决策智能体", "实现“问—算—答—视”的受约束分析闭环",
        [
            ("GEP知识图谱", "生态逻辑、政策法规\n数据血缘与参数关系"),
            ("可信智能体", "自然语言任务解析\n工具调用与边界约束"),
            ("核算与时空分析", "指标计算、变化分析\n驱动力辅助研判"),
            ("交互平台与报告", "问—算—答—视\n专题图、图表、报告"),
        ], "#F0EAF6",
    )
    figure.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    draw()
