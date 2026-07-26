#!/usr/bin/env python3
"""Build editable native PowerPoint slides 8 through 10 for the Pujiang deck."""

from __future__ import annotations

import sys
from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

REPORT_DIR = Path(__file__).resolve().parent
if str(REPORT_DIR) not in sys.path:
    sys.path.insert(0, str(REPORT_DIR))

from pujiang_editable_common import (  # noqa: E402
    BLUE,
    DEEP_BLUE,
    GREEN,
    INK,
    LINE,
    MUTED,
    PALE_BLUE,
    PALE_GRAY,
    PALE_GREEN,
    RED,
    WHITE,
    add_header,
    add_picture,
    add_rect,
    add_text,
    new_blank_slide,
)

ROOT = Path(__file__).resolve().parents[2]
SLIDE01_ROOT = ROOT / "docs/presentations/assets/pujiang_202607/slide01"
PCA_COMPARE = (
    ROOT
    / "docs/production/assets/nonbuilding_fewshot_semantic_20260705"
    / "xuannv_aef_global_pca_compare.png"
)
HAR_BIN_ROOT = ROOT / "docs/bp_deck/assets/user_materials"

PAGE_TITLES = {
    8: "海淀区 3 多边形同协议评测：道路与水体领先",
    9: "嵌入结构与任务表现：优势、边界和下一步",
    10: "哈尔滨新区：从月度嵌入到城市治理专题",
}

METHODS = (
    ("玄女", RED),
    ("AEF", BLUE),
    ("传统", MUTED),
)
METRICS = {
    "F1": (
        ("建筑物", (0.256, 0.311, 0.243)),
        ("道路", (0.368, 0.340, 0.345)),
        ("水体", (0.111, 0.092, 0.088)),
    ),
    "AUC": (
        ("建筑物", (0.780, 0.797, 0.733)),
        ("道路", (0.745, 0.703, 0.694)),
        ("水体", (0.692, 0.665, 0.595)),
    ),
}
AP_ROWS = (
    ("建筑物", "0.221", "0.234", "0.183"),
    ("道路", "0.304", "0.257", "0.259"),
    ("水体", "0.075", "0.057", "0.049"),
)


def _panel(slide, x: float, y: float, w: float, h: float, fill=WHITE) -> None:
    add_rect(slide, x, y, w, h, fill, LINE, MSO_SHAPE.ROUNDED_RECTANGLE)


def _add_legend(slide, x: float, y: float) -> None:
    for index, (name, color) in enumerate(METHODS):
        offset = index * 1.08
        add_rect(slide, x + offset, y, 0.16, 0.16, color, color)
        add_text(slide, name, x + offset + 0.22, y - 0.01, 0.7, 0.18, 10, INK, True)


def _add_metric_chart(slide, metric: str, x: float, y: float, w: float, h: float) -> None:
    """Draw a native horizontal-bar chart for one metric."""
    _panel(slide, x, y, w, h)
    add_text(slide, metric, x + 0.22, y + 0.16, 0.54, 0.26, 17, DEEP_BLUE, True)
    add_text(slide, "越高越好", x + 0.77, y + 0.2, 0.7, 0.18, 10, MUTED)
    _add_legend(slide, x + 2.22, y + 0.22)

    bar_left = x + 1.12
    bar_width = w - 1.42
    for fraction, label in ((0.0, "0.0"), (0.5, "0.5"), (1.0, "1.0")):
        line_x = bar_left + bar_width * fraction
        add_rect(slide, line_x, y + 0.68, 0.01, h - 0.94, LINE, LINE)
        add_text(
            slide,
            label,
            line_x - 0.12,
            y + 0.48,
            0.28,
            0.16,
            9,
            MUTED,
            False,
            PP_ALIGN.CENTER,
        )

    for row_index, (task, values) in enumerate(METRICS[metric]):
        row_y = y + 0.86 + row_index * 0.78
        add_text(slide, task, x + 0.2, row_y + 0.13, 0.72, 0.18, 12, INK, True)
        for method_index, value in enumerate(values):
            bar_y = row_y + method_index * 0.18
            _, color = METHODS[method_index]
            add_rect(slide, bar_left, bar_y, max(0.03, bar_width * value), 0.12, color, color)
            add_text(
                slide,
                f"{value:.3f}",
                bar_left + bar_width * value + 0.05,
                bar_y - 0.01,
                0.45,
                0.14,
                9,
                color,
                True,
            )


def _add_ap_table(slide, x: float, y: float, w: float, h: float) -> None:
    _panel(slide, x, y, w, h)
    add_text(slide, "AP（稀少目标高置信检出）", x + 0.18, y + 0.14, 2.5, 0.22, 15, DEEP_BLUE, True)
    headers = ("任务", "玄女", "AEF", "传统")
    cell_widths = (1.25, 1.15, 1.15, 1.15)
    cell_x = x + 0.18
    header_y = y + 0.48
    for column, (value, cell_width) in enumerate(zip(headers, cell_widths, strict=True)):
        color = RED if value == "玄女" else INK
        add_rect(slide, cell_x, header_y, cell_width, 0.2, PALE_BLUE, LINE)
        add_text(
            slide,
            value,
            cell_x,
            header_y + 0.02,
            cell_width,
            0.14,
            10,
            color,
            True,
            PP_ALIGN.CENTER,
        )
        cell_x += cell_width
    for row_index, row in enumerate(AP_ROWS):
        cell_x = x + 0.18
        row_y = header_y + 0.2 + row_index * 0.2
        for column, (value, cell_width) in enumerate(zip(row, cell_widths, strict=True)):
            color = RED if column == 1 else INK
            add_rect(slide, cell_x, row_y, cell_width, 0.2, WHITE, LINE)
            add_text(
                slide,
                value,
                cell_x,
                row_y + 0.02,
                cell_width,
                0.14,
                10,
                color,
                column == 1,
                PP_ALIGN.CENTER,
            )
            cell_x += cell_width


def _add_evidence_card(
    slide,
    title: str,
    detail: str,
    x: float,
    y: float,
    fill,
    accent,
) -> None:
    _panel(slide, x, y, 3.46, 1.78, fill)
    add_rect(slide, x, y, 0.1, 1.78, accent, accent)
    add_text(slide, title, x + 0.25, y + 0.18, 2.95, 0.25, 14, accent, True)
    add_text(slide, detail, x + 0.25, y + 0.58, 2.92, 0.7, 11, INK)


def build_page08(prs: Presentation):
    """Build native F1/AUC bars, AP table, and fair-comparison boundaries."""
    slide = new_blank_slide(prs)
    add_header(
        slide,
        PAGE_TITLES[8],
        "每类同一组 3 个多边形、同一批 64 个测试 patch、同一 PU+Query 与阈值规则",
    )
    _panel(slide, 0.5, 1.03, 12.28, 0.48, PALE_BLUE)
    add_text(
        slide,
        "三种特征共享标注、多边形顺序、64 个测试 patch、PU 背景挖掘、Query 自适应和阈值选择。",
        0.72,
        1.17,
        11.85,
        0.17,
        11,
        INK,
        True,
    )
    _add_metric_chart(slide, "F1", 0.5, 1.76, 6.0, 3.3)
    _add_metric_chart(slide, "AUC", 6.78, 1.76, 6.0, 3.3)
    _add_ap_table(slide, 0.5, 5.32, 5.3, 1.23)

    _panel(slide, 6.05, 5.32, 3.42, 1.23, PALE_GREEN)
    add_text(slide, "主要结论", 6.28, 5.48, 1.25, 0.2, 14, DEEP_BLUE, True)
    add_text(
        slide,
        "道路：F1 0.368，较 AEF 提升 8.2%\n"
        "水体：F1 0.111，较 AEF 提升 20.7%\n"
        "建筑：F1 0.256，低于 AEF 17.6%",
        6.28,
        5.78,
        2.95,
        0.52,
        10,
        INK,
        True,
    )

    _panel(slide, 9.72, 5.32, 3.06, 1.23, PALE_GRAY)
    add_text(slide, "时间产品边界", 9.92, 5.48, 1.4, 0.2, 13, DEEP_BLUE, True)
    add_text(
        slide,
        "玄女：2026-04 月度嵌入\nAEF：2025 年度嵌入\n传统：2026-04 42 维多源特征",
        9.92,
        5.76,
        2.62,
        0.54,
        10,
        INK,
    )
    _panel(slide, 0.5, 6.78, 12.28, 0.4, PALE_GRAY)
    add_text(
        slide,
        "结果边界：单 fold、单次随机 3 多边形实验；用于平台交互能力展示，"
        "统计结论仍需多 fold、多随机种子复核。",
        0.7,
        6.9,
        11.9,
        0.16,
        10,
        MUTED,
        True,
    )
    return slide


def build_page09(prs: Presentation):
    """Build the PCA comparison and four editable evidence cards."""
    slide = new_blank_slide(prs)
    add_header(
        slide,
        PAGE_TITLES[9],
        "PCA 展示空间结构，第 8 页严格指标说明哪些任务已受益、哪些问题仍需解决",
    )
    _panel(slide, 0.5, 1.35, 4.32, 5.25)
    add_text(
        slide,
        "海淀全域嵌入 PCA｜玄女月度嵌入与 AEF 年度嵌入",
        0.74,
        1.55,
        3.82,
        0.35,
        14,
        DEEP_BLUE,
        True,
    )
    add_picture(slide, PCA_COMPARE, 0.72, 1.95, 3.88, 4.38, "contain")
    _add_evidence_card(
        slide,
        "道路｜当前优势最稳定",
        "玄女 F1 0.368、AUC 0.745、AP 0.304\n三项均高于 AEF 与传统特征。",
        5.12,
        1.35,
        PALE_GREEN,
        GREEN,
    )
    _add_evidence_card(
        slide,
        "水体｜排序能力领先",
        "AUC 0.692、AP 0.075 均领先；\n但 F1 仅 0.111，适合候选发现而非直接交付。",
        8.9,
        1.35,
        PALE_BLUE,
        BLUE,
    )
    _add_evidence_card(
        slide,
        "建筑｜建筑假正例仍偏多",
        "玄女 F1 0.256，低于 AEF 0.311；\n需提升语义分离与跨 patch 阈值稳定性。",
        5.12,
        3.46,
        PALE_GRAY,
        RED,
    )
    _add_evidence_card(
        slide,
        "下一步验证",
        "5-fold × 多随机种子复核\n独立人工标签验证｜建筑阈值校准",
        8.9,
        3.46,
        WHITE,
        DEEP_BLUE,
    )
    _panel(slide, 0.5, 6.82, 12.28, 0.34, PALE_GRAY)
    add_text(
        slide,
        "本页不引入新的评测协议；全部任务结论均引用第 8 页同一 3 多边形 PU+Query 实验。",
        0.7,
        6.92,
        11.85,
        0.14,
        10,
        MUTED,
        True,
    )
    return slide


def _add_case_card(slide, title: str, detail: str, image_path: Path, x: float, y: float) -> None:
    _panel(slide, x, y, 4.02, 2.26)
    add_text(slide, title, x + 0.18, y + 0.15, 3.62, 0.23, 14, DEEP_BLUE, True)
    add_picture(slide, image_path, x + 0.18, y + 0.53, 1.78, 1.45, "contain")
    add_text(slide, detail, x + 2.16, y + 0.66, 1.6, 0.85, 11, MUTED)


def build_page10(prs: Presentation):
    """Build the Harbin cross-region application page with five independent images."""
    slide = new_blank_slide(prs)
    add_header(
        slide,
        PAGE_TITLES[10],
        "同一套月度嵌入可支撑城市建设、地表变化与资源环境专题分析",
    )
    _panel(slide, 0.5, 1.35, 3.22, 5.3)
    add_text(
        slide,
        "哈尔滨新区月度嵌入 PCA｜API V5/v2",
        0.72,
        1.56,
        2.72,
        0.36,
        14,
        DEEP_BLUE,
        True,
    )
    add_picture(slide, SLIDE01_ROOT / "harbin_pca.png", 0.76, 2.0, 2.7, 2.7, "contain")
    add_text(
        slide,
        "跨区域部署关注：\n嵌入结构、专题制图与\n地图复核可在同一工作流中衔接。",
        0.78,
        5.1,
        2.55,
        0.86,
        12,
        INK,
    )
    cases = (
        (
            "建筑物提取｜V5/v2 API",
            "城市建设对象识别\n结果回到地图复核",
            SLIDE01_ROOT / "harbin_building.png",
        ),
        (
            "水体提取｜V5/v2 API",
            "水系候选与范围查看\n服务资源环境专题",
            SLIDE01_ROOT / "harbin_water.png",
        ),
        (
            "土地利用 / 覆盖｜V5/v2 API",
            "地表类型专题制图\n支持空间统计",
            SLIDE01_ROOT / "harbin_landcover.png",
        ),
        (
            "土地利用转换｜V5/v2 API",
            "多时相变化案例\n用于城市治理线索发现",
            HAR_BIN_ROOT / "patch_000217_land_conversion_2025-08_vs_2025-09_detail.png",
        ),
    )
    for index, (title, detail, image_path) in enumerate(cases):
        x = 4.0 if index % 2 == 0 else 8.38
        y = 1.35 if index < 2 else 3.94
        _add_case_card(slide, title, detail, image_path, x, y)
    _panel(slide, 4.0, 6.67, 8.4, 0.45, PALE_GRAY)
    add_text(
        slide,
        "海淀验证方法效果，哈尔滨验证跨区域部署与应用能力｜结果回到地图复核。",
        4.22,
        6.81,
        7.9,
        0.16,
        11,
        INK,
        True,
    )
    return slide
