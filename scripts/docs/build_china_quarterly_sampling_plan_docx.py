#!/usr/bin/env python3
"""Build the China 2020-2021 quarterly sampling plan figures and DOCX."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/plans/china_quarterly_2020_2021_sampling_plan_20260726_zh.md"
ASSET_DIR = ROOT / "docs/plans/assets/china_quarterly_sampling_20260726"
FIGURE = ASSET_DIR / "sampling_composition_and_workflow.png"
OUTPUT = (
    ROOT
    / "docs/plans/word/中国版AlphaEarth全国采样方案_2020_2021季度版_20260726.docx"
)
SONG_FONT = "/usr/share/fonts/google-noto-cjk/NotoSerifCJK-Regular.ttc"


def _set_run_font(run, size: float = 12, bold: bool | None = None) -> None:
    run.font.name = "Times New Roman"
    fonts = run._element.get_or_add_rPr().get_or_add_rFonts()
    fonts.set(qn("w:ascii"), "Times New Roman")
    fonts.set(qn("w:hAnsi"), "Times New Roman")
    fonts.set(qn("w:cs"), "Times New Roman")
    fonts.set(qn("w:eastAsia"), "宋体")
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold


def _set_cell_shading(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = properties.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        properties.append(shading)
    shading.set(qn("w:fill"), fill)


def _set_cell_margins(
    cell,
    top: int = 80,
    start: int = 80,
    bottom: int = 80,
    end: int = 80,
) -> None:
    properties = cell._tc.get_or_add_tcPr()
    margins = properties.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        properties.append(margins)
    for name, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = margins.find(qn(f"w:{name}"))
        if node is None:
            node = OxmlElement(f"w:{name}")
            margins.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def _add_page_number(section) -> None:
    paragraph = section.footer.paragraphs[0]
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("第 ")
    _set_run_font(run, 9)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    paragraph._p.append(field)
    run = paragraph.add_run(" 页")
    _set_run_font(run, 9)


def _configure_document(document: Document) -> None:
    section = document.sections[0]
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.0)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)
    _add_page_number(section)

    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(6)

    for style_name, size, before, after in (
        ("Title", 16, 0, 14),
        ("Heading 1", 14, 14, 7),
        ("Heading 2", 14, 10, 5),
        ("Heading 3", 12, 8, 4),
    ):
        style = document.styles[style_name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
        style.font.size = Pt(size)
        style.font.bold = True
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)

    document.core_properties.title = "中国版 Alpha Earth 全国采样方案"
    document.core_properties.subject = "2020 年和 2021 年季度 10 米全国地理嵌入"
    document.core_properties.author = "玄女 Embedding 项目组"


def _add_inline(paragraph, text: str, size: float = 12) -> None:
    pattern = re.compile(r"(\*\*.+?\*\*|`.+?`)")
    cursor = 0
    for match in pattern.finditer(text):
        if match.start() > cursor:
            _set_run_font(paragraph.add_run(text[cursor : match.start()]), size)
        token = match.group(0)
        if token.startswith("**"):
            _set_run_font(paragraph.add_run(token[2:-2]), size, True)
        else:
            _set_run_font(paragraph.add_run(token[1:-1]), size)
        cursor = match.end()
    if cursor < len(text):
        _set_run_font(paragraph.add_run(text[cursor:]), size)


def _parse_table(lines: list[str]) -> list[list[str]]:
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    if len(rows) > 1 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in rows[1]):
        rows.pop(1)
    return rows


def _add_table(document: Document, rows: list[list[str]]) -> None:
    columns = max(len(row) for row in rows)
    table = document.add_table(rows=len(rows), cols=columns)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for row_index, row in enumerate(rows):
        for column_index in range(columns):
            cell = table.cell(row_index, column_index)
            cell.text = ""
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            _set_cell_margins(cell)
            if row_index == 0:
                _set_cell_shading(cell, "E7E6E3")
            paragraph = cell.paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _add_inline(
                paragraph,
                row[column_index] if column_index < len(row) else "",
                10.5,
            )
            if row_index == 0:
                for run in paragraph.runs:
                    run.bold = True


def _render_figure() -> None:
    font_manager.fontManager.addfont(SONG_FONT)
    chinese_font = font_manager.FontProperties(fname=SONG_FONT)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    labels = ["基础1%空间样本", "UTM空间均衡", "海岸线", "语义与困难区域"]
    values = [57405, 2595, 500, 1500]
    colors = ["#333333", "#777777", "#B44A3D", "#D39A52"]

    figure = plt.figure(figsize=(13.5, 7.4), dpi=220, facecolor="white")
    grid = figure.add_gridspec(2, 1, height_ratios=[1.0, 1.4], hspace=0.42)
    axis = figure.add_subplot(grid[0])
    left = 0
    annotation_levels = {
        "UTM空间均衡": 0.48,
        "海岸线": 0.78,
        "语义与困难区域": 0.30,
    }
    for label, value, color in zip(labels, values, colors, strict=True):
        axis.barh([0], [value], left=[left], color=color, height=0.5, edgecolor="white")
        if label == "基础1%空间样本":
            axis.text(
                left + value / 2,
                0,
                f"{label}\n{value:,}",
                ha="center",
                va="center",
                color="white",
                fontsize=9,
                fontproperties=chinese_font,
            )
        else:
            annotation_y = annotation_levels[label]
            axis.annotate(
                f"{label} {value:,}",
                xy=(left + value / 2, 0.25),
                xytext=(left + value / 2, annotation_y),
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontproperties=chinese_font,
                arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.8},
            )
        left += value
    axis.set_xlim(0, 62000)
    axis.set_ylim(-0.55, 1.08)
    axis.set_yticks([])
    axis.set_xticks([0, 10000, 20000, 30000, 40000, 50000, 62000])
    axis.set_xticklabels(
        ["0", "1万", "2万", "3万", "4万", "5万", "6.2万"],
        fontproperties=chinese_font,
    )
    axis.set_title("最终 62,000 个训练 patch 的组成", fontproperties=chinese_font, fontsize=15)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color("#777777")

    flow = figure.add_subplot(grid[1])
    flow.axis("off")
    steps = [
        ("全国UTM规则网格", "约586万个候选"),
        ("基础1%抽样", "57,405"),
        ("空间均衡补样", "+2,595"),
        ("真实海岸补样", "+500"),
        ("语义与困难补样", "+1,500"),
        ("季度质量审计", "2020Q1—2021Q4"),
        ("冻结注册表", "62,000"),
    ]
    x_positions = [0.075, 0.22, 0.365, 0.51, 0.655, 0.80, 0.94]
    for index, ((title, subtitle), x) in enumerate(zip(steps, x_positions, strict=True)):
        flow.text(
            x,
            0.52,
            f"{title}\n{subtitle}",
            ha="center",
            va="center",
            fontsize=8.2 if index < 6 else 8.6,
            fontproperties=chinese_font,
            bbox={
                "boxstyle": "round,pad=0.55,rounding_size=0.08",
                "facecolor": "#F5F4F1" if index not in (4, 6) else "#F2E5D4",
                "edgecolor": "#555555",
                "linewidth": 0.9,
            },
        )
        if index < len(steps) - 1:
            flow.annotate(
                "",
                xy=(x_positions[index + 1] - 0.065, 0.52),
                xytext=(x + 0.065, 0.52),
                arrowprops={"arrowstyle": "->", "lw": 1.1, "color": "#555555"},
            )
    flow.text(
        0.5,
        0.1,
        "固定空间骨架；补样只增加、不替换；季度影像质量不合格进入补片队列",
        ha="center",
        va="center",
        fontsize=10,
        fontproperties=chinese_font,
        color="#333333",
    )
    figure.savefig(FIGURE, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _build_docx() -> None:
    document = Document()
    _configure_document(document)
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    index = 0
    in_code = False
    code_lines: list[str] = []
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped.startswith("```"):
            if in_code:
                paragraph = document.add_paragraph()
                paragraph.paragraph_format.left_indent = Cm(0.8)
                paragraph.paragraph_format.right_indent = Cm(0.8)
                run = paragraph.add_run("\n".join(code_lines))
                _set_run_font(run, 10.5)
                code_lines = []
                in_code = False
            else:
                in_code = True
            index += 1
            continue
        if in_code:
            code_lines.append(lines[index])
            index += 1
            continue
        if not stripped:
            index += 1
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            _add_table(document, _parse_table(table_lines))
            continue
        image_match = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if image_match:
            image_path = (SOURCE.parent / image_match.group(2)).resolve()
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.add_run().add_picture(str(image_path), width=Inches(6.1))
            index += 1
            continue
        heading = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading:
            level = len(heading.group(1))
            style = "Title" if level == 1 else f"Heading {level - 1}"
            paragraph = document.add_paragraph(style=style)
            paragraph.alignment = (
                WD_ALIGN_PARAGRAPH.CENTER if level == 1 else WD_ALIGN_PARAGRAPH.LEFT
            )
            size = 16 if level == 1 else 14 if level == 2 else 12
            _set_run_font(paragraph.add_run(heading.group(2)), size, True)
            index += 1
            continue
        if stripped.startswith(">"):
            quote_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote_lines.append(lines[index].strip().lstrip("> "))
                index += 1
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.left_indent = Cm(0.7)
            paragraph.paragraph_format.right_indent = Cm(0.7)
            _add_inline(paragraph, " ".join(quote_lines))
            continue
        numbered = re.match(r"^(\d+)[\.、]\s*(.+)$", stripped)
        if numbered:
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.left_indent = Cm(0.6)
            _add_inline(paragraph, f"{numbered.group(1)}. {numbered.group(2)}")
            index += 1
            continue
        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            candidate = lines[index].strip()
            if (
                not candidate
                or candidate.startswith(("#", "|", ">", "```", "!["))
                or re.match(r"^\d+[\.、]\s*", candidate)
            ):
                break
            paragraph_lines.append(candidate)
            index += 1
        paragraph = document.add_paragraph()
        if paragraph_lines[0].startswith("**图"):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _add_inline(paragraph, " ".join(paragraph_lines))

    # Force a new section only when a downstream editor appends appendices.
    document.add_section(WD_SECTION.CONTINUOUS)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT)


def main() -> None:
    _render_figure()
    _build_docx()
    print(FIGURE)
    print(OUTPUT)


if __name__ == "__main__":
    main()
