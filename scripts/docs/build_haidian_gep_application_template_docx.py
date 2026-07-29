from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.etree import ElementTree

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt


ROOT = Path(__file__).resolve().parents[2]
SOURCE_MD = ROOT / "docs/projects/haidian_gep_xuannv_application_formal_20260729_zh.md"
TEMPLATE = Path(
    "/root/.codex/attachments/23a35a89-d436-41dd-b95e-d6fa6b9da987/"
    "中国城市建设研究院有限公司自立科研项目申-生态系统生产总值.docx"
)
OUTPUT = ROOT / "docs/projects/haidian_gep_xuannv_application_formal_template_20260729_zh.docx"


def set_run_font(run, size: Pt = Pt(10.5), bold: bool = False) -> None:
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    run.font.size = size
    run.font.bold = bold
    run.font.color.rgb = None


def set_paragraph(paragraph, text: str, *, size: Pt = Pt(10.5), bold: bool = False) -> None:
    paragraph.clear()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.paragraph_format.line_spacing = 1.25
    run = paragraph.add_run(text)
    set_run_font(run, size, bold)


def set_cell_text(cell, text: str, *, center: bool = False, bold: bool = False) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.1
    run = p.add_run(text)
    set_run_font(run, Pt(10.5), bold)


def extract_section(source: str, heading: str, next_heading: str | None = None) -> list[str]:
    start = source.index(heading) + len(heading)
    end = source.index(next_heading, start) if next_heading else len(source)
    return [line.strip() for line in source[start:end].splitlines() if line.strip()]


def write_markdown_to_cell(cell, lines: list[str]) -> None:
    cell.text = ""
    for index, line in enumerate(lines):
        if line.startswith("## "):
            text, size, bold = line[3:], Pt(12), True
        elif line.startswith("### "):
            text, size, bold = line[4:], Pt(10.5), True
        else:
            text, size, bold = line, Pt(9.5), False
        paragraph = cell.paragraphs[0] if index == 0 else cell.add_paragraph()
        set_paragraph(paragraph, text, size=size, bold=bold)


def remove_template_drawings(document: Document) -> None:
    drawing_tags = {qn("w:drawing"), qn("w:pict")}
    for element in list(document._element.body.iter()):
        if element.tag in drawing_tags:
            parent = element.getparent()
            if parent is not None:
                parent.remove(element)


def add_route_table(cell) -> None:
    heading = cell.add_paragraph()
    set_paragraph(heading, "技术路线图", size=Pt(10.5), bold=True)
    table = cell.add_table(rows=1, cols=6)
    labels = [
        "多源遥感与生态辅助数据",
        "质量控制与时空对齐",
        "月度生态状态基础产品",
        "过程单元核算与独立验证",
        "标准价值量核算",
        "试点报告、专题图与核算工具",
    ]
    for index, label in enumerate(labels):
        target = table.cell(0, index)
        set_cell_text(target, label, center=True, bold=True)
        shade = OxmlElement("w:shd")
        shade.set(qn("w:fill"), "F2F2F2")
        target._tc.get_or_add_tcPr().append(shade)


def replace_cover_title(document: Document, title: str) -> None:
    for paragraph in document.paragraphs:
        if "申报项目名称：" in paragraph.text:
            set_paragraph(paragraph, f"申报项目名称： {title}", size=Pt(12), bold=False)
        elif "预计研究时间：" in paragraph.text:
            set_paragraph(paragraph, "预计研究时间：2026年9月1日 至 2028年8月31日", size=Pt(12))
        elif "二○" in paragraph.text and "年" in paragraph.text and "月" in paragraph.text:
            set_paragraph(paragraph, "二〇二六年七月", size=Pt(12), bold=False)


def strip_template_media(path: Path) -> None:
    """Delete unused image parts carried by the source application template."""
    temporary = path.with_suffix(".tmp.docx")
    with ZipFile(path) as source, ZipFile(temporary, "w", ZIP_DEFLATED) as target:
        for item in source.infolist():
            if item.filename.startswith("word/media/"):
                continue
            data = source.read(item.filename)
            if item.filename.endswith(".rels"):
                root = ElementTree.fromstring(data)
                for relationship in list(root):
                    if relationship.attrib.get("Type", "").endswith("/image"):
                        root.remove(relationship)
                data = ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)
            target.writestr(item, data)
    temporary.replace(path)


def build_document() -> None:
    source = SOURCE_MD.read_text(encoding="utf-8")
    title = source.splitlines()[0].removeprefix("# ").strip()
    overview = extract_section(source, "### 项目概述（300字以内）", "## 一、")
    tasks = extract_section(source, "## 一、目标与研究任务", "## 二、")
    outcomes = extract_section(source, "## 二、项目成果、考核指标及成果转化分析", "## 三、")
    foundation = extract_section(source, "## 三、现有工作基础与优势", "## 四、")
    implementation = extract_section(source, "## 四、研发组织与实施安排", "## 五、")
    budget = extract_section(source, "## 五、经费预算", "## 六、")

    document = Document(TEMPLATE)
    remove_template_drawings(document)
    replace_cover_title(document, title)
    info = document.tables[0]
    set_cell_text(info.cell(0, 3), title, center=True)
    set_cell_text(info.cell(1, 3), "■1.业务化；■2.工程化；□3.产业化；□4.资本化")
    set_cell_text(info.cell(2, 3), "生态环境治理与数字化")
    set_cell_text(info.cell(3, 3), "生态系统服务核算与遥感智能分析")
    set_cell_text(info.cell(4, 3), "□1年；■2年")
    set_cell_text(info.cell(5, 3), "".join(overview))
    set_cell_text(info.cell(6, 3), "主要科技成果及形式：数据台账、三类调节服务价值试点、核算辅助工具、试点报告与技术附件。")
    set_cell_text(info.cell(7, 3), "重点技术突破及产品：月度地理嵌入与服务专题模型协同；标准化实物量与价值量核算；可追溯核算辅助工具。")
    set_cell_text(info.cell(8, 3), "效益预期：支撑海淀区生态空间监测、服务核算、年度更新和结果复核，提升数据整理和成果审查效率。")

    write_markdown_to_cell(document.tables[1].cell(0, 0), tasks)
    add_route_table(document.tables[1].cell(0, 0))
    write_markdown_to_cell(document.tables[2].cell(0, 0), outcomes)
    write_markdown_to_cell(document.tables[3].cell(0, 0), foundation + implementation)

    team = document.tables[4]
    set_cell_text(team.cell(1, 0), "项目负责人（待填）", center=True)
    set_cell_text(team.cell(1, 7), "总体统筹、技术路线与成果质量控制", center=True)
    set_cell_text(team.cell(4, 0), "遥感与数据工程人员（待填）", center=True)
    set_cell_text(team.cell(4, 7), "数据处理、月度产品、质量与版本管理", center=True)
    set_cell_text(team.cell(5, 0), "生态核算与专题模型人员（待填）", center=True)
    set_cell_text(team.cell(5, 7), "三类服务核算、验证与参数管理", center=True)
    set_cell_text(team.cell(7, 0), "平台与质量复核人员（待填）", center=True)
    set_cell_text(team.cell(7, 7), "工具开发、结果复算与成果交付", center=True)

    budget_table = document.tables[5]
    set_cell_text(budget_table.cell(2, 2), "由申报单位核定", center=True)
    set_cell_text(budget_table.cell(14, 2), "由申报单位核定", center=True)
    note = document.add_paragraph()
    set_paragraph(note, "经费预算说明：" + "".join(budget), size=Pt(10.5), bold=True)
    document.save(OUTPUT)
    strip_template_media(OUTPUT)


if __name__ == "__main__":
    build_document()
