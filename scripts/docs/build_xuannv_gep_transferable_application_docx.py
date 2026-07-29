from __future__ import annotations

"""Build one editable GEP application document with Xuannv demonstration figures.

The document treats Haidian as a completed validation example rather than a
mandatory future application area.  All prose, tables and figure captions are
native Word elements and remain editable after delivery.
"""

from pathlib import Path
from shutil import copy2
from zipfile import ZIP_DEFLATED, ZipFile
from xml.etree import ElementTree

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = Path(
    "/root/.codex/attachments/23a35a89-d436-41dd-b95e-d6fa6b9da987/"
    "中国城市建设研究院有限公司自立科研项目申-生态系统生产总值.docx"
)
ASSETS = ROOT / "docs/projects/assets/xuannv_gep_transferable_20260729"
OUTPUT = ROOT / "docs/projects/xuannv_gep_transferable_application_20260729_zh.docx"
TITLE = "基于玄女月度地理嵌入的生态系统生产总值核算技术研究"


def set_run_font(run, size: Pt = Pt(10.5), bold: bool = False) -> None:
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    run.font.size = size
    run.font.bold = bold
    run.font.color.rgb = None


def set_paragraph(
    paragraph,
    text: str = "",
    *,
    size: Pt = Pt(10.5),
    bold: bool = False,
    center: bool = False,
    first_indent: bool = False,
) -> None:
    paragraph.clear()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT
    paragraph.paragraph_format.space_after = Pt(4)
    paragraph.paragraph_format.line_spacing = 1.25
    paragraph.paragraph_format.first_line_indent = Cm(0.74) if first_indent else Cm(0)
    if text:
        run = paragraph.add_run(text)
        set_run_font(run, size, bold)


def set_cell_text(cell, text: str, *, center: bool = False, bold: bool = False) -> None:
    cell.text = ""
    set_paragraph(cell.paragraphs[0], text, center=center, bold=bold)


def add_heading(cell, text: str) -> None:
    paragraph = cell.add_paragraph()
    set_paragraph(paragraph, text, size=Pt(12), bold=True)


def add_body(cell, text: str) -> None:
    paragraph = cell.add_paragraph()
    set_paragraph(paragraph, text, size=Pt(10.5), first_indent=True)


def add_figure(cell, path: Path, caption: str, width: Cm) -> None:
    paragraph = cell.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(2)
    run = paragraph.add_run()
    run.add_picture(str(path), width=width)
    caption_paragraph = cell.add_paragraph()
    set_paragraph(caption_paragraph, caption, size=Pt(9), center=True)


def add_route_table(cell) -> None:
    table = cell.add_table(rows=1, cols=5)
    table.autofit = False
    labels = [
        "多源遥感与\n生态辅助数据",
        "质量控制与\n空间对齐",
        "月度地表状态\n嵌入产品",
        "专题模型、过程单元\n与独立验证",
        "价值核算、专题图\n与查询工具",
    ]
    for index, label in enumerate(labels):
        target = table.cell(0, index)
        target.width = Cm(3.0)
        target.text = ""
        p = target.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        for line_index, line in enumerate(label.split("\n")):
            if line_index:
                p.add_run().add_break()
            run = p.add_run(line)
            set_run_font(run, Pt(9), bold=True)
        shade = OxmlElement("w:shd")
        shade.set(qn("w:fill"), "F2F2F2")
        target._tc.get_or_add_tcPr().append(shade)


def remove_template_drawings(document: Document) -> None:
    drawing_tags = {qn("w:drawing"), qn("w:pict")}
    for element in list(document._element.body.iter()):
        if element.tag in drawing_tags:
            parent = element.getparent()
            if parent is not None:
                parent.remove(element)


def strip_template_media(path: Path) -> None:
    """Remove legacy template images before adding this document's own figures."""
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


def replace_cover(document: Document) -> None:
    for paragraph in document.paragraphs:
        if "申报项目名称：" in paragraph.text:
            set_paragraph(paragraph, f"申报项目名称： {TITLE}", size=Pt(12))
        elif "预计研究时间：" in paragraph.text:
            set_paragraph(paragraph, "预计研究时间：2026年9月1日 至 2028年8月31日", size=Pt(12))
        elif "二○" in paragraph.text and "年" in paragraph.text and "月" in paragraph.text:
            set_paragraph(paragraph, "二〇二六年七月", size=Pt(12))


def build_document() -> None:
    copy2(TEMPLATE, OUTPUT)
    strip_template_media(OUTPUT)
    document = Document(OUTPUT)
    remove_template_drawings(document)
    replace_cover(document)

    info = document.tables[0]
    set_cell_text(info.cell(0, 3), TITLE, center=True)
    set_cell_text(info.cell(1, 3), "■1.业务化；■2.工程化；□3.产业化；□4.资本化")
    set_cell_text(info.cell(2, 3), "生态系统服务核算与遥感智能分析")
    set_cell_text(info.cell(3, 3), "地理嵌入、生态过程核算与数字化工具")
    set_cell_text(info.cell(4, 3), "□1年；■2年")
    set_cell_text(
        info.cell(5, 3),
        "面向可复制的生态系统生产总值核算需求，构建以玄女月度地理嵌入为基础的多源遥感数据处理、生态状态制图、过程单元核算和成果复核技术方法。海淀区现有工作仅作为已完成的技术验证示例，用于说明月度嵌入和少量标注专题制图的可行性；后续应用区域可按项目任务另行确定。",
    )
    set_cell_text(
        info.cell(6, 3),
        "主要科技成果及形式：通用技术方案、数据与质量台账、月度生态状态产品、服务价值试点成果、专题图集、核算与查询工具。",
    )
    set_cell_text(
        info.cell(7, 3),
        "重点技术突破及产品：多源月度地理嵌入、少样本专题制图、过程单元核算、质量控制和可复算成果交付。",
    )
    set_cell_text(
        info.cell(8, 3),
        "效益预期：提高多源数据整理、生态状态更新、核算底图制作、成果复核与年度更新效率，为不同区域生态系统服务价值核算提供可复用技术支撑。",
    )

    purpose = document.tables[1].cell(0, 0)
    purpose.text = ""
    set_paragraph(purpose.paragraphs[0], "项目研究的目的、必要性和可行性", size=Pt(12), bold=True)
    add_body(
        purpose,
        "生态系统生产总值核算需要跨传感器、跨时相且空间一致的生态状态资料。现有工作中，影像、土地覆盖、地形、气象、水文及样地资料来源分散，更新频率和空间尺度不一，导致基础底图制作、过程参数配置和成果复核成本较高。本项目拟构建面向不同试点区域的通用技术方法，以玄女月度地理嵌入为统一的遥感表征底座，服务于生态状态识别、变化追踪、专题制图和核算辅助，而不以嵌入直接替代生态过程观测或价值量计算。",
    )
    add_body(
        purpose,
        "研究按“基础数据—月度地表状态—专题与过程模型—实物量和价值量—质量复核”组织。玄女模型输出128×128×64的月度密集嵌入图，对应1,280米×1,280米数据单元的名义10米等效分析网格；其作用是提供面积、覆盖比例和空间分布信息。固碳、洪水调蓄和局地气候调节等服务仍需分别结合样地、碳通量、降雨径流、气象与参数资料，在生态斑块、子汇水区或绿地影响范围等过程单元开展核算和验证。",
    )
    add_heading(purpose, "拟采用的技术路线")
    add_route_table(purpose)
    add_body(
        purpose,
        "技术路线适用于项目确定的任意试点区。各区域根据数据可得性配置卫星光学、雷达、地形、植被、气象水文和调查资料；统一完成坐标、时间、质量和版本管理后，形成月度生态状态基础产品，并按服务类型接入专题模型与独立验证资料。",
    )
    add_heading(purpose, "海淀区既有验证示例：月度嵌入的连续地表表达")
    add_figure(
        purpose,
        ASSETS / "haidian_embedding_pca_202604.png",
        "图1 海淀区2026年4月月度嵌入 PCA 可视化示例。颜色为64维嵌入的前三主成分组合，用于展示空间连续性和地表差异，不代表生态类别或价值量。",
        Cm(14.8),
    )

    outcomes = document.tables[2].cell(0, 0)
    outcomes.text = ""
    set_paragraph(outcomes.paragraphs[0], "项目预期成果", size=Pt(12), bold=True)
    add_heading(outcomes, "1. 面向区域适配的技术成果")
    add_body(outcomes, "形成可配置的数据清单、预处理规范、质量控制规则、月度嵌入生成流程、生态状态专题制图流程、过程核算接口和成果复核规程。项目实施区域不预设为海淀区，可由任务需求确定，并按当地数据、生态系统类型和核算口径完成适配。")
    add_heading(outcomes, "2. 海淀区既有验证示例：少量标注支持专题制图")
    add_body(outcomes, "在海淀现有数据中，使用少量标注样本训练轻量下游头，可将月度嵌入转换为公园/绿地等专题分布图。该图用于说明“嵌入+少量标注”的工作方式；标签来源为OSM弱标签，结果不作为生态服务实物量或价值量的独立精度结论。")
    add_figure(
        outcomes,
        ASSETS / "haidian_park_green_fewshot_50.png",
        "图2 海淀区公园/绿地50样本专题制图示例。左：OSM弱标签；中：玄女嵌入加轻量分类器；右：传统多源影像加同类分类器。红色越深表示目标类别概率越高，白色为背景。",
        Cm(15.0),
    )
    add_heading(outcomes, "3. 面向GEP核算的专题产品与核算边界")
    add_body(outcomes, "项目拟在目标区域输出生态系统类型与覆盖比例、绿地与水体等基础专题、质量信息、过程单元参数表、三类调节服务实物量与价值量试点结果，以及可复算的参数和版本记录。生态类型、绿地等专题图作为核算底图和空间分配依据；固定碳、洪水调蓄、局地气候调节等服务的实物量和价值量须依照相应方法、参数和独立数据完成计算与验证。")
    add_figure(
        outcomes,
        ASSETS / "haidian_landuse_example.png",
        "图3 海淀区土地利用/覆盖专题制图示例。该示例展示月度嵌入可为生态系统类型划分和面积统计提供空间底图；正式项目需在目标区域采用冻结的类别体系与独立验证数据复核。",
        Cm(14.8),
    )
    add_heading(outcomes, "4. 预期交付")
    add_body(outcomes, "交付通用技术方案1套，区域数据与质量台账1套，月度生态状态基础产品和专题图集1套，三类调节服务核算试点成果1套，参数、版本和复算说明1套，以及面向成果查询、指标追溯和复核的工具原型1套。")

    foundation = document.tables[3].cell(0, 0)
    foundation.text = ""
    set_paragraph(foundation.paragraphs[0], "课题主要研究技术内容的国内外发展现状与趋势，现有技术基础", size=Pt(12), bold=True)
    add_heading(foundation, "国内外发展现状与趋势")
    add_body(foundation, "地球观测基础模型正由单期影像分类转向多源、多时相的通用表征学习。以AlphaEarth Foundations等为代表的研究表明，密集地理嵌入可在统一特征空间支撑多种下游制图任务。生态系统服务核算则强调核算单元、实物量方法、价值转换、质量控制和不确定性说明。两者结合的关键不在于以模型直接输出货币价值，而在于以可更新的遥感状态底图降低多源数据对齐和专题制图成本，并把价值核算置于可验证的过程模型和标准口径之下。")
    add_heading(foundation, "玄女现有技术基础与可迁移边界")
    add_body(foundation, "玄女已在海淀区完成多源月度嵌入的训练与下游验证，现有生产模型使用2025年12月至2026年5月数据，在320个1,280米×1,280米样本单元上生成128×128×64月度密集嵌入。已验证建筑、道路、水体及若干OSM衍生类别的轻量专题制图流程，可作为新区域开展数据对齐、嵌入生成、少样本制图和变化分析的技术参考。")
    add_body(foundation, "现有海淀模型不是其他区域的现成GEP结论，也不等同于全国通用权重。新区域应用应重新完成数据质量检查、空间对齐、类别适配、专题验证和过程参数配置；对于高分辨率数据、生态类型或气候水文条件显著不同的区域，需进行增量训练或再训练，并在独立数据上验证。")
    add_heading(foundation, "组织与实施安排")
    add_body(foundation, "第一阶段完成目标区域数据盘点、标准口径确认和试点单元设计；第二阶段建设月度生态状态产品和生态专题底图；第三阶段开展固碳、洪水调蓄、局地气候调节三类服务的专题模型、实物量核算和独立验证；第四阶段完成价值量核算、质量复核、成果固化和示范应用。")
    add_heading(foundation, "验收与质量控制原则")
    add_body(foundation, "所有成果记录数据来源、处理日期、空间参考、有效观测、模型版本、参数版本和不确定性说明。遥感专题图与生态服务核算分层验收：前者验证分类或覆盖比例，后者分别验证过程模型与实物量；同一冻结输入、参数和舍入规则下，价值量计算应可重复复算。")

    team = document.tables[4]
    set_cell_text(team.cell(1, 0), "项目负责人（待填）", center=True)
    set_cell_text(team.cell(1, 7), "总体统筹、技术路线与成果质量控制", center=True)
    set_cell_text(team.cell(4, 0), "遥感与数据工程人员（待填）", center=True)
    set_cell_text(team.cell(4, 7), "数据处理、月度产品、质量与版本管理", center=True)
    set_cell_text(team.cell(5, 0), "生态核算与专题模型人员（待填）", center=True)
    set_cell_text(team.cell(5, 7), "服务核算、独立验证与参数管理", center=True)
    set_cell_text(team.cell(7, 0), "平台与质量复核人员（待填）", center=True)
    set_cell_text(team.cell(7, 7), "工具开发、结果复算与成果交付", center=True)

    budget = document.tables[5]
    set_cell_text(budget.cell(2, 2), "由申报单位核定", center=True)
    set_cell_text(budget.cell(14, 2), "由申报单位核定", center=True)
    note = document.add_paragraph()
    set_paragraph(note, "经费预算说明：经费金额、人员投入和试点区域范围由申报单位结合实施区域、数据条件和核算服务类别据实核定。", size=Pt(10.5), bold=True)
    document.save(OUTPUT)


if __name__ == "__main__":
    build_document()
