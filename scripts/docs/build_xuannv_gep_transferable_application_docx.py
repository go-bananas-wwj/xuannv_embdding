from __future__ import annotations

"""Generate the final editable Xuannv GEP application from the supplied template.

The parent template's cover, information table, section order, personnel table
and budget table are retained.  Technical-route and Haidian validation figures
are appended after the original application form rather than interrupting its
body text.
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
    "/root/.codex/attachments/a22df072-6028-49d0-b5e4-276baf7fd21a/"
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


def add_heading(parent, text: str) -> None:
    paragraph = parent.add_paragraph()
    set_paragraph(paragraph, text, size=Pt(12), bold=True)


def add_body(parent, text: str) -> None:
    paragraph = parent.add_paragraph()
    set_paragraph(paragraph, text, size=Pt(10.5), first_indent=True)


def add_figure(parent, path: Path, caption: str, width: Cm) -> None:
    paragraph = parent.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_after = Pt(2)
    paragraph.add_run().add_picture(str(path), width=width)
    caption_paragraph = parent.add_paragraph()
    set_paragraph(caption_paragraph, caption, size=Pt(9), center=True)


def add_route_table(parent) -> None:
    table = parent.add_table(rows=1, cols=5)
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
        paragraph = target.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.space_after = Pt(0)
        for line_index, line in enumerate(label.split("\n")):
            if line_index:
                paragraph.add_run().add_break()
            run = paragraph.add_run(line)
            set_run_font(run, Pt(9), bold=True)
        shading = OxmlElement("w:shd")
        shading.set(qn("w:fill"), "F2F2F2")
        target._tc.get_or_add_tcPr().append(shading)


def remove_template_drawings(document: Document) -> None:
    drawing_tags = {qn("w:drawing"), qn("w:pict")}
    for element in list(document._element.body.iter()):
        if element.tag in drawing_tags:
            parent = element.getparent()
            if parent is not None:
                parent.remove(element)


def strip_template_media(path: Path) -> None:
    """Delete legacy template images before adding this document's appendix figures."""
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


def populate_information(document: Document) -> None:
    info = document.tables[0]
    set_cell_text(info.cell(0, 3), TITLE, center=True)
    set_cell_text(info.cell(1, 3), "■1.业务化；■2.工程化；□3.产业化；□4.资本化")
    set_cell_text(info.cell(2, 3), "生态系统服务核算与遥感智能分析")
    set_cell_text(info.cell(3, 3), "地理嵌入、生态过程核算与数字化工具")
    set_cell_text(info.cell(4, 3), "□1年；■2年")
    set_cell_text(
        info.cell(5, 3),
        "面向可复制的生态系统生产总值核算需求，构建以玄女月度地理嵌入为基础的多源遥感数据处理、生态状态制图、过程单元核算和成果复核技术方法。海淀区现有工作仅作为已完成的技术验证示例，用于说明月度嵌入和少量标注专题制图的可行性；后续应用区域由项目任务确定。",
    )
    set_cell_text(info.cell(6, 3), "主要科技成果及形式：通用技术方案、数据与质量台账、月度生态状态产品、服务价值试点成果、专题图集、核算与查询工具。")
    set_cell_text(info.cell(7, 3), "重点技术突破及产品：多源月度地理嵌入、少样本专题制图、过程单元核算、质量控制和可复算成果交付。")
    set_cell_text(info.cell(8, 3), "效益预期：提高多源数据整理、生态状态更新、核算底图制作、成果复核与年度更新效率，为不同区域生态系统服务价值核算提供可复用技术支撑。")
    set_cell_text(info.cell(9, 3), "由申报单位核定", center=True)


def populate_goals_and_tasks(document: Document) -> None:
    cell = document.tables[1].cell(0, 0)
    cell.text = ""
    set_paragraph(cell.paragraphs[0], "项目研究的目的、必要性和可行性", size=Pt(12), bold=True)
    add_body(cell, "生态系统生产总值核算需要统筹生态系统类型、遥感监测、生态过程、统计资料和价值参数。现有工作中，多源数据的时间、空间和质量信息不统一，生态状态制图与服务实物量核算之间衔接不足，难以支撑细尺度、可追溯的区域生态服务评估。本项目面向项目确定的试点区，建立“遥感状态表征—专题过程模型—标准价值核算—成果复核”的通用技术路径。")
    add_body(cell, "项目以GB/T 46869—2025为依据，首期聚焦固定二氧化碳、洪水调蓄和局部气候调节三类服务。玄女月度嵌入用于生态系统面积、覆盖比例和空间分配，不替代生物量、蒸散发、径流和近地气温等过程观测。项目已有海淀区多源月度嵌入验证基础，可作为数据对齐、嵌入生成和少量标注专题制图的技术参考；新区域仍需完成数据质量检查、类别适配、过程参数配置和独立验证。")
    add_heading(cell, "项目目标及内容")
    add_heading(cell, "1. 项目目标")
    add_body(cell, "总目标：建成面向区域生态系统生产总值核算的月度地理嵌入、专题制图、过程核算和成果复核方法体系，实现从多源数据整理到服务价值试点成果的可追溯技术流程。")
    add_body(cell, "具体可考核目标：建立目标区域数据与质量台账，完成一个核算年度12期月度生态状态基础产品；完成固定二氧化碳、洪水调蓄和局部气候调节三类服务的实物量与价值量试点；形成可复算的参数、版本和结果记录，以及面向专题制图、统计汇总和结果查询的工具原型。三类服务试点合计不作为完整GEP总值发布。")
    add_heading(cell, "2. 研究内容")
    add_body(cell, "内容一：多源遥感数据对齐与GEP样本库构建。形成观测数据、生态状态数据、服务实物量数据和价值参数数据四类台账，纳入遥感影像、生态系统类型、地形、气象、土壤、水文、监测、统计和价格参数；建立空间参考、时间范围、质量标识和版本管理规则。")
    add_body(cell, "内容二：玄女月度地理嵌入与生态状态产品构建。以多源光学、雷达和辅助数据为输入，形成月度地表状态嵌入、生态系统分类、覆盖比例等基础产品。名义10米等效分析网格仅用于面积、覆盖比例和空间分配；固碳、洪水调蓄和局部气候调节分别在生态斑块或清查单元、子汇水区、绿地斑块及影响范围等过程单元开展计算。")
    add_body(cell, "内容三：GEP调节服务专题模型与动态核算。固定二氧化碳服务按照GB/T 46869—2025附录A.7，根据资料条件采用生物量法、固碳速率法或净生态系统生产力法；洪水调蓄服务按照附录A.6完成标准核算，城市生态覆盖增量雨洪效益作为补充分析并与标准结果分表；局部气候调节按照附录A.11完成蒸散发能量或实测温差核算。三项服务均结合独立资料开展验证。")
    add_body(cell, "内容四：面向核算辅助的查询、复核与报告工具。将数据来源、处理方法、模型版本、参数版本、空间范围、结果和不确定性纳入统一台账，形成受约束的专题查询、统计汇总、结果复算和报告生成流程。")
    add_heading(cell, "创新点")
    add_body(cell, "范式创新：建立月度地理嵌入与生态服务核算衔接方法，以统一地表状态表征支撑生态系统类型、面积和覆盖比例空间化，同时保持生态过程模型和标准核算口径的独立性。")
    add_body(cell, "机制创新：建立基础模型与专题过程模型协同方法，将月度遥感表征、生态辅助数据和服务专题模型结合，在同一验证集上比较原始影像、嵌入产品与传统过程模型的作用，为多服务复用提供依据。")
    add_body(cell, "应用创新：建立数据、模型、参数和结果可追溯的核算辅助机制，支持质量控制、结果复算、专题图生成和报告编制，降低多源资料重复整理与人工核对成本。")
    add_heading(cell, "关键技术及技术路线图")
    add_heading(cell, "关键技术")
    add_body(cell, "关键技术一：多源遥感数据时空对齐与质量管理。统一空间参考、时间基准、质量标识和数据版本，形成可用于生态服务核算的月度基础数据产品。")
    add_body(cell, "关键技术二：月度地理嵌入与服务专题模型协同。以嵌入产品为空间状态证据，结合服务专题模型完成生态斑块、子汇水区和绿地影响范围等过程单元的实物量核算。")
    add_body(cell, "关键技术三：标准化价值量核算与可追溯工具。依据GB/T 46869—2025完成实物量、价值量、参数和结果记录，形成受约束的查询、制图和报告工具。")
    add_body(cell, "技术路线图及已有验证示例见本申报书末“附图：技术路线与已有验证示例”。")


def populate_outputs_and_evaluation(document: Document) -> None:
    cell = document.tables[2].cell(0, 0)
    cell.text = ""
    set_paragraph(cell.paragraphs[0], "项目预期成果", size=Pt(12), bold=True)
    add_heading(cell, "数据类成果")
    add_body(cell, "区域生态系统调节服务数据台账：形成遥感观测、生态状态、服务实物量、价值参数和质量记录五类可追溯数据。月度生态状态基础产品：完成经批准核算年度12期月度嵌入、生态系统分类或覆盖比例等基础产品，并保留空间参考、时间范围、处理版本和质量标识。")
    add_heading(cell, "模型与算法类成果")
    add_body(cell, "玄女月度地理嵌入与专题制图流程：形成可配置的多源数据预处理、质量控制、月度嵌入生成和少样本专题制图流程。调节服务专题模型与参数台账：形成固定二氧化碳、洪水调蓄、局部气候调节三类服务的过程单元、参数、核算与验证说明。")
    add_heading(cell, "软件平台类成果")
    add_body(cell, "核算辅助工具原型：支持数据范围、时间、版本和参数查询，专题制图、统计汇总、结果复算和报告导出；所有结果返回数据来源、模型版本、参数版本、方法和不确定性说明。")
    add_heading(cell, "知识产权与标准类成果")
    add_body(cell, "根据项目实施情况形成软件著作权、专利或论文等成果；沉淀区域GEP核算技术规程、数据字典和成果交付规范。具体数量由申报单位结合项目管理要求确定。")
    add_heading(cell, "考核指标")
    add_body(cell, "1. 数据与基础产品：完成经批准核算年度12期月度生态状态基础产品，全部产品具有完整的数据来源、时间、空间参考、处理版本和质量标识；形成数据、模型、参数和结果关联台账。")
    add_body(cell, "2. 生态状态专题：生态系统分类采用空间分块独立样本评价，报告总体精度、各类别精度、面积偏差和置信区间；连续变量及水文过程按服务类型报告相应精度、偏差和不确定性。")
    add_body(cell, "3. 服务核算：完成固定二氧化碳、洪水调蓄和局部气候调节三类服务试点。每项服务明确实物量定义、单位、过程单元、价值参数、独立验证资料和核算范围；三类服务合计不替代完整GEP总值。")
    add_body(cell, "4. 可复算与工具：冻结的数据、参数和处理流程可重复复算；工具输出具备数据时间、空间范围、模型版本、参数版本、方法和不确定性字段。")
    add_heading(cell, "项目成果转化分析")
    add_body(cell, "业务化：以数据台账、月度生态状态产品、服务核算流程和专题图集支撑目标试点区生态空间监测、生态保护成效评估、年度更新和成果审查。工程化：将数据处理、专题模型、参数管理、质量复核和报告工具模块化部署，并与既有遥感、自然资源和生态环境数据平台衔接。跨区域应用前重新核验生态类型、数据可得性、参数适用性和独立验证资料。")
    add_body(cell, "产业化和资本化不作为本项目的刚性考核内容。项目可在业务化和工程化验证成熟后，面向生态评估、数据服务和平台集成等实际需求开展后续转化。")


def populate_foundation(document: Document) -> None:
    cell = document.tables[3].cell(0, 0)
    cell.text = ""
    set_paragraph(cell.paragraphs[0], "课题主要研究技术内容的国内外发展现状与趋势，现有技术基础", size=Pt(12), bold=True)
    add_heading(cell, "国内外发展现状与趋势")
    add_heading(cell, "地球观测基础模型")
    add_body(cell, "国际上，Prithvi、SatMAE、SpectralGPT和AlphaEarth Foundations等工作推动遥感由单期影像分类向多源、多时相通用表征学习发展。该类方法通过自监督预训练生成密集地理嵌入，可在统一特征空间支撑不同下游制图任务。发展趋势是增强多源协同、时间连续性、质量感知和跨区域适配能力。")
    add_heading(cell, "生态系统生产总值核算技术")
    add_body(cell, "生态系统服务评估主流技术包括SEEA-EA核算框架、InVEST、CASA等模型。我国GB/T 46869—2025明确了生态产品价值核算的服务类别、实物量和价值量核算原则。遥感产品适合用于生态系统面积、植被状态、水体和土地覆盖的空间化；生态过程与价值量仍应由专题模型、监测、统计资料和参数共同支撑。发展趋势是从年度静态评估向可更新的月/季尺度基础状态产品演进。")
    add_heading(cell, "遥感智能交互决策")
    add_body(cell, "地理空间智能体正逐步将自然语言解析与GIS、遥感和模型工具链结合。本项目采用受约束的查询与核算流程：用户查询被解析为已登记的数据、参数和模型调用，返回结果时同步提供版本、范围、方法和不确定性，不以自由生成内容替代核算。")
    add_heading(cell, "现有技术基础")
    add_body(cell, "玄女项目已在海淀区完成多源月度地理嵌入训练与下游验证。现有生产模型使用2025年12月至2026年5月数据，在320个1,280米×1,280米样本单元上生成128×128×64的名义10米等效分析网格，输入包括Sentinel-1、Sentinel-2、Landsat、高分光学和高分SAR等数据。建筑、道路和水体结果采用与预训练开放地图弱语义同源的标签，属于内部制图验证，不作为生态服务或GEP核算精度依据。")
    add_body(cell, "项目已具备多源遥感数据处理、月度嵌入生成、专题制图、数据版本管理、下游模型评测和多卡训练环境等基础。现有训练配置、数据处理脚本和评测流程可作为新区域研发起点；生态系统类型、气象、水文、监测、统计和价值参数资料将在项目启动阶段与相关数据责任方对接，形成数据取得清单和时间安排。")
    add_heading(cell, "课题申请部门及主要参与单位研究基础")
    add_body(cell, "申报部门既有生态评估、遥感地理信息、工程咨询与软件平台建设经历，以及人员、平台和相关项目成果，请由申报单位在正式申报时据实补充。项目拟设置项目统筹、遥感与数据工程、生态核算、水文与气候专题模型、平台与软件、质量控制与独立复核等岗位。")
    add_body(cell, "实施安排：第一阶段完成核算边界、生态产品清单、数据来源、质量规范和验证方案确认；第二阶段完成月度生态状态基础产品、生态系统分类和三类服务专题模型开发；第三阶段完成实物量与价值量试点、独立验证、不确定性分析、核算工具和示范报告；第四阶段完成成果复核、交付固化和后续推广建议。")
    add_body(cell, "参考依据：GB/T 46869—2025《生态系统评估 陆域生态产品总值核算技术指南》；Ouyang Z. et al. Gross ecosystem product: concept, accounting framework and case study, PNAS, 2020；Potter C. S. et al. Terrestrial ecosystem production: A process model based on global satellite and surface data, Global Biogeochemical Cycles, 1993。")


def populate_people_and_budget(document: Document) -> None:
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


def append_figures(document: Document) -> None:
    document.add_page_break()
    heading = document.add_paragraph()
    set_paragraph(heading, "附图：技术路线与已有验证示例", size=Pt(16), bold=True, center=True)
    note = document.add_paragraph()
    set_paragraph(note, "本附图置于申报书正文之后。海淀区图件仅用于展示玄女现有月度嵌入与专题制图能力，不限定本项目后续实施区域，也不构成生态服务实物量或价值量的独立精度结论。", size=Pt(10.5), first_indent=True)
    add_heading(document, "附图一 技术路线")
    add_route_table(document)
    add_body(document, "项目在目标试点区完成多源数据质量控制和时空对齐后，形成月度地表状态嵌入；以生态斑块、子汇水区和绿地影响范围等过程单元接入专题模型与独立验证资料，最后依据标准开展价值量核算、成果复核和工具化交付。")
    add_heading(document, "附图二 海淀区月度嵌入 PCA 可视化示例")
    add_figure(document, ASSETS / "haidian_embedding_pca_202604.png", "图1 海淀区2026年4月月度嵌入 PCA 可视化。颜色为64维嵌入的前三主成分组合，用于展示空间连续性和地表差异，不代表生态类别或价值量。", Cm(14.8))
    add_heading(document, "附图三 海淀区公园/绿地少样本专题制图示例")
    add_figure(document, ASSETS / "haidian_park_green_fewshot_50.png", "图2 海淀区公园/绿地50样本专题制图示例。左：OSM弱标签；中：玄女嵌入加轻量分类器；右：传统多源影像加同类分类器。红色越深表示目标类别概率越高，白色为背景。", Cm(15.0))
    add_heading(document, "附图四 海淀区土地利用/覆盖专题制图示例")
    add_figure(document, ASSETS / "haidian_landuse_example.png", "图3 海淀区土地利用/覆盖专题制图示例。该示例展示月度嵌入可为生态系统类型划分和面积统计提供空间底图；正式项目需在目标区域采用冻结的类别体系与独立验证数据复核。", Cm(14.8))


def build_document() -> None:
    copy2(TEMPLATE, OUTPUT)
    strip_template_media(OUTPUT)
    document = Document(OUTPUT)
    remove_template_drawings(document)
    replace_cover(document)
    populate_information(document)
    populate_goals_and_tasks(document)
    populate_outputs_and_evaluation(document)
    populate_foundation(document)
    populate_people_and_budget(document)
    append_figures(document)
    document.save(OUTPUT)


if __name__ == "__main__":
    build_document()
