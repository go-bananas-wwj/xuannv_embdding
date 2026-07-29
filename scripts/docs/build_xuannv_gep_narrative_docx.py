from __future__ import annotations

"""Build a detailed, table-free narrative application for the Xuannv GEP project."""

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs/projects/xuannv_gep_application_narrative_20260729_zh.docx"
TITLE = "基于玄女月度地理嵌入的生态系统生产总值核算技术研究"


def set_run_font(run, size: Pt, bold: bool = False) -> None:
    """Chinese uses SimSun; Latin characters in the same run use Times New Roman."""
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    run.font.size = size
    run.font.bold = bold
    run.font.color.rgb = None


def add_paragraph(
    document: Document,
    text: str,
    *,
    size: Pt = Pt(12),
    bold: bool = False,
    center: bool = False,
    first_indent: bool = True,
    before: Pt = Pt(0),
    after: Pt = Pt(6),
) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT
    paragraph.paragraph_format.space_before = before
    paragraph.paragraph_format.space_after = after
    paragraph.paragraph_format.line_spacing = 1.5
    paragraph.paragraph_format.first_line_indent = Cm(0.74) if first_indent else Cm(0)
    run = paragraph.add_run(text)
    set_run_font(run, size, bold)


def add_heading(document: Document, text: str) -> None:
    add_paragraph(
        document,
        text,
        size=Pt(14),
        bold=True,
        first_indent=False,
        before=Pt(14),
        after=Pt(8),
    )


def add_body(document: Document, text: str) -> None:
    add_paragraph(document, text, size=Pt(12), first_indent=True)


def build_document() -> None:
    document = Document()
    section = document.sections[0]
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(2.7)
    section.right_margin = Cm(2.7)

    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing = 1.5

    add_paragraph(
        document,
        TITLE,
        size=Pt(14),
        bold=True,
        center=True,
        first_indent=False,
        after=Pt(24),
    )
    add_paragraph(
        document,
        "自立科研项目申报书（文字稿）",
        size=Pt(14),
        bold=True,
        center=True,
        first_indent=False,
        after=Pt(30),
    )

    add_heading(document, "项目信息")
    add_body(document, "申报部门、项目申报人、参与单位、研究周期和经费预算由申报单位据实填写。建议研究周期为两年，成果转化方式以业务化、工程化为主，对应方向为生态系统服务核算与遥感智能分析。")
    add_heading(document, "项目概述")
    add_body(document, "本项目面向区域生态系统生产总值核算中遥感数据分散、生态状态更新不及时、服务模型与核算口径衔接不足等问题，依托玄女月度地理嵌入，构建多源遥感与生态辅助数据协同处理、生态状态空间化、服务实物量反演、价值量核算和结果复核的技术流程。玄女产品输出名义10米等效分析网格，用于生态系统面积、覆盖比例和空间分配，不替代生物量、蒸散发、径流和近地气温等过程观测。项目依据GB/T 46869—2025，首期完成固定二氧化碳、洪水调蓄和局部气候调节三类服务的实物量与价值量试点，形成数据台账、基础产品、专题成果和可追溯核算工具，为后续完善生态产品清单和开展完整GEP核算提供基础。")

    add_heading(document, "一、目标与研究任务")
    add_heading(document, "1. 项目研究的目的、必要性和可行性")
    add_body(document, "生态系统生产总值核算需要统筹生态系统类型、遥感监测、生态过程、统计资料和价值参数。现有工作中，卫星影像、地形、气象、水文、土壤、生态调查、统计和价格数据的时间、空间和质量信息往往不统一；生态状态制图、服务实物量反演与价值量核算之间缺少稳定的数据接口和版本管理，导致基础底图更新成本高、过程参数难追溯、成果复算困难。")
    add_body(document, "本项目拟建立“多源数据对齐—月度生态状态表征—专题过程模型—标准价值量核算—质量复核与成果查询”的通用技术路径。项目实施区域不预设为海淀区，而根据任务需求确定。海淀区已有工作仅作为前期技术验证：已证明多源月度嵌入、少量标注专题制图和版本化评测流程可以运行；新区域仍需完成数据质量检查、空间对齐、生态类型适配、参数本地化和独立验证，不能直接把海淀结果外推为其他区域的GEP结论。")
    add_body(document, "项目具备实施可行性。一方面，GB/T 46869—2025提供了服务实物量与价值量核算的基本规范；另一方面，玄女已形成多源遥感处理、月度嵌入生成、专题制图、数据版本管理和下游评测的工程基础。项目将在此基础上补充生态系统分类、数字高程模型、气象、水文、土壤、清查、监测、统计和价格资料，按照国家标准和经验证的专题过程模型开展服务核算。")

    add_heading(document, "2. 项目目标")
    add_body(document, "总体目标：建立面向区域生态系统生产总值核算的月度地理嵌入、生态状态专题制图、服务实物量核算、价值量计算和成果复核方法体系，形成从多源数据整理到服务价值试点结果的可追溯技术流程。")
    add_body(document, "目标一：建立目标试点区生态系统调节服务数据台账和月度生态状态基础产品，统一空间参考、时间范围、质量标识、数据血缘和版本管理。")
    add_body(document, "目标二：建立固定二氧化碳、洪水调蓄、局部气候调节三类服务的实物量与价值量核算方法，形成与GB/T 46869—2025衔接的区域试点成果。")
    add_body(document, "目标三：建立数据、模型、参数和结果可追溯的核算辅助工具，支持专题制图、统计汇总、结果查询、复算和报告生成。")
    add_body(document, "目标四：建立完整GEP核算所需的生态产品清单、数据缺口和扩展路径。三类服务试点合计不作为完整GEP总值发布，只有在生态产品清单、适用科目、数据、参数和独立验证条件齐备后才可另行开展完整GEP核算。")

    add_heading(document, "3. 研究内容")
    add_body(document, "第一，数据对齐与GEP样本库构建。建立观测数据、生态状态数据、服务实物量数据和价值参数数据四类台账，纳入遥感影像、生态系统类型、地形、气象、土壤、水文、监测、统计和价格参数。明确原始观测、重采样辅助变量、模型推断结果和人工校核资料的属性；统一空间参考、时间基准、质量标识、缺失值编码和版本记录。")
    add_body(document, "第二，月度地理嵌入与生态状态产品构建。以玄女模型为基础，融合光学、雷达和辅助数据，形成月度地表状态嵌入、生态系统分类、覆盖比例及其质量信息等基础产品。模型输出为1,280米×1,280米数据单元内的128×128×64密集嵌入，即名义10米等效分析网格。该网格用于面积、覆盖比例和空间分配，不代表10米生态过程观测或服务核算精度。")
    add_body(document, "第三，三类调节服务专题模型与实物量核算。固定二氧化碳服务按照GB/T 46869—2025附录A.7，根据资料条件采用生物量法、固碳速率法或净生态系统生产力法；洪水调蓄服务按照附录A.6完成标准核算，同时将城市生态覆盖增量雨洪效益作为补充分析并与标准核算结果分表；局部气候调节服务按照附录A.11，采用蒸散发能量或实测温差方法。固碳以生态斑块或清查单元组织，洪水调蓄以子汇水区组织，局部气候调节以绿地斑块及影响范围组织。")
    add_body(document, "第四，标准化价值量核算、验证与工具化交付。各项服务在明确实物量定义、单位、过程单元、价值参数、空间范围和独立验证资料后开展价值量计算。形成受约束的查询、统计、制图、复算和报告流程，使每项结果均能追溯到数据时间、空间范围、模型版本、参数版本、核算方法和不确定性说明。")

    add_heading(document, "4. 创新点")
    add_body(document, "第一，建立月度地理嵌入与生态服务核算的衔接方法。以统一的月度地表状态表征支撑生态系统类型、面积、覆盖比例和空间分配，同时保持生态过程模型、监测资料和标准核算口径的独立性，避免将嵌入直接等同于服务价值。")
    add_body(document, "第二，建立基础模型与专题过程模型协同方法。在同一空间分块和独立验证资料下，比较原始影像、玄女嵌入和传统过程模型对专题制图与服务核算的作用，确定嵌入在数据对齐、空间化和少样本制图中的实际贡献。")
    add_body(document, "第三，建立可追溯的核算辅助机制。将数据来源、质量信息、处理方法、模型版本、参数版本和结果记录纳入统一台账，支持成果复算、质量审查和年度更新。")

    add_heading(document, "5. 关键技术及技术路线")
    add_body(document, "关键技术一：多源遥感数据时空对齐与质量管理。统一空间参考、时间基准、质量标识和数据版本；对云、缺测、传感器差异和空间错位进行记录与处理，形成可用于生态服务核算的月度基础数据产品。")
    add_body(document, "关键技术二：月度地理嵌入与服务专题模型协同。以玄女嵌入作为可复用的空间状态证据，结合生态系统分类、覆盖比例、绿地、水体等专题模型，为生态斑块、子汇水区和绿地影响范围等过程单元提供空间化输入。")
    add_body(document, "关键技术三：标准化价值量核算与可追溯工具。依据GB/T 46869—2025形成实物量、价值量、参数和结果记录，建立数据、模型和参数冻结机制，支持结果复算和书面审查。")
    add_body(document, "技术路线分为五步：一是汇集多源遥感与生态辅助数据并完成质量控制和时空对齐；二是生成月度地表状态嵌入和生态状态专题产品；三是按服务类型构建生态斑块、子汇水区和绿地影响范围等过程单元；四是开展实物量核算、独立验证和不确定性分析；五是依据标准完成价值量计算，交付专题图、数据台账、试点报告和核算辅助工具。")

    add_heading(document, "二、项目成果、考核指标及成果转化分析")
    add_heading(document, "1. 项目预期成果")
    add_body(document, "数据类成果：形成目标试点区生态系统调节服务数据台账、经批准核算年度12期月度生态状态基础产品、质量记录和数据字典。")
    add_body(document, "模型与方法类成果：形成玄女月度地理嵌入与专题制图流程，固定二氧化碳、洪水调蓄和局部气候调节三类服务的专题模型、参数台账、核算流程和验证方案，以及完整GEP生态产品清单与数据缺口说明。")
    add_body(document, "软件与报告类成果：形成可追溯核算辅助工具原型、三类服务价值试点报告、专题图、附表、方法说明和复算说明。根据实施情况形成软件著作权、专利或论文等成果，具体数量由申报单位结合管理要求确定。")

    add_heading(document, "2. 考核指标")
    add_body(document, "完成经批准核算年度的12期月度生态状态基础产品，每期产品具有数据来源、时间、空间参考、处理版本和质量标识；形成数据、模型、参数和结果关联台账。")
    add_body(document, "完成固定二氧化碳、洪水调蓄和局部气候调节三类服务试点。每项服务明确实物量定义、单位、过程单元、价值参数、独立验证资料和核算范围；对分类任务报告总体精度、各类别精度、面积偏差和置信区间，对连续变量及水文过程按服务类型报告相应精度、偏差和不确定性。")
    add_body(document, "核算结果可在冻结的数据、参数和处理流程下重复复算；工具输出具有数据时间、空间范围、模型版本、参数版本、核算方法和不确定性字段。三类调节服务合计明确表述为服务价值试点合计，不替代完整GEP总值。")

    add_heading(document, "3. 成果转化分析与效益预期")
    add_body(document, "业务化方面，项目成果可服务于目标试点区生态空间监测、生态保护成效评估、年度更新和GEP核算辅助工作，为专题核算和成果审查提供数据与工具支撑。")
    add_body(document, "工程化方面，数据处理、月度嵌入、专题模型、参数管理、质量复核和报告工具可按模块部署，并与既有遥感、自然资源和生态环境数据平台衔接。跨区域应用前应重新核验生态类型、数据可得性、参数适用性和独立验证资料。")
    add_body(document, "项目将减少多源资料重复整理和人工核对工作，提高生态状态产品和服务核算成果的可追溯性，为后续完善生态产品清单、扩大服务种类和开展完整GEP核算提供工程基础。产业化和资本化不作为本项目的刚性考核内容，可在业务化和工程化验证成熟后结合实际需求推进。")

    add_heading(document, "三、现有工作基础与优势")
    add_heading(document, "1. 国内外发展现状与趋势")
    add_body(document, "地球观测基础模型正由单期影像分类向多源、多时相通用表征学习发展。Prithvi、SatMAE、SpectralGPT和AlphaEarth Foundations等研究表明，自监督预训练可生成密集地理嵌入，并在统一特征空间上支撑多种下游制图任务。当前发展重点包括多源协同、时间连续性、质量感知和跨区域适配。")
    add_body(document, "生态系统服务评估的主流方法包括SEEA-EA核算框架、InVEST、CASA等。我国GB/T 46869—2025明确了陆域生态产品价值核算的原则。遥感产品主要承担生态系统面积、植被状态、水体和土地覆盖等空间化任务，生态过程和价值量仍需结合专题模型、监测、统计资料和价值参数。未来技术将从年度静态估算向可更新的月度或季度生态状态产品演进。")
    add_body(document, "地理空间智能体正在与GIS、遥感和模型工具链结合。本项目采用受约束的查询与核算机制：用户查询只能调用已登记的数据、模型和参数，输出同步提供版本、范围、方法和不确定性，避免以自然语言生成替代实际核算。")

    add_heading(document, "2. 现有技术基础")
    add_body(document, "玄女项目已在海淀区完成多源月度地理嵌入训练与下游验证。现有生产模型使用2025年12月至2026年5月数据，在320个1,280米×1,280米样本单元上生成128×128×64的名义10米等效分析网格，输入包括Sentinel-1、Sentinel-2、Landsat、高分光学和高分SAR等数据。建筑、道路和水体结果使用与预训练开放地图弱语义同源的标签，属于内部制图验证，不作为生态服务或GEP核算精度依据。")
    add_body(document, "项目已形成多源遥感数据处理、月度嵌入生成、专题制图、数据版本管理、下游模型评测和多卡训练环境等基础。现有生产模型、训练配置、数据处理脚本和评测流程可以作为新区域研发起点。生态系统类型、气象、水文、监测、统计和价值参数资料将在项目启动阶段与相关数据责任方对接，形成数据取得清单、授权边界和时间安排。")
    add_body(document, "申报单位的既有生态评估、遥感地理信息、工程咨询、软件平台建设经历及项目团队、平台条件和参与人员分工，应由申报单位在正式申报时据实补充。本项目拟设置项目统筹、遥感与数据工程、生态核算、水文与气候专题模型、平台与软件、质量控制与独立复核等岗位。")

    add_heading(document, "四、研发组织与实施安排")
    add_body(document, "第一阶段：完成核算边界、生态产品清单、数据来源、质量规范、验证方案和数据责任方确认。第二阶段：完成月度生态状态基础产品、生态系统分类和三类服务专题模型开发。第三阶段：完成实物量与价值量试点、独立验证、不确定性分析、核算工具和示范报告。第四阶段：完成成果复核、交付固化和后续推广建议。")
    add_body(document, "项目启动后建立服务实物量、过程单元、校准资料、独立验证资料、参数来源和数据责任方的对应关系。每月产品登记有效像元率、数据源可用性、时相差、质量等级和降级原因；质量不满足核算要求的区域不进入价值量计算。各服务使用同口径传统基线和统一验证集开展对比，报告嵌入产品带来的差异、置信区间和消融结果，不预设必须优于基线的结论。")

    add_heading(document, "五、经费预算")
    add_body(document, "经费预算由申报单位依据项目周期、数据获取、外业监测、模型训练、软件开发、测试验证和第三方复核需求，按照单位科研经费管理办法核定。建议将经费与数据获取、生态辅助资料、专题模型开发、计算资源、外业或独立验证、软件工具、成果复核和交付等工作包对应管理。")

    add_heading(document, "六、参考依据")
    add_body(document, "1. GB/T 46869—2025《生态系统评估 陆域生态产品总值核算技术指南》。")
    add_body(document, "2. Ouyang Z. et al. Gross ecosystem product: concept, accounting framework and case study. Proceedings of the National Academy of Sciences, 2020.")
    add_body(document, "3. Potter C. S. et al. Terrestrial ecosystem production: A process model based on global satellite and surface data. Global Biogeochemical Cycles, 1993.")
    add_body(document, "4. 玄女项目生产模型卡、月度嵌入数据说明、下游评测报告和数据布局说明，作为项目前期技术基础材料留存。")

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.paragraph_format.space_after = Pt(0)
    run = footer.add_run(TITLE)
    set_run_font(run, Pt(9))
    document.save(OUTPUT)


if __name__ == "__main__":
    build_document()
