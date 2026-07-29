from __future__ import annotations

"""Build the revised narrative proposal with a technical route and real evidence figures."""

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs/projects/assets/xuannv_gep_transferable_20260729"
OUTPUT = ROOT / "docs/projects/xuannv_gep_application_revised_20260729_zh.docx"
TITLE = "基于自研嵌入场范式模型的生态系统生产总值智能核算关键技术研究"


def set_run_font(run, size: Pt, bold: bool = False) -> None:
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    run.font.size = size
    run.font.bold = bold
    run.font.color.rgb = None


def add_text(
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


def heading(document: Document, text: str) -> None:
    add_text(document, text, size=Pt(14), bold=True, first_indent=False, before=Pt(14), after=Pt(8))


def body(document: Document, text: str) -> None:
    add_text(document, text, size=Pt(12), first_indent=True)


def figure(document: Document, path: Path, caption: str, width: Cm) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(6)
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.add_run().add_picture(str(path), width=width)
    add_text(document, caption, size=Pt(10.5), center=True, first_indent=False, after=Pt(8))


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

    add_text(document, TITLE, size=Pt(14), bold=True, center=True, first_indent=False, after=Pt(24))
    add_text(document, "自立科研项目申报书（修订文字稿）", size=Pt(14), bold=True, center=True, first_indent=False, after=Pt(30))

    heading(document, "项目概述")
    body(document, "本项目拟构建以自主多源遥感数据为基础、以自研嵌入场范式模型为核心、以生态过程模型和独立监测为约束的生态系统生产总值智能核算技术体系。项目围绕数据底座、基础模型、通用解码与可信交互四个环节开展研发，形成全国尺度生态状态制图、调节服务动态反演、成果复核和自然语言辅助查询的工具链，并选择典型区域开展专题模型开发和独立验证。前期海淀区和哈尔滨新区的月度嵌入试验用于证明多源数据组织、密集嵌入生成和下游快速制图流程可行。")
    body(document, "项目将五项核心生态指标或服务指标作为GEP核算的基础产品和研究目标，其中NPP属于基础生产力变量；首期价值核算拟围绕固定二氧化碳、洪水调蓄和局部气候调节三类服务开展，并按照GB/T 46869—2025的适用服务口径、核算边界、实物量至价值量流程、参数来源和版本记录组织实施。三类服务试点合计不替代完整GEP总值。完整GEP仅在生态产品清单、实物量与价值量资料、独立验证和区域外生态产品价值扣减等条件满足后，另行开展汇总核算。")

    heading(document, "一、目标与研究任务")
    heading(document, "1. 项目研究的目的、必要性和可行性")
    body(document, "本项目面向地球观测基础模型与生态系统生产总值核算融合方向。当前遥感基础模型多服务于单一分类或检测任务，GEP核算仍主要依赖人工整理多源资料和静态过程模型，缺少统一、可更新的空间表征层。")
    body(document, "研究目的：研发自研嵌入场范式模型，形成多源遥感统一表征、生态状态制图、GEP关键指标专题反演和可信查询分析工具链。")
    body(document, "研究必要性：针对数据来源多、时相差异大、生态状态更新成本高、指标重复建模和成果复核困难等问题，以月度嵌入降低数据对齐与专题制图成本，并以过程模型、参数台账和独立观测保证核算可追溯。")
    body(document, "研究可行性：项目已具备海淀区月度嵌入生产验证、哈尔滨新区数据和双区域实验基础，并已稳定验证6卡Ascend 910B并行训练；现有16卡本地算力可作为扩展资源，但用于全国训练前须完成多卡通信、I/O和稳定性验收。全国数据底座建设将按数据授权、下载归档、离线预处理、训练、产品导出和区域验证分阶段实施；在本地算力基础上，拟申请济南或郑州超算等扩展资源。全国尺度两小时指标限定为冻结模型上的在线全指标推理目标，不包含原始数据下载、离线预处理和产品入库时间。")

    heading(document, "2. 项目目标及内容")
    heading(document, "（1）项目目标")
    body(document, "总体目标：建成面向生态系统生产总值智能核算的自主数据底座和自研嵌入场范式模型，形成“多源数据—月度嵌入—通用解码—生态核算—可信交互”的技术体系，支撑生态状态制图、关键服务指标动态反演和成果复核。")
    body(document, "目标一：建成自主数据底座。面向高分、资源、环境减灾、风云等自主卫星数据及可依法获取的多源遥感资料，建立覆盖典型生态系统的多源时空谱样本库、质量控制规则和生态过程参数台账；融合通量塔、水文站、样地等地面观测与成熟过程模型，构建GEP关键要素的训练、校准和独立验证样本。")
    body(document, "目标二：预训练自研嵌入场范式模型。模型目标参数规模约1亿，输出64维月度密集嵌入，支持光学、SAR、气象、地形等多模态信息的统一表征；通过时相、空间、模态掩码重建与跨模态一致性学习，形成可迁移、可复用的地表状态特征。模型规模、输入模态和训练范围将结合数据量、算力资源与消融试验结果确定。")
    body(document, "目标三：实现GEP关键指标的动态反演与验证。基于冻结嵌入和轻量化通用解码器，围绕NPP等基础生产力变量以及水源涵养、土壤保持、固碳释氧、生物多样性维护等服务指标开展专题模型开发、月尺度制图和独立验证；其中固定二氧化碳、洪水调蓄和局部气候调节作为首期价值核算试点，生态过程与价值量由服务专用过程模型、监测资料、统计资料和参数台账共同支撑。")
    body(document, "目标四：开发支持自然语言问答的交互平台。构建由知识图谱约束的可信决策智能体，将用户问题解析为已登记的数据、模型、参数和制图工具调用；输出指标、专题图和报告时同步返回数据来源、时间范围、空间范围、模型版本、参数版本、方法条件和不确定性。对缺失数据、超出适用范围或未完成核算的请求，平台应拒绝给出定量结论或降级为定性说明。")

    heading(document, "（2）研究内容")
    body(document, "内容一：自主多源卫星时空谱融合与GEP样本库构建。开展跨传感器辐射一致性处理、时空配准、云与缺失质量控制、月度合成和数据版本管理；整合自主卫星、可用多源遥感数据、通量塔、水文站、样地和生态过程模型，建立生态状态、服务实物量和价值参数的样本与台账体系。")
    body(document, "内容二：物理与知识先验约束的自研嵌入场范式模型构建。设计支持光学、SAR、高分辨率影像、气象和地形等输入的时空建模骨干；通过光谱、时相、空间和模态掩码重建，以及跨模态一致性学习获取通用地表表征；将质量信息、地形条件和生态规律作为软约束或辅助变量，提高表征对缺测、噪声和不同生态场景的鲁棒性。")
    body(document, "内容三：GEP指标的通用映射与动态反演。基于自研嵌入场范式模型的冻结嵌入，设计轻量化通用解码器和指标适配接口，开展生态状态专题制图、NPP等基础变量以及水源涵养、土壤保持、固碳释氧、生物多样性维护等服务指标的时序反演、空间统计和独立验证。固定二氧化碳、洪水调蓄和局部气候调节分别按照适配的生态斑块或清查单元、子汇水区、绿地影响范围组织过程核算；不同指标采用相应参数和验证资料，不以单一嵌入直接替代生态过程观测。")
    body(document, "内容四：融合知识图谱的可信智能决策平台。构建GEP核算逻辑、生态知识、政策规则、数据血缘、模型版本和参数关系的知识图谱；研发受约束的“问—算—答—视”智能体，支持自然语言查询、时空对比、异常分析、专题制图和报告生成，并通过工具调用记录和人工复核机制控制回答边界。")

    heading(document, "（3）创新点")
    body(document, "创新点一：构建面向生态核算的自研嵌入场范式模型。总体上，以连续、可复用的月度密集嵌入替代“一项任务一个模型”的重复建模方式；具体而言，统一组织光学、SAR、高分辨率影像、气象和地形等多源信息，以约1亿参数、64维月度密集嵌入表达地表状态，并通过下游轻量解码器快速适配不同生态专题和服务指标。")
    body(document, "创新点二：建立物理—数据—知识协同的自监督预训练机制。总体上，避免只依赖像素重建学习表面纹理；具体而言，联合时相、空间和模态掩码重建、跨模态一致性学习、质量信息以及地形和生态规律等辅助约束，提升嵌入对云雾、缺测、传感器差异和复杂生态环境的适应能力。")
    body(document, "创新点三：建立“冻结嵌入+通用解码器”的多指标动态反演机制。总体上，以同一嵌入场为多项服务提供共同空间证据；具体而言，针对不同指标配置轻量化解码器、过程单元、参数台账和独立验证资料，在保持基础模型可复用的同时，避免把不同生态过程混入单一黑盒模型。")
    body(document, "创新点四：建立知识图谱约束的可信决策智能体。总体上，将自然语言交互从自由问答转为可审计的工具调度；具体而言，以数据血缘、核算方法、模型版本、参数关系和适用条件构成知识图谱，驱动“问—算—答—视”闭环，并对无数据、超范围和未核算请求实施拒答或降级处理。")
    body(document, "创新点五：建立区域试验与跨区域验证相结合的工程验证路径。总体上，以海淀区和哈尔滨新区已有试验验证多源数据和下游流程的可运行性；具体而言，在新区域应用时重新进行数据检查、模型适配、专题验证和参数本地化，形成可复制而不简单外推的区域化技术规范。")

    heading(document, "（4）关键技术及技术路线")
    body(document, "关键技术一：自主多源卫星时空谱协同归一化与质量控制。面向高分、资源、环境减灾、风云等自主卫星以及可用多源数据，开展跨传感器辐射一致性、时空配准、月度合成、云和缺失质量管理，形成可追溯的数据底座。")
    body(document, "关键技术二：物理与知识先验约束的多模态嵌入场预训练。设计支持光学、SAR、气象、地形等多源输入的时空建模网络，采用多维掩码重建、跨模态一致性学习和质量感知机制，并将生态规律、地形条件和过程变量关系作为辅助约束，学习时空连续的通用地表状态表征。")
    body(document, "关键技术三：基于知识图谱约束的可信决策智能体构建。构建连接GEP核算逻辑、生态知识、政策规则、数据来源、模型版本和参数关系的知识图谱，使自然语言问题只能调用已登记的计算、制图和统计工具，输出结果具备可追溯字段和人工复核接口。")
    figure(document, ASSETS / "technical_route_20260729.png", "图1 自研嵌入场范式模型支撑GEP智能核算技术路线。", Cm(15.4))

    heading(document, "二、项目成果、考核指标及成果转化分析")
    heading(document, "1. 项目预期成果")
    body(document, "数据类成果。国产长时序无缝数据集：完成至少10颗国产卫星的协同处理，形成近10年、月尺度、30米分辨率的全国地表反射率无缝数据产品。GEP像素级真值样本库：构建百万级像素规模的核心生态指标，包括NPP、水源涵养、固碳等高置信度样本集。")
    body(document, "模型与算法类成果。自研嵌入场范式模型：形成参数规模约1亿、嵌入维度64的预训练模型权重与完整网络架构，具备光学、SAR、气象等多模态数据统一表征能力。GEP通用核算解码器套件：形成支持不少于5项核心生态指标或服务指标端到端反演的微调代码与预训练权重，并形成三类服务价值试点的过程模型、参数台账和验证说明。物理与知识约束预训练算法：形成融合光谱、时相、空间、模态掩码与生态知识软约束的自监督学习算法包。")
    body(document, "软件平台类成果。GEP智能核算与可信决策交互系统：形成一套B/S架构软件平台，支持受约束的自然语言驱动自动化核算、驱动力辅助归因与可视化报告生成，产出形式包括软件著作权及可部署的Docker镜像。")
    body(document, "知识产权与标准类成果。围绕嵌入场构建、物理与知识约束训练、通用解码器设计等核心技术申请1至2项专利；在地球观测、遥感、人工智能等领域发表SCI或EI论文1至2篇；编制《基于国产卫星数据的GEP智能核算技术指南》团体或行业标准草案1项。")
    heading(document, "2. 考核指标")
    body(document, "以下指标为项目拟定验收目标，不代表现有原型性能。立项后应在数据核查、样本量确认、独立验证集划分、基线方法和硬件环境确定后，形成书面验收方案并冻结统计口径。")
    body(document, "1. 自主可控数据底座。完成不少于10颗国产卫星，包括高分、资源、环境减灾、风云系列的协同归一化处理，生成近10年、月合成、30米空间分辨率的全国地表反射率无缝数据集；多星辐射一致性误差不高于3%，空间配准误差不高于0.5像元，云区重建结构相似度不低于0.85；构建覆盖5项以上GEP核心指标的像素级真值样本库，规模不低于100万像素，与地面实测值平均相对误差不高于15%。")
    body(document, "2. 自研嵌入场范式模型。模型参数量约1亿，嵌入维度64，支持光学、SAR、气象等不少于3种模态的统一语义表征；自监督预训练时空掩码重建峰值信噪比不低于30 dB，跨模态一致性对比学习准确率不低于80%；在下游GEP指标反演任务中，冻结嵌入场仅微调解码器即可达到全模型训练精度的90%以上，验证其通用性。")
    body(document, "3. GEP智能核算水平。实现NPP等基础生产力变量以及水源涵养、土壤保持、固碳释氧、生物多样性维护等5项核心生态指标或服务指标的月度自动核算，空间分辨率30米；对适用的连续专题变量，核算结果与独立野外观测值的判定系数R²不低于0.85，较传统基于MODIS或统计方法的相对精度提升不低于15%。固定二氧化碳采用年度增量、通量或长期固碳速率资料验证；洪水调蓄采用代表性子汇水区事件过程、年度水量闭合及相应过程指标验证；局部气候调节采用与核算路径一致的温湿度或蒸散发资料验证。三类服务在资料齐备并通过上述独立验证后，按照GB/T 46869—2025形成价值试点；全国尺度单次全指标在线推理时间不超过2小时，使用4卡GPU服务器。五项指标产品和三类服务试点均不直接等同于完整GEP总值。")
    body(document, "4. 可信决策交互平台。支持受约束的自然语言查询，从问答到可视化报告生成平均响应时间小于10秒；驱动力归因限定为基于登记模型与假设的辅助归因，不作为自动因果判定。平台可部署为Docker容器，提供标准API接口，支持并发用户数不少于50。验收使用不少于100条冻结问题集：拒答或降级正确率不低于95%，审计字段完整率不低于99%，数值结论工具调用记录完整率为100%，人工复算一致率不低于99%。")
    body(document, "5. 知识产权与标准。申请专利1至2项，发表论文1至2篇；获得软件著作权2项，编制团体或行业技术标准草案1项。")
    heading(document, "3. 项目成果转化分析")
    body(document, "业务化方面，项目成果可支撑生态空间监测、生态保护成效评估、生态补偿、自然资源管理和GEP核算辅助工作，为不同区域提供可更新的生态状态底图、服务专题结果和审查复算工具。")
    body(document, "工程化方面，将数据处理、嵌入生成、通用解码、生态过程核算、参数管理、质量复核和报告工具封装为可部署模块，与自然资源、生态环境、林草和水利等部门已有数据平台衔接。")
    body(document, "产业化和资本化方面，在完成业务化和工程化验证后，可面向绿色金融、生态补偿评估、自然资本核算和企业ESG服务等场景提供数据、模型和分析服务；本项目阶段以形成可验证、可交付的技术和软件基础为主。")

    heading(document, "三、现有工作基础与优势")
    heading(document, "1. 国内外发展现状与趋势")
    body(document, "地球观测基础模型方面，Prithvi、SatMAE、SpectralGPT、AlphaEarth Foundations等工作表明，基于大规模多时相遥感数据的自监督预训练能够学习可迁移表征，并服务于洪水、烧痕、土地覆盖和农业等多类任务。技术趋势正从单模态、任务专用模型走向光学、SAR、高光谱、气象等多模态协同，从单一分类检测走向连续的地理空间嵌入场，并强化时序、质量和跨区域泛化能力。")
    body(document, "生态系统生产总值核算方面，国际上常用SEEA-EA核算框架、InVEST、ARIES和CASA等方法；国内已形成生态产品价值核算相关标准与实践路径。现有核算普遍依赖遥感监测、地面统计、生态过程模型和价值参数，以年度或行政区单元为主。发展趋势是通过多源遥感和基础模型降低空间化与更新成本，但生态过程、监测资料和参数验证仍是核算结论的主体。")
    body(document, "遥感智能交互方面，大语言模型正在与GIS和遥感工具链结合。可信应用的关键不在于开放式对话，而在于以知识图谱、数据血缘和工具调用记录约束推理，使查询结果可追溯、可复核并符合数据适用条件。")
    heading(document, "2. 现有技术基础")
    body(document, "项目已形成玄女月度地理嵌入原型的区域试验基础。海淀区生产模型采用多源时序数据，在2025年12月至2026年5月、320个1,280米×1,280米样本单元上生成128×128×64的月度密集嵌入，输入包括Sentinel-1、Sentinel-2、Landsat、高分光学和高分SAR等数据。可部署原型约3,394万可训练参数；模型已形成多源数据对齐、掩码重建、弱语义辅助、嵌入导出和下游评测的完整工程链路。")
    figure(document, ROOT / "docs/presentations/assets/pujiang_202607/slide02/haidian_patch_000183_monthly_training_inputs.png", "图2 海淀区单个样本的2025年12月至2026年5月多源月度训练输入与质量掩膜示例。该图用于说明数据组织和质量控制流程，不代表全区统计结果。", Cm(15.2))
    figure(document, ASSETS / "haidian_embedding_pca_202604.png", "图3 海淀区2026年4月月度嵌入 PCA 可视化。颜色为64维嵌入前三主成分组合，用于展示地表状态的空间连续性和差异，不代表生态服务价值。", Cm(14.5))
    body(document, "哈尔滨新区已具备Sentinel-1、Sentinel-2、Landsat、WorldCover和部分高分光学数据与处理管线，并已形成训练权重和海淀—哈尔滨双区域实验配置。该基础用于检验不同地理环境下的数据组织、嵌入提取和专题制图流程可运行性，不应表述为已证明跨区域精度或迁移泛化。")
    figure(document, ROOT / "docs/presentations/assets/pujiang_202607/slide01/harbin_pca.png", "图4 哈尔滨新区2026年5月嵌入 PCA 可视化示例。该图用于说明不同区域的嵌入生成与空间结构展示能力，不代表跨区域精度、土地覆盖验收或生态服务价值结果。", Cm(14.0))
    body(document, "在海淀区建筑、道路和水体三类任务上，已形成以月度嵌入为特征、以轻量化读取器进行快速制图的工程流程。图5保留一组单Patch历史输出：自研原型使用P10C 64维月度嵌入，另列出AEF嵌入Pixel MLP、Sentinel-2随机森林和DINOv3-SAT493M特征的既有结果，用于直观展示不同技术路线可以接入同一类下游制图流程。由于各历史输出的读取器设置、特征时相和支持多边形数量并未统一冻结，图内F1和IoU仅为该Patch的局部核对值；不用于判断模型优劣、全区精度、少标注总体能力或跨区域泛化。该示例不属于5/10/50-shot标准下游评测；总体比较须采用固定划分、多fold或重复抽样及独立标签统计。")
    figure(document, ASSETS / "haidian_fewshot_model_comparison.png", "图5 海淀区单Patch稀疏多边形下游制图的历史输出示例。每行依次展示光学影像、P10C 64维月度嵌入投影、基于OSM弱语义流程选取的支持多边形、现有原型输出、AEF嵌入Pixel MLP、Sentinel-2随机森林和DINOv3-SAT493M特征输出；红色为预测目标，白色为背景。图内训练时相、特征版本、读取器及标注数以图中既有说明为准，未统一为标准5/10/50-shot协议。支持多边形与评价掩膜均来自OSM弱语义相关流程，不是独立人工真值；本图仅说明非GEP代理任务的工程流程可运行，不验证生态系统分类、生态服务实物量或价值量、GEP核算结果。", Cm(15.4))
    body(document, "海淀区和哈尔滨新区试验已形成数据预处理、月度嵌入生成、专题制图、数据版本管理、下游模型评测和多卡训练环境等基础。建筑、道路和水体等结果中部分使用与预训练开放地图弱语义同源的标签，属于内部可分性和流程验证，不作为生态分类、生态服务反演或GEP核算精度依据。正式项目将在全国数据底座建设基础上选择典型区域补充生态系统类型、气象、水文、土壤、地面监测、统计和价值参数资料，并以独立验证资料检验服务专题模型。")
    heading(document, "3. 组织与实施基础")
    body(document, "项目拟设置项目统筹、遥感与数据工程、基础模型研发、生态核算、水文与气候专题模型、知识图谱与智能体、平台软件、质量控制与独立复核等岗位。申报单位既有生态评估、遥感地理信息、工程咨询、软件平台建设经历及参与单位的具体项目成果、人员和平台条件，由申报单位在正式申报时据实补充。")

    heading(document, "四、拟投入研发人员与实施安排")
    body(document, "项目拟设置项目统筹、遥感与数据工程、基础模型研发、生态核算、水文与气候专题模型、知识图谱与智能体、平台软件、质量控制与独立复核等岗位。项目负责人、主要参加人员、管理人员的姓名、专业、投入比例和承担任务由申报单位在正式申报时据实填写。")
    body(document, "第一阶段：完成全国数据底座范围、典型区域、指标范围、数据来源、数据授权、质量规范、样本库、存储计划和验证方案确认。第二阶段：完成多源数据下载归档、离线预处理、自主数据底座建设和自研嵌入场范式模型预训练。第三阶段：完成通用解码器、关键服务专题模型、独立验证、参数台账和智能交互原型，并在冻结模型条件下测试在线推理性能。第四阶段：完成典型区域示范、成果复核、软件交付、技术规范和后续推广建议。")
    body(document, "项目实施中建立数据、样本、模型、参数、工具调用和结果的版本化管理机制。对云雾、缺测、传感器差异、粗分辨率辅助变量和不满足核算要求的区域进行显式质量标记；不以插补结果冒充观测，不以嵌入直接替代生态过程变量，不在未完成核算时由交互平台给出定量结论。")

    heading(document, "五、经费预算")
    body(document, "经费预算由申报单位根据自主卫星及辅助数据授权与获取、数据归档和PB级存储、地面观测和样本整理、16卡本地训练与超算扩展算力、软件开发、测试验证、典型区域示范和第三方复核等工作量据实核定。建议按数据底座、离线预处理、模型训练、专题核算、平台开发、独立验证和成果交付等工作包进行预算与绩效管理，并将全国数据授权、超算资源申请和存储容量作为立项后的前置条件。")

    heading(document, "六、参考依据")
    body(document, "1. GB/T 46869—2025《生态系统评估 陆域生态产品总值核算技术指南》。")
    body(document, "2. Ouyang Z. et al. Gross ecosystem product: concept, accounting framework and case study. Proceedings of the National Academy of Sciences, 2020.")
    body(document, "3. Potter C. S. et al. Terrestrial ecosystem production: A process model based on global satellite and surface data. Global Biogeochemical Cycles, 1993.")
    body(document, "4. 玄女项目生产模型卡、月度嵌入数据说明、下游评测报告和数据布局说明，作为项目前期技术基础材料留存。")

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.paragraph_format.space_after = Pt(0)
    run = footer.add_run(TITLE)
    set_run_font(run, Pt(9))
    document.save(OUTPUT)


if __name__ == "__main__":
    build_document()
