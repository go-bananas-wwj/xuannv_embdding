"""Generate the Xuannv capability case deck.

This follow-up deck starts after the first nine investor BP slides and proves
the foundation model through three concrete cases: Harbin New Area, Haidian,
and Yajiang hydropower monitoring.
"""

from __future__ import annotations

from pathlib import Path

import fitz
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from create_front8_deck_v2 import C, bg, title, text, rect, line, arrow, picture, picture_crop, picture_fit, claim


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "玄女科技BP_能力案例_v0.2.pptx"

CLOUD = Path("/root/workspace/bp_ppt素材汇总/cloud_examples")
MULTI = Path("/root/workspace/bp_ppt素材汇总/multi_task_examples")
MANUAL_REPORT = Path("/root/workspace/哈尔滨新区城市管理卫星遥感监测报告")
MODEL_REPORT = Path("/root/workspace/xuannv/reports/harbin_202512_202605/pdfs")
HARBIN_REPORT_ROOT = Path("/root/workspace/xuannv/reports/harbin_202512_202605")
HAIDIAN = Path("/root/workspace/report/assets")
YAJIANG = ROOT / "assets" / "user_materials" / "yajiang_case"
YAJIANG_AGENT = Path("/root/.codex/attachments/071916f0-9ca0-4d8c-83f1-bcd404e7ea3c/e178b904e06c08479cd215ca230ff245.png")
REGION_OUTLINES = {
    "harbin": ROOT / "assets" / "generated" / "practice_region_harbin_whitefit.png",
    "haidian": ROOT / "assets" / "generated" / "practice_region_haidian_whitefit.png",
    "yajiang": ROOT / "assets" / "generated" / "practice_region_yajiang_whitefit.png",
}
TMP = Path("/tmp/xuannv_capability_deck_assets")
RED = RGBColor(220, 38, 38)
PALE_RED = RGBColor(254, 242, 242)


def section_label(slide, value: str, x: float, y: float, color=C.blue) -> None:
    text(slide, value, x, y, 2.5, 0.16, 9, color, True)
    line(slide, x, y + 0.28, x + 1.34, y + 0.28, color, 1.0)


def small_label(slide, value: str, x: float, y: float, w: float, color=C.body) -> None:
    text(slide, value, x, y, w, 0.13, 7, color, True, PP_ALIGN.CENTER)


def outline_callout(slide, label_text: str, x: float, y: float, w: float, h: float, color=RED) -> None:
    box = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    box.fill.background()
    box.line.color.rgb = color
    box.line.width = Pt(2.0)
    rect(slide, x, y - 0.22, 0.92, 0.22, PALE_RED, color)
    text(slide, label_text, x + 0.06, y - 0.155, 0.80, 0.08, 6, color, True, PP_ALIGN.CENTER)


def image_pill(slide, label_text: str, x: float, y: float, color=C.blue, fill=C.pale_blue) -> None:
    rect(slide, x, y, 0.70, 0.24, fill, color)
    text(slide, label_text, x + 0.07, y + 0.075, 0.56, 0.08, 6, color, True, PP_ALIGN.CENTER)


def bullet(slide, head: str, body: str, x: float, y: float, color=C.blue, fill=C.pale_blue, w: float = 3.66) -> None:
    rect(slide, x, y, w, 0.78, C.white, C.line)
    dot = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL, Inches(x + 0.20), Inches(y + 0.26), Inches(0.26), Inches(0.26))
    dot.fill.solid()
    dot.fill.fore_color.rgb = fill
    dot.line.color.rgb = color
    dot.line.width = Pt(1.0)
    text(slide, head, x + 0.62, y + 0.14, w - 0.92, 0.18, 10, C.ink, True)
    text(slide, body, x + 0.62, y + 0.44, w - 0.92, 0.18, 8, C.body)


def compact_bullet(slide, head: str, body: str, x: float, y: float, color=C.blue, fill=C.pale_blue, w: float = 3.30) -> None:
    rect(slide, x, y, w, 0.58, C.white, C.line)
    dot = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL, Inches(x + 0.18), Inches(y + 0.19), Inches(0.20), Inches(0.20))
    dot.fill.solid()
    dot.fill.fore_color.rgb = fill
    dot.line.color.rgb = color
    dot.line.width = Pt(0.9)
    text(slide, head, x + 0.50, y + 0.10, w - 0.72, 0.14, 9, C.ink, True)
    text(slide, body, x + 0.50, y + 0.34, w - 0.72, 0.12, 7, C.body)


def metric(slide, value: str, label: str, x: float, y: float, color=C.blue, fill=C.pale_blue, w: float = 1.72) -> None:
    rect(slide, x, y, w, 0.78, fill, color)
    text(slide, value, x + 0.10, y + 0.13, w - 0.20, 0.22, 15, color, True, PP_ALIGN.CENTER)
    text(slide, label, x + 0.10, y + 0.47, w - 0.20, 0.14, 7, C.body, True, PP_ALIGN.CENTER)


def case_card(slide, name: str, scene: str, need: str, image: Path, x: float, color, fill) -> None:
    text(slide, name, x + 0.20, 1.90, 3.25, 0.28, 17, C.ink, True, PP_ALIGN.CENTER)
    picture_fit(slide, image, x + 0.06, 2.26, 3.56, 2.42)
    line(slide, x + 0.32, 4.96, x + 3.36, 4.96, color, 1.1)
    rect(slide, x + 0.26, 5.22, 3.14, 0.48, fill, color)
    text(slide, scene, x + 0.44, 5.37, 2.78, 0.14, 10, color, True, PP_ALIGN.CENTER)
    text(slide, need, x + 0.26, 5.94, 3.14, 0.40, 12, C.ink, True, PP_ALIGN.CENTER)


def task_thumb(slide, label: str, path: Path, x: float, y: float, w: float, h: float) -> None:
    picture_fit(slide, path, x, y, w, h)
    small_label(slide, label, x, y + h + 0.08, w)


def render_pdf_page(pdf: Path, page: int, name: str, zoom: float = 1.55) -> Path:
    TMP.mkdir(parents=True, exist_ok=True)
    out = TMP / f"{name}_p{page + 1}.png"
    if out.exists():
        return out
    doc = fitz.open(pdf)
    pix = doc[page].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    pix.save(out)
    return out


def report_page_counts(folder: Path) -> int:
    total = 0
    for pdf in folder.glob("*.pdf"):
        with fitz.open(pdf) as doc:
            total += doc.page_count
    return total


def build() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    cloud_imgs = sorted(CLOUD.glob("*.png"))
    multi_imgs = {
        "水体识别": MULTI / "patch_000041_jrc_water_2025-04_detail.png",
        "建筑提取": MULTI / "patch_000354_building_extraction_2025-04_detail.png",
        "施工工地": MULTI / "patch_000217_construction_2025-08_vs_2025-09_detail.png",
        "耕地变化": MULTI / "patch_000217_land_conversion_2025-08_vs_2025-09_detail.png",
        "地表分类": MULTI / "patch_000339_dynamic_world_2025-04_detail.png",
    }
    manual_total = MANUAL_REPORT / "00-2025年下半年哈尔滨新区城市管理卫星遥感监测总报告.pdf"
    model_total = MODEL_REPORT / "00-2025年12月-2026年5月哈尔滨新区城市管理卫星遥感监测总报告.pdf"
    manual_cover = render_pdf_page(manual_total, 0, "manual_total")
    model_cover = render_pdf_page(model_total, 0, "model_total")
    manual_pages = report_page_counts(MANUAL_REPORT)
    model_pages = report_page_counts(MODEL_REPORT)
    harbin_map = HARBIN_REPORT_ROOT / "pdf_maps" / "all_categories_map.png"
    haidian_chart = HAIDIAN / "aef_vs_v1.0_v2.png"
    haidian_patch = HAIDIAN / "compare_haidian_construction_patch_000198.png"

    # 10. Overview
    s = prs.slides.add_slide(blank)
    bg(s)
    title(s, "10", "现有工程实践案例")
    text(s, "玄女底座已围绕政府城市治理、复杂城市变化评测与水电站周边监测，形成面向真实区域的工程实践。", 1.02, 1.18, 11.20, 0.34, 15, C.blue, True, PP_ALIGN.CENTER)
    case_card(s, "哈尔滨新区", "政府客户 · 城市治理", "多专题持续监测与城市管理报告更新", REGION_OUTLINES["harbin"], 0.86, C.blue, C.pale_blue)
    case_card(s, "海淀区", "城市核心区 · 任务评测", "复杂城市纹理下，对标国际地理嵌入能力", REGION_OUTLINES["haidian"], 4.82, C.green, C.mint)
    case_card(s, "雅江区域", "水电站周边 · 安全监测", "水电站周边态势监测与遥感分析报告", REGION_OUTLINES["yajiang"], 8.78, C.purple, C.pale_purple)

    # 11. Harbin cloud robustness
    s = prs.slides.add_slide(blank)
    bg(s)
    title(s, "11", "哈尔滨新区：有云遮挡，仍能识别变化区域")
    text(s, "政府客户的真实难题", 0.86, 1.24, 3.22, 0.26, 17, C.blue, True)
    line(s, 0.86, 1.66, 2.66, 1.66, C.blue, 1.1)
    text(s, "云、雾、雪遮挡关键地物，城市治理仍需要连续、稳定的变化检测结果。", 0.86, 1.90, 3.36, 0.40, 14, C.ink, True)
    compact_bullet(s, "缺失模态下仍可推理", "不依赖单张完美影像完成判断", 0.86, 2.58, C.blue, C.pale_blue, 3.30)
    compact_bullet(s, "跨模态保留变化信号", "RGB 与嵌入共同支撑变化检测", 0.86, 3.25, C.green, C.mint, 3.30)
    compact_bullet(s, "减少人工重跑", "影像质量波动不直接造成流程停摆", 0.86, 3.92, C.purple, C.pale_purple, 3.30)
    rect(s, 0.86, 4.78, 3.30, 1.28, C.pale_blue, C.blue)
    text(s, "为什么能识别？", 1.06, 4.96, 2.90, 0.14, 10, C.blue, True, PP_ALIGN.CENTER)
    text(s, "训练阶段融合多模态遥感数据；单一模态被云雾遮挡时，其他模态仍可补充约束。对比变化前后地理嵌入差异，即可识别真实变化区域。", 1.06, 5.23, 2.90, 0.42, 8, C.body, True, PP_ALIGN.CENTER)
    col_labels = ["变化前 RGB", "变化后 RGB", "变化前嵌入", "变化后嵌入", "变化概率"]
    for j, label_value in enumerate(col_labels):
        text(s, label_value, 4.88 + j * 1.504, 1.45, 1.42, 0.12, 7, C.muted, True, PP_ALIGN.CENTER)
    sample_names = ["样例一", "样例二", "样例三"]
    for i, img in enumerate(cloud_imgs):
        y = 1.70 + i * 1.60
        picture(s, img, 4.88, y, 7.52, 1.62)
        image_pill(s, sample_names[i], 4.12, y + 0.68, C.blue if i == 0 else (C.green if i == 1 else C.purple), C.pale_blue if i == 0 else (C.mint if i == 1 else C.pale_purple))
    outline_callout(s, "云雾遮挡", 6.46, 1.86, 1.18, 0.78, RED)
    outline_callout(s, "云雾遮挡", 4.98, 3.36, 1.32, 0.90, RED)
    outline_callout(s, "云雾遮挡", 4.98, 5.10, 1.30, 0.78, RED)
    claim(s, "玄女底座提高的是业务可用性：真实遥感数据不完美，但系统仍要稳定产出。", 6.84, C.blue)

    # 12. Harbin multi-task reuse
    s = prs.slides.add_slide(blank)
    bg(s)
    title(s, "12", "同一套地理嵌入，复用到多类城市治理任务")
    text(s, "使用 2021 年 WorldCover 和旧建筑物数据作为监督标签，统一地理嵌入仍能为不同任务提供稳定空间语义线索。", 1.02, 1.14, 11.20, 0.30, 13, C.body, False, PP_ALIGN.CENTER)
    positions = [
        ("水体识别", multi_imgs["水体识别"], 0.82, 1.82, 3.70, 1.62),
        ("建筑提取", multi_imgs["建筑提取"], 4.82, 1.82, 3.70, 1.62),
        ("地表分类", multi_imgs["地表分类"], 8.82, 1.82, 3.70, 1.62),
        ("施工工地变化", multi_imgs["施工工地"], 1.72, 4.08, 4.50, 1.12),
        ("耕地非农非粮", multi_imgs["耕地变化"], 7.12, 4.08, 4.50, 1.12),
    ]
    for name, path, x, y, w, h in positions:
        task_thumb(s, name, path, x, y, w, h)
    metric(s, "5 类", "下游任务示例", 1.58, 5.86, C.blue, C.pale_blue)
    metric(s, "旧标签", "仍可迁移验证", 3.88, 5.86, C.green, C.mint)
    metric(s, "一次表征", "多任务复用", 6.18, 5.86, C.purple, C.pale_purple)
    text(s, "新增任务不再等于重新做一套遥感工程。", 8.70, 6.04, 3.12, 0.20, 10, C.blue, True, PP_ALIGN.CENTER)

    # 13. Harbin report delivery
    s = prs.slides.add_slide(blank)
    bg(s)
    title(s, "13", "从检测结果到政府可用报告：已更新至 2026 年 5 月")
    text(s, "哈尔滨新区是政府客户场景。玄女调用模型自动生成最新监测报告，将多任务检测结果组织成可阅读、可复核、可归档的交付物。", 1.02, 1.14, 11.20, 0.30, 13, C.body, False, PP_ALIGN.CENTER)
    text(s, "传统人工报告", 0.94, 1.78, 3.72, 0.24, 13, C.ink, True, PP_ALIGN.CENTER)
    picture_fit(s, manual_cover, 1.12, 2.14, 3.40, 2.76)
    text(s, "2025 年下半年", 1.20, 5.04, 3.20, 0.16, 9, C.body, True, PP_ALIGN.CENTER)
    text(s, "正式完整，但制作耗时耗力，后续更新压力大", 1.06, 5.34, 3.48, 0.30, 8, C.body, True, PP_ALIGN.CENTER)
    line(s, 4.94, 1.88, 4.94, 5.86, C.line, 0.8)
    text(s, "玄女模型自动生成报告", 5.26, 1.78, 3.72, 0.24, 13, C.blue, True, PP_ALIGN.CENTER)
    picture_fit(s, model_cover, 5.44, 2.14, 3.40, 2.76)
    text(s, "2025 年 12 月 - 2026 年 5 月", 5.52, 5.04, 3.20, 0.16, 9, C.blue, True, PP_ALIGN.CENTER)
    text(s, "自动汇总多任务结果，降低制图、统计和撰写压力", 5.38, 5.34, 3.50, 0.30, 8, C.body, True, PP_ALIGN.CENTER)
    picture_fit(s, harbin_map, 9.38, 2.08, 2.84, 2.60)
    small_label(s, "2026 报告点位分布图", 9.38, 4.82, 2.84)
    metric(s, "6 份", "传统报告", 9.10, 5.42, C.amber, C.pale_amber, 1.34)
    metric(s, f"{manual_pages} 页", "人工规模", 10.62, 5.42, C.amber, C.pale_amber, 1.34)
    metric(s, "5 份", "模型报告", 9.10, 6.16, C.blue, C.pale_blue, 1.34)
    metric(s, f"{model_pages} 页", "自动生成", 10.62, 6.16, C.blue, C.pale_blue, 1.34)
    claim(s, "真正的产品化能力，是把遥感分析结果变成可持续交付的业务文档。", 6.86, C.blue)

    # 14. Haidian benchmark
    s = prs.slides.add_slide(blank)
    bg(s)
    title(s, "14", "海淀区 Benchmark：玄女 V1.0 与 AEF 官方嵌入正面对标")
    text(s, "用真实下游任务指标说话：海淀区 5-fold CV，对比玄女 V1.0 embedding 与 AEF 2025 官方年度 embedding。", 1.02, 1.14, 11.20, 0.30, 13, C.body, False, PP_ALIGN.CENTER)
    picture_fit(s, haidian_chart, 0.78, 1.78, 11.90, 2.94)
    metric(s, "0.8828", "建筑变化 AUC", 1.02, 5.22, C.blue, C.pale_blue)
    metric(s, "0.8876", "耕地变化 AUC", 3.10, 5.22, C.blue, C.pale_blue)
    metric(s, "0.8876", "垃圾堆放 AUC", 5.18, 5.22, C.blue, C.pale_blue)
    metric(s, "AEF 更强", "施工工地任务", 7.26, 5.22, C.amber, C.pale_amber)
    text(s, "坦诚结论：施工工地上 AEF 更强；建筑变化、耕地变化、垃圾堆放等变化类任务上，玄女 V1.0 更强。", 9.28, 5.34, 2.88, 0.40, 9, C.ink, True, PP_ALIGN.CENTER)
    text(s, "注：5-fold CV，海淀区下游任务评测；指标包括 AUC-ROC、F1-best、F1@0.5、mIoU。", 1.00, 6.18, 11.30, 0.16, 8, C.muted, False, PP_ALIGN.CENTER)
    claim(s, "玄女不是追随 AlphaEarth，而是在中国城市变化类任务上形成可量化的本土优势。", 6.58, C.blue)

    # 15. Haidian business meaning
    s = prs.slides.add_slide(blank)
    bg(s)
    title(s, "15", "本土数据和本土任务，会形成自己的模型优势")
    picture_fit(s, haidian_patch, 0.90, 1.34, 5.48, 2.86)
    small_label(s, "海淀下游任务样例：模型输出与标注对比", 0.90, 4.36, 5.48)
    bullet(s, "国际模型证明方向", "AlphaEarth/AEF 说明地理嵌入是全球前沿方向", 7.00, 1.54, C.blue, C.pale_blue, 4.82)
    bullet(s, "本土任务需要本土优化", "城市变化、建设活动、垃圾堆放具有明显区域业务定义", 7.00, 2.72, C.green, C.mint, 4.82)
    bullet(s, "差距指向迭代路线", "施工工地仍需补数据、补标签、补任务定义", 7.00, 3.90, C.amber, C.pale_amber, 4.82)
    line(s, 0.96, 5.16, 12.12, 5.16, C.line, 0.8)
    text(s, "为什么中国需要自己的遥感数据通用嵌入底座？", 1.10, 5.58, 4.00, 0.24, 14, C.ink, True)
    text(s, "不是只出于国产替代，而是因为本土数据、本土标签和本土业务目标会带来真实模型优势。", 5.36, 5.56, 6.48, 0.28, 12, C.blue, True)
    claim(s, "海淀 Benchmark 把“地理嵌入方向正确”推进到“本土底座有必要、有优势、有迭代空间”。", 6.58, C.blue)

    # 16. Yajiang agent
    s = prs.slides.add_slide(blank)
    bg(s)
    title(s, "16", "雅江区域：把地理嵌入接入水电站监测智能体")
    text(s, "雅江案例面向水电站监测：用户选择区域、任务和时间，用自然语言触发模型与下游任务，自动生成结构化遥感分析报告。", 1.02, 1.14, 11.20, 0.30, 13, C.body, False, PP_ALIGN.CENTER)
    picture_fit(s, YAJIANG_AGENT, 0.76, 1.70, 8.12, 4.58)
    bullet(s, "自然语言发起任务", "用户直接提出报告需求，而不是操作脚本", 9.20, 1.78, C.blue, C.pale_blue, 3.10)
    bullet(s, "模型与指标可追溯", "展示模型版本、精度、置信度、低置信比例", 9.20, 2.82, C.green, C.mint, 3.10)
    bullet(s, "报告可继续对话", "历史会话保留，可追加月份和修改任务", 9.20, 3.86, C.purple, C.pale_purple, 3.10)
    metric(s, "96.8%", "总体精度", 9.20, 5.12, C.blue, C.pale_blue, 1.40)
    metric(s, "96.3%", "平均置信度", 10.88, 5.12, C.green, C.mint, 1.40)
    claim(s, "玄女底座可以变成用户可直接使用的智能体产品界面，而不是停留在模型 API。", 6.58, C.blue)

    # 17. Yajiang downstream to report
    s = prs.slides.add_slide(blank)
    bg(s)
    title(s, "17", "从下游任务结果，到可阅读的遥感分析报告")
    text(s, "雅江水电站监测需要的不只是预测图，而是能持续生成指标、图表、风险提示和后续行动建议的工作流。", 1.02, 1.14, 11.20, 0.30, 13, C.body, False, PP_ALIGN.CENTER)
    y_tasks = [
        ("雅江区域", YAJIANG / "yajiang_area.png", 0.86, 1.74, 1.82, 1.64),
        ("地物分类", YAJIANG / "land_classification.png", 3.10, 1.74, 2.48, 1.64),
        ("空间聚类", YAJIANG / "spatial_cluster.png", 6.00, 1.74, 2.48, 1.64),
        ("高程回归", YAJIANG / "elevation_regression.png", 8.90, 1.74, 2.48, 1.64),
        ("变化检测", YAJIANG / "change_detection.png", 1.32, 4.10, 1.62, 1.56),
        ("嵌入检索", YAJIANG / "embedding_retrieval.png", 3.64, 4.10, 1.62, 1.56),
        ("坡度风险", YAJIANG / "slope_risk.png", 5.96, 4.10, 1.62, 1.56),
    ]
    for name, path, x, y, w, h in y_tasks:
        task_thumb(s, name, path, x, y, w, h)
    rect(s, 8.70, 4.18, 3.26, 1.42, C.pale_blue, C.blue)
    text(s, "智能体报告", 9.00, 4.48, 2.66, 0.22, 14, C.blue, True, PP_ALIGN.CENTER)
    text(s, "汇总预测图、置信度、误差图、关键指标与行动建议", 9.02, 4.92, 2.62, 0.30, 9, C.body, True, PP_ALIGN.CENTER)
    arrow(s, 7.76, 4.86, 8.52, 4.86, C.line, 1.2)
    claim(s, "玄女的最终产品形态：地理嵌入底座 + 下游任务库 + 报告智能体。", 6.58, C.blue)

    # 18. Product capability matrix
    s = prs.slides.add_slide(blank)
    bg(s)
    title(s, "18", "从三个案例抽象出玄女底座的四层产品能力")
    text(s, "案例不是孤立项目，而是在验证玄女可复制的产品架构。", 1.02, 1.14, 11.20, 0.30, 13, C.body, False, PP_ALIGN.CENTER)
    layers = [
        ("地理嵌入层", "多源、多时相、跨模态、缺失模态下可用", "哈尔滨云遮挡样例", C.blue, C.pale_blue),
        ("任务适配层", "分类、检测、变化监测、检索、风险识别共享底座", "哈尔滨多任务 / 雅江多任务", C.green, C.mint),
        ("评测对标层", "与 AEF 官方嵌入进行下游任务量化比较", "海淀 V1.0 Benchmark", C.amber, C.pale_amber),
        ("报告交付层", "生成政务报告或智能体报告，进入客户工作流", "哈尔滨 2026 报告 / 雅江智能体", C.purple, C.pale_purple),
    ]
    for i, (head, body, evidence, color, fill) in enumerate(layers):
        y = 1.78 + i * 1.02
        rect(s, 1.02, y, 11.30, 0.76, C.white, C.line)
        rect(s, 1.22, y + 0.18, 0.40, 0.40, fill, color)
        text(s, head, 1.88, y + 0.17, 1.74, 0.18, 12, color, True, PP_ALIGN.CENTER)
        text(s, body, 4.02, y + 0.17, 4.22, 0.20, 10, C.ink, True, PP_ALIGN.CENTER)
        text(s, evidence, 8.72, y + 0.17, 2.94, 0.20, 9, C.body, True, PP_ALIGN.CENTER)
    line(s, 1.72, 5.98, 11.64, 5.98, C.line, 0.8)
    text(s, "商业化承接", 1.62, 6.28, 2.00, 0.20, 12, C.ink, True, PP_ALIGN.CENTER)
    text(s, "按区域、按任务、按更新频率、按 API / 平台授权收费。", 3.92, 6.25, 6.70, 0.24, 12, C.blue, True, PP_ALIGN.CENTER)
    claim(s, "玄女不是遥感项目公司，而是在搭建可复用、可评测、可交付的地理智能基础设施。", 6.86, C.blue)

    return prs


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs = build()
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
