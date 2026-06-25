"""Generate the Xuannv capability case deck.

This is a follow-up deck after the first nine investor BP slides.  It keeps
the same visual language and focuses on concrete capability evidence from the
Harbin New Area case.
"""

from __future__ import annotations

from pathlib import Path

import fitz
from pptx import Presentation
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN

from create_front8_deck_v2 import C, bg, title, text, rect, line, picture_fit, claim


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "玄女科技BP_能力案例_哈尔滨新区_v0.1.pptx"
CLOUD = Path("/root/workspace/bp_ppt素材汇总/cloud_examples")
MULTI = Path("/root/workspace/bp_ppt素材汇总/multi_task_examples")
MANUAL_REPORT = Path("/root/workspace/哈尔滨新区城市管理卫星遥感监测报告")
MODEL_REPORT = Path("/root/workspace/xuannv/reports/harbin_202512_202605/pdfs")
TMP = Path("/tmp/xuannv_capability_deck_assets")


def add_section(slide, label: str, x: float, y: float, color=C.blue) -> None:
    text(slide, label, x, y, 2.30, 0.16, 9, color, True)
    line(slide, x, y + 0.28, x + 1.42, y + 0.28, color, 1.0)


def bullet(slide, head: str, body: str, x: float, y: float, color=C.blue, fill=C.pale_blue) -> None:
    rect(slide, x, y, 3.62, 0.78, C.white, C.line)
    dot = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL, Inches(x + 0.20), Inches(y + 0.26), Inches(0.26), Inches(0.26))
    dot.fill.solid()
    dot.fill.fore_color.rgb = fill
    dot.line.color.rgb = color
    dot.line.width = Pt(1.0)
    text(slide, head, x + 0.62, y + 0.14, 2.68, 0.18, 10, C.ink, True)
    text(slide, body, x + 0.62, y + 0.44, 2.70, 0.18, 8, C.body)


def metric(slide, value: str, label: str, x: float, y: float, color=C.blue, fill=C.pale_blue) -> None:
    rect(slide, x, y, 1.72, 0.78, fill, color)
    text(slide, value, x + 0.12, y + 0.13, 1.48, 0.22, 15, color, True, PP_ALIGN.CENTER)
    text(slide, label, x + 0.12, y + 0.47, 1.48, 0.14, 7, C.body, True, PP_ALIGN.CENTER)


def label(slide, value: str, x: float, y: float, w: float, color=C.body) -> None:
    text(slide, value, x, y, w, 0.13, 7, color, True, PP_ALIGN.CENTER)


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

    # 10. Capability case overview
    s = prs.slides.add_slide(blank)
    bg(s)
    title(s, "10", "能力验证：哈尔滨新区城市治理场景")
    text(
        s,
        "用一个真实区域验证玄女底座的三类能力：复杂观测条件下可用、多任务可复用、结果可进入政府交付。",
        1.03,
        1.18,
        11.20,
        0.28,
        13,
        C.body,
        False,
        PP_ALIGN.CENTER,
    )
    bullet(s, "复杂观测可用", "云遮挡、缺失模态下仍能保留变化信号", 0.86, 2.08, C.blue, C.pale_blue)
    bullet(s, "一套嵌入多任务", "施工工地、建筑、水体、耕地、地表分类共享同一底座", 0.86, 3.18, C.green, C.mint)
    bullet(s, "报告交付闭环", "从模型结果直接形成可汇报、可复核的监测报告", 0.86, 4.28, C.purple, C.pale_purple)
    picture_fit(s, cloud_imgs[1], 5.10, 1.96, 6.90, 1.48)
    label(s, "云遮挡场景下的变化识别", 5.10, 3.56, 6.90, C.body)
    picture_fit(s, multi_imgs["建筑提取"], 5.10, 3.92, 3.30, 1.58)
    picture_fit(s, model_cover, 8.70, 3.92, 3.10, 1.58)
    label(s, "多任务识别", 5.10, 5.63, 3.30, C.body)
    label(s, "自动化报告", 8.70, 5.63, 3.10, C.body)
    claim(s, "哈尔滨新区案例证明：地理嵌入不是单点模型，而是可进入业务流程的遥感智能底座。", 6.34, C.blue)

    # 11. Missing modality and cross-modal robustness
    s = prs.slides.add_slide(blank)
    bg(s)
    title(s, "11", "能力一：缺失模态与跨模态处理")
    add_section(s, "关键结论", 0.86, 1.42, C.blue)
    text(
        s,
        "遥感应用经常遇到云遮挡、时相不齐、模态缺失。玄女底座通过多源观测预训练，让嵌入在不完整观测下仍保留可用于变化检测的空间信号。",
        0.86,
        1.92,
        3.58,
        0.86,
        13,
        C.ink,
        True,
    )
    bullet(s, "有云不等于不可用", "云覆盖时仍能识别到潜在变化区域", 0.86, 3.20, C.blue, C.pale_blue)
    bullet(s, "跨模态补偿", "不同传感器信息在统一嵌入空间中互相补足", 0.86, 4.18, C.green, C.mint)
    bullet(s, "减少人工重跑", "不因单次影像质量不足而完全中断流程", 0.86, 5.16, C.purple, C.pale_purple)
    for i, img in enumerate(cloud_imgs):
        y = 1.62 + i * 1.58
        picture_fit(s, img, 5.18, y, 6.96, 1.18)
        label(s, f"云遮挡变化检测样例 {i + 1}", 5.18, y + 1.28, 6.96, C.body)
    claim(s, "复杂天气和缺失观测不再直接转化为业务停摆。", 6.48, C.blue)

    # 12. Multi-task reuse
    s = prs.slides.add_slide(blank)
    bg(s)
    title(s, "12", "能力二：一套嵌入支撑多类下游任务")
    text(
        s,
        "即使部分监督标签来自 2021 年 WorldCover 或旧建筑数据，统一地理嵌入仍能为不同任务提供稳定的空间语义线索。",
        1.02,
        1.18,
        11.20,
        0.30,
        13,
        C.body,
        False,
        PP_ALIGN.CENTER,
    )
    positions = [
        ("水体识别", 0.82, 1.86, 3.72, 1.62),
        ("建筑提取", 4.82, 1.86, 3.72, 1.62),
        ("地表分类", 8.82, 1.86, 3.72, 1.62),
        ("施工工地变化", 1.80, 4.16, 4.35, 1.16),
        ("耕地非农非粮", 7.18, 4.16, 4.35, 1.16),
    ]
    for name, x, y, w, h in positions:
        path = multi_imgs.get(name.replace("变化", "").replace("非农非粮", "变化"), None)
        if name == "施工工地变化":
            path = multi_imgs["施工工地"]
        elif name == "耕地非农非粮":
            path = multi_imgs["耕地变化"]
        else:
            path = multi_imgs[name]
        picture_fit(s, path, x, y, w, h)
        label(s, name, x, y + h + 0.12, w, C.body)
    metric(s, "5 类", "下游任务示例", 1.88, 5.88, C.blue, C.pale_blue)
    metric(s, "旧标签", "仍可迁移验证", 4.08, 5.88, C.green, C.mint)
    metric(s, "一次表征", "多任务复用", 6.28, 5.88, C.purple, C.pale_purple)
    text(s, "从“一个任务训练一个模型”转向“同一地理嵌入适配多个任务”。", 8.68, 6.06, 3.24, 0.20, 10, C.blue, True, PP_ALIGN.CENTER)

    # 13. Report delivery loop
    s = prs.slides.add_slide(blank)
    bg(s)
    title(s, "13", "能力三：从模型结果到政府监测报告")
    text(
        s,
        "哈尔滨新区的价值不只在模型精度，更在于把检测结果持续组织成政府可阅读、可复核、可归档的报告。",
        1.02,
        1.18,
        11.20,
        0.30,
        13,
        C.body,
        False,
        PP_ALIGN.CENTER,
    )
    text(s, "传统人工报告", 1.12, 1.82, 4.72, 0.26, 14, C.ink, True, PP_ALIGN.CENTER)
    picture_fit(s, manual_cover, 1.36, 2.22, 4.20, 3.28)
    text(s, "2025 年下半年更新", 1.44, 5.70, 4.04, 0.16, 9, C.body, True, PP_ALIGN.CENTER)
    text(s, "流程重、周期长，后续更新压力大", 1.44, 6.02, 4.04, 0.18, 9, C.body, True, PP_ALIGN.CENTER)
    line(s, 6.66, 2.04, 6.66, 6.15, C.line, 0.9)
    text(s, "玄女模型生成报告", 7.50, 1.82, 4.72, 0.26, 14, C.blue, True, PP_ALIGN.CENTER)
    picture_fit(s, model_cover, 7.74, 2.22, 4.20, 3.28)
    text(s, "已更新至 2026 年 5 月", 7.82, 5.70, 4.04, 0.16, 9, C.blue, True, PP_ALIGN.CENTER)
    text(s, "自动汇总多任务结果，显著降低人工整理压力", 7.82, 6.02, 4.04, 0.18, 9, C.body, True, PP_ALIGN.CENTER)
    metric(s, "6 份", "传统报告文件", 5.76, 2.34, C.amber, C.pale_amber)
    metric(s, f"{manual_pages} 页", "人工交付规模", 5.76, 3.28, C.amber, C.pale_amber)
    metric(s, "5 份", "模型报告文件", 5.76, 4.42, C.blue, C.pale_blue)
    metric(s, f"{model_pages} 页", "自动生成规模", 5.76, 5.36, C.blue, C.pale_blue)
    claim(s, "真正的产品化能力，是把遥感分析结果变成可持续交付的业务文档。", 6.48, C.blue)

    return prs


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs = build()
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
