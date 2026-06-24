"""Generate the revised first eight Xuannv BP slides.

This deck is a stage-presentation draft, not a document handout.  It avoids
internal notes, keeps each slide to one claim, and favors real project images
plus clean editable diagrams.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import MSO_AUTO_SIZE, PP_ALIGN, MSO_VERTICAL_ANCHOR
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
USER = ASSETS / "user_materials"
CACHE = ASSETS / "render_cache"
OUT = ROOT / "outputs" / "玄女科技BP_前8页视觉返工_v0.2.pptx"


class C:
    bg = RGBColor(255, 255, 255)
    off = RGBColor(248, 250, 252)
    ink = RGBColor(15, 23, 42)
    body = RGBColor(51, 65, 85)
    muted = RGBColor(100, 116, 139)
    line = RGBColor(226, 232, 240)
    blue = RGBColor(37, 99, 235)
    sky = RGBColor(14, 165, 233)
    green = RGBColor(16, 185, 129)
    mint = RGBColor(236, 253, 245)
    pale_blue = RGBColor(239, 246, 255)
    purple = RGBColor(124, 58, 237)
    pale_purple = RGBColor(245, 243, 255)
    amber = RGBColor(245, 158, 11)
    pale_amber = RGBColor(255, 251, 235)
    dark = RGBColor(2, 6, 23)
    white = RGBColor(255, 255, 255)


FONT = "Noto Sans SC"

IMG = {
    "cover": ASSETS / "geo_embedding_cover_white_v2.png",
    "s2": USER / "patch_000339_s2_2025-04.png",
    "s2hr": USER / "patch_000339_s2_hr_2025-04.png",
    "s1": USER / "patch_000339_s1_2025-04.png",
    "landsat": USER / "patch_000339_landsat_2025-04.png",
    "dem": USER / "patch_000339_dem.png",
    "worldcover": USER / "patch_000339_worldcover.png",
    "dynamic": USER / "patch_000339_dynamic_world_2025-04_detail.png",
    "construction": USER / "patch_000217_construction_2025-08_vs_2025-09_detail.png",
    "land": USER / "patch_000217_land_conversion_2025-08_vs_2025-09_detail.png",
    "building": USER / "patch_000354_building_extraction_2025-04_detail.png",
    "cloud": USER / "patch_000146_2025-06_vs_2025-08_detail.png",
    "semantic": USER / "harbin_semantic_preset_frame.png",
    "migration": USER / "harbin_migration_frame.png",
    "persona_gov": ASSETS / "persona_government_v2.png",
    "persona_biz": ASSETS / "persona_enterprise_v2.png",
    "persona_uni": ASSETS / "persona_university_v2.png",
}


def bg(slide, color=C.bg) -> None:
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def text(
    slide,
    value: str,
    x: float,
    y: float,
    w: float,
    h: float,
    size: int = 14,
    color=C.ink,
    bold: bool = False,
    align=PP_ALIGN.LEFT,
):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.clear()
    frame.word_wrap = True
    frame.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    frame.vertical_anchor = MSO_VERTICAL_ANCHOR.MIDDLE
    frame.margin_left = 0
    frame.margin_right = 0
    frame.margin_top = 0
    frame.margin_bottom = 0
    p = frame.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = value
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return box


def rect(slide, x, y, w, h, fill=C.white, line=C.line, rounded=True):
    kind = MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if rounded else MSO_AUTO_SHAPE_TYPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(0.8)
    return shape


def line(slide, x1, y1, x2, y2, color=C.line, width=1.2):
    shape = slide.shapes.add_connector(1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    shape.line.color.rgb = color
    shape.line.width = Pt(width)
    return shape


def picture(slide, path: Path, x, y, w, h):
    if path.exists():
        return slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w), height=Inches(h))
    rect(slide, x, y, w, h, C.off)
    text(slide, "素材待补", x, y, w, h, 11, C.muted, True, PP_ALIGN.CENTER)
    return None


def crop_to_ratio(path: Path, ratio: float) -> Path:
    """Return a center-cropped copy that matches the target aspect ratio."""
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{path.stem}_{int(ratio * 1000)}.png"
    if out.exists():
        return out
    if not path.exists():
        return path
    im = Image.open(path).convert("RGB")
    iw, ih = im.size
    current = iw / ih
    if current > ratio:
        nw = int(ih * ratio)
        left = (iw - nw) // 2
        box = (left, 0, left + nw, ih)
    else:
        nh = int(iw / ratio)
        top = (ih - nh) // 2
        box = (0, top, iw, top + nh)
    im.crop(box).save(out)
    return out


def picture_crop(slide, path: Path, x, y, w, h):
    return picture(slide, crop_to_ratio(path, w / h), x, y, w, h)


def title(slide, no: str, heading: str, sub: str | None = None) -> None:
    text(slide, no, 0.62, 0.42, 0.42, 0.22, 8, C.sky, True)
    text(slide, heading, 1.04, 0.28, 11.4, 0.62, 24, C.ink, True)
    if sub:
        text(slide, sub, 1.06, 0.93, 10.6, 0.28, 10, C.muted)


def chip(slide, label: str, x, y, w, color=C.blue, fill=C.pale_blue) -> None:
    rect(slide, x, y, w, 0.34, fill, color)
    text(slide, label, x + 0.05, y + 0.02, w - 0.1, 0.30, 8, color, True, PP_ALIGN.CENTER)


def claim(slide, value: str, y=6.05, color=C.ink) -> None:
    text(slide, value, 0.86, y, 11.6, 0.42, 18, color, True, PP_ALIGN.CENTER)


def arrow(slide, x1, y1, x2, y2, color=C.line, width=1.4) -> None:
    shape = line(slide, x1, y1, x2, y2, color, width)
    shape.line.end_arrowhead = True


def image_card(slide, label: str, path: Path, x, y, w, h, color=C.line) -> None:
    rect(slide, x, y, w, h, C.white, color)
    picture_crop(slide, path, x + 0.06, y + 0.06, w - 0.12, h - 0.44)
    text(slide, label, x + 0.08, y + h - 0.30, w - 0.16, 0.20, 8, C.body, True, PP_ALIGN.CENTER)


def icon_database(slide, x, y, color=C.green) -> None:
    for i in range(3):
        shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.CAN, Inches(x), Inches(y + i * 0.13), Inches(0.36), Inches(0.34))
        shape.fill.solid()
        shape.fill.fore_color.rgb = C.white
        shape.line.color.rgb = color
        shape.line.width = Pt(1.0)


def icon_layers(slide, x, y, color=C.purple) -> None:
    for i in range(3):
        rect(slide, x + i * 0.08, y + i * 0.10, 0.46, 0.26, C.white, color, rounded=False)


def icon_loop(slide, x, y, color=C.blue) -> None:
    arc = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ARC, Inches(x), Inches(y), Inches(0.58), Inches(0.58))
    arc.line.color.rgb = color
    arc.line.width = Pt(2.0)
    text(slide, "↻", x + 0.10, y + 0.07, 0.38, 0.30, 17, color, True, PP_ALIGN.CENTER)


def pain_row(slide, icon_fn, head: str, body: str, x, y, color, fill) -> None:
    rect(slide, x, y, 5.05, 0.92, fill, color)
    icon_fn(slide, x + 0.22, y + 0.20, color)
    text(slide, head, x + 0.86, y + 0.16, 1.45, 0.20, 11, C.ink, True)
    text(slide, body, x + 2.05, y + 0.14, 2.78, 0.46, 9, C.body)


def flat_avatar(slide, x, y, role: str, color, accent) -> None:
    rect(slide, x, y, 3.55, 3.0, C.white, C.line)
    # soft background
    circle = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL, Inches(x + 0.74), Inches(y + 0.28), Inches(2.05), Inches(2.05))
    circle.fill.solid()
    circle.fill.fore_color.rgb = color
    circle.fill.transparency = 18
    circle.line.fill.background()
    # head and body
    head = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL, Inches(x + 1.43), Inches(y + 0.62), Inches(0.66), Inches(0.66))
    head.fill.solid()
    head.fill.fore_color.rgb = RGBColor(248, 214, 190)
    head.line.fill.background()
    body = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, Inches(x + 1.12), Inches(y + 1.36), Inches(1.28), Inches(0.94))
    body.fill.solid()
    body.fill.fore_color.rgb = accent
    body.line.fill.background()
    # role props: badge / laptop / book
    if role == "gov":
        rect(slide, x + 1.50, y + 1.55, 0.24, 0.30, C.white, C.white)
        text(slide, "证", x + 1.50, y + 1.56, 0.24, 0.16, 6, accent, True, PP_ALIGN.CENTER)
        rect(slide, x + 2.28, y + 1.75, 0.50, 0.62, C.white, C.line)
        line(slide, x + 2.36, y + 1.94, x + 2.70, y + 1.94, C.line, 1)
    elif role == "biz":
        rect(slide, x + 2.28, y + 1.72, 0.62, 0.42, C.white, C.line)
        line(slide, x + 2.25, y + 2.18, x + 2.93, y + 2.18, C.line, 1.2)
        rect(slide, x + 0.64, y + 1.88, 0.48, 0.30, C.white, C.line)
    else:
        rect(slide, x + 2.25, y + 1.72, 0.56, 0.44, C.white, C.line)
        line(slide, x + 2.53, y + 1.72, x + 2.53, y + 2.16, C.line, 1)
        text(slide, "()", x + 1.37, y + 0.82, 0.80, 0.18, 11, C.ink, True, PP_ALIGN.CENTER)


def persona_card(slide, path: Path, x, y, w, h, fill, color) -> None:
    rect(slide, x, y, w, h, fill, color)
    picture_crop(slide, path, x + 0.42, y + 0.22, w - 0.84, h - 0.28)


def metric(slide, number: str, label: str, x, y, w, color, fill) -> None:
    rect(slide, x, y, w, 1.0, fill, color)
    text(slide, number, x + 0.18, y + 0.13, w - 0.36, 0.30, 18, color, True, PP_ALIGN.CENTER)
    text(slide, label, x + 0.16, y + 0.58, w - 0.32, 0.22, 8, C.body, align=PP_ALIGN.CENTER)


def build() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # 1. Cover
    s = prs.slides.add_slide(blank)
    bg(s)
    picture(s, IMG["cover"], 0, 0, 13.333, 7.5)
    text(s, "玄女科技", 0.82, 1.42, 4.5, 0.42, 24, C.ink, True)
    text(s, "用地理嵌入\n赋能遥感智能应用", 0.82, 2.08, 5.45, 1.15, 31, C.ink, True)
    text(s, "地球观测 - 一次表征 - 多任务复用", 0.86, 3.64, 5.4, 0.32, 15, C.body)
    text(s, "做中国的遥感数据通用嵌入底座", 0.86, 4.10, 5.4, 0.30, 13, C.blue, True)

    # 2. Company intro
    s = prs.slides.add_slide(blank)
    bg(s)
    title(s, "02", "企业简介")
    text(s, "玄女科技是一家面向地球空间智能的人工智能基础底座公司。", 1.04, 1.16, 8.7, 0.36, 17, C.ink, True)
    text(s, "面向政府、遥感企业和行业客户，把遥感项目制交付沉淀成可复用的地理智能能力。", 1.04, 1.62, 10.8, 0.42, 12, C.body)
    rect(s, 0.98, 2.46, 11.35, 3.15, C.white, C.line)
    chip(s, "多源观测", 1.34, 2.78, 1.10, C.blue, C.pale_blue)
    chip(s, "地理嵌入", 5.72, 2.78, 1.10, C.green, C.mint)
    chip(s, "下游任务", 9.72, 2.78, 1.10, C.purple, C.pale_purple)
    for i, (label, key) in enumerate([("哨兵二号", "s2"), ("雷达影像", "s1"), ("陆地卫星", "landsat"), ("高程数据", "dem")]):
        image_card(s, label, IMG[key], 1.20 + (i % 2) * 1.26, 3.20 + (i // 2) * 1.03, 1.08, 0.90, C.line)
    arrow(s, 3.72, 4.02, 5.10, 4.02, C.line, 1.7)
    picture_crop(s, IMG["semantic"], 5.06, 3.18, 2.86, 2.08)
    arrow(s, 7.90, 4.02, 9.16, 4.02, C.line, 1.7)
    for i, (label, key) in enumerate([("施工变化", "construction"), ("建筑提取", "building")]):
        image_card(s, label, IMG[key], 9.10, 3.08 + i * 1.08, 2.86, 0.90, [C.blue, C.purple][i])
    claim(s, "不是接一个项目做一个模型，而是沉淀一套可复用的地理智能基础设施。", 6.22, C.blue)

    # 3. Same place, multiple observations
    s = prs.slides.add_slide(blank)
    bg(s)
    title(s, "03", "卫星越多，理解地球的方式反而越碎片化")
    rect(s, 0.92, 1.30, 6.10, 4.62, C.white, C.line)
    text(s, "同一地理位置", 3.03, 1.58, 1.72, 0.22, 11, C.ink, True, PP_ALIGN.CENTER)
    pin = slide_pin = slide = s
    center = slide_pin.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL, Inches(3.28), Inches(2.12), Inches(1.28), Inches(1.28))
    center.fill.solid()
    center.fill.fore_color.rgb = C.mint
    center.line.color.rgb = C.green
    text(s, "同一\n地块", 3.38, 2.38, 1.08, 0.34, 12, C.green, True, PP_ALIGN.CENTER)
    sources = [
        ("哨兵二号", "s2", 1.20, 1.80),
        ("高分光学", "s2hr", 5.02, 1.80),
        ("雷达卫星", "s1", 1.20, 4.15),
        ("陆地卫星", "landsat", 5.02, 4.15),
    ]
    for label, key, x, y in sources:
        image_card(s, label, IMG[key], x, y, 1.45, 1.20, C.line)
        arrow(s, x + 0.74, y + 0.60, 3.92, 2.76, C.line, 1.1)
    pain_row(s, icon_loop, "重复造轮子", "每个任务都重新找数据、清洗、标注、训练。", 7.62, 1.54, C.blue, C.pale_blue)
    pain_row(s, icon_database, "数据孤岛", "光学、雷达、高程、气象各自成体系，难复用。", 7.62, 2.88, C.green, C.mint)
    pain_row(s, icon_layers, "结果难复用", "一个项目一个结果，难沉淀成下次可调用的资产。", 7.62, 4.22, C.purple, C.pale_purple)
    claim(s, "真正缺的不是更多影像，而是把同一块地连续、统一理解的能力。", 6.20, C.ink)

    # 4. Benchmark and timing
    s = prs.slides.add_slide(blank)
    bg(s)
    title(s, "04", "谷歌已经验证地理嵌入方向，中国需要本土版本")
    rect(s, 0.94, 1.30, 5.55, 4.65, C.off, C.line)
    text(s, "谷歌卫星嵌入", 1.26, 1.72, 4.70, 0.28, 16, C.ink, True, PP_ALIGN.CENTER)
    metric(s, "10 米", "全球年度嵌入", 1.30, 2.38, 1.28, C.blue, C.pale_blue)
    metric(s, "64 维", "机器可调用表征", 2.80, 2.38, 1.28, C.green, C.mint)
    metric(s, "八年", "连续年度覆盖", 4.30, 2.38, 1.38, C.purple, C.pale_purple)
    text(s, "海外头部公司已将卫星影像从图像资产升级为机器可调用的基础表征。", 1.26, 4.05, 4.80, 0.42, 13, C.body, True, PP_ALIGN.CENTER)
    rect(s, 6.84, 1.30, 5.55, 4.65, C.white, C.line)
    text(s, "玄女的中国路径", 7.16, 1.72, 4.70, 0.28, 16, C.ink, True, PP_ALIGN.CENTER)
    metric(s, "11 天", "支持高频巡查", 7.20, 2.38, 1.28, C.sky, C.pale_blue)
    metric(s, "高分", "看清城市小目标", 8.70, 2.38, 1.28, C.green, C.mint)
    metric(s, "国产算力", "可进入本土部署", 10.20, 2.38, 1.38, C.purple, C.pale_purple)
    text(s, "中国客户需要本土数据、本土算力、本土业务流程里的地理智能底座。", 7.16, 4.05, 4.80, 0.46, 13, C.body, True, PP_ALIGN.CENTER)
    claim(s, "商业航天解决“看见地球”，地理嵌入解决“理解地球”。", 6.25, C.blue)

    # 5. Commercial space window
    s = prs.slides.add_slide(blank)
    bg(s)
    title(s, "05", "商业航天把数据推向爆发，遥感应用还卡在重流程")
    steps = [("制造", C.pale_blue, C.blue), ("发射", C.pale_blue, C.sky), ("组网", C.mint, C.green), ("下行", C.mint, C.green), ("处理", C.pale_amber, C.amber), ("应用", C.pale_purple, C.purple)]
    for i, (label, fill, color) in enumerate(steps):
        x = 0.92 + i * 1.88
        rect(s, x, 1.78, 1.26, 0.72, fill, color)
        text(s, label, x, 1.95, 1.26, 0.20, 11, color, True, PP_ALIGN.CENTER)
        if i < len(steps) - 1:
            arrow(s, x + 1.28, 2.14, x + 1.72, 2.14, C.line, 1.1)
    rect(s, 8.48, 2.84, 3.58, 0.96, RGBColor(255, 247, 237), C.amber)
    text(s, "真正卡住商业化的是应用层", 8.78, 3.05, 2.98, 0.22, 12, C.amber, True, PP_ALIGN.CENTER)
    text(s, "处理、标注、建模、核查仍重度项目制", 8.78, 3.42, 2.98, 0.20, 9, C.body, True, PP_ALIGN.CENTER)
    picture_crop(s, IMG["migration"], 0.92, 3.18, 3.15, 2.08)
    picture_crop(s, IMG["dynamic"], 4.30, 3.18, 3.15, 2.08)
    picture_crop(s, IMG["cloud"], 8.14, 4.26, 3.20, 1.08)
    text(s, "新区级连续监测", 0.92, 5.42, 3.15, 0.20, 9, C.body, True, PP_ALIGN.CENTER)
    text(s, "多任务结果复用", 4.30, 5.42, 3.15, 0.20, 9, C.body, True, PP_ALIGN.CENTER)
    text(s, "云遮挡与缺失模态", 8.14, 5.50, 3.20, 0.20, 9, C.body, True, PP_ALIGN.CENTER)
    claim(s, "窗口期的核心机会：更低成本、更快交付、更高频更新。", 6.22, C.ink)

    # 6. Three industry frictions
    s = prs.slides.add_slide(blank)
    bg(s)
    title(s, "06", "遥感应用没有大规模爆发：用不起、用不好、用不快")
    for i, (head, body, color, fill) in enumerate(
        [
            ("用不起", "项目制、专家制、标注贵，任务越多成本越高。", C.blue, C.pale_blue),
            ("用不好", "换区域、换地物、换传感器，模型泛化就变差。", C.green, C.mint),
            ("用不快", "从需求到交付链条长，结果常常错过决策窗口。", C.purple, C.pale_purple),
        ]
    ):
        x = 0.94 + i * 4.13
        rect(s, x, 1.50, 3.45, 1.34, fill, color)
        text(s, head, x + 0.22, 1.77, 3.02, 0.28, 18, color, True, PP_ALIGN.CENTER)
        text(s, body, x + 0.28, 2.24, 2.90, 0.32, 9, C.body, align=PP_ALIGN.CENTER)
    text(s, "传统流程", 1.08, 3.65, 1.10, 0.22, 11, C.muted, True)
    old = ["采购", "预处理", "标注", "训练", "核查", "报告"]
    for i, label in enumerate(old):
        x = 2.06 + i * 0.98
        rect(s, x, 3.50, 0.74, 0.48, C.off, C.line)
        text(s, label, x, 3.62, 0.74, 0.16, 8, C.body, True, PP_ALIGN.CENTER)
        if i < len(old) - 1:
            arrow(s, x + 0.76, 3.74, x + 0.92, 3.74, C.line, 0.9)
    text(s, "玄女流程", 1.08, 4.80, 1.10, 0.22, 11, C.green, True)
    new = ["地理嵌入", "任务适配", "候选图斑", "证据包"]
    for i, label in enumerate(new):
        x = 2.06 + i * 1.28
        rect(s, x, 4.65, 1.00, 0.48, C.mint, C.green)
        text(s, label, x, 4.77, 1.00, 0.16, 8, C.green, True, PP_ALIGN.CENTER)
        if i < len(new) - 1:
            arrow(s, x + 1.02, 4.89, x + 1.22, 4.89, C.green, 1.0)
    picture_crop(s, IMG["construction"], 8.28, 3.32, 3.58, 1.04)
    picture_crop(s, IMG["land"], 8.28, 4.66, 3.58, 1.04)
    claim(s, "玄女要做的，是把一次性项目流程变成可复用的地理智能流水线。", 6.18, C.ink)

    # 7. Angel users
    s = prs.slides.add_slide(blank)
    bg(s)
    title(s, "07", "天使用户：政府、企业、高校，痛点都指向同一件事")
    users = [
        ("gov", "政府部门", "变化图斑太多", "需要证据链、优先级、核查闭环", C.pale_blue, C.blue),
        ("biz", "遥感企业", "项目越多越重", "需要复用底座、少样本适配、降低人力", C.mint, C.green),
        ("edu", "高校团队", "数据工程太重", "需要现成嵌入、预标注、难例检索", C.pale_purple, C.purple),
    ]
    for i, (role, name, pain, value, fill, color) in enumerate(users):
        x = 0.90 + i * 4.16
        path = {"gov": IMG["persona_gov"], "biz": IMG["persona_biz"], "edu": IMG["persona_uni"]}[role]
        persona_card(s, path, x, 1.40, 3.55, 3.0, fill, color)
        text(s, name, x + 0.16, 4.58, 3.23, 0.24, 16, C.ink, True, PP_ALIGN.CENTER)
        text(s, pain, x + 0.22, 5.08, 3.10, 0.20, 11, color, True, PP_ALIGN.CENTER)
        text(s, value, x + 0.34, 5.48, 2.86, 0.34, 9, C.body, align=PP_ALIGN.CENTER)
    claim(s, "三类用户表面不同，本质都是：不要每个任务重做一次遥感工程。", 6.23, C.ink)

    # 8. HMW and token analogy
    s = prs.slides.add_slide(blank)
    bg(s)
    title(s, "08", "我们的答案：把地球观测转成地理智能底座")
    text(
        s,
        "我们如何把爆发式增长的地球观测数据转化为可复用的地理智能底座，\n让不同用户不再为每个遥感任务重复采购、标注、建模和计算？",
        1.04,
        1.30,
        11.25,
        0.96,
        24,
        C.ink,
        True,
        PP_ALIGN.CENTER,
    )
    rect(s, 1.30, 3.22, 4.75, 1.38, C.pale_blue, C.blue)
    text(s, "文本智能", 1.62, 3.48, 4.10, 0.24, 14, C.blue, True, PP_ALIGN.CENTER)
    text(s, "文本 → 语义单元 → 理解 / 检索 / 生成", 1.62, 3.94, 4.10, 0.24, 11, C.body, True, PP_ALIGN.CENTER)
    rect(s, 7.28, 3.22, 4.75, 1.38, C.mint, C.green)
    text(s, "地球智能", 7.60, 3.48, 4.10, 0.24, 14, C.green, True, PP_ALIGN.CENTER)
    text(s, "地球观测 → 地理嵌入 → 监测 / 核查 / 预警 / 决策", 7.60, 3.94, 4.10, 0.24, 10, C.body, True, PP_ALIGN.CENTER)
    arrow(s, 6.18, 3.92, 7.05, 3.92, C.line, 1.4)
    text(
        s,
        "大模型先把文本转成语义单元，才能进行理解、检索、生成；\n玄女先把地球观测转成地理嵌入，才能进行监测、核查、预警和决策。",
        1.16,
        5.45,
        11.0,
        0.58,
        15,
        C.blue,
        True,
        PP_ALIGN.CENTER,
    )

    return prs


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs = build()
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
