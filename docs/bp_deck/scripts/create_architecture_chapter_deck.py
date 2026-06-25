"""Generate the Xuannv technical-route chapter deck.

This chapter intentionally starts after the general "one embedding, many
downstream tasks" story already covered in the previous two decks.  It focuses
on the underlying technical route: what is different, why it works, and why it
can scale.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps
from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches

from create_front8_deck_v2 import C, IMG, bg, picture_fit, text, title


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
GENERATED = ASSETS / "generated"
OLD = ASSETS / "old_bp_media"
OUT = ROOT / "outputs" / "玄女科技BP_架构原理章节_v0.5.pptx"
FONT_PATH = Path("/root/workspace/xuannv/fonts/NotoSansCJKsc-Regular.otf")
ROADMAP_IMAGEGEN = GENERATED / "arch_roadmap_imagegen_base.png"
CLOSING_IMAGEGEN = GENERATED / "arch_closing_satellite_imagegen.png"


W, H = 2400, 1260
BLUE = (37, 99, 235)
SKY = (14, 165, 233)
GREEN = (16, 185, 129)
PURPLE = (124, 58, 237)
AMBER = (245, 158, 11)
INK = (15, 23, 42)
BODY = (51, 65, 85)
MUTED = (100, 116, 139)
LINE = (226, 232, 240)
OFF = (248, 250, 252)
WHITE = (255, 255, 255)
PALE_BLUE = (239, 246, 255)
MINT = (236, 253, 245)
PALE_PURPLE = (245, 243, 255)
PALE_AMBER = (255, 251, 235)


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size=size)


def draw_text(
    d: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    size: int,
    fill=INK,
    anchor: str | None = None,
    align: str = "left",
    width: int | None = None,
    spacing: int = 8,
) -> None:
    f = font(size)
    if width is None:
        d.text(xy, value, font=f, fill=fill, anchor=anchor)
        return
    lines: list[str] = []
    current = ""
    for ch in value:
        test = current + ch
        if d.textlength(test, font=f) <= width or not current:
            current = test
        else:
            lines.append(current)
            current = ch
    if current:
        lines.append(current)
    d.multiline_text(xy, "\n".join(lines), font=f, fill=fill, spacing=spacing, align=align, anchor=anchor)


def rounded(d: ImageDraw.ImageDraw, box, radius=28, fill=WHITE, outline=LINE, width=2) -> None:
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(d: ImageDraw.ImageDraw, start, end, fill=BLUE, width=7) -> None:
    d.line([start, end], fill=fill, width=width)
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    size = 24
    points = [
        (x2, y2),
        (x2 - ux * size + px * size * 0.55, y2 - uy * size + py * size * 0.55),
        (x2 - ux * size - px * size * 0.55, y2 - uy * size - py * size * 0.55),
    ]
    d.polygon(points, fill=fill)


def circle(d: ImageDraw.ImageDraw, cx, cy, r, fill=WHITE, outline=LINE, width=2) -> None:
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=outline, width=width)


def paste_fit(canvas: Image.Image, path: Path, box: tuple[int, int, int, int], rounded_radius: int = 0) -> None:
    if not path.exists():
        return
    x1, y1, x2, y2 = box
    img = Image.open(path).convert("RGB")
    img = ImageOps.fit(img, (x2 - x1, y2 - y1), method=Image.Resampling.LANCZOS)
    if rounded_radius:
        mask = Image.new("L", img.size, 0)
        md = ImageDraw.Draw(mask)
        md.rounded_rectangle((0, 0, img.size[0], img.size[1]), radius=rounded_radius, fill=255)
        canvas.paste(img, (x1, y1), mask)
    else:
        canvas.paste(img, (x1, y1))


def label_pill(d, x, y, value, fill=PALE_BLUE, color=BLUE, w=None):
    f = font(28)
    if w is None:
        w = int(d.textlength(value, font=f)) + 50
    rounded(d, (x, y, x + w, y + 54), 27, fill, color, 2)
    d.text((x + w // 2, y + 27), value, font=f, fill=color, anchor="mm")


def base_canvas() -> Image.Image:
    im = Image.new("RGB", (W, H), WHITE)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, H), fill=WHITE)
    return im


def save(im: Image.Image, name: str) -> Path:
    GENERATED.mkdir(parents=True, exist_ok=True)
    out = GENERATED / name
    im.save(out, quality=95)
    return out


def draw_satellite(d, x, y, color=BLUE) -> None:
    rounded(d, (x - 28, y - 18, x + 28, y + 18), 8, WHITE, color, 3)
    d.rectangle((x - 92, y - 12, x - 38, y + 12), outline=color, width=3, fill=(239, 246, 255))
    d.rectangle((x + 38, y - 12, x + 92, y + 12), outline=color, width=3, fill=(239, 246, 255))
    d.line((x - 38, y, x - 28, y), fill=color, width=3)
    d.line((x + 28, y, x + 38, y), fill=color, width=3)


def make_paradigm() -> Path:
    im = base_canvas()
    d = ImageDraw.Draw(im)
    draw_text(d, (220, 90), "传统任务模型", 54, INK)
    draw_text(d, (1420, 90), "玄女底座模型", 54, INK)
    draw_text(d, (220, 165), "一个任务、一套流程，能力难沉淀", 30, MUTED)
    draw_text(d, (1420, 165), "一个底座、多任务调用，能力持续复用", 30, MUTED)

    task_y = [340, 590, 840]
    tasks = [("变化检测", AMBER), ("地物分类", PURPLE), ("报告生成", GREEN)]
    for i, ((name, color), y) in enumerate(zip(tasks, task_y)):
        paste_fit(im, [IMG["cloud"], IMG["worldcover"], OLD / "image15.jpeg"][i], (220, y, 430, y + 150), 22)
        rounded(d, (480, y + 5, 780, y + 145), 34, WHITE, color, 3)
        d.text((630, y + 56), "单任务模型", font=font(34), fill=color, anchor="mm")
        d.text((630, y + 96), name, font=font(24), fill=BODY, anchor="mm")
        arrow(d, (430, y + 75), (480, y + 75), color, 5)
        arrow(d, (780, y + 75), (890, y + 75), color, 5)
        draw_text(d, (905, y + 48), "重新标注\n重新训练", 25, BODY)
    d.line((1050, 270, 1050, 1060), fill=LINE, width=4)

    for i, path in enumerate([IMG["s2"], IMG["s1"], IMG["dem"], IMG["landsat"], IMG["s2hr"]]):
        paste_fit(im, path, (1280 + i * 135, 300, 1385 + i * 135, 405), 18)
    arrow(d, (1580, 430), (1580, 545), BLUE, 7)
    rounded(d, (1330, 550, 1830, 720), 48, PALE_BLUE, BLUE, 4)
    d.text((1580, 610), "DV 地理嵌入向量", font=font(46), fill=BLUE, anchor="mm")
    d.text((1580, 670), "统一表征层", font=font(28), fill=BODY, anchor="mm")
    for i, (name, color) in enumerate(tasks):
        x = 1230 + i * 300
        arrow(d, (1580, 720), (x + 105, 850), color, 5)
        rounded(d, (x, 860, x + 210, 960), 28, WHITE, color, 3)
        d.text((x + 105, 910), name, font=font(30), fill=color, anchor="mm")
    draw_text(d, (1230, 1040), "新增任务主要训练轻量任务头，不重做完整底座", 34, INK)
    return save(im, "arch_paradigm_task_to_foundation.png")


def make_multimodal() -> Path:
    im = base_canvas()
    d = ImageDraw.Draw(im)
    draw_text(d, (210, 85), "多模态自学习机制", 56, INK)
    draw_text(d, (210, 165), "把同一地理位置的多源、多时序观测对齐到同一语义空间", 31, MUTED)
    center = (1190, 640)
    circle(d, *center, 190, PALE_BLUE, BLUE, 5)
    d.text(center, "DV\n地理嵌入", font=font(58), fill=BLUE, anchor="mm", align="center")

    inputs = [
        ("光学影像", IMG["s2"], (420, 310), GREEN),
        ("SAR 雷达", IMG["s1"], (420, 760), BLUE),
        ("多光谱", IMG["landsat"], (1770, 310), PURPLE),
        ("DEM 地形", IMG["dem"], (1770, 760), AMBER),
    ]
    for label, path, (x, y), color in inputs:
        paste_fit(im, path, (x, y, x + 310, y + 230), 26)
        rounded(d, (x, y, x + 310, y + 230), 26, fill=None, outline=color, width=4)
        d.text((x + 155, y + 270), label, font=font(30), fill=color, anchor="mm")
        arrow(d, (x + (310 if x < center[0] else 0), y + 115), center, color, 6)

    for i, (label, color) in enumerate([("跨模态对齐", BLUE), ("时序一致性", GREEN), ("缺失模态补全", PURPLE)]):
        label_pill(d, 865 + i * 300, 1000, label, [PALE_BLUE, MINT, PALE_PURPLE][i], color, 245)
    draw_text(d, (875, 1110), "当云、雪、传感器缺测导致单一模态不完整时，模型仍可利用其他模态推断地理状态。", 34, INK, width=900)
    return save(im, "arch_multimodal_self_learning.png")


def make_same_land() -> Path:
    im = base_canvas()
    d = ImageDraw.Draw(im)
    draw_text(d, (190, 80), "为什么遥感适合做通用嵌入", 56, INK)
    draw_text(d, (190, 160), "同一地理位置会被不同卫星、不同时间、不同模态反复观测", 31, MUTED)

    paste_fit(im, IMG["s2hr"], (190, 420, 760, 990), 34)
    rounded(d, (190, 420, 760, 990), 34, fill=None, outline=LINE, width=4)
    d.text((475, 1035), "同一地块", font=font(36), fill=INK, anchor="mm")

    sat_positions = [(1080, 280), (1370, 210), (1660, 280)]
    for x, y in sat_positions:
        draw_satellite(d, x, y, BLUE)
        d.line((x, y + 45, 480, 420), fill=(186, 230, 253), width=4)

    thumbs = [(980, 470, IMG["s2"]), (1270, 470, IMG["s1"]), (1560, 470, IMG["landsat"]), (1125, 755, IMG["dem"]), (1415, 755, IMG["worldcover"])]
    for x, y, path in thumbs:
        paste_fit(im, path, (x, y, x + 230, y + 180), 22)
        rounded(d, (x, y, x + 230, y + 180), 22, fill=None, outline=LINE, width=3)
        arrow(d, (x + 115, y + 180), (1860, 735), SKY, 4)

    rounded(d, (1840, 580, 2200, 900), 54, PALE_BLUE, BLUE, 5)
    d.text((2020, 685), "DV", font=font(72), fill=BLUE, anchor="mm")
    d.text((2020, 765), "地理嵌入向量", font=font(38), fill=BLUE, anchor="mm")
    d.text((2020, 830), "位置稳定 · 语义可比", font=font(26), fill=BODY, anchor="mm")
    draw_text(d, (945, 1045), "自然语言用 token 表示文本单元；玄女用地理嵌入表示地球观测单元。", 38, INK, width=980)
    return save(im, "arch_same_land_to_embedding.png")


def make_model_architecture() -> Path:
    im = base_canvas()
    d = ImageDraw.Draw(im)
    draw_text(d, (190, 80), "模型架构：编码器、嵌入空间与任务头", 56, INK)
    draw_text(d, (190, 160), "底层统一编码，中层沉淀通用嵌入，上层通过轻量任务头适配场景", 31, MUTED)

    x0 = 190
    for i, (label, path) in enumerate([("光学", IMG["s2"]), ("SAR", IMG["s1"]), ("DEM", IMG["dem"]), ("多光谱", IMG["landsat"])]):
        y = 320 + i * 165
        paste_fit(im, path, (x0, y, x0 + 170, y + 125), 18)
        d.text((x0 + 235, y + 63), label, font=font(30), fill=BODY, anchor="mm")
        arrow(d, (x0 + 330, y + 63), (720, 605), LINE, 4)

    rounded(d, (720, 360, 1120, 850), 62, PALE_BLUE, BLUE, 5)
    d.text((920, 510), "时空多模态\n编码器", font=font(52), fill=BLUE, anchor="mm", align="center")
    for i, label in enumerate(["时空对齐", "跨模态融合", "自监督预训练"]):
        label_pill(d, 785, 610 + i * 70, label, WHITE, BLUE, 270)

    arrow(d, (1120, 605), (1340, 605), BLUE, 8)
    rounded(d, (1340, 460, 1685, 750), 58, MINT, GREEN, 5)
    d.text((1512, 555), "DV", font=font(70), fill=GREEN, anchor="mm")
    d.text((1512, 635), "地理嵌入向量", font=font(40), fill=GREEN, anchor="mm")
    d.text((1512, 700), "可复用表征资产", font=font(27), fill=BODY, anchor="mm")

    for i, (label, color) in enumerate([("变化检测", BLUE), ("地物分类", GREEN), ("高程回归", AMBER), ("检索问答", PURPLE), ("报告生成", BLUE)]):
        y = 300 + i * 145
        arrow(d, (1685, 605), (1900, y + 55), color, 4)
        rounded(d, (1900, y, 2200, y + 110), 32, WHITE, color, 3)
        d.text((2050, y + 55), label, font=font(31), fill=color, anchor="mm")

    d.line((210, 1055, 2190, 1055), fill=LINE, width=4)
    draw_text(d, (230, 1100), "新增任务不重建整套模型，只在统一嵌入上训练或调用任务头。", 38, INK)
    return save(im, "arch_model_encoder_embedding_heads.png")


def make_scale_path() -> Path:
    if ROADMAP_IMAGEGEN.exists():
        im = ImageOps.fit(Image.open(ROADMAP_IMAGEGEN).convert("RGB"), (W, H), method=Image.Resampling.LANCZOS)
        # Keep the image airy like the cover page, but add a very light veil so
        # the stage labels stay legible.
        veil = Image.new("RGBA", (W, H), (255, 255, 255, 72))
        im = Image.alpha_composite(im.convert("RGBA"), veil).convert("RGB")
    else:
        im = base_canvas()
    d = ImageDraw.Draw(im)

    stages = [
        ("2026", "技术验证", "中国区域底座验证", "标杆案例 + 试点收入", BLUE, PALE_BLUE, 420),
        ("2027", "商业验证", "行业模型产品化", "API / License / 智能体报告", GREEN, MINT, 1200),
        ("2028", "规模化", "地理智能普惠化", "融合社会经济数据", PURPLE, PALE_PURPLE, 1980),
    ]
    for year, stage, head, result, color, fill, cx in stages:
        rounded(d, (cx - 245, 88, cx + 245, 246), 42, WHITE, (226, 232, 240), 2)
        d.text((cx - 145, 142), year, font=font(58), fill=color, anchor="mm")
        d.text((cx + 92, 128), stage, font=font(30), fill=INK, anchor="mm")
        d.text((cx + 92, 174), head, font=font(24), fill=BODY, anchor="mm")
        rounded(d, (cx - 178, 1000, cx + 178, 1070), 35, fill, color, 2)
        d.text((cx, 1035), result, font=font(25), fill=color, anchor="mm")

    draw_text(d, (365, 1140), "三年路径：技术验证 → 商业验证 → 规模化应用；从遥感底座走向融合社会经济数据的地理智能服务。", 34, INK, width=1680)
    return save(im, "arch_scale_path_social_econ.png")


def make_figures() -> dict[str, Path]:
    return {
        "paradigm": make_paradigm(),
        "multimodal": make_multimodal(),
        "same_land": make_same_land(),
        "architecture": make_model_architecture(),
        "scale": make_scale_path(),
    }


def bottom_claim(slide, value: str) -> None:
    text(slide, value, 0.86, 6.78, 11.60, 0.28, 15, C.blue, True, PP_ALIGN.CENTER)


def add_slide(prs: Presentation, no: str, heading: str, sub: str, figure: Path, claim: str) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg(slide)
    title(slide, no, heading)
    text(slide, sub, 1.06, 1.02, 11.10, 0.34, 12, C.body, False, PP_ALIGN.CENTER)
    picture_fit(slide, figure, 0.76, 1.46, 11.82, 5.08)
    bottom_claim(slide, claim)


def add_closing_slide(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg(slide)
    if CLOSING_IMAGEGEN.exists():
        slide.shapes.add_picture(str(CLOSING_IMAGEGEN), Inches(0), Inches(0), width=Inches(13.333), height=Inches(7.5))
    text(slide, "玄女科技", 0.86, 0.86, 2.20, 0.24, 13, C.blue, True)
    text(slide, "重新定义认知地球的方式", 0.86, 1.36, 6.90, 0.64, 28, C.ink, True)
    text(slide, "做中国的遥感数据通用嵌入底座", 0.90, 2.18, 6.35, 0.38, 17, C.body, True)
    text(slide, "地球观测一次表征 - 多次复用", 0.92, 6.60, 5.90, 0.26, 14, C.blue, True)


def build() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    figures = make_figures()

    add_slide(
        prs,
        "19",
        "技术范式差异：从任务模型到底座模型",
        "传统遥感 AI 围绕单一任务重复建设；玄女将能力沉淀在统一地理嵌入中。",
        figures["paradigm"],
        "路线差异：不是多训练几个模型，而是建立可复用的遥感基础表征。",
    )
    add_slide(
        prs,
        "20",
        "为什么能做：多模态自学习机制",
        "模型通过同一地理位置的多源观测互相校准，学习稳定的地理语义。",
        figures["multimodal"],
        "能力来源：跨模态对齐、时序一致性与缺失模态补全共同形成鲁棒表征。",
    )
    add_slide(
        prs,
        "21",
        "核心原理：同一地块的长期、跨源表征学习",
        "遥感不是孤立图片，而是同一地球表面的反复观测；这使通用嵌入成为可能。",
        figures["same_land"],
        "GPT 将文本单元 token 化；玄女将地球观测单元嵌入化。",
    )
    add_slide(
        prs,
        "22",
        "模型架构：编码器、嵌入空间与任务头",
        "底层统一编码，中层形成 DV 地理嵌入向量，上层用轻量任务头服务不同场景。",
        figures["architecture"],
        "新增任务复用同一表征资产，训练成本和交付周期随任务数量下降。",
    )
    add_slide(
        prs,
        "23",
        "发展规划：三年三阶段里程碑",
        "围绕技术成熟度、商业转化和规模化应用，形成清晰可验证的三年推进路径。",
        figures["scale"],
        "三年路径：技术验证 → 商业验证 → 规模化应用，风险可控、节奏清晰。",
    )
    add_closing_slide(prs)
    return prs


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs = build()
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
