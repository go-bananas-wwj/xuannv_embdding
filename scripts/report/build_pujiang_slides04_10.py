#!/usr/bin/env python3
"""Build slides 4-10 for the Pujiang Xuannv embedding presentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Inches

ROOT = Path("/root/workspace/xuannv")
PRESENTATION_ROOT = ROOT / "docs/presentations/pujiang_202607"
ASSET_ROOT = ROOT / "docs/presentations/assets/pujiang_202607"
SOURCE_PPTX = Path(
    "/root/.codex/attachments/d0a9024d-03de-463f-83fa-675e2a235467/Alpha Earth介绍.pptx"
)
PLATFORM_POSTER = ROOT / "docs/bp_deck/assets/video/custom_annotation_demo_poster.png"
PLATFORM_SCREENSHOT = Path(
    "/data/xuannv_embedding/experiments/presentation_pujiang_202607/frontend/"
    "haidian_platform_thum.png"
)
WORKFLOW_IMAGE = ROOT / "docs/bp_deck/assets/remote_sensing_team_workflow.png"
FEWSHOT_IMAGE = ROOT / "docs/production/assets/haidian_v1_20260708/fewshot_5shot_examples.png"
PCA_COMPARE = (
    ROOT
    / "docs/production/assets/nonbuilding_fewshot_semantic_20260705"
    / "xuannv_aef_global_pca_compare.png"
)
HAR_BIN_ROOT = ROOT / "docs/bp_deck/assets/user_materials"
SLIDE01_ROOT = ASSET_ROOT / "slide01"

BLUE = (0, 97, 170)
DEEP_BLUE = (0, 69, 122)
CYAN = (0, 143, 174)
GREEN = (34, 128, 94)
RED = (201, 47, 55)
INK = (25, 40, 55)
MUTED = (80, 99, 114)
LINE = (205, 215, 223)
PALE_BLUE = (236, 246, 252)
PALE_GRAY = (245, 247, 249)
WHITE = (255, 255, 255)

PAGE_TITLES = {
    4: "玄女嵌入平台：浏览、分析、训练与生成",
    5: "少量标注，快速形成区域级专题图",
    6: "嵌入底座进入遥感智能体工作流",
    7: "云遮挡或观测缺失时，生成指定时刻的遥感参考影像",
    8: "海淀区同协议评测：64 维月度嵌入达到 AEF 同量级",
    9: "标注越少，月度嵌入的复用价值越明显",
    10: "哈尔滨新区：从月度嵌入到城市治理专题",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        (
            "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Bold.ttc"
            if bold
            else "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"
        ),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def centered(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    text_font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = box
    bounds = draw.multiline_textbbox((0, 0), text, font=text_font, spacing=5, align="center")
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.multiline_text(
        (x0 + (x1 - x0 - width) / 2, y0 + (y1 - y0 - height) / 2),
        text,
        font=text_font,
        fill=fill,
        spacing=5,
        align="center",
    )


def fit_image(image_path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    ratio = min(size[0] / image.width, size[1] / image.height)
    resized = image.resize(
        (max(1, int(image.width * ratio)), max(1, int(image.height * ratio))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", size, WHITE)
    canvas.paste(resized, ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2))
    return canvas


def draw_header(canvas: Image.Image, title: str, kicker: str) -> ImageDraw.ImageDraw:
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 382, 58), fill=BLUE)
    draw.text((42, 10), "玄女月度地理嵌入", font=font(25, True), fill=WHITE)
    draw.rectangle((0, 58, 1600, 139), fill=BLUE)
    centered(draw, title, (30, 58, 1570, 139), font(36, True), WHITE)
    centered(draw, kicker, (55, 141, 1545, 186), font(20, True), INK)
    return draw


def draw_number_badge(
    draw: ImageDraw.ImageDraw,
    number: int,
    x: int,
    y: int,
    title: str,
    detail: str,
) -> None:
    draw.rounded_rectangle((x, y, x + 410, y + 130), radius=6, fill=PALE_BLUE, outline=LINE, width=2)
    draw.ellipse((x + 22, y + 30, x + 84, y + 92), fill=BLUE)
    centered(draw, f"{number:02d}", (x + 22, y + 30, x + 84, y + 92), font(20, True), WHITE)
    draw.text((x + 105, y + 23), title, font=font(22, True), fill=DEEP_BLUE)
    draw.multiline_text((x + 105, y + 60), detail, font=font(15), fill=MUTED, spacing=5)


def page04() -> tuple[Image.Image, dict]:
    canvas = Image.new("RGB", (1600, 900), WHITE)
    draw = draw_header(
        canvas,
        PAGE_TITLES[4],
        "同一份月度嵌入，从离线数据文件变成可浏览、可训练、可调用的业务能力",
    )
    platform = fit_image(PLATFORM_SCREENSHOT, (970, 560))
    pca = fit_image(SLIDE01_ROOT / "haidian_pca.png", (530, 370))
    platform.paste(pca, (300, 125))
    poster = fit_image(PLATFORM_POSTER, (250, 180))
    platform.paste(poster, (705, 365))
    platform_draw = ImageDraw.Draw(platform)
    platform_draw.rounded_rectangle((286, 110, 844, 510), radius=6, outline=(226, 170, 65), width=5)
    for number, xy in [(1, (800, 48)), (2, (250, 95)), (3, (685, 345)), (4, (878, 48))]:
        x, y = xy
        platform_draw.ellipse((x, y, x + 42, y + 42), fill=RED)
        centered(platform_draw, str(number), (x, y, x + 42, y + 42), font(16, True), WHITE)
    canvas.paste(platform, (55, 220))
    draw.rounded_rectangle((55, 220, 1025, 780), radius=6, outline=LINE, width=2)
    draw.rectangle((55, 220, 1025, 263), fill=(249, 251, 252))
    draw.text(
        (76, 229),
        "玄女底座遥感模型展示平台｜海淀区主界面与关键入口组合",
        font=font(18, True),
        fill=DEEP_BLUE,
    )

    items = [
        ("嵌入浏览", "按区域、月份、任务和 patch\n查看嵌入与专题结果"),
        ("专题分析", "建筑、道路、水体、变化等\n结果回填地图统一查看"),
        ("自定义训练", "少量标注后创建或加载\n轻量下游模型"),
        ("时相生成", "选择目标时刻，生成缺测或\n云遮挡场景的参考影像"),
    ]
    for index, (title, detail) in enumerate(items, start=1):
        draw_number_badge(draw, index, 1080, 220 + (index - 1) * 142, title, detail)

    draw.rounded_rectangle((55, 808, 1545, 860), radius=5, fill=PALE_GRAY)
    centered(
        draw,
        "平台入口：http://112.111.7.74:1114/embedEarth?regionId=haidian　｜　"
        "嵌入浏览 → 任务训练 → 全域推理 → 地图复核",
        (70, 808, 1530, 860),
        font(16, True),
        INK,
    )
    metadata = {
        "slide": 4,
        "title": PAGE_TITLES[4],
        "functions": ["嵌入浏览", "专题分析", "自定义训练", "时相生成"],
        "claim_boundary": "平台展示现有功能，不把生成结果表述为真实观测。",
    }
    return canvas, metadata


def paste_panel(
    canvas: Image.Image,
    image_path: Path,
    box: tuple[int, int, int, int],
    title: str | None = None,
) -> None:
    x0, y0, x1, y1 = box
    inner_top = y0 + (42 if title else 0)
    panel = fit_image(image_path, (x1 - x0, y1 - inner_top))
    canvas.paste(panel, (x0, inner_top))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(box, radius=5, outline=LINE, width=2)
    if title:
        draw.rectangle((x0, y0, x1, inner_top), fill=(249, 251, 252))
        centered(draw, title, (x0 + 8, y0, x1 - 8, inner_top), font(17, True), DEEP_BLUE)


def paste_panel_image(
    canvas: Image.Image,
    image: Image.Image,
    box: tuple[int, int, int, int],
    title: str | None = None,
) -> None:
    x0, y0, x1, y1 = box
    inner_top = y0 + (42 if title else 0)
    panel = fit_image_from_image(image, (x1 - x0, y1 - inner_top))
    canvas.paste(panel, (x0, inner_top))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(box, radius=5, outline=LINE, width=2)
    if title:
        draw.rectangle((x0, y0, x1, inner_top), fill=(249, 251, 252))
        centered(draw, title, (x0 + 8, y0, x1 - 8, inner_top), font(17, True), DEEP_BLUE)


def page05() -> tuple[Image.Image, dict]:
    canvas = Image.new("RGB", (1600, 900), WHITE)
    draw = draw_header(
        canvas,
        PAGE_TITLES[5],
        "冻结月度嵌入，只训练轻量下游头；标注成本从全域描绘缩减为少量代表性样例",
    )
    steps = [
        ("01", "选择月度嵌入", "2026年4月｜128×128×64"),
        ("02", "标注少量样例", "平台提交目标多边形标注"),
        ("03", "训练轻量任务头", "冻结底座｜PU / Binary Conv"),
        ("04", "全域推理与复核", "320个patch形成专题初图"),
    ]
    for index, (number, title, detail) in enumerate(steps):
        x = 55 + index * 382
        draw.rounded_rectangle((x, 210, x + 345, 315), radius=6, fill=PALE_BLUE, outline=LINE, width=2)
        draw.ellipse((x + 18, 231, x + 78, 291), fill=BLUE)
        centered(draw, number, (x + 18, 231, x + 78, 291), font(18, True), WHITE)
        draw.text((x + 92, 225), title, font=font(19, True), fill=DEEP_BLUE)
        draw.text((x + 92, 262), detail, font=font(14), fill=MUTED)
        if index < 3:
            centered(draw, "→", (x + 345, 235, x + 382, 292), font(25, True), BLUE)
    with Image.open(FEWSHOT_IMAGE) as source:
        xuannv_case = source.convert("RGB").crop((0, 65, 2025, 690))
    paste_panel_image(
        canvas,
        xuannv_case,
        (55, 350, 1110, 806),
        "5-shot建筑案例｜原始影像、GT、玄女概率与玄女掩膜",
    )
    draw.rounded_rectangle((1150, 350, 1545, 806), radius=6, fill=PALE_GRAY, outline=LINE, width=2)
    centered(draw, "“5-shot”是什么意思？", (1175, 375, 1520, 425), font(23, True), DEEP_BLUE)
    centered(
        draw,
        "论文评测口径：\n5 个正样本 patch\n+ 5 个负样本 patch\n\n平台产品口径：\n提交目标多边形标注\n由 PU / Query 或 Binary Conv\n形成轻量任务头\n\n随后对海淀全域推理",
        (1180, 440, 1515, 665),
        font(19, True),
        INK,
    )
    draw.rounded_rectangle((1190, 690, 1505, 770), radius=5, fill=WHITE, outline=LINE, width=2)
    centered(
        draw,
        "本页只说明业务流程\n模型公平对比见第8页：\n同标签｜同划分｜同头｜同阈值规则",
        (1200, 692, 1495, 768),
        font(15, True),
        RED,
    )
    centered(
        draw,
        "业务价值：新增类别时无需重训嵌入底座，只需补少量样例并更新轻量任务头。",
        (55, 826, 1545, 875),
        font(18, True),
        INK,
    )
    return canvas, {
        "slide": 5,
        "title": PAGE_TITLES[5],
        "protocol": (
            "Paper evaluation: 5-shot = 5 positive patches + 5 negative patches. "
            "Platform: polygon labels with PU/query or binary conv."
        ),
        "fairness": ["same labels", "same split", "same head", "same threshold rule"],
    }


def page06() -> tuple[Image.Image, dict]:
    canvas = Image.new("RGB", (1600, 900), WHITE)
    draw = draw_header(
        canvas,
        PAGE_TITLES[6],
        "现有数据与推理 API 可封装为智能体工具；嵌入底座负责提供可复用地理表示",
    )
    paste_panel(canvas, WORKFLOW_IMAGE, (55, 215, 905, 758), "自然语言需求到地图结果的协同流程")
    tool_rows = [
        ("理解任务", "区域、月份、目标类别与输出形式"),
        ("读取嵌入", "embedding_tool：加载月度64维嵌入"),
        ("选择工具", "系统任务头｜变化分析｜任务摘要"),
        ("空间计算", "全域推理、统计汇总与质量检查"),
        ("交付结果", "地图图层、案例图与结构化报告"),
    ]
    for index, (title, detail) in enumerate(tool_rows, start=1):
        y = 215 + (index - 1) * 105
        draw.rounded_rectangle((955, y, 1545, y + 82), radius=6, fill=PALE_BLUE, outline=LINE, width=2)
        draw.ellipse((975, y + 14, 1027, y + 66), fill=BLUE)
        centered(draw, str(index), (975, y + 14, 1027, y + 66), font(18, True), WHITE)
        draw.text((1045, y + 10), title, font=font(19, True), fill=DEEP_BLUE)
        draw.text((1045, y + 43), detail, font=font(14), fill=MUTED)
    draw.rounded_rectangle((955, 754, 1545, 816), radius=5, fill=PALE_GRAY)
    centered(
        draw,
        "边界：智能体调用模型与工具，不替代模型推理；关键结果仍需地图复核。",
        (970, 756, 1530, 814),
        font(15, True),
        RED,
    )
    centered(
        draw,
        "当前状态：具备数据、推理与结构化分析接口；正式智能体编排属于下一步集成工作。",
        (55, 832, 1545, 875),
        font(18, True),
        INK,
    )
    return canvas, {
        "slide": 6,
        "title": PAGE_TITLES[6],
        "tool": "embedding_tool",
        "boundary": (
            "API-ready conceptual integration; no claim that a formal Xuannv agent "
            "orchestration has already been deployed."
        ),
    }


def page07() -> tuple[Image.Image, dict]:
    canvas = Image.new("RGB", (1600, 900), WHITE)
    draw = draw_header(
        canvas,
        PAGE_TITLES[7],
        "融合目标时刻前后的多源观测，为缺测场景提供可解释的参考影像",
    )
    source = HAR_BIN_ROOT / "patch_000027_2025-08_vs_2025-09_detail.png"
    with Image.open(source) as raw:
        strip = raw.convert("RGB")
    crop_w = strip.width // 5
    panels = [
        (
            strip.crop((0, 0, crop_w, strip.height - 34)),
            "相邻时刻有效观测",
            "提供空间结构与地物背景",
        ),
        (
            strip.crop((crop_w, 0, crop_w * 2, strip.height - 34)),
            "目标时刻观测缺失",
            "云雾、空洞或传感器未覆盖",
        ),
        (
            Image.open(HAR_BIN_ROOT / "patch_000146_2025-06_vs_2025-08_detail.png")
            .convert("RGB")
            .crop((strip.width // 5, 0, strip.width // 5 * 2, strip.height - 34)),
            "生成参考影像示意（非实测）",
            "仅说明平台输出形态｜不作为生成质量证据",
        ),
    ]
    for index, (panel, title, detail) in enumerate(panels):
        x = 55 + index * 510
        holder = Image.new("RGB", (450, 410), WHITE)
        panel = fit_image_from_image(panel, (450, 330))
        holder.paste(panel, (0, 0))
        canvas.paste(holder, (x, 280))
        draw.rounded_rectangle((x, 220, x + 450, 720), radius=6, outline=LINE, width=2)
        centered(draw, title, (x + 10, 225, x + 440, 275), font(21, True), DEEP_BLUE)
        centered(draw, detail, (x + 15, 630, x + 435, 708), font(16), MUTED)
        if index < 2:
            centered(draw, "→", (x + 450, 410, x + 510, 500), font(32, True), BLUE)
    draw.rounded_rectangle((55, 756, 1545, 850), radius=6, fill=PALE_GRAY)
    centered(
        draw,
        "使用边界｜本页为平台能力流程示意，不是生成精度对比；模型输出是“参考影像”，"
        "不是该时刻真实观测。重要结论仍需真实影像核验。",
        (80, 760, 1520, 846),
        font(18, True),
        RED,
    )
    return canvas, {
        "slide": 7,
        "title": PAGE_TITLES[7],
        "output": "指定时刻参考影像",
        "boundary": "Reference generation, not recovery of a true observation.",
    }


def fit_image_from_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    ratio = min(size[0] / image.width, size[1] / image.height)
    resized = image.resize(
        (max(1, int(image.width * ratio)), max(1, int(image.height * ratio))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", size, WHITE)
    canvas.paste(resized, ((size[0] - resized.width) // 2, (size[1] - resized.height) // 2))
    return canvas


def page08() -> tuple[Image.Image, dict]:
    canvas = Image.new("RGB", (1600, 900), WHITE)
    draw = draw_header(
        canvas,
        PAGE_TITLES[8],
        "相同标签、相同划分、相同 conv3×3 轻量头与相同阈值规则；F1 越高越好",
    )
    tasks = ["建筑物", "道路", "水体"]
    values = {
        "玄女 P10C（64维）": [0.4840, 0.5198, 0.6229],
        "Google AEF（64维）": [0.4855, 0.5165, 0.6520],
        "DINOv3（1024维）": [0.4983, 0.5999, 0.6750],
    }
    colors = [RED, BLUE, (92, 112, 129)]
    chart = (70, 245, 1000, 720)
    x0, y0, x1, y1 = chart
    draw.line((x0, y1, x1, y1), fill=INK, width=2)
    for tick in range(0, 8):
        value = tick / 10
        y = y1 - int(value / 0.8 * (y1 - y0))
        draw.line((x0, y, x1, y), fill=(225, 231, 235), width=1)
        draw.text((20, y - 10), f"{value:.1f}", font=font(13), fill=MUTED)
    group_w = (x1 - x0) / len(tasks)
    bar_w = 48
    for task_index, task in enumerate(tasks):
        center_x = x0 + group_w * (task_index + 0.5)
        for model_index, (_, model_values) in enumerate(values.items()):
            value = model_values[task_index]
            left = int(center_x - 84 + model_index * 60)
            top = y1 - int(value / 0.8 * (y1 - y0))
            draw.rectangle((left, top, left + bar_w, y1), fill=colors[model_index])
            centered(draw, f"{value:.3f}", (left - 5, top - 30, left + bar_w + 5, top - 3), font(12, True), colors[model_index])
        centered(draw, task, (int(center_x - 95), y1 + 12, int(center_x + 95), y1 + 52), font(16, True), INK)
    for index, label in enumerate(values):
        x = 100 + index * 280
        draw.rectangle((x, 200, x + 24, 224), fill=colors[index])
        draw.text((x + 34, 197), label, font=font(15, True), fill=INK)
    paste_panel(canvas, PCA_COMPARE, (1050, 225, 1545, 690), "全域嵌入 PCA｜玄女与 AEF")
    draw.rounded_rectangle((1045, 712, 1545, 830), radius=6, fill=PALE_GRAY)
    centered(
        draw,
        "结论\n建筑基本持平，道路 F1 略高\n水体仍有明确差距\nDINOv3 为 1024 维高预算参照\n单 fold 探索性比较，不宣称统计显著",
        (1065, 715, 1525, 827),
        font(16, True),
        INK,
    )
    return canvas, {
        "slide": 8,
        "title": PAGE_TITLES[8],
        "metric": "F1",
        "p10c": dict(zip(tasks, values["玄女 P10C（64维）"])),
        "building_f1": 0.4840,
        "protocol": "same labels/split/conv3x3 head/threshold selection",
    }


def page09() -> tuple[Image.Image, dict]:
    canvas = Image.new("RGB", (1600, 900), WHITE)
    draw = draw_header(
        canvas,
        PAGE_TITLES[9],
        "与原始多源影像相比，玄女在 5-shot 和 10-shot 条件下的优势最稳定",
    )
    shots = [5, 10, 50]
    gains = {
        "建筑物": [9.8, 3.5, 5.0],
        "道路": [18.3, 11.8, 3.3],
        "水体": [41.5, 25.6, 0.0],
    }
    colors = [RED, BLUE, GREEN]
    x0, y0, x1, y1 = 75, 260, 980, 725
    draw.line((x0, y1, x1, y1), fill=INK, width=2)
    for tick in range(0, 6):
        value = tick * 10
        y = y1 - int(value / 50 * (y1 - y0))
        draw.line((x0, y, x1, y), fill=(225, 231, 235), width=1)
        draw.text((22, y - 10), f"{value}%", font=font(14), fill=MUTED)
    group_w = (x1 - x0) / 3
    for shot_index, shot in enumerate(shots):
        center_x = x0 + group_w * (shot_index + 0.5)
        for task_index, (_, task_values) in enumerate(gains.items()):
            value = task_values[shot_index]
            left = int(center_x - 95 + task_index * 65)
            top = y1 - int(value / 50 * (y1 - y0))
            draw.rectangle((left, top, left + 52, y1), fill=colors[task_index])
            centered(draw, f"+{value:.1f}%" if value else "持平", (left - 8, top - 32, left + 60, top - 3), font(13, True), colors[task_index])
        centered(draw, f"{shot}-shot", (int(center_x - 85), y1 + 12, int(center_x + 85), y1 + 55), font(18, True), INK)
    for index, task in enumerate(gains):
        x = 115 + index * 250
        draw.rectangle((x, 205, x + 24, 229), fill=colors[index])
        draw.text((x + 34, 202), task, font=font(16, True), fill=INK)
    paste_panel(canvas, FEWSHOT_IMAGE, (1030, 235, 1545, 650), "少样本预测案例")
    draw.rounded_rectangle((1030, 675, 1545, 820), radius=6, fill=PALE_GRAY)
    centered(
        draw,
        "shot 定义\nN 个正样本 patch + N 个负样本 patch\n\nfold-0、seed 42 探索性结果\n每种输入取五类候选头中的最佳值\n全量标签 + 重型 UNet 仍是原始影像强上限",
        (1050, 680, 1525, 815),
        font(16, True),
        INK,
    )
    centered(
        draw,
        "纵轴为 F1 相对提升百分比：（玄女 F1 − 原始影像 F1）÷ 原始影像 F1；"
        "原始影像允许使用更强任务头。",
        (60, 832, 1540, 875),
        font(16, True),
        MUTED,
    )
    return canvas, {
        "slide": 9,
        "title": PAGE_TITLES[9],
        "metric": "relative F1 gain (%)",
        "water_5shot_gain_percent": 41.5,
        "gains": gains,
    }


def page10() -> tuple[Image.Image, dict]:
    canvas = Image.new("RGB", (1600, 900), WHITE)
    draw = draw_header(
        canvas,
        PAGE_TITLES[10],
        "同一套月度嵌入可支撑城市建设、地表变化与资源环境专题分析",
    )
    paste_panel(canvas, SLIDE01_ROOT / "harbin_pca.png", (55, 220, 660, 790), "哈尔滨新区月度嵌入 PCA｜API V5/v2")
    cases = [
        ("建筑物提取｜V5/v2 API", SLIDE01_ROOT / "harbin_building.png"),
        ("水体提取｜V5/v2 API", SLIDE01_ROOT / "harbin_water.png"),
        ("土地覆盖｜V5/v2 API", SLIDE01_ROOT / "harbin_landcover.png"),
        (
            "土地利用转换｜V5/v2 API",
            HAR_BIN_ROOT / "patch_000217_land_conversion_2025-08_vs_2025-09_detail.png",
        ),
    ]
    for index, (title, path) in enumerate(cases):
        col = index % 2
        row = index // 2
        x = 700 + col * 430
        y = 220 + row * 285
        paste_panel(canvas, path, (x, y, x + 400, y + 250), title)
    draw.rounded_rectangle((700, 804, 1530, 860), radius=5, fill=PALE_GRAY)
    centered(
        draw,
        "月度更新｜轻量任务头｜可由平台和智能体调用｜结果回到地图复核",
        (715, 806, 1515, 858),
        font(17, True),
        INK,
    )
    return canvas, {
        "slide": 10,
        "title": PAGE_TITLES[10],
        "harbin_source": "V5/v2 API",
        "tasks": ["建筑物提取", "水体提取", "土地覆盖", "土地利用转换"],
    }


PAGE_BUILDERS = {
    4: page04,
    5: page05,
    6: page06,
    7: page07,
    8: page08,
    9: page09,
    10: page10,
}


def image_to_pptx(image_path: Path, output: Path) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333333)
    prs.slide_height = Inches(7.5)
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.shapes.add_picture(str(image_path), 0, 0, width=prs.slide_width, height=prs.slide_height)
    output.parent.mkdir(parents=True, exist_ok=True)
    prs.save(output)


def build_page(page: int, output: Path, preview: Path) -> dict:
    if page not in PAGE_BUILDERS:
        raise ValueError(f"Unsupported page: {page}")
    canvas, metadata = PAGE_BUILDERS[page]()
    preview.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(preview, optimize=True)
    image_to_pptx(preview, output)
    (preview.parent / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def default_paths(page: int) -> tuple[Path, Path]:
    output = PRESENTATION_ROOT / f"玄女月度地理嵌入_浦江交流_第{page:02d}页_20260726.pptx"
    preview = ASSET_ROOT / f"slide{page:02d}" / f"slide{page:02d}_preview.png"
    return output, preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=int, required=True, choices=range(4, 11))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output, preview = default_paths(args.page)
    build_page(args.page, output, preview)
    print(output)


if __name__ == "__main__":
    main()
