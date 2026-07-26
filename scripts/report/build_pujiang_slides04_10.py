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
COMPLETE_OUTPUT = PRESENTATION_ROOT / "玄女月度地理嵌入_浦江交流_10页完整版_20260726.pptx"
PLATFORM_POSTER = ROOT / "docs/bp_deck/assets/video/custom_annotation_demo_poster.png"
PLATFORM_SCREENSHOT = Path(
    "/data/xuannv_embedding/experiments/presentation_pujiang_202607/frontend/"
    "haidian_platform_thum.png"
)
RECON_EXAMPLE = Path(
    "/data/xuannv_embedding/experiments/presentation_pujiang_202607/frontend/"
    "patch_000093_recon.png"
)
WORKFLOW_IMAGE = ROOT / "docs/bp_deck/assets/remote_sensing_team_workflow.png"
FEWSHOT_IMAGE = ROOT / "docs/production/assets/haidian_v1_20260708/fewshot_5shot_examples.png"
PU_QUERY_ASSET_ROOT = (
    ROOT
    / "docs/production/assets/haidian_pu_query_3polygon_compare_20260726"
)
PU_QUERY_RESULTS = Path(
    "/data/xuannv_embedding/experiments/production/"
    "haidian_pu_query_3polygon_compare_20260726/results.json"
)
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
    8: "海淀区 3 多边形同协议评测：道路与水体领先",
    9: "探索性结果：标注越少，月度嵌入优势越明显",
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
        "用户每类只圈 3 个目标多边形；玄女、AEF 与传统特征共享同一标注和 PU+Query 流程",
    )
    draw.rounded_rectangle((55, 198, 1545, 246), radius=5, fill=PALE_BLUE, outline=LINE, width=2)
    centered(
        draw,
        "PU：从未标区域中自动寻找可靠背景；Query：在新 patch 中用高置信目标轻量校准。"
        "候选得分白色低、红色高；右侧为真实标签。",
        (70, 200, 1530, 244),
        font(16, True),
        INK,
    )
    for index, task in enumerate(("building", "road", "water")):
        source_path = PU_QUERY_ASSET_ROOT / f"{task}_3polygon_compare.png"
        with Image.open(source_path) as source:
            row = source.convert("RGB")
        row = row.resize((1490, 190), Image.Resampling.LANCZOS)
        y = 260 + index * 200
        canvas.paste(row, (55, y))
        draw.rounded_rectangle((55, y, 1545, y + 190), radius=4, outline=LINE, width=2)
    draw.rounded_rectangle((55, 864, 1545, 892), radius=4, fill=PALE_GRAY)
    centered(
        draw,
        "三行依次为建筑物、道路、水体；本页显示候选排序，第 8 页报告独立测试集二值指标。",
        (70, 864, 1530, 892),
        font(14, True),
        MUTED,
    )
    return canvas, {
        "slide": 5,
        "title": PAGE_TITLES[5],
        "protocol": "每个任务 3 个目标多边形；PU + Query；64 个独立测试 patch",
        "tasks": ["建筑物", "道路", "水体"],
        "features": ["玄女 P10C 2026-04", "AEF 2025", "传统 2026-04 多源特征"],
        "fairness": ["same polygons", "same test patches", "same PU+Query", "same threshold rule"],
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
            Image.open(RECON_EXAMPLE).convert("RGB"),
            "真实运行输出｜patch_000093",
            "平台已保存生成结果｜目标日期元数据未公开",
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
        if index == 0:
            centered(draw, "→", (x + 450, 410, x + 510, 500), font(32, True), BLUE)
        elif index == 1:
            centered(draw, "独立\n案例", (x + 452, 405, x + 508, 505), font(15, True), MUTED)
    draw.rounded_rectangle((55, 756, 1545, 850), radius=6, fill=PALE_GRAY)
    centered(
        draw,
        "使用边界｜左侧两图用于说明缺测场景，右图为独立真实运行样例，三者不构成同一 patch 的对照；"
        "模型输出是“参考影像”，不是该时刻真实观测，重要结论仍需真实影像核验。",
        (80, 760, 1520, 846),
        font(18, True),
        RED,
    )
    return canvas, {
        "slide": 7,
        "title": PAGE_TITLES[7],
        "output": "指定时刻参考影像",
        "verified_example": "patch_000093_recon.png",
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
        "每类同一组 3 个多边形、同一批 64 个测试 patch、同一 PU+Query 与阈值规则",
    )
    paste_panel(
        canvas,
        PU_QUERY_ASSET_ROOT / "pu_query_3polygon_metrics.png",
        (55, 210, 1085, 745),
        "F1 与 AUC｜越高越好",
    )
    draw.rounded_rectangle((1125, 210, 1545, 435), radius=6, fill=PALE_BLUE, outline=LINE, width=2)
    centered(draw, "主要结论", (1150, 226, 1520, 268), font(22, True), DEEP_BLUE)
    centered(
        draw,
        "道路：F1 0.368，较 AEF 提升 8.2%\n"
        "水体：F1 0.111，较 AEF 提升 20.7%\n"
        "建筑：F1 0.256，低于 AEF 17.6%\n\n"
        "玄女在道路、水体的 F1 / AUC / AP 均领先；\n建筑假正例仍是下一步重点。",
        (1150, 276, 1520, 420),
        font(16, True),
        INK,
    )
    draw.rounded_rectangle((1125, 455, 1545, 650), radius=6, fill=WHITE, outline=LINE, width=2)
    centered(draw, "AP（稀少目标高置信检出）", (1145, 470, 1525, 510), font(18, True), DEEP_BLUE)
    rows = [
        ("建筑物", "0.221", "0.234", "0.183"),
        ("道路", "0.304", "0.257", "0.259"),
        ("水体", "0.075", "0.057", "0.049"),
    ]
    headers = ("任务", "玄女", "AEF", "传统")
    for column, value in enumerate(headers):
        centered(
            draw,
            value,
            (1140 + column * 95, 518, 1235 + column * 95, 548),
            font(14, True),
            RED if value == "玄女" else INK,
        )
    for row_index, row in enumerate(rows):
        y = 550 + row_index * 30
        for column, value in enumerate(row):
            centered(
                draw,
                value,
                (1140 + column * 95, y, 1235 + column * 95, y + 28),
                font(13, column == 1),
                RED if column == 1 else INK,
            )
    draw.rounded_rectangle((1125, 672, 1545, 790), radius=6, fill=PALE_GRAY)
    centered(
        draw,
        "时间产品边界\n玄女：2026-04 月度嵌入\nAEF：2025 年度嵌入\n传统：2026-04 多源特征",
        (1145, 678, 1525, 784),
        font(15, True),
        INK,
    )
    centered(
        draw,
        "结果边界：单 fold、单次随机 3 多边形实验；用于平台交互能力展示，"
        "统计结论仍需多 fold、多随机种子复核。",
        (55, 805, 1545, 868),
        font(15, True),
        MUTED,
    )
    tasks = ["建筑物", "道路", "水体"]
    p10c = [0.2564693118, 0.3680430440, 0.1105819153]
    return canvas, {
        "slide": 8,
        "title": PAGE_TITLES[8],
        "metrics": ["F1", "AUC", "AP"],
        "p10c": dict(zip(tasks, p10c)),
        "building_f1": p10c[0],
        "road_f1": p10c[1],
        "water_f1": p10c[2],
        "protocol": "same 3 polygons/test split/PU+Query/threshold selection",
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
        "海淀验证方法效果，哈尔滨验证跨区域部署与应用能力｜结果回到地图复核",
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


def merge_complete_deck(output: Path = COMPLETE_OUTPUT) -> None:
    prs = Presentation()
    prs.slide_width = Inches(13.333333)
    prs.slide_height = Inches(7.5)
    for page in range(1, 11):
        preview = ASSET_ROOT / f"slide{page:02d}" / f"slide{page:02d}_preview.png"
        if not preview.exists():
            raise FileNotFoundError(f"Missing reviewed preview for page {page}: {preview}")
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        slide.shapes.add_picture(
            str(preview),
            0,
            0,
            width=prs.slide_width,
            height=prs.slide_height,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    prs.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--page", type=int, choices=range(4, 11))
    group.add_argument("--merge", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.merge:
        merge_complete_deck()
        print(COMPLETE_OUTPUT)
        return
    output, preview = default_paths(args.page)
    build_page(args.page, output, preview)
    print(output)


if __name__ == "__main__":
    main()
