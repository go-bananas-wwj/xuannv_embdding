"""Generate the Xuannv architecture and paradigm chapter deck."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

from create_front8_deck_v2 import C, bg, title, text, rect, line, arrow, picture_fit, claim, IMG


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "玄女科技BP_架构原理章节_v0.1.pptx"

HAIDIAN = Path("/root/workspace/report/assets")
CAP = ROOT / "outputs" / "玄女科技BP_能力案例_v0.2.pptx"
YAJIANG_AGENT = Path("/root/.codex/attachments/071916f0-9ca0-4d8c-83f1-bcd404e7ea3c/e178b904e06c08479cd215ca230ff245.png")


def small_label(slide, value: str, x: float, y: float, w: float, color=C.body) -> None:
    text(slide, value, x, y, w, 0.13, 7, color, True, PP_ALIGN.CENTER)


def chip(slide, value: str, x: float, y: float, w: float, color=C.blue, fill=C.pale_blue) -> None:
    rect(slide, x, y, w, 0.34, fill, color)
    text(slide, value, x + 0.05, y + 0.08, w - 0.10, 0.10, 8, color, True, PP_ALIGN.CENTER)


def node(slide, value: str, x: float, y: float, w: float, h: float = 0.54, color=C.blue, fill=C.pale_blue, size: int = 9) -> None:
    rect(slide, x, y, w, h, fill, color)
    text(slide, value, x + 0.08, y + 0.14, w - 0.16, h - 0.20, size, color, True, PP_ALIGN.CENTER)


def card(slide, head: str, body: str, x: float, y: float, w: float, h: float, color=C.blue, fill=C.pale_blue) -> None:
    rect(slide, x, y, w, h, fill, color)
    text(slide, head, x + 0.16, y + 0.18, w - 0.32, 0.18, 11, color, True, PP_ALIGN.CENTER)
    text(slide, body, x + 0.18, y + 0.52, w - 0.36, h - 0.68, 8, C.body, True, PP_ALIGN.CENTER)


def metric(slide, value: str, label: str, x: float, y: float, color=C.blue, fill=C.pale_blue, w: float = 1.54) -> None:
    rect(slide, x, y, w, 0.72, fill, color)
    text(slide, value, x + 0.08, y + 0.14, w - 0.16, 0.18, 14, color, True, PP_ALIGN.CENTER)
    text(slide, label, x + 0.08, y + 0.46, w - 0.16, 0.12, 7, C.body, True, PP_ALIGN.CENTER)


def layer(slide, name: str, items: list[str], x: float, y: float, w: float, color=C.blue, fill=C.pale_blue) -> None:
    rect(slide, x, y, w, 0.82, fill, color)
    text(slide, name, x + 0.18, y + 0.14, 1.56, 0.16, 11, color, True)
    text(slide, " / ".join(items), x + 1.66, y + 0.17, w - 1.88, 0.18, 9, C.ink, True, PP_ALIGN.RIGHT)


def build() -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    haidian_patch = HAIDIAN / "compare_haidian_construction_patch_000198.png"
    haidian_chart = HAIDIAN / "aef_vs_v1.0_v2.png"
    arch_img = HAIDIAN / "architecture_diagram.png"

    # 1. From cases to paradigm
    s = prs.slides.add_slide(blank)
    bg(s)
    title(s, "19", "从案例到范式：遥感应用正在从项目制走向底座化")
    text(s, "哈尔滨、海淀、雅江证明的不是单点应用，而是同一套地理嵌入可以持续复用到多类真实任务。", 1.02, 1.14, 11.20, 0.30, 13, C.body, False, PP_ALIGN.CENTER)
    text(s, "传统项目制", 0.78, 1.68, 5.32, 0.22, 15, C.amber, True, PP_ALIGN.CENTER)
    steps = ["数据采购", "预处理", "专家标注", "单任务建模", "制图报告", "下个任务重来"]
    xs = [0.86, 2.34, 3.82, 0.86, 2.34, 3.82]
    ys = [2.08, 2.08, 2.08, 3.16, 3.16, 3.16]
    for i, (st, x, y) in enumerate(zip(steps, xs, ys)):
        node(s, st, x, y, 1.18, 0.50, C.amber, C.pale_amber, 8)
        if i in (0, 1, 3, 4):
            arrow(s, x + 1.18, y + 0.25, x + 1.44, y + 0.25, C.amber, 1.2)
        if i == 2:
            arrow(s, x + 0.60, y + 0.50, x + 0.60, y + 1.04, C.amber, 1.2)
    text(s, "一个需求一套流程，新增任务意味着重复投入。", 1.00, 4.20, 4.66, 0.18, 10, C.body, True, PP_ALIGN.CENTER)
    line(s, 6.55, 1.56, 6.55, 5.58, C.line, 0.9)
    text(s, "玄女底座化", 7.20, 1.68, 5.26, 0.22, 15, C.blue, True, PP_ALIGN.CENTER)
    node(s, "多源观测", 7.16, 2.30, 1.30, 0.58, C.green, C.mint, 9)
    arrow(s, 8.46, 2.59, 8.86, 2.59, C.line, 1.3)
    node(s, "统一地理嵌入", 8.88, 2.18, 1.76, 0.82, C.blue, C.pale_blue, 10)
    arrow(s, 10.64, 2.59, 11.02, 2.59, C.line, 1.3)
    node(s, "多任务复用", 11.04, 2.30, 1.30, 0.58, C.purple, C.pale_purple, 9)
    for label, x, color, fill in [
        ("变化检测", 7.12, C.blue, C.pale_blue),
        ("地物分类", 8.48, C.green, C.mint),
        ("报告生成", 9.84, C.purple, C.pale_purple),
        ("智能体问答", 11.20, C.amber, C.pale_amber),
    ]:
        chip(s, label, x, 3.58, 1.04, color, fill)
    picture_fit(s, IMG["dynamic"], 7.24, 4.18, 2.22, 1.04)
    picture_fit(s, IMG["cloud"], 9.82, 4.18, 2.22, 1.04)
    claim(s, "玄女的核心不是做更多项目，而是把遥感项目沉淀为可复用的地理智能基础设施。", 6.64, C.blue)

    # 2. Technical paradigm
    s = prs.slides.add_slide(blank)
    bg(s)
    title(s, "20", "技术范式：从“图像学 + 专家规则”到“多模态自学习”")
    text(s, "传统遥感依赖显性规则；玄女让模型从多模态、多时序观测中学习隐性地理规律。", 1.02, 1.14, 11.20, 0.28, 13, C.body, False, PP_ALIGN.CENTER)
    text(s, "传统图像学范式", 0.86, 1.62, 5.30, 0.20, 15, C.amber, True, PP_ALIGN.CENTER)
    for i, label in enumerate(["专家定义特征", "人工融合数据", "单任务训练", "换任务重做"]):
        node(s, label, 1.04 + i * 1.18, 2.24, 1.02, 0.58, C.amber, C.pale_amber, 8)
        if i < 3:
            arrow(s, 2.06 + i * 1.18, 2.53, 2.20 + i * 1.18, 2.53, C.amber, 1.2)
    text(s, "慢：交付依赖专家经验和项目人力。", 1.10, 3.22, 4.90, 0.18, 10, C.body, True, PP_ALIGN.CENTER)
    picture_fit(s, IMG["s2"], 1.28, 3.82, 1.26, 1.10)
    picture_fit(s, IMG["worldcover"], 2.72, 3.82, 1.26, 1.10)
    picture_fit(s, IMG["building"], 4.16, 3.82, 1.26, 1.10)
    line(s, 6.56, 1.56, 6.56, 5.78, C.line, 0.9)
    text(s, "玄女自学习范式", 7.18, 1.62, 5.36, 0.20, 15, C.blue, True, PP_ALIGN.CENTER)
    for i, label in enumerate(["多模态对齐", "时空规律学习", "DV 地理嵌入", "多任务激活"]):
        node(s, label, 7.10 + i * 1.26, 2.24, 1.12, 0.58, C.blue if i != 1 else C.green, C.pale_blue if i != 1 else C.mint, 8)
        if i < 3:
            arrow(s, 8.22 + i * 1.26, 2.53, 8.36 + i * 1.26, 2.53, C.blue, 1.2)
    text(s, "快：底座先学习通用表征，下游任务轻量适配。", 7.20, 3.22, 5.02, 0.18, 10, C.body, True, PP_ALIGN.CENTER)
    picture_fit(s, IMG["s2hr"], 7.38, 3.82, 1.26, 1.10)
    picture_fit(s, IMG["dem"], 8.82, 3.82, 1.26, 1.10)
    picture_fit(s, IMG["semantic"], 10.26, 3.82, 1.26, 1.10)
    claim(s, "技术变化的本质：从“人工解释影像”转向“模型自学习地理规律”。", 6.58, C.blue)

    # 3. Digital earth to embedded earth
    s = prs.slides.add_slide(blank)
    bg(s)
    title(s, "21", "核心原理：从“数字地球”到“嵌入地球”")
    text(s, "数字地球解决数据存储；嵌入地球解决知识编码。玄女把同一地理位置的多源观测转成可计算的 DV 地理嵌入向量。", 1.02, 1.14, 11.20, 0.30, 13, C.body, False, PP_ALIGN.CENTER)
    card(s, "数字地球", "影像、矢量、DEM、专题数据分散存储，使用时仍需人工解读。", 0.86, 1.84, 3.20, 1.16, C.amber, C.pale_amber)
    picture_fit(s, IMG["s2"], 1.14, 3.30, 0.92, 0.82)
    picture_fit(s, IMG["s1"], 2.06, 3.52, 0.92, 0.82)
    picture_fit(s, IMG["dem"], 2.98, 3.30, 0.92, 0.82)
    arrow(s, 4.28, 3.06, 5.10, 3.06, C.line, 1.4)
    card(s, "嵌入地球", "把时间、空间、频谱和语义压入统一表征，形成可比较、可检索、可复用的地理知识。", 5.28, 1.74, 3.06, 1.32, C.blue, C.pale_blue)
    rect(s, 5.72, 3.36, 2.18, 0.82, C.pale_blue, C.blue)
    text(s, "DV 地理嵌入向量", 5.94, 3.62, 1.74, 0.12, 11, C.blue, True, PP_ALIGN.CENTER)
    for i, label in enumerate(["T", "X/Y", "谱", "义"]):
        chip(s, label, 5.54 + i * 0.58, 4.48, 0.38, C.blue, C.white)
    arrow(s, 8.54, 3.06, 9.30, 3.06, C.line, 1.4)
    card(s, "下游调用", "变化检测、地物分类、检索、预测、问答和报告生成共用同一套底座表征。", 9.50, 1.84, 2.90, 1.16, C.green, C.mint)
    for label, x, y in [("变化", 9.58, 3.44), ("分类", 10.36, 3.44), ("检索", 11.14, 3.44), ("报告", 10.00, 4.10), ("问答", 10.78, 4.10)]:
        chip(s, label, x, y, 0.62, C.green, C.white)
    claim(s, "GPT 先把文本转成 token；玄女先把地球观测转成地理嵌入。", 6.58, C.blue)

    # 4. Model architecture
    s = prs.slides.add_slide(blank)
    bg(s)
    title(s, "22", "模型架构：多模态观测进入统一地理嵌入空间")
    text(s, "玄女底座通过时空多模态编码器，将不同传感器、不同时间、不同分辨率的数据对齐到同一套 DV 地理嵌入向量。", 1.02, 1.14, 11.20, 0.30, 13, C.body, False, PP_ALIGN.CENTER)
    text(s, "Input", 0.86, 1.70, 2.28, 0.18, 13, C.blue, True, PP_ALIGN.CENTER)
    for i, (label, img) in enumerate([("光学", IMG["s2"]), ("SAR", IMG["s1"]), ("DEM", IMG["dem"]), ("多光谱", IMG["landsat"])]):
        picture_fit(s, img, 0.92 + (i % 2) * 1.16, 2.06 + (i // 2) * 1.20, 0.92, 0.84)
        small_label(s, label, 0.92 + (i % 2) * 1.16, 2.96 + (i // 2) * 1.20, 0.92, C.body)
    arrow(s, 3.18, 3.00, 3.90, 3.00, C.line, 1.4)
    rect(s, 4.08, 2.12, 2.02, 1.76, C.pale_blue, C.blue)
    text(s, "时空多模态编码器", 4.26, 2.48, 1.66, 0.18, 13, C.blue, True, PP_ALIGN.CENTER)
    text(s, "时空对齐 / 跨模态融合 / 时序学习", 4.28, 2.92, 1.62, 0.28, 8, C.body, True, PP_ALIGN.CENTER)
    arrow(s, 6.10, 3.00, 6.86, 3.00, C.line, 1.4)
    rect(s, 7.04, 2.10, 1.98, 1.80, C.mint, C.green)
    text(s, "DV 地理嵌入向量", 7.22, 2.50, 1.62, 0.18, 13, C.green, True, PP_ALIGN.CENTER)
    text(s, "像素级 / 格网级\n可复用表征", 7.30, 2.94, 1.46, 0.34, 9, C.body, True, PP_ALIGN.CENTER)
    arrow(s, 9.02, 3.00, 9.74, 3.00, C.line, 1.4)
    text(s, "Task Heads", 9.90, 1.70, 2.56, 0.18, 13, C.blue, True, PP_ALIGN.CENTER)
    for i, label in enumerate(["LUCC", "变化检测", "高程回归", "地物分类", "检索问答", "报告生成"]):
        chip(s, label, 9.78 + (i % 2) * 1.16, 2.08 + (i // 2) * 0.62, 1.02, C.purple if i % 2 else C.blue, C.pale_purple if i % 2 else C.pale_blue)
    if arch_img.exists():
        picture_fit(s, arch_img, 1.10, 4.76, 3.12, 1.08)
        small_label(s, "工程化模型架构示意", 1.10, 5.92, 3.12, C.body)
    text(s, "一套底座表征，多个任务出口。新增业务不必从原始影像重新开始。", 4.58, 5.22, 6.90, 0.24, 14, C.blue, True, PP_ALIGN.CENTER)
    claim(s, "能力的复用来自统一嵌入空间，而不是为每个场景重新堆模型。", 6.58, C.blue)

    # 5. Product and business architecture
    s = prs.slides.add_slide(blank)
    bg(s)
    title(s, "23", "产品架构：一个通用底座，支撑多类场景应用")
    text(s, "玄女不是交付一个模型文件，而是把数据、模型、任务头、智能体和报告工作流封装成可调用的产品能力。", 1.02, 1.14, 11.20, 0.30, 13, C.body, False, PP_ALIGN.CENTER)
    layer(s, "数据层", ["遥感", "DEM", "气象", "社会经济", "本土标签"], 0.96, 1.82, 11.28, C.green, C.mint)
    layer(s, "底座层", ["多模态编码器", "DV 地理嵌入库", "向量检索", "任务头"], 0.96, 3.02, 11.28, C.blue, C.pale_blue)
    layer(s, "应用层", ["政务监测", "城市治理", "水电站监测", "智能体报告", "API/SDK"], 0.96, 4.22, 11.28, C.purple, C.pale_purple)
    arrow(s, 6.60, 2.66, 6.60, 2.98, C.line, 1.2)
    arrow(s, 6.60, 3.86, 6.60, 4.18, C.line, 1.2)
    text(s, "业务范式变化", 0.96, 5.46, 2.20, 0.18, 13, C.blue, True)
    for i, (old, new) in enumerate([("专家项目制", "底座调用式"), ("一次性交付", "持续服务"), ("人力扩张", "数据闭环")]):
        text(s, old, 3.08 + i * 2.86, 5.44, 1.04, 0.14, 9, C.amber, True, PP_ALIGN.CENTER)
        arrow(s, 4.10 + i * 2.86, 5.52, 4.48 + i * 2.86, 5.52, C.line, 1.0)
        text(s, new, 4.54 + i * 2.86, 5.44, 1.04, 0.14, 9, C.blue, True, PP_ALIGN.CENTER)
    claim(s, "商业价值来自底座复用：把高成本定制服务，变成可持续调用的地理智能能力。", 6.58, C.blue)

    # 6. Roadmap and public benefit
    s = prs.slides.add_slide(blank)
    bg(s)
    title(s, "24", "三年路径：从技术验证到地理智能普惠化")
    text(s, "未来的玄女底座不仅理解遥感影像，也将接入人口、产业、交通、用地、夜光等社会经济数据，让地理智能进入更多公共服务和商业场景。", 1.02, 1.14, 11.20, 0.30, 13, C.body, False, PP_ALIGN.CENTER)
    years = [
        ("2026", "技术验证", "完成中国区域 DV 地理嵌入底座验证；形成哈尔滨、海淀、雅江等标杆案例。", "试点收入"),
        ("2027", "商业验证", "行业专用模型、API/License 和智能体报告产品化，覆盖政务与企业客户。", "千万级目标"),
        ("2028", "规模化", "融合社会经济数据，向城市治理、农业保险、自然资源、公众服务扩展。", "亿元级目标"),
    ]
    for i, (yr, head, body, money) in enumerate(years):
        x = 0.92 + i * 4.02
        rect(s, x, 1.98, 3.34, 2.30, C.white, C.line)
        text(s, yr, x + 0.18, 2.20, 0.82, 0.26, 18, C.blue if i != 2 else C.green, True)
        text(s, head, x + 1.08, 2.24, 1.86, 0.18, 13, C.ink, True, PP_ALIGN.RIGHT)
        text(s, body, x + 0.22, 2.78, 2.90, 0.54, 9, C.body, True, PP_ALIGN.CENTER)
        chip(s, money, x + 0.82, 3.66, 1.60, C.blue if i != 2 else C.green, C.pale_blue if i != 2 else C.mint)
        if i < 2:
            arrow(s, x + 3.34, 3.12, x + 3.76, 3.12, C.line, 1.2)
    text(s, "社会经济数据接入", 1.00, 4.76, 2.12, 0.18, 13, C.blue, True)
    for i, label in enumerate(["人口", "产业", "交通", "用地", "夜光", "POI"]):
        chip(s, label, 3.14 + i * 0.84, 4.72, 0.62, C.green, C.mint)
    text(s, "让模型从“看见地表变化”，进一步走向“理解变化背后的社会经济活动”。", 1.00, 5.34, 6.34, 0.22, 12, C.ink, True)
    metric(s, "基层治理", "更低门槛", 8.00, 4.80, C.blue, C.pale_blue, 1.36)
    metric(s, "中小企业", "按需调用", 9.64, 4.80, C.green, C.mint, 1.36)
    metric(s, "公众服务", "普惠应用", 11.28, 4.80, C.purple, C.pale_purple, 1.36)
    claim(s, "长期愿景：让遥感能力像用地图一样简单，让地理智能成为社会经济运行的基础服务。", 6.58, C.blue)

    return prs


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs = build()
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
