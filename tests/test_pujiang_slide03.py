from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation

SCRIPT = Path(__file__).parents[1] / "scripts/report/build_pujiang_slide03.py"
SPEC = importlib.util.spec_from_file_location("build_pujiang_slide03", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_slide03_states_p10c_adaptations_without_overclaiming(tmp_path: Path) -> None:
    output = tmp_path / "slide03.pptx"

    MODULE.build_ppt(MODULE.DEFAULT_SOURCE_PPTX, output)

    presentation = Presentation(output)
    assert len(presentation.slides) == 1
    text = "\n".join(
        shape.text for shape in presentation.slides[0].shapes if hasattr(shape, "text")
    )
    for expected in [
        "参考 AEF 的 embedding field 范式",
        "不是完全复现 AEF",
        "128×128×64",
        "连续 6 期",
        "跨月份注意力",
        "模态遮挡",
        "月份遮挡 0.65",
        "样本启用概率",
        "空间块遮挡 0.65",
        "期望遮挡面积约 0.32",
        "availability mask",
        "pixel mask",
        "人工困难遮挡保留 availability mask",
        "uniformity 0.06",
        "13 类 OSM 弱语义 0.14",
        "困难负样本",
        "高分光学重建 0.90",
        "高分观测先聚合为单帧，再逐月注入月度嵌入",
        "S1 0.25",
        "高分 SAR 0.35",
        "人工变化监督 0",
        "时间对比损失 0",
        "配置权重，不代表损失贡献",
    ]:
        assert expected in text
    font_sizes = [
        run.font.size.pt
        for shape in presentation.slides[0].shapes
        if hasattr(shape, "text_frame")
        for paragraph in shape.text_frame.paragraphs
        for run in paragraph.runs
        if run.text.strip() and run.font.size is not None
    ]
    assert min(font_sizes) >= 12


def test_preview_has_widescreen_dimensions(tmp_path: Path) -> None:
    output = tmp_path / "preview.png"

    MODULE.build_preview(output)

    with Image.open(output) as image:
        assert image.size == (1600, 900)
