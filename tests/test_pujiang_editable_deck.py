from __future__ import annotations

import hashlib
import importlib.util
import sys
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.util import Inches

SCRIPT = Path(__file__).parents[1] / "scripts/report/pujiang_editable_common.py"
PAGES04_07_SCRIPT = Path(__file__).parents[1] / "scripts/report/pujiang_editable_pages04_07.py"
PAGES08_10_SCRIPT = Path(__file__).parents[1] / "scripts/report/pujiang_editable_pages08_10.py"
DECK_SCRIPT = Path(__file__).parents[1] / "scripts/report/build_pujiang_editable_deck.py"
PRESENTATION_ROOT = Path(__file__).parents[1] / "docs/presentations/pujiang_202607"
EDITABLE_SOURCE_SLIDES = [
    PRESENTATION_ROOT / f"玄女月度地理嵌入_浦江交流_第{page:02d}页_20260726.pptx"
    for page in range(1, 4)
]
SPEC = importlib.util.spec_from_file_location("pujiang_editable_common", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_helpers_create_separate_text_and_picture_shapes(tmp_path: Path) -> None:
    image_path = tmp_path / "observation.png"
    Image.new("RGB", (80, 40), (33, 128, 94)).save(image_path)
    presentation = Presentation()
    presentation.slide_width = Inches(13.333333)
    presentation.slide_height = Inches(7.5)

    slide = MODULE.new_blank_slide(presentation)
    text_shape = MODULE.add_text(slide, "可编辑标题", 0.6, 0.7, 3.2, 0.5, 22)
    picture_shape = MODULE.add_picture(slide, image_path, 4.0, 1.0, 2.4, 1.2)

    assert text_shape is not picture_shape
    assert text_shape.text == "可编辑标题"
    assert picture_shape.shape_type == MSO_SHAPE_TYPE.PICTURE
    assert len(slide.shapes) == 2


def test_full_slide_picture_detector_uses_presentation_geometry(tmp_path: Path) -> None:
    image_path = tmp_path / "background.png"
    Image.new("RGB", (160, 90), (236, 246, 252)).save(image_path)
    presentation = Presentation()
    presentation.slide_width = Inches(13.333333)
    presentation.slide_height = Inches(7.5)
    slide = MODULE.new_blank_slide(presentation)

    full_slide_picture = slide.shapes.add_picture(
        str(image_path),
        0,
        0,
        width=presentation.slide_width,
        height=presentation.slide_height,
    )
    oversized_picture = slide.shapes.add_picture(
        str(image_path),
        -Inches(0.2),
        -Inches(0.1),
        width=presentation.slide_width + Inches(0.4),
        height=presentation.slide_height + Inches(0.2),
    )
    inset_picture = MODULE.add_picture(slide, image_path, 0.6, 0.7, 3.2, 1.8)

    assert MODULE.is_full_slide_picture(full_slide_picture, presentation)
    assert MODULE.is_full_slide_picture(oversized_picture, presentation)
    assert not MODULE.is_full_slide_picture(inset_picture, presentation)


def test_clone_first_slide_keeps_picture_relationships(tmp_path: Path) -> None:
    image_path = tmp_path / "source.png"
    source_path = tmp_path / "source.pptx"
    destination_path = tmp_path / "destination.pptx"
    Image.new("RGB", (80, 40), (0, 97, 170)).save(image_path)

    source = Presentation()
    source.slide_width = Inches(13.333333)
    source.slide_height = Inches(7.5)
    source_slide = MODULE.new_blank_slide(source)
    MODULE.add_text(source_slide, "原生文字", 0.6, 0.7, 3.2, 0.5, 22)
    MODULE.add_picture(source_slide, image_path, 4.0, 1.0, 2.4, 1.2)
    source.save(source_path)

    destination = Presentation()
    MODULE.clone_first_slide(source_path, destination)
    destination.save(destination_path)

    cloned = Presentation(destination_path).slides[0]
    assert any(shape.text == "原生文字" for shape in cloned.shapes if hasattr(shape, "text"))
    assert any(shape.shape_type == MSO_SHAPE_TYPE.PICTURE for shape in cloned.shapes)


def test_clone_real_editable_slides_uses_unique_parts_and_preserves_content(tmp_path: Path) -> None:
    output = tmp_path / "editable_deck.pptx"
    destination = Presentation()

    for source in EDITABLE_SOURCE_SLIDES:
        MODULE.clone_first_slide(source, destination)
    destination.save(output)

    reopened = Presentation(output)
    assert len(reopened.slides) == len(EDITABLE_SOURCE_SLIDES)
    with zipfile.ZipFile(output) as archive:
        member_counts = Counter(archive.namelist())
    assert not [member for member, count in member_counts.items() if count > 1]

    for source_path, cloned_slide in zip(EDITABLE_SOURCE_SLIDES, reopened.slides, strict=True):
        source_slide = Presentation(source_path).slides[0]
        assert _slide_text(source_slide) == _slide_text(cloned_slide)
        assert _picture_hashes(source_slide) == _picture_hashes(cloned_slide)


def _slide_text(slide) -> list[str]:
    return [shape.text for shape in slide.shapes if hasattr(shape, "text")]


def _picture_hashes(slide) -> list[str]:
    return sorted(
        hashlib.sha256(shape.image.blob).hexdigest()
        for shape in slide.shapes
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE
    )


def _pages04_07_module():
    spec = importlib.util.spec_from_file_location("pujiang_editable_pages04_07", PAGES04_07_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _editable_pages04_07() -> Presentation:
    presentation = Presentation()
    pages = _pages04_07_module()
    for builder in (pages.build_page04, pages.build_page05, pages.build_page06, pages.build_page07):
        builder(presentation)
    return presentation


def test_pages04_to_07_have_required_titles_shapes_and_no_full_slide_picture() -> None:
    presentation = _editable_pages04_07()
    expected = [
        ("玄女嵌入平台：浏览、分析、训练与生成", 20),
        ("少量标注，快速形成区域级专题图", 35),
        ("嵌入底座进入遥感智能体工作流", 22),
        ("云遮挡或观测缺失时，生成指定时刻的遥感参考影像", 18),
    ]

    assert len(presentation.slides) == len(expected)
    for slide, (title, minimum_shapes) in zip(presentation.slides, expected, strict=True):
        assert title in _slide_text(slide)
        assert len(slide.shapes) >= minimum_shapes
        assert not any(MODULE.is_full_slide_picture(shape, presentation) for shape in slide.shapes)


def test_page05_has_18_independent_raster_panels_and_native_labels() -> None:
    presentation = _editable_pages04_07()
    slide = presentation.slides[1]
    pictures = [shape for shape in slide.shapes if shape.shape_type == MSO_SHAPE_TYPE.PICTURE]

    assert len(pictures) >= 18
    assert all(not MODULE.is_full_slide_picture(shape, presentation) for shape in pictures)
    text = _slide_text(slide)
    for label in (
        "建筑物",
        "道路",
        "水体",
        "3 个圈选",
        "测试影像",
        "玄女候选",
        "AEF 候选",
        "传统候选",
        "真实标签",
    ):
        assert label in text


def test_page07_splits_the_observation_timeline_into_independent_pictures() -> None:
    presentation = _editable_pages04_07()
    slide = presentation.slides[3]
    pictures = [shape for shape in slide.shapes if shape.shape_type == MSO_SHAPE_TYPE.PICTURE]

    dimensions = [_picture_dimensions(shape) for shape in pictures]
    timeline_dimensions = [size for size in dimensions if size[1] == 262]

    assert len(pictures) >= 6
    assert Counter(timeline_dimensions) == Counter([(275, 262)] * 4 + [(276, 262)])
    timeline_pictures = [shape for shape in pictures if _picture_dimensions(shape)[1] == 262]
    assert not any(_has_all_black_edge(shape) for shape in timeline_pictures)
    assert any("使用边界｜左侧两图用于说明缺测场景" in text for text in _slide_text(slide))


def test_page06_marks_the_workflow_as_a_conceptual_diagram() -> None:
    presentation = _editable_pages04_07()

    assert "概念流程示意（非真实系统界面）" in _slide_text(presentation.slides[2])


def test_page06_workflow_text_boxes_have_sufficient_height_and_spacing() -> None:
    presentation = _editable_pages04_07()
    slide = presentation.slides[2]
    text_shapes = {shape.text: shape for shape in slide.shapes if hasattr(shape, "text")}

    concept_label = text_shapes["概念流程示意（非真实系统界面）"]
    assert concept_label.width >= Inches(2.5)
    assert concept_label.height >= Inches(0.24)

    rows = (
        ("理解任务", "区域、月份、目标类别与输出形式"),
        ("读取嵌入", "embedding_tool：加载月度 64 维嵌入"),
        ("选择工具", "系统任务头、变化分析、任务摘要"),
        ("空间计算", "全域推理、统计汇总与质量检查"),
        ("交付结果", "地图图层、案例图与结构化报告"),
    )
    for title, detail in rows:
        title_shape = text_shapes[title]
        detail_shape = text_shapes[detail]
        assert title_shape.height >= Inches(0.22)
        assert title_shape.top + title_shape.height <= detail_shape.top


def _pages08_10_module():
    spec = importlib.util.spec_from_file_location("pujiang_editable_pages08_10", PAGES08_10_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_pages08_to_10_use_native_metrics_cards_and_independent_pictures() -> None:
    presentation = Presentation()
    pages = _pages08_10_module()
    for builder in (pages.build_page08, pages.build_page09, pages.build_page10):
        builder(presentation)

    page08, page09, page10 = presentation.slides
    assert "海淀区 3 多边形同协议评测：道路与水体领先" in _slide_text(page08)
    assert all(shape.shape_type != MSO_SHAPE_TYPE.PICTURE for shape in page08.shapes)
    assert sum(shape.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE for shape in page08.shapes) >= 35
    for value in ("F1", "AUC", "0.368", "0.745", "0.304", "单 fold、单次随机 3 多边形实验"):
        assert any(value in text for text in _slide_text(page08))

    assert "嵌入结构与任务表现：优势、边界和下一步" in _slide_text(page09)
    assert len(
        [shape for shape in page09.shapes if shape.shape_type == MSO_SHAPE_TYPE.PICTURE]
    ) == 1
    for card in (
        "道路｜当前优势最稳定",
        "水体｜排序能力领先",
        "建筑｜建筑假正例仍偏多",
        "下一步验证",
    ):
        assert card in _slide_text(page09)

    assert "哈尔滨新区：从月度嵌入到城市治理专题" in _slide_text(page10)
    assert len(
        [shape for shape in page10.shapes if shape.shape_type == MSO_SHAPE_TYPE.PICTURE]
    ) >= 5
    assert any("土地利用 / 覆盖" in text for text in _slide_text(page10))


def test_final_editable_deck_has_ten_native_slides_without_duplicate_zip_members(
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location("build_pujiang_editable_deck", DECK_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    output = tmp_path / "pujiang_editable_deck.pptx"

    module.build_deck(output)

    presentation = Presentation(output)
    expected_titles = [
        "玄女月度地理嵌入",
        "两个区域的多源数据处理与训练组织",
        "从 AEF 思路到城市尺度月度嵌入",
        "玄女嵌入平台：浏览、分析、训练与生成",
        "少量标注，快速形成区域级专题图",
        "嵌入底座进入遥感智能体工作流",
        "云遮挡或观测缺失时，生成指定时刻的遥感参考影像",
        "海淀区 3 多边形同协议评测：道路与水体领先",
        "嵌入结构与任务表现：优势、边界和下一步",
        "哈尔滨新区：从月度嵌入到城市治理专题",
    ]
    assert len(presentation.slides) == len(expected_titles)
    for slide, title in zip(presentation.slides, expected_titles, strict=True):
        assert any(title in text for text in _slide_text(slide))
        assert len(slide.shapes) > 1
        assert any(shape.has_text_frame and shape.text.strip() for shape in slide.shapes)
        assert not any(MODULE.is_full_slide_picture(shape, presentation) for shape in slide.shapes)

    page05 = presentation.slides[4]
    assert len(
        [shape for shape in page05.shapes if shape.shape_type == MSO_SHAPE_TYPE.PICTURE]
    ) >= 18
    page08 = presentation.slides[7]
    assert all(shape.shape_type != MSO_SHAPE_TYPE.PICTURE for shape in page08.shapes)
    assert sum(shape.shape_type == MSO_SHAPE_TYPE.AUTO_SHAPE for shape in page08.shapes) >= 35
    with zipfile.ZipFile(output) as archive:
        member_counts = Counter(archive.namelist())
    assert not [member for member, count in member_counts.items() if count > 1]


def _picture_dimensions(shape) -> tuple[int, int]:
    with Image.open(BytesIO(shape.image.blob)) as image:
        return image.size


def _has_all_black_edge(shape) -> bool:
    with Image.open(BytesIO(shape.image.blob)) as image:
        picture = image.convert("RGB")
        edges = (
            picture.crop((0, 0, picture.width, 1)),
            picture.crop((0, picture.height - 1, picture.width, picture.height)),
            picture.crop((0, 0, 1, picture.height)),
            picture.crop((picture.width - 1, 0, picture.width, picture.height)),
        )
    return any(all(pixel == (0, 0, 0) for pixel in edge.get_flattened_data()) for edge in edges)
