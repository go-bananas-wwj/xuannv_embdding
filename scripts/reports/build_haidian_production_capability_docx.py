#!/usr/bin/env python3
"""Build the shareable Haidian production capability report as a DOCX file."""

from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor


REPO_ROOT = Path("/root/workspace/xuannv")
ASSET_ROOT = REPO_ROOT / "docs/production/assets"
DEFAULT_OUTPUT = REPO_ROOT / "docs/reports/haidian_production_capability_report_20260720.docx"


def set_cell_text(cell, text: str, bold: bool = False) -> None:
    cell.text = ""
    run = cell.paragraphs[0].add_run(text)
    run.bold = bold
    run.font.size = Pt(9)


def add_table(document: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Light Shading Accent 1"
    for cell, header in zip(table.rows[0].cells, headers):
        set_cell_text(cell, header, bold=True)
    for row in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            set_cell_text(cell, value)


def add_figure(document: Document, relative_path: str, caption: str, width_cm: float = 15.8) -> None:
    image_path = ASSET_ROOT / relative_path
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(image_path), width=Cm(width_cm))
    caption_paragraph = document.add_paragraph(caption)
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in caption_paragraph.runs:
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(89, 101, 121)


def add_bullet_list(document: Document, items: list[str]) -> None:
    for item in items:
        document.add_paragraph(item, style="List Bullet")


def add_heading(document: Document, text: str, level: int = 1) -> None:
    document.add_heading(text, level=level)


def build(output_path: Path) -> None:
    document = Document()
    section = document.sections[0]
    section.top_margin = Cm(1.8)
    section.bottom_margin = Cm(1.8)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)

    normal = document.styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal.font.size = Pt(10.5)

    title = document.add_heading("玄女海淀月度地理 Embedding\n生产能力成果报告", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_paragraph("当前生产版：P10C epoch 800 | 更新日期：2026-07-20").alignment = (
        WD_ALIGN_PARAGRAPH.CENTER
    )
    document.add_paragraph(
        "本报告仅汇总已完成的产品训练、下游验证、可视化与交付能力；不包含论文实验计划或投稿进度。"
    ).alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_paragraph()

    add_heading(document, "一、成果摘要")
    add_bullet_list(
        document,
        [
            "当前生产版本为 P10C epoch 800：在多任务、few-shot 与原型检索的综合横评中最均衡。P2A/P1B 为历史实验，P10A 为 P10 训练链早期候选。",
            "模型将 Sentinel-2、Sentinel-1、Landsat、高分光学与高分 SAR 的月度观测压缩为 128×128×64 稠密地理 embedding，可由轻量下游头复用于多类制图任务。",
            "核心业务价值在少量标注快速制图：5-shot 和 10-shot 条件下，建筑、道路、水体均优于直接使用原始多源影像训练的强基线。",
            "零训练向量检索已经具备候选发现能力：水体检索精度较高，道路召回较高；建筑仍建议叠加轻量头或人工核查。",
        ],
    )

    add_heading(document, "二、模型、数据与训练方法")
    document.add_paragraph(
        "训练窗口覆盖 2025-12 至 2026-05。主时序输入为 Sentinel-2、Sentinel-1 和 Landsat；高分光学与高分 SAR 作为独立模态。"
        "训练前检查云雾、阴影、无效像素、同月多景质量和跨源空间对齐。"
    )
    document.add_paragraph(
        "P10C 采用多模态重建、月份/模态/空间块遮挡、跨月时空融合，以及清洗后的 OSM 弱语义探针和困难负样本。"
        "模型保持 128×128 空间输出，避免将建筑边界和道路等细节压缩到低分辨率格网。"
    )

    add_heading(document, "三、核心下游任务表现")
    document.add_paragraph(
        "评测使用海淀 2026-04 embedding，冻结特征后训练轻量 conv3×3 探针。阈值仅由验证集选取，再报告测试集 F1、AP 和 AUC。"
    )
    add_table(
        document,
        ["任务", "F1", "AP", "AUC", "解读"],
        [
            ["施工工地检测", "0.550", "0.544", "0.959", "可用于监测候选筛选与变化核查。"],
            ["建筑物提取", "0.484", "0.450", "0.896", "屋顶、硬化地面与道路纹理相近时仍有假正例。"],
            ["道路提取", "0.522", "0.566", "0.834", "局部连通性可用，细长目标仍受 10 m 网格影响。"],
            ["水体提取", "0.624", "0.604", "0.891", "三项基础地物中最稳定。"],
        ],
    )
    add_table(
        document,
        ["任务（F1 / AP）", "P10C 64维", "AEF 官方 64维", "DINOv3 1024维"],
        [
            ["施工工地", "0.550 / 0.544", "0.545 / 0.567", "0.582 / 0.522"],
            ["建筑", "0.484 / 0.450", "0.486 / 0.422", "0.499 / 0.482"],
            ["道路", "0.522 / 0.566", "0.517 / 0.564", "0.602 / 0.673"],
            ["水体", "0.624 / 0.604", "0.666 / 0.687", "0.677 / 0.720"],
        ],
    )
    document.add_paragraph(
        "P10C 与 AEF 均为 64 维并接入同一轻量头。DINOv3 为 1024 维特征，作为高信息预算能力参照，不能表述为同成本对比。"
    )

    add_heading(document, "四、少样本快速制图")
    document.add_paragraph(
        "5/10/50-shot 分别指使用 5/10/50 个正样本 patch，并配同数负样本 patch 训练下游头。"
        "所有对比使用相同标签、划分与验证集阈值选择。"
    )
    add_table(
        document,
        ["任务", "5-shot：玄女 / 原始影像", "10-shot：玄女 / 原始影像", "50-shot：玄女 / 原始影像"],
        [
            ["建筑", "0.459 / 0.418（+9.8%）", "0.454 / 0.439（+3.5%）", "0.490 / 0.467（+5.0%）"],
            ["道路", "0.474 / 0.401（+18.3%）", "0.487 / 0.435（+11.8%）", "0.517 / 0.501（+3.3%）"],
            ["水体", "0.613 / 0.433（+41.5%）", "0.612 / 0.487（+25.6%）", "0.631 / 0.631（持平）"],
        ],
    )
    add_figure(document, "haidian_v1_20260708/fewshot_best_f1.png", "图 1. 建筑、道路、水体的少样本最佳 F1 对比。")
    add_figure(document, "haidian_v1_20260708/fewshot_5shot_examples.png", "图 2. 只提供 5 个正样本 patch 时的代表性预测对比。")
    add_figure(document, "haidian_v1_20260708/fewshot_head_heatmap.png", "图 3. 不同下游头的少样本表现；轻量 conv3×3 通常更稳定。")

    add_heading(document, "五、全域 embedding 与核心任务可视化")
    add_figure(document, "nonbuilding_fewshot_semantic_20260705/xuannv_aef_global_pca_compare.png", "图 4. 玄女与 AEF 的海淀全域 embedding PCA。用于检查空间连续性与地物边界，不单独作为性能结论。")
    add_figure(document, "haidian_fine_osm_leadership_20260705/foundation_building_water_road_320patch.png", "图 5. 建筑、道路、水体的海淀 320 patch 全域制图。")
    add_figure(document, "haidian_v1_20260708/building_xuannv_vs_raw_examples.png", "图 6. 建筑任务的原始影像、真值与预测对照。")
    add_figure(document, "haidian_v1_20260708/road_xuannv_vs_raw_examples.png", "图 7. 道路任务的原始影像、真值与预测对照。")
    add_figure(document, "haidian_v1_20260708/water_xuannv_vs_raw_examples.png", "图 8. 水体任务的原始影像、真值与预测对照。")

    document.add_section(WD_SECTION.NEW_PAGE)
    add_heading(document, "六、零训练向量检索报告")
    document.add_paragraph(
        "向量检索不训练下游分割头。只在训练划分的少量目标像素上构造类别 prototype，"
        "在测试区域逐像素计算 cosine similarity。验证集选阈值后，输出测试指标和全域相似度图。"
        "因此它衡量的是 embedding 自身是否将语义相似的地物组织在相近位置。"
    )
    add_table(
        document,
        ["目标", "最佳原型方式", "F1", "Precision", "Recall", "IoU", "AUC", "AP"],
        [
            ["建筑物", "单类全量 + whitening", "0.3546", "0.2935", "0.4479", "0.2155", "0.8374", "0.2912"],
            ["道路", "单类全量", "0.4492", "0.3555", "0.6101", "0.2897", "0.7721", "0.3760"],
            ["水体", "正负多原型（8 个）", "0.5858", "0.7811", "0.4687", "0.4143", "0.8506", "0.5547"],
        ],
    )
    add_bullet_list(
        document,
        [
            "水体：原型检索精度与 AP 较高，可直接用于缩小人工排查范围。",
            "道路：召回较高，适合作为候选发现图；硬化地面会造成误检，建议后续接轻量头细化。",
            "建筑：相似屋顶、道路与硬化地面仍会混淆，适合做候选筛选而非替代下游制图头。",
        ],
    )
    add_figure(document, "haidian_v1_vector_retrieval_20260710/building_method_metrics_bar.png", "图 9. 建筑的不同原型构造方式比较。")
    add_figure(document, "haidian_v1_vector_retrieval_20260710/road_method_metrics_bar.png", "图 10. 道路的不同原型构造方式比较。")
    add_figure(document, "haidian_v1_vector_retrieval_20260710/water_method_metrics_bar.png", "图 11. 水体的不同原型构造方式比较。")
    add_figure(document, "haidian_v1_vector_retrieval_20260710/building_similarity_320patch.png", "图 12. 建筑原型在海淀 320 patch 的全域相似度图。")
    add_figure(document, "haidian_v1_vector_retrieval_20260710/road_similarity_320patch.png", "图 13. 道路原型在海淀 320 patch 的全域相似度图。")
    add_figure(document, "haidian_v1_vector_retrieval_20260710/water_similarity_320patch.png", "图 14. 水体原型在海淀 320 patch 的全域相似度图。")
    add_figure(document, "haidian_v1_vector_retrieval_20260710/building_sample_comparison.png", "图 15. 建筑原型检索的单 patch 对照。")
    add_figure(document, "haidian_v1_vector_retrieval_20260710/road_sample_comparison.png", "图 16. 道路原型检索的单 patch 对照。")
    add_figure(document, "haidian_v1_vector_retrieval_20260710/water_sample_comparison.png", "图 17. 水体原型检索的单 patch 对照。")

    add_heading(document, "七、扩展类别、交付与使用建议")
    add_table(
        document,
        ["类别", "代表性指标", "建议"],
        [
            ["林地", "F1 0.814", "可直接用于区域级制图。"],
            ["耕地", "F1 0.544", "可用于区域覆盖制图与人工复核。"],
            ["湖泊", "F1 0.539", "大尺度水体较稳定。"],
            ["火车站", "检索 P@10 = 0.5", "适合辅助人工普查，不建议直接分割。"],
            ["停车场、湿地、垃圾场、机场", "样本不足或检索失败", "应先补充高质量人工标注。"],
        ],
    )
    add_figure(document, "haidian_fine_osm_leadership_20260705/advantage_categories_f1_auc.png", "图 18. 细粒度 OSM 类别的 F1 与 AUC 概览。")
    add_figure(document, "nonbuilding_fewshot_semantic_20260705/pitch_fewshot_5_10_50_visual.png", "图 19. 运动场类别的 5/10/50-shot 全域制图，展示基础三类以外的拓展能力。")
    document.add_paragraph("推荐业务流程：生成指定月份 embedding → 标注少量正负样例或先用检索找候选 → 训练 conv3×3 轻量头 → 验证集选阈值 → 输出全域概率图、掩膜和人工复核清单。")
    document.add_paragraph("适用边界：P10C 是海淀区域生产模型；OSM 空白不等于真实负类；充分标注时原始影像 + 强分割头仍可能取得更高单任务上限；云雾、数据缺失、配准偏差和 10 m 小目标仍是主要误差来源。")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(output_path)
    print(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.output)
