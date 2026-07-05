#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import tempfile
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


IMAGE_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)")


def clean_inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def split_table_row(line: str) -> list[str]:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return [clean_inline(cell) for cell in cells]


def is_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(set(cell) <= {"-", ":"} and "-" in cell for cell in cells)


def make_image(path: Path, max_width: float, max_height: float, cache_dir: Path) -> Image:
    with PILImage.open(path) as img:
        width, height = img.size
        scale = min(max_width / width, max_height / height, 1.0)
        display_width = width * scale
        display_height = height * scale

        max_pixels_w = 2200
        max_pixels_h = 1600
        pixel_scale = min(max_pixels_w / width, max_pixels_h / height, 1.0)
        out_w = max(1, int(width * pixel_scale))
        out_h = max(1, int(height * pixel_scale))
        rgb = img.convert("RGB")
        if pixel_scale < 1.0:
            rgb = rgb.resize((out_w, out_h), PILImage.Resampling.LANCZOS)
        cached = cache_dir / f"{path.stem}_{abs(hash(path))}.jpg"
        rgb.save(cached, "JPEG", quality=86, optimize=True, progressive=True)
    return Image(str(cached), width=display_width, height=display_height)


def flush_paragraph(buffer: list[str], story: list, style: ParagraphStyle) -> None:
    if not buffer:
        return
    text = " ".join(part.strip() for part in buffer if part.strip())
    if text:
        story.append(Paragraph(clean_inline(text), style))
        story.append(Spacer(1, 4))
    buffer.clear()


def build_story(
    markdown_path: Path,
    styles: dict[str, ParagraphStyle],
    max_width: float,
    max_height: float,
    cache_dir: Path,
) -> list:
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    story: list = []
    paragraph: list[str] = []
    idx = 0
    base_dir = markdown_path.parent

    while idx < len(lines):
        line = lines[idx].rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph(paragraph, story, styles["Body"])
            idx += 1
            continue

        if stripped.startswith("# "):
            flush_paragraph(paragraph, story, styles["Body"])
            story.append(Paragraph(clean_inline(stripped[2:].strip()), styles["Title"]))
            story.append(Spacer(1, 8))
            idx += 1
            continue

        if stripped.startswith("## "):
            flush_paragraph(paragraph, story, styles["Body"])
            if story:
                story.append(Spacer(1, 6))
            story.append(Paragraph(clean_inline(stripped[3:].strip()), styles["Heading2"]))
            story.append(Spacer(1, 4))
            idx += 1
            continue

        image_match = IMAGE_RE.match(stripped)
        if image_match:
            flush_paragraph(paragraph, story, styles["Body"])
            image_path = (base_dir / image_match.group("path")).resolve()
            if image_path.exists():
                img = make_image(image_path, max_width, max_height, cache_dir)
                story.append(KeepTogether([img, Spacer(1, 4)]))
            else:
                story.append(Paragraph(f"[Missing image: {clean_inline(str(image_path))}]", styles["Caption"]))
            idx += 1
            continue

        if stripped.startswith("|") and idx + 1 < len(lines) and is_separator(lines[idx + 1].strip()):
            flush_paragraph(paragraph, story, styles["Body"])
            table_lines = [stripped]
            idx += 2
            while idx < len(lines) and lines[idx].strip().startswith("|"):
                table_lines.append(lines[idx].strip())
                idx += 1
            rows = [[Paragraph(cell, styles["TableCell"]) for cell in split_table_row(row)] for row in table_lines]
            if rows:
                column_count = len(rows[0])
                col_widths = [max_width / column_count] * column_count
                table = Table(rows, colWidths=col_widths, repeatRows=1)
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#10233f")),
                            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c7d2e0")),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 3),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                            ("TOPPADDING", (0, 0), (-1, -1), 3),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ]
                    )
                )
                story.append(table)
                story.append(Spacer(1, 8))
            continue

        if stripped == "---":
            flush_paragraph(paragraph, story, styles["Body"])
            story.append(PageBreak())
            idx += 1
            continue

        paragraph.append(stripped)
        idx += 1

    flush_paragraph(paragraph, story, styles["Body"])
    return story


def make_styles() -> dict[str, ParagraphStyle]:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    base = getSampleStyleSheet()
    return {
        "Title": ParagraphStyle(
            "ChineseTitle",
            parent=base["Title"],
            fontName="STSong-Light",
            fontSize=24,
            leading=30,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "Heading2": ParagraphStyle(
            "ChineseHeading2",
            parent=base["Heading2"],
            fontName="STSong-Light",
            fontSize=15,
            leading=20,
            textColor=colors.HexColor("#12355b"),
            spaceBefore=8,
            spaceAfter=4,
        ),
        "Body": ParagraphStyle(
            "ChineseBody",
            parent=base["BodyText"],
            fontName="STSong-Light",
            fontSize=10.5,
            leading=16,
            alignment=TA_LEFT,
            firstLineIndent=0,
            spaceAfter=4,
        ),
        "Caption": ParagraphStyle(
            "ChineseCaption",
            parent=base["BodyText"],
            fontName="STSong-Light",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#374151"),
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "TableCell": ParagraphStyle(
            "ChineseTableCell",
            parent=base["BodyText"],
            fontName="STSong-Light",
            fontSize=6.8,
            leading=8.2,
            alignment=TA_LEFT,
        ),
    }


def add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("STSong-Light", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawRightString(doc.pagesize[0] - 14 * mm, 8 * mm, f"{doc.page}")
    canvas.restoreState()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a Markdown report with local images to PDF.")
    parser.add_argument("markdown", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    page_size = landscape(A4)
    margin = 13 * mm
    doc = SimpleDocTemplate(
        str(args.output),
        pagesize=page_size,
        rightMargin=margin,
        leftMargin=margin,
        topMargin=12 * mm,
        bottomMargin=13 * mm,
        title=args.markdown.stem,
    )
    styles = make_styles()
    max_width = page_size[0] - 2 * margin
    max_height = page_size[1] - 34 * mm
    with tempfile.TemporaryDirectory(prefix="xuannv_pdf_") as tmp:
        story = build_story(
            args.markdown,
            styles,
            max_width=max_width,
            max_height=max_height,
            cache_dir=Path(tmp),
        )
        doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


if __name__ == "__main__":
    main()
