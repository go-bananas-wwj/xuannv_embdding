#!/usr/bin/env python3
"""Inject the V1.1 vs AEF 2025 section into /root/workspace/report/index.html.

Reads:
  - /root/workspace/report/v1.1_benchmark/V1.1_vs_AEF_REPORT.md
  - /root/workspace/report/v1.1_benchmark/v1.1_vs_aef_metrics.png
  - /root/workspace/report/index.html

Writes:
  - /root/workspace/report/index.html (with V1.1 section inserted/updated)
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


SECTION_MARKER = "<!-- V1.1_VS_AEF_SECTION -->"


def parse_markdown_table(md: str) -> tuple[list[str], list[list[str]]]:
    """Return (headers, rows) from the first markdown table in md."""
    lines = [line.strip() for line in md.splitlines()]
    try:
        start = next(i for i, line in enumerate(lines) if line.startswith("| 任务"))
    except StopIteration:
        raise ValueError("No markdown table found in report")

    table_lines: list[str] = []
    for line in lines[start:]:
        if line.startswith("|"):
            table_lines.append(line)
        else:
            break

    # Skip separator line (second line)
    header_line = table_lines[0]
    data_lines = [ln for ln in table_lines[2:] if ln.strip() and "---" not in ln]

    headers = [h.strip() for h in header_line.split("|")[1:-1]]
    rows = []
    for line in data_lines:
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if cells:
            rows.append(cells)
    return headers, rows


def md_table_to_html(headers: list[str], rows: list[list[str]]) -> str:
    thead = "\n".join(f"    <th>{h}</th>" for h in headers)
    tbody_lines: list[str] = []
    for row in rows:
        cells = "\n".join(f"      <td>{c}</td>" for c in row)
        tbody_lines.append(f"    <tr>\n{cells}\n    </tr>")
    tbody = "\n".join(tbody_lines)
    return (
        "<table>\n"
        "  <thead>\n"
        "    <tr>\n"
        f"{thead}\n"
        "    </tr>\n"
        "  </thead>\n"
        "  <tbody>\n"
        f"{tbody}\n"
        "  </tbody>\n"
        "</table>"
    )


def extract_conclusion(md: str) -> str:
    m = re.search(r"## 结论\n+(.+?)(?:\n\n|\Z)", md, re.DOTALL)
    if not m:
        return ""
    return m.group(1).strip()


def update_index(
    index_path: Path,
    report_path: Path,
    figure_path: Path,
) -> None:
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    if not figure_path.exists():
        raise FileNotFoundError(figure_path)

    md = report_path.read_text(encoding="utf-8")
    headers, rows = parse_markdown_table(md)
    table_html = md_table_to_html(headers, rows)
    conclusion = extract_conclusion(md)

    # Convert conclusion plain text to HTML paragraphs.
    conclusion_html = "\n".join(f"      <p>{p}</p>" for p in conclusion.split("\n") if p.strip())

    figure_rel = figure_path.relative_to(index_path.parent).as_posix()

    section_html = f"""  <section>
    <h2>十一、V1.1 AEF 蒸馏模型与 AEF 2025 对比</h2>
    <p>使用 50-epoch AEF 蒸馏训练得到的 V1.1 embedding（双时相 202512+202605，64 维）与 AEF 2025 官方 embedding（单时相 2025，64 维）在 5 个下游任务上的 5-fold 交叉验证对比。</p>

    <h3>11.1 指标对比表</h3>
{table_html}

    <h3>11.2 结论</h3>
{conclusion_html}

    <h3>11.3 可视化对比</h3>
    <img src="{figure_rel}" alt="V1.1 vs AEF 2025 指标对比" class="img-full">
    <p class="center">图：V1.1 与 AEF 2025 在 5 个下游任务上的多指标对比。</p>
  </section>

"""

    index_html = index_path.read_text(encoding="utf-8")

    if SECTION_MARKER in index_html:
        # Replace existing section block.
        pattern = re.compile(re.escape(SECTION_MARKER) + r".*?" + re.escape(SECTION_MARKER), re.DOTALL)
        index_html = pattern.sub(SECTION_MARKER + "\n" + section_html + SECTION_MARKER, index_html)
    else:
        # Insert before the appendix section.
        appendix_marker = "    <section>\n      <h2>十、附录与数据来源</h2>"
        if appendix_marker in index_html:
            index_html = index_html.replace(appendix_marker, SECTION_MARKER + "\n" + section_html + "    " + SECTION_MARKER + "\n\n" + appendix_marker)
        else:
            # Fallback: insert before </body>.
            index_html = index_html.replace("</body>", SECTION_MARKER + "\n" + section_html + SECTION_MARKER + "\n</body>")

    index_path.write_text(index_html, encoding="utf-8")
    print(f"Updated {index_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Inject V1.1 vs AEF section into report/index.html")
    p.add_argument("--index", type=Path, default=Path("/root/workspace/report/index.html"))
    p.add_argument("--report", type=Path, default=Path("/root/workspace/report/v1.1_benchmark/V1.1_vs_AEF_REPORT.md"))
    p.add_argument("--figure", type=Path, default=Path("/root/workspace/report/v1.1_benchmark/v1.1_vs_aef_metrics.png"))
    args = p.parse_args()
    update_index(args.index, args.report, args.figure)


if __name__ == "__main__":
    main()
