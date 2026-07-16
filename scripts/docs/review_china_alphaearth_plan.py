#!/usr/bin/env python3
"""Create a Word tracked-change review of the China AlphaEarth planning draft.

The source file remains untouched.  The output uses native WordprocessingML
``w:ins``/``w:del`` markup and Word comments, so it opens in Word's Review
view rather than becoming a separate rewritten document.
"""

from __future__ import annotations

import copy
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from lxml import etree


SOURCE = Path(
    "/root/.codex/attachments/acf524a9-3b5d-4486-8851-3b7fa61849bc/"
    "china_alpha_earth_three_plans_report (1).docx"
)
OUTPUT = Path(
    "/root/workspace/xuannv/docs/reviews/"
    "china_alpha_earth_three_plans_report_reviewed_20260716.docx"
)
AUTHOR = "Codex 审阅"
DATE = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
CONTENT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
NS = {"w": W_NS, "r": OFFICE_REL_NS, "ct": CONTENT_NS}


def w(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def xml_text(element: etree._Element) -> str:
    return "".join(element.xpath(".//w:t/text()", namespaces=NS))


def plain_run(text: str, deleted: bool = False) -> etree._Element:
    run = etree.Element(w("r"))
    text_node = etree.SubElement(run, w("delText") if deleted else w("t"))
    if text.startswith(" ") or text.endswith(" "):
        text_node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    text_node.text = text
    return run


def revision(tag: str, revision_id: int, text: str) -> etree._Element:
    element = etree.Element(w(tag))
    element.set(w("id"), str(revision_id))
    element.set(w("author"), AUTHOR)
    element.set(w("date"), DATE)
    element.append(plain_run(text, deleted=tag == "del"))
    return element


def replace_paragraph(paragraph: etree._Element, new_text: str, revision_id: int) -> None:
    old_text = xml_text(paragraph)
    if old_text == new_text:
        return
    properties = paragraph.find(w("pPr"))
    for child in list(paragraph):
        if child is not properties:
            paragraph.remove(child)
    paragraph.append(revision("del", revision_id, old_text))
    paragraph.append(revision("ins", revision_id + 1, new_text))


def find_paragraph(root: etree._Element, exact_text: str) -> etree._Element:
    matches = [p for p in root.xpath(".//w:p", namespaces=NS) if xml_text(p) == exact_text]
    if len(matches) != 1:
        raise ValueError(f"Expected one paragraph for {exact_text!r}, found {len(matches)}")
    return matches[0]


def add_comment_anchor(paragraph: etree._Element, comment_id: int) -> None:
    start = etree.Element(w("commentRangeStart"))
    start.set(w("id"), str(comment_id))
    end = etree.Element(w("commentRangeEnd"))
    end.set(w("id"), str(comment_id))
    reference_run = etree.Element(w("r"))
    reference = etree.SubElement(reference_run, w("commentReference"))
    reference.set(w("id"), str(comment_id))
    insert_at = 1 if paragraph.find(w("pPr")) is not None else 0
    paragraph.insert(insert_at, start)
    paragraph.append(end)
    paragraph.append(reference_run)


def make_comments(comments: list[tuple[str, str]], root: etree._Element) -> bytes:
    comments_root = etree.Element(w("comments"), nsmap={"w": W_NS})
    for comment_id, (anchor_text, comment_text) in enumerate(comments):
        paragraph = find_paragraph(root, anchor_text)
        add_comment_anchor(paragraph, comment_id)
        comment = etree.SubElement(comments_root, w("comment"))
        comment.set(w("id"), str(comment_id))
        comment.set(w("author"), AUTHOR)
        comment.set(w("date"), DATE)
        comment_p = etree.SubElement(comment, w("p"))
        comment_r = etree.SubElement(comment_p, w("r"))
        comment_t = etree.SubElement(comment_r, w("t"))
        comment_t.text = comment_text
    return etree.tostring(comments_root, xml_declaration=True, encoding="UTF-8", standalone=True)


def update_relationships(xml: bytes) -> bytes:
    root = etree.fromstring(xml)
    exists = root.xpath(
        ".//rel:Relationship[@Type=$type]",
        namespaces={"rel": R_NS},
        type=f"{OFFICE_REL_NS}/comments",
    )
    if not exists:
        relationship = etree.SubElement(root, f"{{{R_NS}}}Relationship")
        relationship.set("Id", "rIdCodexComments")
        relationship.set("Type", f"{OFFICE_REL_NS}/comments")
        relationship.set("Target", "comments.xml")
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def update_content_types(xml: bytes) -> bytes:
    root = etree.fromstring(xml)
    exists = root.xpath(
        ".//ct:Override[@PartName='/word/comments.xml']", namespaces=NS
    )
    if not exists:
        override = etree.SubElement(root, f"{{{CONTENT_NS}}}Override")
        override.set("PartName", "/word/comments.xml")
        override.set(
            "ContentType",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml",
        )
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(SOURCE) as source_zip:
        document_root = etree.fromstring(source_zip.read("word/document.xml"))

        # High-impact text edits.  Numeric cards/costs are deliberately not
        # invented here: they are converted into auditable estimates via notes.
        changes = [
            (
                "中国版Alpha Earth 三个落地方案",
                "中国版 AlphaEarth：10 m 国家级地球观测嵌入的分阶段建设方案",
            ),
            (
                "现有资源：16张昇腾910B | DDL：2026年11月30日 | 启动日期：2026年8月1日",
                "拟可用资源：16 张昇腾 910B（须完成多节点 HCCL 拓扑、排期与稳定性验收；本项目已验证 6 卡并行） | 目标里程碑：2026 年 11 月 30 日 | 拟启动：2026 年 8 月 1 日",
            ),
            (
                "资源不变，缩小范围，三档分辨率验证",
                "资源不变，先完成 10 m 多模态月度嵌入试点；5 m/2 m 仅在高分数据授权、覆盖和配准通过后作为区域增强验证",
            ),
            (
                "东北训练面积：80.8万km² × 30% = 24.2万km² | 可用训练窗口：107天",
                "东北训练范围：80.8 万 km² × 30% = 24.2 万 km²；须说明 30% 的分层抽样规则、覆盖生态区和每月有效观测阈值 | 名义训练窗口：107 天",
            ),
            (
                "建议：优先做10m东北底座，22天即可完成，作为团队技术验证和首版成果。",
                "建议：优先建设 10 m 东北试点底座，先用固定优化步数的 1,000-step 吞吐基准和数据就绪率复算工期，再作为团队技术验证与首版成果。",
            ),
            (
                "方案三：采购扩充算力，全国三档分辨率，61天DDL",
                "方案三：在通过数据与吞吐阶段门后扩充算力，建设全国 10 m 主底座；5 m/2 m 作为有授权高分数据支撑的区域增强路线",
            ),
            (
                "可采购H100/H800/A100/A800/910C，10/1-11/30完成训练",
                "候选算力包括 910C 与合规可采购 GPU；10/1-11/30 仅作为目标训练窗口，须在前置数据就绪、集群网络、HCCL 扩展效率与 1,000-step 基准验收后确认",
            ),
            (
                "10m/5m目标：910C即可满足，64-256卡，成本仅0.04-0.18亿，强烈推荐。",
                "10 m 目标：以实测吞吐、并行效率和全口径 TCO 为依据确定 910C 集群规模；5 m 只有在全国高分数据的授权、覆盖、时相与配准均满足时才进入预算评审。",
            ),
            (
                "2m目标：成本跃升至1.8-4.2亿，卡数需求1,600-2,800张。受8/1启动、61天训练窗口限制，卡数需求比6/22启动方案增加约60%。",
                "2 m 目标不应与全国 10 m 主预训练并列承诺：其主要瓶颈是高分数据授权与覆盖、存储和 I/O，而非单纯卡数；建议改为完成 10 m 主底座后的重点城市/行业区域增强路线。",
            ),
            (
                "如需107天完成，需追加",
                "如需 107 天完成（线性外推，未计多节点效率损失）",
            ),
            (
                "29卡910B（共45卡）",
                "约需追加 24 张 910B（共 40 张）；最终数值须由固定步数吞吐基准复算",
            ),
            (
                "有效算力",
                "有效训练吞吐（待实测）",
            ),
            (
                "单卡参考价",
                "采购估算单价（须询价）",
            ),
        ]
        revision_id = 1
        for old, new in changes:
            paragraph = find_paragraph(document_root, old)
            replace_paragraph(paragraph, new, revision_id)
            revision_id += 2

        comments = [
            (
                "拟可用资源：16 张昇腾 910B（须完成多节点 HCCL 拓扑、排期与稳定性验收；本项目已验证 6 卡并行） | 目标里程碑：2026 年 11 月 30 日 | 拟启动：2026 年 8 月 1 日",
                "资源口径需要拆开写：当前节点可见 8 张 910B4，项目已稳定验证 6 卡并行；P10C 海淀生产基线（320 patch、6 个月、800 epoch、每卡 batch=3、梯度累积=2）从 2026-07-04 16:53:29 至 2026-07-05 10:43:01 保存 epoch_800，实测 17 小时49分32秒。该时间包含验证/保存，不含导出和下游评测，且从 P9B 权重继续训练，不能写作从零预训练。“16 卡”若为两节点集群，须再写明节点数、HCCL 拓扑、网络带宽、可独占时段和故障替补策略。",
            ),
            (
                "资源不变，先完成 10 m 多模态月度嵌入试点；5 m/2 m 仅在高分数据授权、覆盖和配准通过后作为区域增强验证",
                "关键技术修改：Sentinel-2/Sentinel-1/Landsat 的原生空间信息不是 5 m 或 2 m。把全国主预训练直接改成 5 m/2 m 会把上采样当成新增信息。建议明确：10 m 是主 embedding 网格；高分光学/SAR 是稀疏辅助模态；5 m/2 m 仅依赖另行取得且可合法训练的高分数据。",
            ),
            (
                "东北训练范围：80.8 万 km² × 30% = 24.2 万 km²；须说明 30% 的分层抽样规则、覆盖生态区和每月有效观测阈值 | 名义训练窗口：107 天",
                "“30%”不能只是面积比例。需在附录列出分层：省份/生态区/土地覆盖/城乡梯度/云量带，并固定每月最小有效像素比例。否则训练集可能集中在易获取、低云区域，无法代表中国尺度泛化。",
            ),
            (
                "建议：优先建设 10 m 东北试点底座，先用固定优化步数的 1,000-step 吞吐基准和数据就绪率复算工期，再作为团队技术验证与首版成果。",
                "请删去把“小数据 800 epoch”直接放大到全国的做法。P10C 海淀实测为 320×800=25.6 万 patch-visits，6 卡 17小时49分32秒，即 2,393.6 patch-visits/卡·小时；全国30%约175.8万 patch，若仍跑800 epoch，16卡完全线性也约需1,530天。样本规模改变后，epoch不是可比单位；应统一为优化步数、全局 batch、有效像素数、混合精度、数据读取命中率和实际 samples/s。先跑 1,000 step 基准，再得出卡数和 ETA。",
            ),
            (
                "候选算力包括 910C 与合规可采购 GPU；10/1-11/30 仅作为目标训练窗口，须在前置数据就绪、集群网络、HCCL 扩展效率与 1,000-step 基准验收后确认",
                "采购表要把“加速卡数量”改为“完整集群”：计算卡、节点 CPU/RAM、本地 NVMe 缓存、RoCE/IB 网络、对象存储、备份、运维和电力。全国数据训练常先被 I/O 限速，单卡 TFLOPS 不能直接推导完成天数。",
            ),
            (
                "有效训练吞吐（待实测）",
                "建议删除或弱化这一列的纯 TFLOPS 比较。不同卡的 BF16/FP16、算子、显存、通信、软件栈和数据管线不同；请改为同一训练脚本、同一全局 batch 下的实测 step/s、tokens/pixels/s 与 16/32/64 卡扩展效率。",
            ),
            (
                "采购估算单价（须询价）",
                "价格不能作为固定事实。补充“报价日期、含税/维保年限、服务器与网络是否包含、采购或租赁、交期、替代料”。尤其 H800/A800 的可得性与合规性需要供应商书面确认。",
            ),
            (
                "2 m 目标不应与全国 10 m 主预训练并列承诺：其主要瓶颈是高分数据授权与覆盖、存储和 I/O，而非单纯卡数；建议改为完成 10 m 主底座后的重点城市/行业区域增强路线。",
                "按当前项目 64 维、128×128、FP16 月度 embedding 的输出格式线性估算，30% 全国、6 个月的 10 m embedding 约 22 TB；5 m 约 88 TB；2 m 约 553 TB，尚未包括输入、临时缓存、版本和副本。因此 2 m 必须单列数据、存储和交付方案。",
            ),
            (
                "如需 107 天完成（线性外推，未计多节点效率损失）",
                "原表存在算术不一致：265 天在 16 卡上的工作量若压缩到 107 天，线性比例为 265/107×16≈40 张，总追加约 24 张，不是 45 张总量。更重要的是，多节点通信损失会使线性估算偏乐观。",
            ),
            (
                "现有16卡910B：可纳入方案三集群中作为补充算力，降低采购量。",
                "请勿默认 910B 与 H 系列/A 系列可以混入同一个分布式训练作业并线性抵扣采购卡数。若不能完成混合 HCCL/NCCL 和算子兼容性压测，现有 910B 应单独承担数据预处理、评测、消融或独立训练。",
            ),
            (
                "方案三：在通过数据与吞吐阶段门后扩充算力，建设全国 10 m 主底座；5 m/2 m 作为有授权高分数据支撑的区域增强路线",
                "建议新增“阶段门与验收指标”一页：G0 数据许可/覆盖/云质量通过；G1 10 万 patch 数据管线通过；G2 16→24→32 卡扩展效率验收；G3 跨区域轻量 probe 和少样本任务优于 raw/传统基线；G4 发布前独立标签与时相留出评测。以当前P10C实测速率估算，40万 updates 在16卡、70%效率下含20%缓冲约71.6天，61天窗口不足；24张同构卡、68%效率下约49.1天，建议作为该目标的最小规模。全国数据约71TB筛选训练包、22TB embedding，应配0.5PB可用/1PB原始容量、节点NVMe缓存和至少2×200Gb/s互连。没有这些门，采购规模无法被审计。",
            ),
            (
                "中国版 AlphaEarth：10 m 国家级地球观测嵌入的分阶段建设方案",
                "建议在首页增加一句定位：本项目借鉴 AlphaEarth 的“多源观测→统一 embedding→少标签下游制图”范式，但不是对官方模型的复刻或性能等价声明。官方公开产品是年度 10 m、64 维 embedding；本项目可突出中国数据源和月度更新能力。",
            ),
            (
                "方案一5m可行性",
                "此行有明确逻辑矛盾：6/22 启动时的可用训练窗口为 122 天，表二给出的 5 m 训练需 161 天，仍超期 39 天，不能标为“可接受”。应改为不可行，或列出新增资源/不同训练配方的假设。",
            ),
            (
                "方案三5m需卡数",
                "表八称 76 天缩至 61 天后所需卡数由 128 增至 256；若其他条件不变，线性倍率仅为 76/61≈1.25。该表混用了不同芯片、不同配方或未说明裕量，必须按同一实测吞吐公式重算。",
            ),
            (
                "方案三2m需卡数",
                "1,280 到 1,664–2,816 的增幅为约 30%–120%，不是表中写的“30–55%”。更重要的是，当前数据只有约 3 m 高分光学/SAR，并不能验证全国 2 m 训练数据是否存在。",
            ),
        ]
        comments_xml = make_comments(comments, document_root)
        document_xml = etree.tostring(
            document_root, xml_declaration=True, encoding="UTF-8", standalone=True
        )

        with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as output_zip:
            for item in source_zip.infolist():
                if item.filename in {
                    "word/document.xml",
                    "word/comments.xml",
                    "word/_rels/document.xml.rels",
                    "[Content_Types].xml",
                }:
                    continue
                output_zip.writestr(item, source_zip.read(item.filename))
            output_zip.writestr("word/document.xml", document_xml)
            output_zip.writestr("word/comments.xml", comments_xml)
            output_zip.writestr(
                "word/_rels/document.xml.rels",
                update_relationships(source_zip.read("word/_rels/document.xml.rels")),
            )
            output_zip.writestr(
                "[Content_Types].xml",
                update_content_types(source_zip.read("[Content_Types].xml")),
            )
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
