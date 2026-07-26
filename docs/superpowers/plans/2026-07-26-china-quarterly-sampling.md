# 中国版季度地理嵌入采样方案实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 生成 2020--2021 年季度中国版地理嵌入的 62,000 点可执行采样方案。

**Architecture:** 以独立 JSON 配置冻结采样参数，以 Markdown 作为内容源，以
DOCX 作为正式交付格式，并复用已验证的全国基础 1% 抽样图。文档明确区分基础
预览、最终采样注册表和季度数据质量筛选。

**Tech Stack:** JSON、Markdown、Python 3.11、python-docx、现有全国采样脚本。

## Global Constraints

- 最终样本严格为 62,000 个 patch。
- 单 patch 为 1280 m x 1280 m、128 x 128 像素、10 米等效分辨率。
- 2020Q1 至 2021Q4 共八期产品共用同一空间采样骨架。
- 不使用最终下游人工标签进行主模型采样。
- 数据与代码分离，大型注册表不进入 Git。

---

### Task 1: 冻结采样策略

**Files:**
- Create: `configs/national/china_quarterly_2020_2021_sampling_policy_20260726.json`
- Test: `tests/test_china_quarterly_sampling_plan.py`

**Interfaces:**
- Consumes: 参考技术方案中的 62,000 点构成。
- Produces: 文档和采样脚本可读取的自包含策略。

- [ ] 写策略结构和数量一致性测试。
- [ ] 新增自包含 JSON，包含网格、四层采样、质量门槛和划分规则。
- [ ] 运行测试并确认总量、比例和季度列表正确。
- [ ] 提交并推送。

### Task 2: 生成正式方案与 DOCX

**Files:**
- Create: `docs/plans/china_quarterly_2020_2021_sampling_plan_20260726_zh.md`
- Create: `scripts/docs/build_china_quarterly_sampling_plan_docx.py`
- Create: `docs/plans/word/中国版AlphaEarth全国采样方案_2020_2021季度版_20260726.docx`

**Interfaces:**
- Consumes: Task 1 的 JSON 和现有基础 1% 抽样图。
- Produces: 可阅读、可修改、可审计的正式方案。

- [ ] 编写结论先行的 Markdown 正文。
- [ ] 编写 DOCX 构建脚本，使用宋体、Times New Roman 和黑白表格。
- [ ] 构建 DOCX 并验证标题、表格、图片和页数。
- [ ] 提交并推送。

### Task 3: 独立审查与修订

**Files:**
- Modify: Task 1 和 Task 2 产物。
- Create: `docs/reviews/china_quarterly_sampling_plan_review_20260726.md`

**Interfaces:**
- Consumes: 完整方案、配置、测试和 DOCX。
- Produces: 多视角审查记录和修订后的最终版本。

- [ ] 分别发起空间统计、遥感数据、工程复现和文档表达审查。
- [ ] 汇总问题，逐项修改或记录不采纳理由。
- [ ] 运行测试、DOCX 解包检查和 Git 差异检查。
- [ ] 提交并推送最终版本。
