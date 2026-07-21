# Codex 对话记录总结：xuannv_embedding 项目

> 来源：`/root/.codex/sessions/` 会话日志（2026-07-07 ~ 2026-07-20）中与 **xuannv_embedding 项目**直接相关的对话线索，已按"对话线索"归并去重（同一条长对话被 Codex 压缩/恢复机制拆成多个文件）。
> 整理时间：2026-07-21。其余与本项目无关的会话（rs_words 河流笔画、AI 纯音乐、系统安全审查、连通性测试等）已从本文档移除。

## 会话清单概览

| 线索 | 时间 | 规模 | 内容 |
|---|---|---|---|
| E. **主线程：模型升级长对话** | 07-07 ~ 07-20 | 约 1.4 GB、531 条独立用户消息 | 项目核心：P 系列实验 → 生产版 → 论文 → 全国版 |
| F. 论文严苛审查子会话 | 07-07 11:29 ×2、07-15 ×4、07-16 ×45、07-17 ×3 | 每个 0.2~0.9 MB | 主线程派生的论文/实验/下载管线审查 |

---

## E. 主线程：模型升级长对话（07-07 ~ 07-20）

本项目最核心的工作线索，始于"参考这个分支里面的内容，然后继续做吧，升级模型"，横跨两周、531 条用户消息。分四个阶段：

### E1. 07-07 起：bug 修复与 P 系列实验迭代

- **战略转型**：从"v2 变化检测导向"转向"海淀专用通用月度地理嵌入"。子代理审查发现 P0 级错位：低分辨率月度输入未正确进入 monthly embedding head，高分影像被时间加权平均成单帧复制到所有月。用户决策：先修 P0，确认 bug-fixed baseline 再升级。
- **Round1 变化感知训练**：Macro AUC 0.7843→0.8005 小幅提升，但 construction_joint 远弱于 AEF（mIoU 0.0379 vs 0.1776）。审查发现 val_loss 只在 rank0 计算、best checkpoint 选择有"指标幻觉"、fold0 快评与 AEF 5-fold 不可比等问题。
- **重大方向决策**：否决"稳定+变化双通道 embedding"，主张单一时间条件嵌入 E(x,y,t)；否决用下游人工标注训练主嵌入（影响泛化）；采纳 OSM 弱语义 + 月度状态重建增强路线，参考 AEF、OlmoEarth、TESSERA、CACo 等。
- **数据整改**：下载 AEF 官方大图本地切 patch 做公平对比；诊断出玄女嵌入 PCA 大块色斑的根因是云雾被当普通成像重建，随后落实像素级 mask、同月多帧质量筛选平均、WorldCover 替换为 OSM 合并土地分类。
- **P5H→P10C 迭代**：修复 STP 块 channels-last reshape 错位、train/val patch 重叠泄漏、editable install 指向旧 worktree 等；P8A/P9A 800 epoch 后建筑/道路指标超 AEF；6 卡分组并行实验（3+3 或 2+2+2）。
- **生产版**：最佳模型定为 **P10C epoch800**，打包为 `haidian-embedding-v1`，上传 ModelScope 数据集 `WeijieWu/xuannv_haidian_embdding`。
- **汇报材料**：HTML 报告（8001 端口）、论文风格 Markdown/PDF；few-shot（5/10/50-shot）测评建筑/道路/水体及 OSM 细粒度类别，并加入传统 ML（RF/ExtraTrees/SVM）与强监督基线（U-Net/DeepLab-lite/SegFormer-lite）横向对比。
- **P11/P12 探索**：P11 提升 64 维嵌入有效维度（当时仅约 9.7 维）未超 P10C；P12 尝试 patch 边缘重叠采样消除边界割裂。
- **关键结论**：P10C 在 5-shot 下相比 raw 多源特征基线，建筑/道路/水体 F1 分别提升 9.8%/18.3%/41.5%；云雾必须作异常 mask 而非重建目标；simple probe 不提升即不算嵌入变好。
- **AP-GARSS 摘要**：定稿 "XuannvEarth: Toward China-Scale Monthly Geospatial Embeddings for Robust Mapping and Change Detection"（作者吴伟杰、赵龙、范欣怡）。

### E2. 07-10 ~ 07-16：生产版定型、下游头深化与论文框架

- **P12C 导出与调试**：发现下游概率图"一片红"（概率挤在 0.465~0.475），排查下游头表达能力；试验五层 MLP；用 2026-04 原始 42 通道特征 + UNet 等头做基线，回答"embedding 意义在哪"的质疑并写成汇报文档。
- **项目记忆沉淀**：训练/评测经验写入 AGENTS.md 与 `docs/agent_memory/haidian_embedding_project_memory.md`。
- **P13 实验**：P13A+P13C 混合 6 卡训练；审查发现 `best.pt` 只在 save_every 保存点更新、可能漏掉真正最佳验证点，`--init-from` strict=False 隐蔽风险等 bug。
- **生产发布**：P10C epoch800 定型，2025-12 ~ 2026-05 每月嵌入打包上传 ModelScope；GitHub 独立分支放代码；新增土地利用与施工工地检测两个下游头。
- **论文转向 JRS 专刊**（DDL 2026-12-31）：多子代理审查结论——不能称 "foundation model"，定位 **city-scale monthly multimodal embedding framework**；三大硬伤：OSM 同源标签泄露、随机 patch 切分未控制空间自相关、主要基线仅 fold 0。产出 `docs/paper/jrs_2026/` 全套框架（大纲、写作蓝图、claim-evidence 矩阵、注册评测协议、先前发表审计、cover letter）。
- **注册评测协议**：空间 5-fold 连续带 + 1-patch 缓冲区、固定 conv3x3 主头、5/10-shot 为主、三 seed、10000 次配对分层空间 bootstrap；40/80/150 嵌套子集注册；P10C 生产模型隔离为传导式案例不进科学评测；旧 fold-0 结果全部标记 `paper_eligible=false`。
- **方法事实核实**：P10C 实为 1 节点 ×2 NPU、单卡 batch 3 累积 2，谱系 P7A→P8A→P9A e400→P9B e800→P10C；修复 best.pt 选择 bug 为每次验证即时更新。

### E3. 07-15 ~ 07-16：JRS 论文框架多轮门禁审查

07-15 四个并行审查子会话给出一致结论：选题与专刊契合，但核心证据有硬伤，"Reject，重做核心验证后再投"；术语定为 "monthly-indexed"、"labeled-patch efficiency"。随后在 07-16 组织 10 轮 × 5 角色的并行审查子代理群（详见 F 节），逐轮 PASS/FAIL，直至全部问题关闭。

### E4. 07-17 ~ 07-20：全国版（china_v1）数据准备与汇报包装

- **全国数据方案**：空间 1% 采样（每 10×10 宏格网采 1 个）+ 稀少区/海岸线补采至约 6 万 patch；归档 2025-04 ~ 2026-04 共 13 个月；存 `/data2/xuannv_embedding/`；产出 `docs/plans/china_v1_national_data_preparation_20260717.md` 与采样策略配置。用户要求先给方案、同意后再下载，并用多子代理交叉审查。
- **子代理审查发现多项 P0**：采样脚本实际入样概率偏向低可用性区域；OSM 快照晚于影像期存在前视泄漏；OSM 空白≠背景（nodata 无法区分负类与未测绘）；Landsat `qa_pixel` 被当反射率缩放导致云 bitmask 失效；训练实际 import 旧 worktree 代码；下载器只支持 haidian/harbin 单 UTM 带。
- **下载加速**：初始 ETA 158 天，用户要求压到一周内。否决"少下载"和新购服务器，采纳"同景只打开一次、COG 窗口批量读取、STAC 清单缓存按需签名"的优化路径；tmux 后台下载 + watchdog 处理 SAS 签名过期；曾测试 Sentinel-2 官方 S3 密钥后回退 PC 下载。
- **物化管线审查**：`materialize_china_v1_shard.py` 存在 S1 无效景 `nanmean` 污染、缺 fail-closed 门禁、无断点续传、39 份全国 JSONL 目录一次性解析的内存风险等问题。
- **china_v1 计划文档**：2021 年为多源数据共同年份、约 6.2 万 patch、16 张 910B 卡的训练时长与存储保守测算、按 11 月初 DDL 倒排工期（`docs/plans/china_v1_2021_multisource_data_and_label_plan_20260720_zh.md`）。
- **对外技术方案 DOCX**：基于领导的两份 docx 生成"现有数据版"与"2 米高分辨率版"两份中文方案（2020/2021 年季度嵌入），严格排版要求。
- **周报汇报材料**：模型进展 + 论文进度两个 HTML 汇报（后被批不够专业，迭代加入全国采样计划、论文框架图、业界基线对比——P10C 与 AEF 同 64 维总体持平、construction/road 占优；PU Query 标注统计网页），部署 9001 端口，另产出 Word 版生产能力报告。

## F. 论文与管线审查子会话

主线程派生的只读审查子会话，按时间排列：

### F1. 07-07 11:29（×2）：论文可行性与可复现性初审

- 以遥感/地理基础模型论文严苛审稿人视角，基于 P10/P11/P12、AEF 对比、few-shot/下游任务/全域 PCA 等线索，列出最易被质疑的点、必须补的实验证据、拒稿风险与推荐论文结构。
- 以工程真实性/可复现性审查员视角，盘点已有报告/配置/日志资产，判断哪些结果可信、哪些需重新统一评测，给出可复现性 checklist。

### F2. 07-15 13:30（×4）：JRS 投稿前四角色审查

- **实验完整性审查**：当前实验包尚不具备直接投稿条件，三处硬伤——训练/验证重叠、评测标签同源泄露、主要基线仅 fold 0；选题与 JRS 专刊契合度高。
- **论文叙事与结构**：专刊范围覆盖多模态融合、EO 表征学习、时序应用（DDL 2026-12-31）；不应宣称"全国基础模型"或"全面超过 AEF"，最稳定位是城市尺度月度多模态嵌入框架。
- **苛刻审稿人模拟**：结论"Reject，重做核心验证后再投"；320 patch 单城区 6 个月 64 维模型本质上是城市尺度任务知情预训练，不能支持 "foundation model"、"跨城市泛化"或"可靠月度变化检测"的主张。
- **方法创新性与专刊契合度**：三级表述——标题/摘要用 "a city-scale monthly multimodal embedding framework"，正文定位 "foundation-model-inspired regional representation learning"，应用角色 "a reusable urban embedding backbone"。

### F3. 07-16 15:19 ~ 16:38（×45）：JRS 论文框架 10 轮 × 5 角色门禁审查

45 个小会话是以 `<recommended_plugins>` 开头的 Kimi Code 审查子代理，按 10 个轮次组织（第一轮 → 第二轮 → 第三轮 → 终审 → 最终复审 → 通过性终审 → 终局审查 → 封板终审 → 最终 PASS 审查 → 闭环确认），每轮 5 个角色并行：

- **方法事实审查**：对照 P10C 配置与 `src/` 实现，检查时间语义、高分路径、损失、OSM 监督、采样、训练/checkpoint 描述的事实错误（OSM 有放回加权采样、10m 高分来源、vMF-style、跨月注意、P9B 热启动谱系、名义 batch 等）。
- **统计与评测协议审查**：空间 5-fold、exact shot、40/80/150 冻结、64D 同头、阈值、CI、seed 层 bootstrap、checkpoint 后验选择、`paper_eligible` 标记。
- **JRS 合规编辑**：标题字符数、单段五句摘要 ≤250 词、章节结构、图表预算 ≤10、声明、cover letter、专刊契合、AI 披露。
- **叙事审查**：单一主线、figure-first、RQ、What/Why/So What、术语一致（monthly-indexed、labeled-patch efficiency）、避免过度主张。
- **对抗性复现审查**：假设要拒稿，攻击标签泄露、基线不公平、旧结果混入、测试集窥探、checkpoint 祖先泄露、Track B 换名泄露、APGARSS 重复发表、artifact provenance 缺失。

规则：未实现但明确标 PENDING 且阻止投稿的实验不算框架失败；每轮第一行严格给出 PASS 或 FAIL，FAIL 只列阻塞项。多轮迭代直至各类攻击路径均有明确禁止与 PENDING 验证门。

### F4. 07-17 07:50（×3）：全国下载管线审查

- 审查全国下载数据质量与可恢复性（`materialize_china_v1_shard.py`、`cache_national_stac_catalog.py`、`/data2` 已下载数据）。
- 审查 `materialize_china_v1_shard.py` 的性能与正确性：6 万 patch 规模下目录索引内存、COG 读取并发、Zarr 写入、断点与错误处理，按 P0/P1 排序。
- 外部最佳实践调研：Planetary Computer/COG 窗口批量读取、GDAL/rasterio 并发、STAC 缓存、Zarr 分片、速率限制与重试。
