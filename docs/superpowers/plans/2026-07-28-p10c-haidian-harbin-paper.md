# P10C 海淀—哈尔滨月度嵌入论文实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 把既有的 P10C 海淀生产模型、哈尔滨同配方复现和冻结编码器跨城市试验组织为一篇可审计的论文；核心论点是，玄女生成的是可按月更新的城市密集嵌入场，并在统一的稀疏标注读取协议下与年度 AEF 进行对照。

**架构：** 论文由三个互补证据构成：海淀 P10C 的月度应用案例、哈尔滨从零复现以检验训练配方可复制性、以及“海淀训练、哈尔滨冻结推理”的跨城编码器迁移。下游阶段均冻结 embedding，只训练同一个轻量 `Conv3x3` 读取头；AEF 亦接入相同读取头、支持样本、空间划分和阈值选择流程。

**技术栈：** PyTorch + torch_npu、6 张 Ascend NPU、现有 `scripts/eval/` 导出与 probe 框架、AEF 官方 2025 annual COG 缓存、OSM 栅格弱标签。

## 全局约束

- 只评测海淀、哈尔滨两个已有区域；不重新下载、不重新切片、不重训已经完成的 P10C 海淀或哈尔滨 scratch 模型。
- 玄女的“月度”是方法能力与产品能力：每个月均可从该月可用观测和固定时序窗口生成一个 embedding；AEF 官方可比产品是年度 embedding。论文必须写明“月度玄女 vs 年度 AEF”，不把它伪装成同日期、同输入或同规模预训练的对照。
- 任何 AEF 对比在下游层面必须共用：类别、patch 清单、空间划分、shot 清单、probe 架构、优化预算、随机种子、支持集标准化和仅验证集阈值选择。
- P10C 的上游训练使用了 OSM 弱语义；OSM 标签也用于部分下游读取。因此主文称为“OSM-assisted sparse mapping”，不称独立人工标注泛化。已有人类/外部独立标签时可作为补充验证，不得把 OSM 结果写成无监督转移。
- “冻结迁移”中编码器、归一化统计和 P10C 参数完全冻结；哈尔滨只允许训练下游读取头。任何对 P10C 编码器的哈尔滨微调必须单列为“区域适配”，不能叫零样本迁移。
- 所有新代码、配置、协议与轻量报告完成一个独立步骤后立即提交并推送；大文件、embedding、权重和预测图仅存 `/data/xuannv_embedding/`。
- 论文图使用仓库内的相对路径资源；禁止在论文 Markdown 引用 `/data/...`。

---

## 1. 论文故事与三个实验臂

### 1.1 论文一句话主张

> XuannvEarth 将多源遥感观测压缩为 10 m、64 维、可月度更新的城市 embedding field；少量支持标注即可用统一轻量读取头快速制作建筑、道路与水体地图，并可把在海淀学到的编码器直接迁移到哈尔滨。

这不是“AEF 不具备年度产品就不公平”的说法。正确表述是：**年度 AEF 是强大的公开参考表示；玄女的差异化贡献是城市级月度更新与高分辨率多源融合。** 同一个下游读取头的比较保证读取阶段公平；两种上游信息条件不同，因而结果解释为“产品/表示对照”，而非严格的同信息因果消融。

### 1.2 三个实验臂

| 实验臂 | 要回答的问题 | 玄女模型 | 新训练 | 已有资源 | 论文定位 |
| --- | --- | --- | --- | --- | --- |
| A：海淀月度应用 | P10C 在已部署城市能否用少量标注制图，并与 AEF 年度表示对照？ | P10C epoch 800，2025-12 至 2026-05 六月窗口，目标月 2026-04 | 否 | P10C checkpoint、320 patch 月度 embedding、P10C/AEF 对比资产 | 主结果、应用价值 |
| B：哈尔滨配方复现 | 同一训练配方换到第二个城市，是否仍能产出可读 embedding？ | `paper_e5_p10c_recipe_harbin_scratch_20260714` best 与 epoch 800 | 否，已跑完 | 424 patch、六个月数据、800 epoch checkpoints | 可复制性，不叫跨城零样本 |
| C：冻结跨城迁移 | 不让编码器看哈尔滨训练数据时，海淀 P10C 是否仍保留可由少量哈尔滨标注读出的结构？ | 冻结 P10C epoch 800；仅哈尔滨 probe 训练 | 否；只需兼容性检查与推理/评测 | 海淀 P10C checkpoint + 哈尔滨共同模态 + 哈尔滨标签 | 跨城市迁移主证据 |

### 1.3 不再重复的工作

以下均已存在，计划中只核验来源、复制论文所需图表或做统一重测，**不重新训练**：

- 海淀 P10C 的 800 epoch 训练和 checkpoint：`/data/xuannv_embedding/outputs/v2_p10c_haidian_202512_202605_osm_semantic_hardneg_20260704/epoch_800.pt`。
- 海淀 P10C 的训练配置：`configs/v2_p10c_haidian_202512_202605_osm_semantic_hardneg_20260704.yaml`。
- 海淀 320 patch、2025-12 至 2026-05 的清洗数据清单：配置中的 `manifest_p6a_202512_202605_pixelmask_clean_osm_landcover.json`。
- 哈尔滨 scratch 训练已跑完 800 epoch，已有 `best.pt`、`epoch_400.pt`、`epoch_600.pt`、`epoch_800.pt`：`/data/xuannv_embedding/outputs/paper_e5_p10c_recipe_harbin_scratch_20260714/`。
- 哈尔滨 424 patch 六个月清单：`/data/xuannv_embedding/processed/harbin/manifest_e5_harbin424_202512_202605.json`。
- AEF 官方 annual 2025 COG 缓存及来源索引：`/data/xuannv_embedding/cache/aef_official_2025_cogs/`。
- 海淀 P10C 的下游/可视化经验、few-shot 与 raw baseline：`docs/production/haidian_fewshot_raw_vs_xuannv_20260708.md`、`docs/production/haidian_raw_image_baseline_comparison_20260708.md`。
- 已完成但协议不同的随机 fold P10C 三类结果只能作为开发记录；不可和新的空间划分结果混写。

## 2. 实验协议冻结

### 2.1 任务、标签和评测月份

- 核心任务：建筑、道路、水体三类二值提取。建筑/道路/水体均从同一份按 patch 对齐的 OSM 栅格标签读取。
- 目标月：玄女优先使用 `2026-04` 月度 embedding。模型可在同一配方下输出 2025-12 至 2026-05 各月产品；图中展示 2026-04，附录展示三个月稳定性，而不是把六个月拼为一个伪月度指标。
- AEF：官方 `annual_2025`，明确标识为年度产品，绝不伪造为 2026-04 AEF。
- 标签解释：OSM 为当前可用弱语义栅格，不是 2025/2026 同期实测真值。因此这些结果证明“OSM-assisted mapping readout”，不证明月度变化真值检测。

### 2.2 公平的读取阶段

- 读取头：固定 `Conv3x3`，输入 64 维 embedding，输出单类概率图。禁止在 AEF 一侧使用不同头或额外学习适配器。
- 支持集：5-shot、10-shot 两档。一个 shot = 一个含目标正样本的训练 patch，并配同数量负 patch；同一类别/折/seed 的 patch ID 文件对玄女、AEF、raw 保持字节一致。
- 划分：优先使用连续空间 block 的 5-fold 及缓冲区。若哈尔滨现有数据尚未生成合格的空间 split，先生成并固定该 split，再跑任何模型；绝不沿用海淀随机 fold。
- 训练：同一优化器、学习率、batch、epoch、随机种子 `42/43/44`，不早停，以最后一个 epoch 的读取头为准。
- 阈值：仅在验证 patch 的 `0.001..0.999` 网格挑每类别阈值；测试集只报告该阈值下 F1/mIoU/Precision/Recall，同时报告无阈值的 AP 与 ROC-AUC。
- 统计：每任务 × shot × 5 folds × 3 seeds。统计摘要为均值±标准差，玄女—AEF差异用同 patch 的分层空间 bootstrap 95% CI。不得用 AUC 单独宣称更好。

### 2.3 三种模型条件

| 条件 | 编码器 | 输入 | 允许训练 | 用途 |
| --- | --- | --- | --- | --- |
| `P10C-Haidian` | 海淀 P10C e800 | 海淀 P10C 原始六源输入 | 下游 Conv3x3 | 实验臂 A |
| `P10C-Harbin-scratch` | 哈尔滨同配方 best/e800 | 哈尔滨 S2、S1、Landsat、高分光学、WorldCover | 无需再训编码器；下游 Conv3x3 | 实验臂 B |
| `P10C-Haidian-frozen→Harbin` | 海淀 P10C e800，所有参数冻结 | 仅通过兼容性闸门后可用的哈尔滨模态 | 仅哈尔滨 Conv3x3 | 实验臂 C |
| `AEF-annual-2025` | 官方 AEF 64D 年度产品 | AEF official COG，非学习重投影到参考网格 | 相同 Conv3x3 | A/B/C 的参照 |

## 3. P0：冻结迁移的输入兼容性闸门

海淀 P10C 原本配置为 S2(12)、S1(2)、Landsat(7)、高分光学(3)、高分 SAR(1)；哈尔滨已有 S2(12)、S1(2)、Landsat(7)、高分光学(3)，**没有高分 SAR**。因此在任何迁移分数之前，必须做以下无训练检查。

### Task 1：生成并冻结跨城输入契约

**Files:**
- Create: `configs/paper_p10c_haidian_frozen_to_harbin_20260728.yaml`
- Create: `docs/paper/rse_2026/p10c_haidian_harbin_input_contract_20260728.md`
- Test: `tests/test_p10c_harbin_transfer_contract.py`

**Consumes:** 海淀 P10C e800、海淀训练配置、哈尔滨 manifest 和 source statistics。

**Produces:** 一个显式说明每个 P10C 输入槽位在哈尔滨如何提供/缺失的不可变 YAML 契约，以及机器可读的 source/shape/normalization 审计记录。

- [ ] **Step 1: 写失败测试**

```python
def test_transfer_contract_rejects_silent_sensor_rename_or_shape_change():
    contract = load_transfer_contract(CONFIG)
    assert contract["s2"]["channels"] == 12
    assert contract["s1"]["channels"] == 2
    assert contract["landsat"]["channels"] == 7
    assert contract["highres_optical"]["channels"] == 3
    assert contract["highres_sar"]["availability"] == "explicit_missing"
    assert contract["normalization"] == "haidian_p10c_training_statistics"
```

- [ ] **Step 2: 运行测试并确认其先失败**

Run: `python -m pytest tests/test_p10c_harbin_transfer_contract.py -v`  
Expected: FAIL，因为迁移配置与契约尚不存在。

- [ ] **Step 3: 建立最小可审计迁移配置**

配置必须包含：

```yaml
transfer:
  encoder_checkpoint: /data/xuannv_embedding/outputs/v2_p10c_haidian_202512_202605_osm_semantic_hardneg_20260704/epoch_800.pt
  freeze_encoder: true
  target_region: harbin
  target_manifest: /data/xuannv_embedding/processed/harbin/manifest_e5_harbin424_202512_202605.json
  normalization: haidian_p10c_training_statistics
  modality_contract:
    s2: {channels: 12, availability: provided}
    s1: {channels: 2, availability: provided}
    landsat: {channels: 7, availability: provided}
    highres_optical: {channels: 3, availability: provided_with_physical_source_provenance}
    highres_sar: {channels: 1, availability: explicit_missing}
  forbidden:
    - encoder_finetuning
    - learned_adapter
    - target_statistics_fitting
    - silent_modality_rename
```

“provided_with_physical_source_provenance”只表示 3 通道高分光学占据同一模型输入槽位；文档必须保留海淀和哈尔滨物理来源、获取时间、分辨率差异。它不是把两个数据源宣称为相同产品。

- [ ] **Step 4: 运行 one-batch 前向 dry-run**

Run: 使用 2 个哈尔滨 patch 和冻结 e800，依次测试：共同低分模态；加入哈尔滨 3 通道高分光学；显式标记高分 SAR 缺失。记录每种条件是否有 `64×128×128` 非 NaN 输出、输出范数分布及可用性 mask。

Expected: 只有当“高分 SAR 缺失”被模型显式处理而不是静默零填充为真实观测，才接受迁移试验。

- [ ] **Step 5: 运行测试并确认通过**

Run: `python -m pytest tests/test_p10c_harbin_transfer_contract.py -v`  
Expected: PASS。

- [ ] **Step 6: 写入审计结论并提交**

审计文档必须列出接受或拒绝的输入条件及原因。若失败，实验 C 标为“不适用：训练时未覆盖该缺模态合同”，论文只保留 A+B，绝不改变 P10C 配方后声称冻结迁移。

Run: `git add configs/paper_p10c_haidian_frozen_to_harbin_20260728.yaml docs/paper/rse_2026/p10c_haidian_harbin_input_contract_20260728.md tests/test_p10c_harbin_transfer_contract.py && git commit -m "test: seal P10C Harbin transfer input contract" && git push`

## 4. 哈尔滨与 AEF 资源完整性

### Task 2：为哈尔滨建立统一的评测清单和 AEF 覆盖审计

**Files:**
- Create: `configs/eval/paper_harbin_424_spatial_folds_20260728.json`
- Create: `configs/eval/paper_harbin_aef_coverage_20260728.json`
- Create: `docs/paper/rse_2026/harbin_evaluation_inventory_20260728.md`
- Modify: `scripts/eval/export_registered_paper_encoders.sh` 或新增小型哈尔滨导出入口
- Test: `tests/test_harbin_paper_inventory.py`

**Consumes:** 424 patch 哈尔滨 manifest、AEF 官方 COG index、OSM 标签根目录。

**Produces:** 完整 424 patch 空间分块、训练/验证/测试 buffer、每 patch 的 AEF 2025 覆盖与有效像素清单。

- [ ] **Step 1: 写失败测试**

```python
def test_harbin_inventory_has_exact_patch_coverage_and_disjoint_spatial_folds():
    inventory = load_inventory(INVENTORY)
    assert len(inventory["patch_ids"]) == 424
    assert inventory["aef_full_valid_coverage"] is True
    assert every_test_patch_is_outside_train_buffer(inventory)
```

- [ ] **Step 2: 运行测试并确认其先失败**

Run: `python -m pytest tests/test_harbin_paper_inventory.py -v`  
Expected: FAIL，因为哈尔滨尚无论文级冻结空间清单。

- [ ] **Step 3: 生成空间 5-fold 与标签审计**

每折应为连续地理 block；测试 block 周边留出一个 patch 宽缓冲区，验证 block 与训练 block 不重叠。统计每任务的正 patch、正像素比、空标签 patch 和 OSM 覆盖。若任一任务在某折少于 5 个可用正 patch，则该任务不进入该城市主表。

- [ ] **Step 4: 导出/验证哈尔滨 AEF annual 2025 embedding**

从官方索引与 COG 缓存按 label-free 424 patch 清单切出 64×128×128 map；记录来源哈希、投影规则、valid mask 与输出哈希。任何缺像素、零填充 nodata 或标签驱动选图均失败。

- [ ] **Step 5: 运行测试并确认通过**

Run: `python -m pytest tests/test_harbin_paper_inventory.py -v`  
Expected: PASS。

- [ ] **Step 6: 提交**

Run: `git add configs/eval/paper_harbin_424_spatial_folds_20260728.json configs/eval/paper_harbin_aef_coverage_20260728.json docs/paper/rse_2026/harbin_evaluation_inventory_20260728.md scripts/eval tests/test_harbin_paper_inventory.py && git commit -m "data: register Harbin paper evaluation inventory" && git push`

## 5. 复用现有模型，完成两组哈尔滨 embedding

### Task 3：导出哈尔滨 scratch 复现与冻结迁移 embedding

**Files:**
- Create: `scripts/eval/export_p10c_harbin_paper_embeddings.py`
- Create: `configs/eval/paper_harbin_embedding_exports_20260728.json`
- Test: `tests/test_p10c_harbin_export.py`

**Consumes:** Task 1 输入契约、Task 2 清单、哈尔滨 scratch best/e800 checkpoint、冻结海淀 P10C e800 checkpoint。

**Produces:** 每一个条件的 2026-04 `64×128×128` embedding、source manifest、输出哈希、PCA 仅用于定性检查。

- [ ] **Step 1: 写失败测试**

```python
def test_export_registry_requires_frozen_encoder_and_complete_424_patch_set():
    registry = load_export_registry(REGISTRY)
    assert registry["p10c_harbin_scratch"]["shape"] == [64, 128, 128]
    assert registry["p10c_haidian_frozen_harbin"]["encoder_frozen"] is True
    assert registry["p10c_haidian_frozen_harbin"]["patch_count"] == 424
```

- [ ] **Step 2: 运行测试并确认其先失败**

Run: `python -m pytest tests/test_p10c_harbin_export.py -v`  
Expected: FAIL，因为导出注册表尚不存在。

- [ ] **Step 3: 先导出哈尔滨 scratch 两个 checkpoint 的小样本**

对 `best.pt` 和 `epoch_800.pt` 各导出 16 个固定 patch，比较重建统计、embedding 范数、PCA、跨 patch 边界。选择下游表现更好的一个；不得根据测试集指标选择，选择依据为独立验证 reconstruction loss 加上训练前固定的质量检查。

- [ ] **Step 4: 导出完整 424 patch**

只为选定的哈尔滨 scratch checkpoint 导出全量。若 Task 1 接受迁移合同，再导出海淀 e800 冻结迁移 embedding；两次导出均采用 NPU 并行、固定 patch 分片和可恢复日志。

- [ ] **Step 5: 执行输出质量检查**

检查：每个 patch/条件形状为 64×128×128、无 NaN/Inf、没有零方差通道、patch ID 与 Task 2 清单完全相同；输出 manifest 写入 checkpoint/config/normalization/source 的 SHA-256。

- [ ] **Step 6: 运行测试并确认通过**

Run: `python -m pytest tests/test_p10c_harbin_export.py -v`  
Expected: PASS。

- [ ] **Step 7: 提交**

Run: `git add scripts/eval/export_p10c_harbin_paper_embeddings.py configs/eval/paper_harbin_embedding_exports_20260728.json tests/test_p10c_harbin_export.py && git commit -m "feat: export registered P10C Harbin paper embeddings" && git push`

## 6. 统一的哈尔滨下游对照

### Task 4：运行 scratch、冻结迁移、AEF 的同协议稀疏读取

**Files:**
- Create: `scripts/eval/run_p10c_harbin_sparse_benchmark.py`
- Create: `configs/eval/paper_harbin_sparse_mapping_20260728.json`
- Create: `docs/paper/rse_2026/harbin_sparse_mapping_results_20260728.md`
- Test: `tests/test_p10c_harbin_sparse_benchmark.py`

**Consumes:** Task 2 spatial split、Task 3 embeddings、同一 shot schedule 与 OSM 标签根。

**Produces:** 每任务 × 5/10 shot × 5 fold × 3 seed 的 scratch、frozen-transfer、AEF 指标、阈值与预测哈希。

- [ ] **Step 1: 写失败测试**

```python
def test_all_harbin_families_share_the_same_reader_schedule_and_threshold_rule():
    matrix = load_matrix(MATRIX)
    assert matrix["families"] == ["p10c_harbin_scratch", "p10c_haidian_frozen", "aef_annual_2025"]
    assert matrix["reader"] == "conv3x3_64d"
    assert matrix["threshold_source"] == "validation_only"
    assert matrix["seed_list"] == [42, 43, 44]
```

- [ ] **Step 2: 运行测试并确认其先失败**

Run: `python -m pytest tests/test_p10c_harbin_sparse_benchmark.py -v`  
Expected: FAIL，因为矩阵尚未注册。

- [ ] **Step 3: 固定三方共用的训练日程**

每个任务/shot/fold/seed 生成一个不可变 JSON：支持 patch、负 patch、验证 patch、测试 patch、类别、随机种子。所有特征的标准化只在 support 像素计算；禁止 AEF 或任何玄女条件使用额外 target 统计量。

- [ ] **Step 4: 并行运行 probe**

在 6 张 NPU 上按“任务/模型族”分片，确保每个 worker 写独立输出目录。每个 cell 使用固定 `Conv3x3`、固定训练轮数、固定验证阈值网格。先跑一个 fold 的三方 smoke，核对预测图/概率范围/GT 对齐后再提交全矩阵。

- [ ] **Step 5: 聚合统计与失败处理**

输出 F1、AP、ROC-AUC、mIoU、Precision、Recall 的 mean±std；对 `frozen - AEF` 和 `scratch - AEF` 执行 paired spatial bootstrap。如果迁移合同被拒绝，只聚合 scratch 与 AEF，并明确说明跨城冻结迁移未声称。

- [ ] **Step 6: 生成论文图**

必须生成：三类任务的 5/10-shot 均值及置信区间图；每类至少 2 个测试 patch 的“光学—GT—P10C scratch—P10C frozen—AEF”行图；哈尔滨全域 P10C PCA 仅作定性图。图注写清 annual/monthly 与 OSM-assisted 标签边界。

- [ ] **Step 7: 运行测试并确认通过**

Run: `python -m pytest tests/test_p10c_harbin_sparse_benchmark.py -v`  
Expected: PASS。

- [ ] **Step 8: 提交**

Run: `git add scripts/eval/run_p10c_harbin_sparse_benchmark.py configs/eval/paper_harbin_sparse_mapping_20260728.json docs/paper/rse_2026/harbin_sparse_mapping_results_20260728.md tests/test_p10c_harbin_sparse_benchmark.py && git commit -m "feat: benchmark P10C Haidian Harbin sparse mapping" && git push`

## 7. 海淀主结果的复用与严谨表述

### Task 5：把海淀 P10C 结果规范化为论文证据包

**Files:**
- Create: `docs/paper/rse_2026/haidian_p10c_evidence_inventory_20260728.md`
- Modify: `docs/paper/rse_2026/manuscript_draft_en.md`
- Modify: `docs/paper/rse_2026/manuscript_draft_zh.md` only after preserving the user’s edits and explicit review

**Consumes:** P10C e800、海淀 320 patch、已完成的 P10C/AEF 对比、海淀 few-shot raw 对照、PCA 与下游图。

**Produces:** 海淀实验的模型卡、协议卡、图表清单和严格的可用/不可用结论。

- [ ] **Step 1: 建立证据登记表**

每一张表/图绑定：checkpoint hash、embedding export hash、任务标签 hash、fold/shot schedule、reader、阈值规则、生成脚本和证据级别。旧的单 prototype + query adaptive 图不能混入新的多 prototype/验证阈值协议图。

- [ ] **Step 2: 选择两类互不混淆的海淀结果**

1. 产品展示：P10C 2026-04 月度 PCA、全域建筑/道路/水体图以及 5/10-shot 局部案例。
2. 定量对照：只使用与 AEF 完全同 reader/split/schedule 的一套结果。若当前 P10C 结果是随机 patch fold，则在正文标注为随机 patch readout，不与哈尔滨空间 fold 合并平均；优先重新读取已导出的 embedding，而非重训 encoder。

- [ ] **Step 3: 更新文稿中的公平性语言**

主文固定使用：

> We compare a monthly Xuannv embedding with the publicly released annual AEF product under an identical downstream readout. This evaluates practical representation utility under different product cadences; it is not a matched-time or matched-input pretraining comparison.

并固定使用：

> OSM-derived masks are weak semantic supervision and downstream readout targets; results are therefore reported as OSM-assisted sparse mapping rather than independent label transfer.

- [ ] **Step 4: 自检与提交**

Run: `git diff --check && python -m py_compile scripts/eval/*.py`  
Run: `git add docs/paper/rse_2026 && git commit -m "docs: register P10C Haidian paper evidence" && git push`

## 8. 论文成稿结构与图表

### Task 6：用 A+B+C 结果填充论文，而不是先写强结论

**Files:**
- Modify: `docs/paper/rse_2026/manuscript_draft_en.md`
- Modify: `docs/paper/rse_2026/manuscript_draft_zh.md` only after user review
- Create: `docs/paper/rse_2026/figure_table_manifest_20260728.md`
- Create: `scripts/report/build_p10c_haidian_harbin_paper_figures.py`

**Consumes:** Tasks 1–5 的冻结产物与可视化。

**Produces:** 英文主稿、中文对照稿、可复建的图表脚本与资产清单。

- [ ] **Step 1: 固定六张主图和四张主表**

| 编号 | 内容 | 数据来源 |
| --- | --- | --- |
| Fig. 1 | 月度 embedding field 工作流：多源输入→P10C→64D map→少量标注读取 | 已有架构图改造 |
| Fig. 2 | 海淀 2026-04：光学、PCA、建筑/道路/水体 5/10-shot 局部制图 | 已有 P10C/AEF 图，按同协议重画 |
| Fig. 3 | 海淀 P10C 与年度 AEF 的稀疏映射柱状/区间图 | P10C 统一结果 |
| Fig. 4 | 哈尔滨 scratch 与 AEF 的三任务稀疏映射图 | Task 4 |
| Fig. 5 | 海淀冻结 P10C→哈尔滨、哈尔滨 scratch、AEF 的迁移图 | Task 4；若合同拒绝移入补充材料 |
| Fig. 6 | 月度产品序列：同一海淀区域三个月 embedding/PCA 与静态地物边界 | 已有六个月 export；定性展示 |
| Table 1 | 两城市、传感器、月份、patch 数、缺失模态 | 配置/manifest 审计 |
| Table 2 | P10C 架构、损失与训练配方 | P10C YAML |
| Table 3 | 海淀与 AEF 统一稀疏映射定量指标 | Task 5 |
| Table 4 | 哈尔滨 scratch/冻结/AEF 指标与跨城解释 | Task 4 |

- [ ] **Step 2: 章节写作顺序**

1. Introduction：问题是“年度全球 embedding 难以满足城市月度更新”；贡献是月度多源 embedding field、少量标注读取、两个城市的配方/冻结迁移证据。
2. Data and method：城市、六个月、传感器、128×128×64、重建+OSM 弱语义+困难重建。不要把 64 维逐维解释成类别。
3. Evaluation protocol：AEF 年度对照、共同 `Conv3x3`、5/10-shot、空间 folds、validation threshold、OSM-assisted 限制。
4. Results：先海淀，后哈尔滨 scratch，再冻结迁移；定量指标与案例图一起给。
5. Discussion：月度和年度的产品定位差异、云/缺模态风险、OSM 泄漏边界、哈尔滨输入缺 SAR 的现实限制。
6. Conclusion：只说证据支持的结论，不写“全球 SOTA”或“独立泛化”。

- [ ] **Step 3: 图表复现与审查**

每幅图要有：输入数据版本、脚本、输出 hash、论文标题、单位、月份、色条/图例、是否定量或定性。生成 PDF 前检查 Markdown 所有图片为仓库内相对路径且渲染成功。

- [ ] **Step 4: 论文审查与提交**

至少从方法、统计、遥感应用、期刊格式四个视角审查，逐条关闭 P0/P1 问题。每次文稿实质更新独立 commit/push。

## 9. 执行顺序、并行与停止条件

1. 先完成 Task 1 的一批次 dry-run。它决定是否能做冻结迁移，成本极低。
2. Task 2 的哈尔滨空间清单和 AEF 覆盖审计可与 Task 1 并行。
3. Task 3 只在 Task 1/2 接受后执行；Scratch 导出与 AEF export 可各占 2–3 张 NPU 并行。
4. Task 4 先做一个三方 smoke；无对齐或阈值问题立即停，不开全矩阵。Smoke 通过后用 6 张 NPU 按模型族/任务分片。
5. Task 5 与 Task 4 并行整理海淀证据；Task 6 只填入已冻结的数值。

**停止条件：**

- P10C 在哈尔滨无法显式处理缺失高分 SAR：停止冻结迁移，保留哈尔滨 scratch 复现，不改写为“跨城泛化”。
- AEF 2025 对任一评测 patch 缺有效覆盖：停止该城市 AEF 对照，不能零填充补齐。
- OSM 某任务无法支撑 5 个空间 fold 的 5-shot：从主表移除该任务；不靠重复采样伪造样本数。
- 任何一方的 reader、split、阈值或 support schedule 不一致：该组结果不可比较，先修协议再跑。

## 10. 论文可作出的结论边界

完成 A+B+C 后，最稳妥的论文结论为：

1. P10C 能在海淀输出连续的 64 维月度城市 embedding，并用少量支持 patch 读取建筑、道路和水体。
2. 相同训练配方在哈尔滨可从零复现，说明并非单一海淀数据偶然有效。
3. 如果冻结迁移合同和 Task 4 成立，海淀训练的 P10C 在未接触哈尔滨上游训练的条件下仍保留部分稀疏读取能力；这一点是“区域迁移能力”，不是全球无监督泛化。
4. 与 AEF 的对照是相同下游读取流程下的**月度产品与年度产品**对照。玄女的核心差异化是更新频率与城市多源设计；两者时间、输入与预训练规模不同，结果不作绝对 global SOTA 宣称。

## 自检

- 需求覆盖：包含了用户指定的海淀 P10C vs AEF、哈尔滨 scratch vs AEF、冻结海淀 encoder 到哈尔滨、现有资源清单、避免重复训练与完整论文叙事。
- 不重复实验：所有 P10C 海淀和哈尔滨 scratch encoder 训练均明确复用；只做缺失的兼容性/导出/统一读取。
- 公平性：将“相同 downstream readout”与“上游信息不等价”分开陈述；月度能力作为创新与产品优势，而非隐藏时间差。
- 风险闭环：输入缺失、高分 SAR、AEF 覆盖、OSM 弱标签与空间划分均有失败即停的规则。
