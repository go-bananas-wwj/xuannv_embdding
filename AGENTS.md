# Agent Instructions for xuannv_embedding

> 本文件对协助本项目的 AI agent 提出强制规范。所有规范优先级低于用户在该对话中的直接指令，但高于默认系统提示。
> 本文件同时承担"项目导览"职责：假设读者对本项目一无所知。

## 1. 项目概览

**玄女（xuannv）embedding** 是一个参考 AlphaEarth Foundations（AEF）思路实现的月度地理嵌入项目，包名 `xuannv_embedding`（`pyproject.toml` 描述："Clean re-implementation of AlphaEarth Foundations with monthly embeddings"）。

- **目标**：不是为每个下游任务训练独立分割模型，而是训练一个可复用的月度地理 embedding。对每个 1280 m × 1280 m 的 patch 生成 **128×128 空间分辨率、64 维** 的密集 embedding map，下游只需少量标注 + 轻量头（linear / MLP / conv3x3 / UNet / deeplab_lite）即可完成建筑提取、道路提取、水体提取、土地覆盖等多任务。
- **输入**：多源时序遥感数据 —— Sentinel-2（12 通道）、Sentinel-1（2 通道）、Landsat（7 通道）、高分辨率光学（海淀 4 通道 / 哈尔滨 3 通道）、高分辨率 SAR，以及 OSM 弱语义标签、WorldCover 分类标签。
- **当前生产版**：`haidian-embedding-v1`（分支 `haidian-embedding-v1-production`），实验 `P10C epoch800`，海淀区 320 个 patch、2025-12 至 2026-05 月度数据。大文件（checkpoints、embeddings、下游头、可视化）发布在 ModelScope 数据集 `WeijieWu/xuannv_haidian_embdding`，详见 `README.md` 与 `docs/production/haidian_artifacts_manifest.json`。
- **实验区**：海淀（haidian）与哈尔滨（harbin）；另有全国尺度的 china_v1 数据准备计划（`configs/national/`、`docs/plans/`）。

### 模型架构（`src/xuannv_embedding/models/`）

`AEFModel`（`models/model.py`）的前向链路：

```
per-sensor stem（sensor_encoders.py）
  → 多分辨率 STP（Space-Time-Precision）编码器（blocks.py）
  → 月度嵌入模块 + 嵌入上采样头
  → 可选高分辨率融合（highres_fusion.py，逐月）
  → vMF 瓶颈（bottleneck.py，单位球面）
  → 多目标解码器（decoders.py：continuous / categorical）
```

训练目标（`training/losses.py`）包括多源重建、高分辨率重建、OSM 弱语义、困难重建（随机遮挡模态/月份/空间块）等，各目标权重在配置的 `model.target_heads` 中指定。

## 2. 工作流要求（强制）

### 2.0 项目长期记忆

在处理海淀区 embedding 训练、下游评测、报告整理或相关实验设计前，先阅读：

- `docs/agent_memory/haidian_embedding_project_memory.md`

该文档记录了本项目参考 AEF 路线构建通用地理 embedding 的目标、关键实验结论（few-shot 评测结果）、数据/训练经验、评测规范与后续升级方向。它用于避免长对话后上下文遗失或重复踩坑。

### 2.1 每完成一步，立即同步到 GitHub

本项目的代码仓库托管在 `git@github.com:go-bananas-wwj/xuannv_embdding.git`。
**任何对代码、配置、脚本、文档的实质性修改，必须在完成该步骤后立即 `git commit` 并 `git push`**，禁止等到一个完整大阶段结束后再一次性提交。

什么算"一步"：

- 完成一个独立功能/修复（例如：重构模型 bottleneck、新增预处理脚本、修改配置、更新文档）。
- 完成一次数据预处理或验证后，将对应的脚本/配置/日志文档提交。
- 跑通一组测试并确认通过后，立即提交。
- 任何对 `AGENTS.md`、README、数据布局文档的更新。

提交规范：

- 使用中文或英文 commit message 均可，但必须清晰说明本次改动。
- 推荐前缀：`feat:`、`fix:`、`docs:`、`refactor:`、`test:`、`data:`。
- 不要提交数据文件本身（数据在 `/data/xuannv_embedding/`，不进入 git）。

如果提交前需要确认远程分支或处理冲突，先尝试 `git pull --rebase`；遇到无法自动解决的冲突时，再向用户汇报。

### 2.2 数据与代码分离

- 代码、配置、文档必须保存在 `/root/workspace/xuannv/` 并进入 git。
- 原始/预处理数据、训练输出、大体积日志必须保存在 `/data/xuannv_embedding/` 或 `outputs/`，不进入 git。
- 在数据盘产生的文档（如 `/data/xuannv_embedding/README.md`）应在项目侧保留同步副本或说明（如 `docs/data_layout.md`、`docs/data_inventory.md`）。
- 报告图片必须复制到仓库内 `docs/production/assets/...` 并用相对路径引用；面向 GitHub/PDF 的 Markdown 中禁止引用 `/data/...` 路径。

### 2.3 状态同步

每完成一个子任务，向用户简要汇报：

- 完成了什么；
- 修改了哪些文件；
- 是否已提交/推送；
- 下一步是什么。

## 3. 技术约束

- 使用 **PyTorch + torch_npu**，目标硬件为 **6 张 Ascend NPU**。使用 NPU 前须加载 CANN 环境：`source /usr/local/Ascend/cann-9.0.0/set_env.sh`（若 `torch_npu` 报 `tbe` 相关错误，通常是 CANN 环境未加载）。
- 禁止使用 `einops`，所有 reshape 使用 PyTorch 原生操作。
- 模型输出必须保持 **128×128 空间分辨率**（patch 尺寸 1280 m × 1280 m，10 m 等效分辨率）。
- 一般情况下不使用 L2 归一化作为 bottleneck 输出约束；**VMF（von Mises-Fisher）bottleneck 例外**，允许通过 L2 归一化将 embedding 约束到单位球面。
- 支持多源时序数据（S2、S1、Landsat）与稀疏高分辨率数据（光学、SAR）作为独立模态。

## 4. 仓库结构与模块划分

```
src/xuannv_embedding/          # 核心包（训练侧）
  config.py                    # 配置加载与校验（Config / ConfigError）
  data/                        # builder / dataset / sampler / transforms / collate
  models/                      # model.py(AEFModel) / blocks(STP编码器) / sensor_encoders
                               #   / bottleneck(vMF) / decoders / highres_fusion / time_encoding
  training/                    # trainer / losses / masking / optimizer / checkpoint / amp_utils
  utils/                       # device / geo / manifest
downstreams/downstreams/       # 下游评测框架（与 src 并列，作为独立包安装）
  data/                        # embedding_dataset / label_loaders / split
  heads/                       # linear_probe / segmentation_head / classification_head / change_detection_head
  tasks/  metrics/             # 任务定义与 mIoU/F1/AP/AUC 等指标、可视化
  inference.py
scripts/
  train/                       # train.py（DDP/单卡训练入口）、launch*.sh（torchrun 启动脚本）
  eval/                        # extract_embeddings / knn_eval / 各类 benchmark 脚本
  production/                  # 生产导出、下游 probe、打包、ModelScope 上传
  data/ qa/ report/ monitor/ scale/ release/  # 数据预处理、质量检查、报告、监控、实验管理
configs/                       # 实验配置（全部自包含，见第 5 节）
  production/                  # 生产配置 haidian_embedding_v1.yaml
  eval/ national/ regions/ paper_*/  # 评测划分、全国采样策略、论文实验
tests/                         # pytest 测试（见第 6 节）
docs/                          # 全部项目文档（见第 8 节）
```

数据盘布局（`/data/xuannv_embedding/`）详见 `docs/data_layout.md`：`raw/<region>/<source>/`（原始下载）、`processed/<region>/`（对齐切 patch 后的模型输入，含 `manifest.json`）、`outputs/`（训练产物）、`embeddings/`（可重建的推理产物）、`statistics/<region>/`（各 source 的 mean/std）、`downstream/labels/<region>/<task>/`（下游标注与 5-fold 划分）。

## 5. 配置文件规范

- **所有实验配置文件必须自包含**。禁止在实验 YAML 中使用 `_base_` 字段继承其他配置文件；`scripts/train/train.py` 中的 `reject_base_config` 会显式拒绝含 `_base_` 的配置。
- 每个实验配置文件都应包含该实验所需的完整配置（`experiment` / `model` / `training` / `data` 等全部字段），以便直接打开查看，无需跳转或合并多个文件。结构可参考 `configs/base.yaml`。
- 公共模板（如 `base.yaml`、`_base_.yaml`）可以作为参考保留，但**任何实际运行的实验配置都必须去掉 `_base_` 并写全所有字段**。
- 当需要基于现有实验做变体时，复制一份完整配置并直接修改其中的字段，而不是通过继承覆盖。
- 实验配置命名约定：`<版本>_<实验号>_<主题>_<区域>_<月份范围>_<日期>.yaml`， sanity 检查版本加 `_sanity` 后缀。

## 6. 构建、训练与测试命令

安装（Python ≥ 3.11）：

```bash
pip install -e .                  # 训练侧
pip install -e ".[downstream]"    # 含下游评测依赖（matplotlib/seaborn/scikit-image/shapely）
export PYTHONPATH=$PWD/src:$PWD/downstreams:$PYTHONPATH
```

训练（6×NPU DDP，入口 `scripts/train/train.py`）：

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
bash scripts/train/launch_6card.sh configs/<experiment>.yaml   # torchrun, 默认 6 卡
# 单卡调试：python scripts/train/train.py --config configs/smoke_tiny.yaml --device npu:0
```

测试（pytest，测试文件位于 `tests/`，含数据集、模型、训练、评测、全国数据准备等）：

```bash
python -m pytest tests/                  # 全量
python -m pytest tests/test_model.py     # 单文件
```

注意：`tests/conftest.py` 会将 OpenBLAS/OMP/MKL 线程数限制为 16，避免超多核机器上线程元数据耗尽崩溃。

Embedding 导出与下游评测（生产流程，详见 `README.md`）：

```bash
python scripts/production/export_haidian_embedding.py --checkpoint <ckpt> --output-root ... --months 202512 ... --device npu:0
python scripts/production/run_haidian_downstream_probe.py --embedding-root ... --month 202604 --tasks building road water --head mlp
```

下游框架的完整用法（数据准备、embedding 预生成、训练+评测、可视化）见 `downstreams/README.md`。

## 7. 代码风格

- 行宽 100：`black`（`target-version py311`）与 `ruff`（lint 规则 `E, F, I, W`）已在 `pyproject.toml` 中配置。
- Python 代码使用 `from __future__ import annotations` 与现代类型标注；docstring 与注释以中文为主。
- 新代码应与周围文件风格一致，不做无关重构；改动保持最小、聚焦。
- 依赖以 `pyproject.toml` 为准，不要假设未声明的库可用（如 `einops` 被显式禁止）。

## 8. 评测与文档规范

- 严肃的下游对比必须使用相同标签、相同 fold 划分、相同 shot 协议、相同阈值选择方法，报告 F1/AP/AUC 并附可视化；严格区分 full-label 与 few-shot（5/10/50-shot）结论。核心任务为建筑、道路、水体提取；标准下游头为 `wide_mlp`、`deep_wide_mlp`、`conv3x3`、`unet`、`deeplab_lite`。详见 `docs/agent_memory/haidian_embedding_project_memory.md` 第 2、6 节。
- 关键文档索引：
  - `docs/agent_memory/haidian_embedding_project_memory.md` —— 项目长期记忆（必读）
  - `docs/data_layout.md` / `docs/data_inventory.md` —— 数据盘布局与数据清单
  - `docs/production/` —— 生产版模型卡、数据卡、指标报告、artifacts 清单
  - `docs/checkpoint_registry.md` —— 需长期保护的 checkpoint 登记
  - `docs/downstream_eval_protocol.md` —— 下游评测协议
  - `docs/plans/` —— china_v1 全国数据准备等计划
  - `CHANGELOG.md` —— 变更记录（重大变更应同步更新）

## 9. 联系方式与路径

- 仓库：`git@github.com:go-bananas-wwj/xuannv_embdding.git`
- 数据根目录：`/data/xuannv_embedding/`
- 项目根目录：`/root/workspace/xuannv/`
- ModelScope 数据集：`WeijieWu/xuannv_haidian_embdding`
