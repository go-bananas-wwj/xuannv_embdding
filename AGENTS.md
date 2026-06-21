# Agent Instructions for xuannv_embedding

> 本文件对协助本项目的 AI agent 提出强制规范。所有规范优先级低于用户在该对话中的直接指令，但高于默认系统提示。

## 1. 工作流要求

### 1.1 每完成一步，立即同步到 GitHub

本项目的代码仓库托管在 `git@github.com:go-bananas-wwj/xuannv_embdding.git`。
**任何对代码、配置、脚本、文档的实质性修改，必须在完成该步骤后立即 `git commit` 并 `git push`**，禁止等到一个完整大阶段结束后再一次性提交。

什么算“一步”：

- 完成一个独立功能/修复（例如：重构模型 bottleneck、新增预处理脚本、修改配置、更新文档）。
- 完成一次数据预处理或验证后，将对应的脚本/配置/日志文档提交。
- 跑通一组测试并确认通过后，立即提交。
- 任何对 `AGENTS.md`、README、数据布局文档的更新。

提交规范：

- 使用中文或英文 commit message 均可，但必须清晰说明本次改动。
- 推荐前缀：`feat:`、`fix:`、`docs:`、`refactor:`、`test:`、`data:`。
- 不要提交数据文件本身（数据在 `/data/xuannv_embedding/`，不进入 git）。

如果提交前需要确认远程分支或处理冲突，先尝试 `git pull --rebase`；遇到无法自动解决的冲突时，再向用户汇报。

### 1.2 数据与代码分离

- 代码、配置、文档必须保存在 `/root/workspace/xuannv/` 并进入 git。
- 原始/预处理数据、训练输出、大体积日志必须保存在 `/data/xuannv_embedding/` 或 `outputs/`，不进入 git。
- 在数据盘产生的文档（如 `/data/xuannv_embedding/README.md`）应在项目侧保留同步副本或说明（如 `docs/data_layout.md`、`docs/data_inventory.md`）。

### 1.3 状态同步

每完成一个子任务，向用户简要汇报：

- 完成了什么；
- 修改了哪些文件；
- 是否已提交/推送；
- 下一步是什么。

## 2. 技术约束

- 使用 **PyTorch + torch_npu**，目标硬件为 **6 张 Ascend NPU**。
- 禁止使用 `einops`，所有 reshape 使用 PyTorch 原生操作。
- 模型输出必须保持 **128×128 空间分辨率**（patch 尺寸 1280 m × 1280 m，10 m 等效分辨率）。
- 一般情况下不使用 L2 归一化作为 bottleneck 输出约束；**VMF（von Mises-Fisher）bottleneck 例外**，允许通过 L2 归一化将 embedding 约束到单位球面。
- 支持多源时序数据（S2、S1、Landsat）与稀疏高分辨率数据（光学、SAR）作为独立模态。

## 3. 联系方式

- 仓库：`git@github.com:go-bananas-wwj/xuannv_embdding.git`
- 数据根目录：`/data/xuannv_embedding/`
- 项目根目录：`/root/workspace/xuannv/`

## 4. 配置文件规范

- **所有实验配置文件必须自包含**。禁止在实验 YAML 中使用 `_base_` 字段继承其他配置文件。
- 每个实验配置文件都应包含该实验所需的完整配置（模型、数据、训练参数等），以便直接打开查看，无需跳转或合并多个文件。
- 公共模板（如 `base.yaml`、`_base_.yaml`）可以作为参考保留，但**任何实际运行的实验配置都必须去掉 `_base_` 并写全所有字段**。
- 当需要基于现有实验做变体时，复制一份完整配置并直接修改其中的字段，而不是通过继承覆盖。
