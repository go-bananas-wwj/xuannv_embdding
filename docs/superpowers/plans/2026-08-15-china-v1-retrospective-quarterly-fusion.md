# China V1 年度回溯季度融合 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 2020、2021 两个完整年度的 S2/S1、AEF 年度 embedding 和国产 2 米一张图，训练并导出 2020Q1--2021Q4 共八期全国 10 米、64 维季度回溯 embedding。

**Architecture:** 每年使用一次冻结的 OlmoEarth v1.2 提取带全年上下文的月度 token，再按三个月池化为四个回溯季度；轻量 10 米季度支路保留季度定位和边界，年度 2 米支路提供静态空间细节，AEF 首版只作为年度教师。所有外部特征离线分片缓存，融合模型使用零初始化门控残差和 vMF 瓶颈输出 `128 x 128 x 64`。

**Tech Stack:** Python 3.11、PyTorch、torch_npu、6 x Ascend NPU、CANN 9.0、Rasterio、Zarr 2.x、NumPy、PyYAML、pytest；OlmoEarth 提取器使用独立锁定环境，训练主环境不原地升级。

## Global Constraints

- 产品语义固定为“年度数据完成后，利用全年信息回溯当年四个季度”，不是截止季度末实时产品。
- 输入年份严格为 2020、2021；输出季度严格为 2020Q1--2021Q4，共八期。
- 单 patch 为 `1280 m x 1280 m`，输出为 `128 x 128 x 64`，等效 10 米分辨率。
- 训练空间骨架固定为 62,000 个位置；两年和八个季度共用相同 `patch_id`。
- OlmoEarth 第一候选为 v1.2 Tiny/P4，Tiny/P8 为吞吐基线，Small/P4 为精度候选。
- 第一生产候选只使用 S2 12 bands + S1 VV/VH；Landsat 在满足官方 11-band 合同前不得接入 OlmoEarth。
- AEF 首版只作为年度教师，不作为推理输入；AEF-input 只保留为有复制风险的消融实验。
- 年度 2 米影像只提供静态空间细节，不作为季度变化真值；缺失年份必须显式置 `year_valid=0`。
- 使用 PyTorch + torch_npu；训练目标硬件为 6 张 Ascend NPU。
- 禁止使用 `einops`；空间折叠使用原生 `reshape`、`permute` 和 `contiguous`。
- VMF bottleneck 允许 L2 归一化；其他位置不额外把 L2 当作默认训练约束。
- 所有实际运行 YAML 必须自包含，禁止 `_base_`。
- 代码、配置、文档保存在 `/root/workspace/xuannv/`；数据和产物保存在 `/data/xuannv_embedding/`，不进入 Git。
- 每个独立任务完成测试后立即 commit 并 push，不积压多个任务一次提交。
- 容量统一使用十进制，`1 TB = 10^12 Byte`；时间估算均为规划区间，必须由 4/200/2,000/10,000 patch 实测逐级替换。
- OlmoEarth 衍生物对外发布前必须完成许可证审查；未取得适用授权时，不得把研究原型直接作为无用途限制的通用产品发布。

---

## 1. 已冻结的产品定义

### 1.1 允许和不允许的时间信息

允许每个年份的 Q1--Q4 token 在 OlmoEarth 内部看见同年 12 个月，因为本产品是年度结束后的回溯结果。2020 与 2021 必须分别前向，不能把 24 个月放入同一个序列：

```text
2020 月度输入 -> Olmo 2020 全年上下文 -> 2020Q1/Q2/Q3/Q4
2021 月度输入 -> Olmo 2021 全年上下文 -> 2021Q1/Q2/Q3/Q4
```

AEF 2020 只监督 2020 四季度的年度聚合，AEF 2021 只监督 2021 四季度。高分数据按年份独立提供；若实际只有一张跨年全国图，只能登记为 `static_reference`，不得伪造两个年度观测。

### 1.2 输出语义

每个季度 embedding 同时表达：

1. 全年上下文下该季度的多源遥感语义；
2. 该季度 10 米 S2/S1 的局部状态；
3. 同年度 2 米一张图提供的静态边界和纹理；
4. 与同年 AEF 年度语义一致、但不被迫与其他季度相同的 64 维单位向量。

产品说明必须使用“10 米网格季度回溯 embedding”，不得表述为“季度 2 米 embedding”或“季度 2 米观测重建”。

### 1.3 第一版成功标准

第一版不是以总训练 loss 最低为成功，而是同时满足：

- 4-patch NPU gate 无 NaN、OOM 和热路径 CPU fallback；
- BF16 与 FP32 输出余弦相似度不低于 0.995；
- 200-patch 中，推荐融合方案在 building/road/water 三项的两个以上任务改善 10-shot F1；
- 任一核心任务相对最强无高分基线下降不超过 2 个 F1 点；
- 相邻季度 embedding 余弦中位数不得高于 0.995；
- 季度方差不得低于 raw-quarter 基线的 10%；
- 高分分支缺失时输出必须数值等于基线容差范围，验证旁路为真实 identity；
- patch seam 指标不劣于海淀 P10C/P12 生产 embedding；
- 10,000 patch 连续运行 24--48 小时后能够从中断点精确续跑；
- 许可证、署名、数据版本和输入哈希可追溯。

---

## 2. 模型和数据接口

### 2.1 PatchYear 样本

训练数据以“一个位置、一个年份”为一个样本；Dataset 长度为 `62,000 x 2 = 124,000`：

```python
class PatchYearSample(TypedDict):
    patch_id: str
    year: int
    product_mode: Literal["annual-retrospective-quarterly"]
    realtime_asof_valid: Literal[False]
    olmo_quarters: torch.Tensor       # [4, D, 32, 32]，Tiny/P4 时 D=192
    olmo_valid_months: torch.Tensor   # [4]，每季度有效月份数 0..3
    raw_quarters: torch.Tensor        # [4, C_raw, 128, 128]
    raw_valid: torch.Tensor           # [4, 1, 128, 128]
    highres_static: torch.Tensor      # [C_static, 128, 128]，首版 C_static=32
    highres_valid: torch.Tensor       # [1, 128, 128]
    highres_year_valid: torch.Tensor  # [1]
    aef_int8: torch.Tensor            # [64, 128, 128]
    aef_valid: torch.Tensor           # [1, 128, 128]
    affine: torch.Tensor              # [6]
    crs_epsg: int
```

`raw_quarters` 首版由每季度 S2/S1 稳健复合及质量统计组成，不重复保存 12 个月全量张量。推荐首版通道为 S2 RGB/NIR/SWIR 复合、S1 VV/VH、有效观测数和季度内离散度；最终通道顺序写入 manifest 并纳入 fingerprint。

### 2.2 Olmo 缓存接口

独立提取环境只写数据，不被训练进程 import：

```text
olmo_features/shard_00000.zarr/
  quarter_features  [N, 4, D, Hs, Ws]  bfloat16/float16
  valid_month_count [N, 4]             uint8
  patch_id          [N]                fixed unicode/index
  year              [N]                int16
  done              [N]                uint8
  attrs:
    schema_version
    model_id
    model_commit
    weights_sha256
    input_manifest_sha256
    band_order
    normalization_version
    patch_size
    extractor_env_sha256
```

每个 shard 控制在 512 MB--2 GB；不得创建 124,000 个独立 `.pt` 文件。写入采用 `.partial + done bitmap + atomic rename`。

### 2.3 融合模型接口

`RetrospectiveQuarterlyFusionModel.forward` 接收
`olmo_quarters [B,4,D,32,32]`、`raw_quarters [B,4,C_raw,128,128]`、
`raw_valid [B,4,1,128,128]`、`highres_static [B,C_static,128,128]`、
`highres_valid [B,1,128,128]`、`highres_year_valid [B,1]`、
`aef_float [B,64,128,128]`、`aef_valid [B,1,128,128]` 和布尔开关
`use_aef_input`。返回字典必须包含
`embedding [B,4,64,128,128]`、`pre_vmf [B,4,hidden_dim,128,128]`、
`highres_gate`、`raw_gate` 和 `aef_gate`。

融合采用：

```text
base = olmo_upsample(olmo_quarters)
raw_delta = raw_branch(raw_quarters, raw_valid)
static_delta = highres_branch(highres_static, highres_valid)
aef_delta = aef_low_rank_adapter(aef_float, aef_valid)  # 仅消融启用
pre_vmf = base + tanh(alpha_raw) * raw_delta
               + tanh(alpha_hr) * static_delta[:, None]
               + use_aef_input * tanh(alpha_aef) * aef_delta[:, None]
embedding = vmf(project(pre_vmf))
```

`alpha_raw`、`alpha_hr` 和 `alpha_aef` 初始化为 0。AEF input adapter 固定先用
bias-free `64 -> 16` 压缩，再投影到内部 hidden space，禁止存在 `64 -> 64` 直达输出的复制通路。
高分或 AEF 输入缺失时必须绕过整个对应分支，不能只把输入乘 0 后继续经过有偏置卷积。

### 2.4 2 米空间折叠和离线结构特征

640 与 128 的比例严格为 5。使用原生 PyTorch 保留每个 5 x 5 子像素：

```python
def space_to_depth_5(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    if (h, w) != (640, 640):
        raise ValueError(f"expected 640x640, got {(h, w)}")
    return (
        x.reshape(b, c, 128, 5, 128, 5)
        .permute(0, 1, 3, 5, 2, 4)
        .contiguous()
        .reshape(b, c * 25, 128, 128)
    )
```

若原始高分 GSD 不是精确 2 米，先使用带抗混叠的地理重投影对齐到固定 2 米网格，再执行空间折叠；不得直接把任意尺寸 reshape 成 5 倍关系。

为了快速稳定训练，首版不在每个训练 step 读取并反传 `640 x 640` 原图。预处理阶段使用上述 5 倍空间折叠计算 32 通道结构缓存，内容固定为波段均值/标准差、Sobel x/y、梯度幅值和方向、局部纹理、有效比例、拼接缝强度及配准置信度。融合模型读取 `[32,128,128]` FP16 结构特征；200-patch 阶段保留“原图可学习 stride-5 encoder”作为精度上限消融，只有其核心指标显著优于结构缓存时才升级正式方案。

### 2.5 AEF 年度教师损失

AEF 先按官方规则反量化：

```python
valid = raw != -128
aef = torch.sign(raw.float()) * (raw.float() / 127.5).square()
```

对季度 embedding 做年度向量聚合并重新归一化：

```text
z_year = normalize(sum_q quarter_valid[q] * z_q)
L_aef_cos = mean_valid(1 - cosine(project(z_year), aef_year))
```

AEF loss 只作用于不含 AEF 输入的 raw-only forward，不允许模型复制教师。首轮 loss：

```text
L = L_masked_recon
  + lambda_aef * L_aef_cos
  + lambda_hr_boundary * L_hr_boundary
  + lambda_temporal * L_adjacent_temporal
  + lambda_var * L_quarter_variance_floor
  + lambda_cov * L_feature_covariance
```

时间平滑只在原始影像判定稳定且有效的像素上施加；变化像素使用变化敏感性损失，不对全国所有像素盲目平滑。

---

## 3. 文件结构

### 3.1 新建文件

```text
environments/olmoearth_npu/
  README.md                         # 独立环境、版本、权重和运行方式
  constraints.txt                  # torch/torch_npu/依赖锁定
src/xuannv_embedding/integrations/
  __init__.py
  olmoearth_v12.py                  # 无 einops 的输入/输出适配器
src/xuannv_embedding/data/
  quarterly_fusion_dataset.py       # PatchYear Zarr 数据集
  aef_vectors.py                    # AEF 反量化、重采样、归一化
src/xuannv_embedding/models/
  highres_static.py                 # 2 米空间折叠编码器
  quarterly_fusion.py               # 10 米动态 + 2 米静态门控融合
src/xuannv_embedding/training/
  quarterly_fusion_losses.py        # 年度教师和动态保真损失
  quarterly_fusion_trainer.py       # 分组 LR、恢复状态、评测选 best
scripts/data/
  audit_china_v1_2020_2021_inventory.py
  prepare_china_v1_aef_annual.py
  prepare_china_v1_highres_static.py
  precompute_olmoearth_quarter_features.py
scripts/train/
  train_china_v1_quarterly_fusion.py
scripts/eval/
  eval_china_v1_quarterly_fusion.py
scripts/production/
  export_china_v1_quarterly_embeddings.py
configs/national/
  china_v1_retrospective_quarterly_fusion_200_sanity_20260815.yaml
  china_v1_retrospective_quarterly_fusion_2000_20260815.yaml
  china_v1_retrospective_quarterly_fusion_10000_20260815.yaml
  china_v1_retrospective_quarterly_fusion_62000_20260815.yaml
tests/
  test_aef_vectors.py
  test_china_v1_inventory_audit.py
  test_olmoearth_v12_adapter.py
  test_quarterly_fusion_dataset.py
  test_highres_static.py
  test_quarterly_fusion_model.py
  test_quarterly_fusion_losses.py
  test_quarterly_fusion_checkpoint.py
  test_china_v1_quarterly_export.py
```

### 3.2 修改文件

```text
pyproject.toml                       # national 可选依赖：zarr>=2.18,<3
src/xuannv_embedding/config.py      # 新配置字段；补齐既有 distill decay 解析回归
src/xuannv_embedding/training/checkpoint.py
                                      # global_step/GradScaler/RNG/sampler/cursor
src/xuannv_embedding/training/optimizer.py
                                      # base/raw/highres/head 参数组
README.md                            # China V1 入口和产品语义索引
docs/data_inventory.md              # 真实数据审计结果，不复制大数据
CHANGELOG.md                         # 记录新管线和兼容性边界
```

现有 `AEFModel`、`MonthlyEmbeddingDataset` 和生产 P10C 配置不改语义；季度融合走独立入口，避免破坏海淀生产版。

---

## 4. 时间计划

### 4.1 快速版和正式候选分开排期

| 里程碑 | 交付物 | 工程日 | 日历目标 |
| --- | --- | ---: | ---: |
| M0 合同冻结 | 数据对象清单、许可结论、产品卡草稿 | 2--3 | 第 1 周前半 |
| M1 算子闸门 | 4-patch Tiny/Small CPU-NPU 对照报告 | 2--4 | 第 1 周 |
| M2 可跑原型 | 200-patch 缓存、模型、loss、下游评测 | 6--8 | 第 2 周 |
| M3 决策版 | 200-patch 完整消融和 Go/No-Go | 3--5 | 第 3 周前半 |
| M4 系统版 | 2,000 patch 可恢复流水线 | 3--4 | 第 3 周 |
| M5 生产演练 | 10,000 patch、24--48h 稳定性 | 4--6 | 第 4 周 |
| M6 62k 候选 | 全量特征、融合训练、核心评测 | 6--10 | 第 5--6 周 |
| M7 版本冻结 | 模型卡、数据卡、量化和区域导出 | 3--5 | 第 6--7 周 |

串行总工程量为约 29--45 工程日。模型、数据、评测三条线并行时：

- **最快可判断方向的 200-patch 版本：8--12 个工作日；**
- **62k 可评测 China V1 候选：20--30 个工作日；**
- **具备发布文档和区域生产能力的 V0：约 6--7 周。**

以上不包含采购、许可证谈判、远程数据补带和全国 50 TB 级产品写盘。若 S2 缺失 B01/B09 需要重新获取，数据准备可能增加 3--10 个日历日，取决于数据源吞吐。

### 4.2 关键路径

```text
真实数据清单
  -> S2 12-band/S1 合同通过
  -> Olmo 4-patch NPU gate
  -> 200-patch 模型选择
  -> 2k 恢复测试
  -> 10k 吞吐锁定
  -> 62k 缓存
  -> 融合训练和下游评测
  -> 量化与生产导出
```

AEF 和 2 米预处理可与 Olmo 环境适配并行。62k 全量不得绕过 10k 演练直接启动。

---

## 5. 算力、存储和人员预算

### 5.1 训练样本和特征缓存

| 内容 | 计算公式 | 62k 估算 |
| --- | --- | ---: |
| PatchYear 样本 | `62,000 x 2` | 124,000 |
| 最终季度样本 | `62,000 x 8` | 496,000 |
| Tiny/P4 FP16 缓存 | `62k x 8 x 192 x 32 x 32 x 2` | 195 GB |
| Small/P4 FP16 缓存 | `62k x 8 x 384 x 32 x 32 x 2` | 390 GB |
| AEF 两年 int8 | `62k x 2 x 64 x 128 x 128` | 130 GB |
| 2 米两年 4-band uint16 原始裁片 | `62k x 2 x 4 x 640 x 640 x 2` | 406 GB |
| 2 米两年 D32 FP16 结构缓存 | `62k x 2 x 32 x 128 x 128 x 2` | 130 GB |
| 最终 62k FP16 embedding | `62k x 8 x 64 x 128 x 128 x 2` | 1.04 TB |
| 最终 62k int8 embedding | 上一行减半 | 0.52 TB |

训练区建议：

- 最小可运行：2 TB 可用空间，只保留单一 Tiny 缓存、D32高分结构和 int8 最终结果；
- 推荐：4--6 TB 可用空间，可保留 Tiny/Small 对照、AEF、高分、失败 shard 和一个完整输出版本；
- 不允许按 patch 写小文件；Zarr shard 采用 512 MB--2 GB 粒度。

### 5.2 Olmo 提取时间公式

定义 `tau` 为单卡完成一个 patch 的“两次年度前向、八季度池化和写盘”实测秒数，六卡有效利用率先按 0.8 规划：

```text
hours = patch_count * tau / (6 * 3600 * 0.8)
```

| Patch 数 | tau=10s | tau=30s | tau=60s | tau=120s |
| ---: | ---: | ---: | ---: | ---: |
| 200 | 0.12h | 0.35h | 0.69h | 1.39h |
| 2,000 | 1.16h | 3.47h | 6.94h | 13.89h |
| 10,000 | 5.79h | 17.36h | 34.72h | 69.44h |
| 62,000 | 35.9h | 107.6h | 215.3h | 430.6h |

规划区间只用于预留：Tiny/P8 按 10--30s，Tiny/P4 按 30--60s，Small/P4 按 60--120s。若 P4 必须使用 `64 x 64 crop + 32 overlap`，每个 128 patch 会产生 9 个 crop，实际时间可能再扩大数倍；4-patch gate 必须记录是否能直接处理 128 输入。

### 5.3 融合训练时间公式

融合训练不再反传 Olmo，主要负担为 10 米动态支路、2 米支路和 64 维头。用实测 `step_seconds` 计算：

```text
steps_per_epoch = ceil(124000 / effective_global_batch)
train_hours = epochs * steps_per_epoch * step_seconds / 3600
```

第一轮预算采用 20--50 epoch + early stopping，不固定追求 800 epoch。若 6 卡有效全局 batch 为 12：

```text
steps_per_epoch = ceil(124000 / 12) = 10,334
20 epochs = 206,680 steps
50 epochs = 516,700 steps
```

在 200-patch 阶段实测单 step 之后，直接代入公式。没有该实测前，不承诺全量训练天数。

### 5.4 全国生产容量

全国按 960 万平方千米估算，约 960 亿个 10 米像素：

| 产品 | 净数据 | 推荐在线容量 | 双副本容量 |
| --- | ---: | ---: | ---: |
| 8 季度 x 64 维 int8 | 49.2 TB | 60 TB | 120 TB |
| 8 季度 x 64 维 FP16 | 98.3 TB | 110 TB | 220 TB |

全国两年四波段 uint16 2 米原图理论净数据约 38.4 TB，加入云影/拼接缝/来源图、索引和重投影中间量后建议按 50--80 TB 管理。由于生产模型读取 D32 静态结构而非原始 2 米图，建议把全国结构特征与最终 int8 产品分区共址流式生成，不长期同时保留所有中间版本。连同 60--120 TB 最终产品、低分源缓存、失败重跑和工作余量，全国生产最低需要约 150--200 TB 可用空间；若保留多版本和原始遥感归档，沿用 0.3--0.5 PB 级对象存储规划。

全国生产默认使用向量保持型 int8 量化、分 UTM/行政区 sharding 和中心裁边。训练特征缓存不推广到全国：生产时采用 Olmo -> 融合 -> 量化流式流水线，只保存最终 embedding。

全国陆地约对应 586 万个 `1280 m x 1280 m` patch。生产时的实际耗时公式为：

```text
days = 5_860_000 * measured_seconds_per_patch / (N_npu * 86400 * utilization)
```

在 6 卡、80% 利用率下，即使每 patch 只需 10 秒也约 141 天。因此 6 卡适合研发和 62k 训练，不适合快速完成全国八季度生产。全国导出建议：

- 最低生产配置：32 张同构 NPU + 60 TB 在线存储；
- 推荐生产配置：64 张同构 NPU + 120 TB 双副本存储；
- 先用 10,000 patch 实测 `seconds_per_patch`，再决定是否申请 64/128 卡；
- 128 卡以上优先拆成多个独立 UTM shard worker，不对冻结提取器做 DDP AllReduce。

### 5.5 人员配置

| 角色 | 建议投入 | 主要职责 |
| --- | ---: | --- |
| 模型/训练工程 | 1.0 人 | Olmo适配、融合模型、loss、NPU训练 |
| 遥感数据工程 | 1.0 人 | 12-band/S1、AEF、2米配准、Zarr和QA |
| 评测工程 | 0.5 人 | few-shot、边界、变化、seam、版本准入 |
| 平台/运维 | 0.25 人 | 独立环境、6/64卡调度、监控和恢复 |
| 许可/产品 | 0.1--0.2 人 | Olmo条款、数据授权、产品卡和发布范围 |

两名全职工程师加评测兼职是 6--7 周目标的最低稳定配置；单人串行应按 29--45 工程日并增加 20% 上下文切换余量。

---

## 6. 详细实施任务

### Task 1: 冻结数据合同和真实库存

**Files:**
- Create: `scripts/data/audit_china_v1_2020_2021_inventory.py`
- Create: `tests/test_china_v1_inventory_audit.py`
- Modify: `docs/data_inventory.md`

**Interfaces:**
- Consumes: ModelScope 对象列表、本地 `/data/xuannv_embedding/`、62,000 patch 注册表。
- Produces: `inventory.json`，逐源报告年份、月份、波段、shape、dtype、CRS、缺失率、hash 和许可字段。

- [ ] **Step 1: 写失败测试，拒绝 README 代替真实库存**

```python
def test_inventory_requires_all_expected_bands(tmp_path):
    report = audit_inventory(fake_s2_10_band_root(tmp_path), years=[2020, 2021])
    assert report["olmo_ready"] is False
    assert report["missing_s2_bands"] == ["B01", "B09"]
```

- [ ] **Step 2: 运行定点测试并确认失败**

Run: `PYTHONPATH=$PWD/src:$PWD/downstreams python -m pytest tests/test_china_v1_inventory_audit.py -v`

Expected: FAIL，因为 `audit_inventory` 尚不存在。

- [ ] **Step 3: 实现对象级扫描和合同判定**

脚本固定检查 S2 12-band 顺序、S1 VV/VH、2020-01--2021-12、AEF 两年、2 米逐年可用性和每个对象的实际字节数；任何字段缺失写入 `blocking_issues`，不自动猜测。

- [ ] **Step 4: 生成 200-patch 冻结清单和完整报告**

Run: `python scripts/data/audit_china_v1_2020_2021_inventory.py --registry /data/xuannv_embedding/raw/china_v1/registry/china_v1_quarterly_62000.jsonl --sample-size 200 --output /data/xuannv_embedding/china_v1_2020_2021/inventory/inventory.json`

Expected: 报告包含 `olmo_ready`、逐月缺失率、真实容量和 200 个固定 patch ID。

- [ ] **Step 5: 测试、更新数据清单、提交并推送**

Run: `python -m pytest tests/test_china_v1_inventory_audit.py -v`

Commit: `data: audit China V1 2020-2021 source inventory`

### Task 2: 建立独立 OlmoEarth NPU 提取环境

**Files:**
- Create: `environments/olmoearth_npu/README.md`
- Create: `environments/olmoearth_npu/constraints.txt`
- Create: `src/xuannv_embedding/integrations/__init__.py`
- Create: `src/xuannv_embedding/integrations/olmoearth_v12.py`
- Create: `tests/test_olmoearth_v12_adapter.py`

**Interfaces:**
- Consumes: 官方 v1.2 Tiny/Small checkpoint、S2/S1 年度张量和 mask。
- Produces: `encode_year(s2, s2_mask, s1, s1_mask, timestamps) -> tuple[quarter_features, valid_month_count]`。

- [ ] **Step 1: 写 shape、月份池化和缺月测试**

```python
def test_pool_month_tokens_to_quarters():
    tokens = torch.arange(12.0).view(1, 1, 1, 12, 1, 1)
    valid = torch.ones(1, 1, 1, 12, 1, dtype=torch.bool)
    quarters, counts = pool_quarters(tokens, valid)
    assert quarters.flatten().tolist() == [1.0, 4.0, 7.0, 10.0]
    assert counts.tolist() == [[3, 3, 3, 3]]
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `PYTHONPATH=$PWD/src python -m pytest tests/test_olmoearth_v12_adapter.py -v`

Expected: FAIL，因为 adapter 尚不存在。

- [ ] **Step 3: 用原生 PyTorch 实现 band/mask/timestamp 适配和季度池化**

输入固定为 `[B,H,W,12,C]`，时间戳固定 `[day, zero_based_month, year]`；输出读取原始 `tokens_and_masks`，禁止调用官方默认的跨时间平均接口。

- [ ] **Step 4: 固定环境版本和权重 hash**

主候选环境记录 Python 3.11、CANN 9.0、匹配的 torch/torch_npu wheel、Olmo commit 和 checkpoint SHA-256；不修改 Xuannv 核心环境。环境文档必须包含 `python -c` 版本检查命令和 CANN 加载命令。

- [ ] **Step 5: 完成 4-patch CPU/NPU gate**

Run: `python scripts/data/precompute_olmoearth_quarter_features.py --registry /data/xuannv_embedding/china_v1_2020_2021/registries/olmo_smoke4.jsonl --models tiny small --patch-sizes 8 4 --dtypes fp32 bf16 --device npu:0 --report /data/xuannv_embedding/china_v1_2020_2021/qa/olmo_gate4.json`

Expected: 报告含 shape、NaN、cosine、HBM、延迟、fallback 和重复运行差异。

- [ ] **Step 6: 测试、提交并推送**

Run: `python -m pytest tests/test_olmoearth_v12_adapter.py -v`

Commit: `feat: add isolated OlmoEarth v1.2 quarter extractor`

### Task 3: 正确处理 AEF 年度向量

**Files:**
- Create: `src/xuannv_embedding/data/aef_vectors.py`
- Create: `scripts/data/prepare_china_v1_aef_annual.py`
- Create: `tests/test_aef_vectors.py`

**Interfaces:**
- Consumes: AEF signed int8 COG、目标 patch CRS/affine。
- Produces: `[64,128,128]` int8 原值、valid mask、版本元数据；训练时按 batch 反量化。

- [ ] **Step 1: 写反量化、nodata 和向量聚合测试**

```python
def test_dequantize_and_normalize_aef():
    raw = torch.tensor([127, -127, -128], dtype=torch.int8)
    values, valid = dequantize_aef(raw)
    assert torch.allclose(values[:2], torch.tensor([0.99217224, -0.99217224]), atol=1e-6)
    assert valid.tolist() == [True, True, False]
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/test_aef_vectors.py -v`

- [ ] **Step 3: 实现 vector-aware 反量化、重采样和重新归一化**

连续向量先反量化到 float，重投影/聚合后按 64 维重新归一化；禁止对原始 int8 使用 bilinear 或 average pyramid。

- [ ] **Step 4: 处理 200 patch 并做数值 QA**

Expected: 有效像素向量范数 P50/P95 接近 1；`-128` 不参与统计；2020/2021 版本字段完整。

- [ ] **Step 5: 测试、提交并推送**

Commit: `data: prepare vector-safe AEF annual teachers`

### Task 4: 建立 2 米年度静态支路数据

**Files:**
- Create: `scripts/data/prepare_china_v1_highres_static.py`
- Create: `src/xuannv_embedding/models/highres_static.py`
- Create: `tests/test_highres_static.py`

**Interfaces:**
- Consumes: 年度 2 米影像、逐景/逐像素日期、质量和供应商元数据。
- Produces: 对齐后的 D32 `[32,128,128]` FP16 结构特征、valid mask、year_valid 和轻量 `StaticHighResAdapter`。

- [ ] **Step 1: 写空间折叠和缺失旁路测试**

```python
def test_space_to_depth_5_preserves_all_pixels():
    x = torch.arange(640 * 640).reshape(1, 1, 640, 640).float()
    y = space_to_depth_5(x)
    assert y.shape == (1, 25, 128, 128)
    assert torch.equal(torch.sort(y.flatten()).values, torch.sort(x.flatten()).values)
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/test_highres_static.py -v`

- [ ] **Step 3: 实现 2 米对齐、mask 下采样和离线结构特征**

mask 以有效面积比例折叠；影像不先 bilinear 到 128。结构提取器先保留 `C_hr x 25` 子像素，再计算固定的波段统计、梯度、纹理、seam和配准置信度，写出 `[D_static,128,128]`，首版 `D_static=32`。

- [ ] **Step 4: 加入配准和供应商风格 QA**

200-patch 报告计算与 S2 边缘偏移、拼接缝、云影、年份/供应商分类 AUC。空间偏移超过 5 米或稳定区供应商 AUC 超过 0.8 时不进入融合实验。

- [ ] **Step 5: 测试、提交并推送**

Commit: `feat: add native 2m static spatial branch`

### Task 5: 构建可恢复 Olmo 特征缓存

**Files:**
- Create: `scripts/data/precompute_olmoearth_quarter_features.py`
- Modify: `pyproject.toml`
- Test: `tests/test_olmoearth_v12_adapter.py`

**Interfaces:**
- Consumes: Task 1 registry、Task 2 extractor。
- Produces: 512 MB--2 GB Zarr shard 和全局 feature manifest。

- [ ] **Step 1: 写 fingerprint 不一致和 partial 恢复测试**

```python
def test_feature_cache_rejects_changed_weights(tmp_path):
    cache = create_cache(tmp_path, weights_sha256="aaa")
    with pytest.raises(ValueError, match="weights_sha256"):
        resume_cache(cache, weights_sha256="bbb")
```

- [ ] **Step 2: 增加 `national` 可选依赖并实现 shard writer**

`pyproject.toml` 增加 `national = ["zarr>=2.18,<3"]`；writer 复用现有 `done` bitmap、input fingerprint、`.partial` 和原子 rename 模式。

- [ ] **Step 3: 先运行 200 patch 三候选缓存**

Tiny/P8、Tiny/P4、Small/P4 分别写独立版本目录，不允许覆盖。记录每 patch 两年度总秒数和 P50/P95。

- [ ] **Step 4: 完成 2,000 patch 中断恢复演练**

运行到 30% 后终止进程，再用相同命令恢复；已完成 chunk 不重复计算，最终 manifest hash 与无中断小样本一致。

- [ ] **Step 5: 测试、提交并推送**

Commit: `feat: cache recoverable Olmo quarterly features`

### Task 6: 新建 PatchYear Dataset

**Files:**
- Create: `src/xuannv_embedding/data/quarterly_fusion_dataset.py`
- Create: `tests/test_quarterly_fusion_dataset.py`

**Interfaces:**
- Consumes: Olmo、AEF、2 米、raw-quarter 四类分片 manifest。
- Produces: `PatchYearSample` 和确定性的空间 split。

- [ ] **Step 1: 写跨源 patch/year join、缺失和 import provenance 测试**

```python
def test_dataset_joins_only_same_patch_and_year(fake_stores):
    sample = QuarterlyFusionDataset(fake_stores)[0]
    assert sample["year"] in (2020, 2021)
    assert sample["olmo_quarters"].shape[0] == 4
    assert sample["aef_int8"].shape == (64, 128, 128)
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `PYTHONPATH=$PWD/src:$PWD/downstreams python -m pytest tests/test_quarterly_fusion_dataset.py -v`

- [ ] **Step 3: 实现 manifest join、lazy Zarr read 和 mask 传播**

所有 join key 固定为 `(patch_id, year)`；缺少高分允许 `year_valid=0`，缺少 Olmo/AEF 不允许静默补零。Dataset 启动时记录实际 import 文件路径和 Git commit。

- [ ] **Step 4: 加入分片级缓存和 worker 初始化**

每个 DataLoader worker 独立打开 Zarr handle，不跨进程 pickle 打开的 store；避免每个样本重复打开文件。

- [ ] **Step 5: 测试、提交并推送**

Commit: `feat: add patch-year quarterly fusion dataset`

### Task 7: 实现门控季度融合模型

**Files:**
- Create: `src/xuannv_embedding/models/quarterly_fusion.py`
- Create: `tests/test_quarterly_fusion_model.py`
- Modify: `src/xuannv_embedding/models/__init__.py`

**Interfaces:**
- Consumes: `PatchYearSample` 的模型张量。
- Produces: `[B,4,64,128,128]` 单位向量和门控诊断。

- [ ] **Step 1: 写 shape、单位范数、zero-gate identity 和 mask 测试**

```python
def test_zero_initialized_side_gates_preserve_base():
    model = tiny_fusion_model()
    out_with_sides = model(**full_inputs())["embedding"]
    out_without_sides = model(**masked_side_inputs())["embedding"]
    assert torch.allclose(out_with_sides, out_without_sides, atol=1e-6)
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest tests/test_quarterly_fusion_model.py -v`

- [ ] **Step 3: 实现 coarse upsample、raw 10 米支路、static 2 米支路和低秩 AEF 消融 adapter**

Olmo 特征使用 learned 4x upsampling；raw branch 每季度独立但共享权重；static adapter 每年读取一次 D32 结构缓存后广播四季度。AEF adapter 只有 `use_aef_input=True` 时执行，先压到16维，默认生产配置关闭。所有 reshape 使用原生 PyTorch。

- [ ] **Step 4: 实现零初始化残差门和 vMF 输出**

旁路关闭时在投影前后都保持可验证的基线；门控值写入训练日志，避免训练后发现高分分支从未启用或完全支配输出。

- [ ] **Step 5: CPU/NPU 单元测试、提交并推送**

Commit: `feat: add retrospective quarterly gated fusion model`

### Task 8: 实现年度教师和季度保真损失

**Files:**
- Create: `src/xuannv_embedding/training/quarterly_fusion_losses.py`
- Create: `tests/test_quarterly_fusion_losses.py`
- Modify: `src/xuannv_embedding/config.py`

**Interfaces:**
- Consumes: 模型输出、AEF、raw change evidence、高分边界和 valid masks。
- Produces: 总 loss、逐项 loss 和有效样本计数。

- [ ] **Step 1: 写年度聚合、无效像素、季度不塌缩测试**

```python
def test_aef_teacher_supervises_year_mean_not_each_quarter():
    loss = build_loss(aef_weight=1.0)
    terms = loss(quarter_embeddings=different_quarters_same_year_mean(), **targets())
    assert terms["aef_cos"] < 1e-5
    assert terms["quarter_variance"] == 0
```

- [ ] **Step 2: 写配置回归测试，覆盖既有 distill decay 漏解析**

加载 V5 配置后断言 `distill_final_weight == 0.0`、`distill_decay_start_epoch == 200`、`distill_decay_end_epoch == 400`；该修复独立记录，避免继续错误解释历史实验。

- [ ] **Step 3: 实现 masked FP32 loss 计算**

余弦、关系矩阵、方差和协方差损失在 autocast 外转 FP32；每项 loss 返回 `valid_count`，有效像素为 0 时返回可微的 0，不产生 NaN。

- [ ] **Step 4: 实现 side dropout 和负面对照开关**

配置支持 AEF teacher、AEF input 消融、高分 broadcast 错误对照和高分 mask dropout；默认生产配置关闭 AEF input。

- [ ] **Step 5: 测试、提交并推送**

Commit: `feat: add annual teacher and quarter-preserving losses`

### Task 9: 建立独立训练入口和完整恢复

**Files:**
- Create: `src/xuannv_embedding/training/quarterly_fusion_trainer.py`
- Create: `scripts/train/train_china_v1_quarterly_fusion.py`
- Create: `configs/national/china_v1_retrospective_quarterly_fusion_200_sanity_20260815.yaml`
- Create: `tests/test_quarterly_fusion_checkpoint.py`
- Modify: `src/xuannv_embedding/training/checkpoint.py`
- Modify: `src/xuannv_embedding/training/optimizer.py`

**Interfaces:**
- Consumes: 自包含 YAML、PatchYear Dataset、融合模型和 loss。
- Produces: best/last/stage checkpoints、训练日志、固定 probe 指标和完整恢复状态。

- [ ] **Step 1: 写 checkpoint 精确恢复测试**

checkpoint 必须恢复 `epoch`、`global_step`、optimizer、scheduler、GradScaler、Python/NumPy/Torch/NPU RNG、sampler epoch、shard cursor 和 best metric。恢复后下一个 batch ID 与不中断运行一致。

- [ ] **Step 2: 实现 base/raw/highres/head 参数组**

Olmo 不进入 optimizer；upsample/base projection、raw branch、highres branch、64维 head 使用独立 LR，配置中明确记录。梯度累积时非同步 step 使用 DDP `no_sync()`。

- [ ] **Step 3: 实现以固定下游分数选 best**

best score 使用 building/road/water 10-shot 的预注册加权指标；总 val loss 只做监控，不作为唯一 best 标准。

- [ ] **Step 4: 写自包含 V0 配置并做 Config round-trip 测试**

配置包含数据路径、模型、损失、优化、训练、评测、缓存版本和所有门槛，无 `_base_`。

- [ ] **Step 5: 6卡 smoke、测试、提交并推送**

Run: `source /usr/local/Ascend/cann-9.0.0/set_env.sh && torchrun --nproc_per_node=6 scripts/train/train_china_v1_quarterly_fusion.py --config configs/national/china_v1_retrospective_quarterly_fusion_200_sanity_20260815.yaml --max-steps 20`

Commit: `feat: add recoverable China V1 quarterly fusion training`

### Task 10: 完成 200-patch 模型选择

**Files:**
- Create: `scripts/eval/eval_china_v1_quarterly_fusion.py`
- Create: `configs/national/ablation/china_v1_qfusion_200_b0_raw10_20260815.yaml`
- Create: `configs/national/ablation/china_v1_qfusion_200_b1_olmo_20260815.yaml`
- Create: `configs/national/ablation/china_v1_qfusion_200_b2_olmo_raw10_20260815.yaml`
- Create: `configs/national/ablation/china_v1_qfusion_200_b3_highres_20260815.yaml`
- Create: `configs/national/ablation/china_v1_qfusion_200_b4_aef_teacher_20260815.yaml`
- Create: `configs/national/ablation/china_v1_qfusion_200_b5_aef_input_20260815.yaml`
- Create: `configs/national/ablation/china_v1_qfusion_200_n1_highres_broadcast_20260815.yaml`
- Create: `configs/national/ablation/china_v1_qfusion_200_n2_aef_copy_20260815.yaml`
- Modify: `docs/downstream_eval_protocol.md`
- Test: `tests/test_eval.py`

**Interfaces:**
- Consumes: 200-patch 固定空间 folds 和候选 checkpoint。
- Produces: 同协议消融矩阵、速度/HBM/缓存报告和 Go/No-Go 决策。

- [ ] **Step 1: 冻结八组消融**

```text
B0 raw10-only
B1 Olmo-only
B2 Olmo + raw10
B3 Olmo + raw10 + highres-static
B4 Olmo + raw10 + highres-static + AEF-teacher（推荐）
B5 B4 + AEF-input（复制风险研究）
N1 B4 但高分无约束广播（负面对照）
N2 B5 且 teacher 在 AEF-visible view（复制负面对照）
```

八组各自使用 Files 中对应的自包含 YAML，配置之间不使用 `_base_`，也不依赖运行时命令行覆盖来定义模型结构。

- [ ] **Step 2: 固定评测和阈值协议**

building/road/water 使用相同 fold、相同 5/10/50-shot、相同头、相同阈值选择；报告 F1/AP/AUC、boundary F1、窄边界 IoU、seam、季度稳定性和变化敏感性。

- [ ] **Step 3: 运行 Tiny/P8、Tiny/P4、Small/P4**

六卡作为六个独立 worker，不使用 DDP；每个候选记录总卡时和最慢 shard 时间。

- [ ] **Step 4: 按第一版成功标准做 Go/No-Go**

没有任何候选同时通过效果、季度差异和算子门槛时，停止 62k；保留 B0/B2 结果定位是基座、AEF还是高分支路的问题。

- [ ] **Step 5: 提交评测协议和结果摘要并推送**

Commit: `eval: select 200-patch China V1 fusion candidate`

### Task 11: 2,000 和 10,000 patch 生产演练

**Files:**
- Modify: `scripts/data/precompute_olmoearth_quarter_features.py`
- Modify: `scripts/train/train_china_v1_quarterly_fusion.py`
- Create: `configs/national/china_v1_retrospective_quarterly_fusion_2000_20260815.yaml`
- Create: `configs/national/china_v1_retrospective_quarterly_fusion_10000_20260815.yaml`
- Create: `docs/production/china_v1_quarterly_system_gate.md`

**Interfaces:**
- Consumes: Task 10 唯一胜出候选。
- Produces: P50/P95 吞吐、故障恢复、数据质量和 62k 锁定排期。

- [ ] **Step 1: 2,000 patch 执行进程终止、坏 patch 和磁盘满模拟**

每类故障恢复后检查 done bitmap、manifest hash 和重复计算率；坏 patch 进入 quarantine，不阻断其他 shard。

- [ ] **Step 2: 2,000 patch 完成跨区域人工 QA**

城市、农业、森林、荒漠、高原、海岸各抽固定样本，检查云、配准、高分拼接缝和季度顺序。

- [ ] **Step 3: 10,000 patch 使用与 62k 完全相同目录和调度运行 24--48 小时**

记录 NPU 利用率、HBM、CPU fallback、Zarr read/write、失败率、重试率、单 patch P50/P95 和六卡负载偏差。

- [ ] **Step 4: 用 10k 实测锁定 62k 时间和容量**

```text
ETA_62k = elapsed_10k * 6.2 * 1.20
capacity_62k = written_10k * 6.2 * 1.15
```

20% 时间余量覆盖长尾和重试，15% 容量余量覆盖索引和元数据。

- [ ] **Step 5: 提交系统 gate 报告并推送**

Commit: `docs: admit China V1 10k production rehearsal`

### Task 12: 运行 62k 全量缓存和融合训练

**Files:**
- Create: `configs/national/china_v1_retrospective_quarterly_fusion_62000_20260815.yaml`
- Create: `docs/production/china_v1_quarterly_training_log.md`

**Interfaces:**
- Consumes: 10k gate 冻结的模型、缓存和训练配置。
- Produces: 62k Tiny/Small 单一版本缓存、best/last checkpoint 和完整训练记录。

- [ ] **Step 1: 冻结配置和全部输入 hash**

训练启动后不原地修改 YAML；任何超参数变更复制新自包含配置并使用新实验名。

- [ ] **Step 2: 六卡独立完成 Olmo feature shards**

每卡约 10,334 patch；完成一个 shard 立即校验并原子入库。随机复算 1% 特征，BF16 容差内一致。

- [ ] **Step 3: 运行 6卡融合训练并按 probe early stop**

每 4--8 小时和每个阶段保存 checkpoint；持续报告门控值、各 loss 有效计数、季度方差和固定 probe。

- [ ] **Step 4: 对 best/last 做完整同协议评测**

若 best 仅降低 val loss 但核心下游退化，拒绝该 checkpoint；只登记通过门槛的模型。

- [ ] **Step 5: 更新 checkpoint registry、提交日志并推送**

Commit: `docs: register China V1 retrospective quarterly candidate`

### Task 13: 实现向量保持型量化和区域导出

**Files:**
- Create: `scripts/production/export_china_v1_quarterly_embeddings.py`
- Create: `tests/test_china_v1_quarterly_export.py`
- Create: `docs/production/china_v1_quarterly_model_card.md`
- Create: `docs/production/china_v1_quarterly_data_card.md`

**Interfaces:**
- Consumes: admitted checkpoint、区域 patch registry。
- Produces: 八季度 int8 embedding、valid mask、索引、hash、模型卡和数据卡。

- [ ] **Step 1: 写量化 round-trip、nodata 和拼接测试**

量化后解码向量与 FP16 向量余弦 P50/P95 必须达标；nodata 使用保留码并有独立 mask；相邻 patch 中心裁边后不重复、不缺口。

- [ ] **Step 2: 实现流式 backbone -> fusion -> quantize -> shard**

全国生产不写 Olmo 中间缓存；单 patch 完成后立即量化写入目标 UTM shard，记录 checkpoint/input/output hash。

- [ ] **Step 3: 导出城市、农业、山区、海岸四类区域试产品**

每类区域同时保留小范围 FP16 参考，验证 int8 下游差异、seam 和空间索引。

- [ ] **Step 4: 写清年度回溯和许可证限制**

模型卡必须声明四季度可使用同年全年观测、2米为年度静态细节、AEF为年度教师，以及 Olmo 衍生发布约束。

- [ ] **Step 5: 测试、提交并推送**

Commit: `feat: export quantized China V1 quarterly embeddings`

### Task 14: 全国生产准入和分区导出

**Files:**
- Create: `docs/production/china_v1_national_export_runbook.md`
- Create: `docs/production/china_v1_national_export_manifest.json`

**Interfaces:**
- Consumes: 区域导出 gate、全国 registry、32/64 NPU 和 60/120 TB 存储。
- Produces: 可恢复的 UTM/省级导出任务、全国索引和验收清单。

- [ ] **Step 1: 用 10,000 全国生产 patch 实测含写盘的端到端吞吐**

不得使用仅模型前向速度外推；指标包含数据读取、Olmo、融合、量化、写盘、校验和失败重试。

- [ ] **Step 2: 按 UTM zone 和行政区生成不重叠 job manifest**

每个 job 明确输入 patch ID、输出范围、NPU worker、重试次数、expected bytes 和 predecessor hash。

- [ ] **Step 3: 先导出 1% 全国覆盖并验收**

检查容量误差、空间缺口、重复像素、季度顺序、nodata、下游抽测和随机反量化。

- [ ] **Step 4: 通过许可和资源准入后启动全量**

准入条件为 60 TB 单副本或 120 TB 双副本空间已就绪、至少 32 张同构 NPU、Olmo发布条款已批准、1% 试产全部通过。

- [ ] **Step 5: 每完成一个区域立即更新 manifest、提交小型索引并推送**

Commit: `release: register China V1 national export shard`

---

## 7. 决策门和停损规则

| Gate | 通过条件 | 不通过动作 |
| --- | --- | --- |
| 数据合同 | S2 12-band、S1、年份、mask、hash 可验证 | 补带或缩小可用样本，不补零冒充 |
| 许可 | 原型和发布用途均有明确结论 | Olmo只做内部基准，替换生产 backbone |
| 4-patch NPU | 无 fallback/NaN/OOM，数值一致 | 停止200 patch，修环境/算子 |
| 200-patch 效果 | 核心任务、季度差异、高分收益同时通过 | 回退 B0/B2 定位，不扩大数据 |
| 2k 恢复 | 可断点续跑、坏patch隔离、hash一致 | 不进入10k |
| 10k 系统 | 24--48h稳定、ETA/容量锁定 | 不进入62k |
| 62k 模型 | 同协议优于基线且无任务明显退化 | 不注册生产checkpoint |
| 区域导出 | int8、seam、索引、许可全部通过 | 不申请全国资源 |
| 全国1% | 容量和吞吐误差在预算内 | 调整shard/并发/存储后重跑1% |

---

## 8. 不纳入第一版的内容

- 不重新预训练完整 OlmoEarth 或完整 Xuannv STP 主干。
- 不把 Landsat 6/7-band 直接塞入 Olmo 11-band 接口。
- 不使用 `patch_size=1` 生成所谓原生 10 米 Olmo token。
- 不对 AEF 每个季度分别做相同年度向量蒸馏。
- 不把同一张 2 米影像当作八个季度的变化真值。
- 不把 40 米 Olmo token 双线性放大后直接宣称为 10 米语义。
- 不在 62k 之前运行全国 586 万 patch 的特征缓存。
- 不在单一 `.pt` 文件或每 patch 小文件中保存全国特征。
- 不以总 val loss 作为唯一 best checkpoint 标准。
- 不在未完成许可证审查时公开发布无用途限制的 Olmo 衍生 embedding。

---

## 9. 执行建议

推荐使用 subagent-driven-development，按 Task 1--14 顺序执行；Task 2、3、4 在 Task 1 的 200-patch 清单冻结后可并行。每个任务由独立实现 agent 完成，主 agent 在合并前进行规格审查和代码质量审查，并严格执行“测试通过 -> commit -> push -> 下一任务”。

第一笔资源只批准到 M2：2 名工程师、6 张 Ascend、2 TB 临时空间、8--12 个工作日。只有 200-patch 证明 Olmo、10 米支路、2 米支路和 AEF 教师确有组合收益后，再批准 4--6 TB 和 62k 全量。全国生产资源最后批准，避免在模型尚未通过区域验证时提前建设 60--120 TB 存储。
