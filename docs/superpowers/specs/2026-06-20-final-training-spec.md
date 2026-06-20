# AEF 月度地理嵌入训练最终方案

> 日期：2026-06-20
> 依据：chat4、chat5、AEF 论文（Brown et al., arXiv:2507.22291）、子代理代码/复现/最佳实践调研。
> 目标：在 6× Ascend NPU 上，于 24 h 内完成一版可用训练，输出月度 embedding field。

---

## 1. 用户拍板的关键决策

| 问题 | 决策 |
|---|---|
| WorldCover 类别 11 | 方案 A：当前标签已只有 0–8，无需 remap；`channels=9`，`0` 为 ignore/nodata |
| 高分辨率 SAR | 阶段一（低分辨率预训练）不加；阶段二（持续预训练）再加入 |
| vMF 逼真度 | 方案 B：可学习 `log_kappa`，不再使用固定 `kappa=100` |
| 监控 | 使用 WANDB 实时记录 loss / 学习率 / 各源重建指标 |
| Checkpoint 策略 | 每 10 epoch 保存；仅保留 1 个最佳权重 + 最近 3 个权重 |
| 训练硬件 | 6× Ascend 910B，DDP + AMP + gradient checkpointing |

---

## 2. 数据源与配置命名

### 2.1 低分辨率源（10 m，128×128）

| source | 波段 | 说明 |
|---|---|---|
| `s2` | 12 | Sentinel-2 L2A，经 SCL 掩膜，DN/10000 反射率 |
| `s1` | 2 | Sentinel-1 GRD，VV/VH，dB 归一化到 [0,1] |
| `landsat` | 7 | Landsat Collection 2 SR，QA_PIXEL 掩膜 |
| `worldcover` | 1 | ESRI 2023 标签，target-only，0=ignore，1–8 有效 |

### 2.2 高分辨率源（原生分辨率，按源独立编码）

| source | 区域 | 波段 | 原生尺寸 | GSD |
|---|---|---|---|---|
| `highres_optical_haidian` | 海淀 | 4（BGR+NIR） | 427×427 | ~3 m |
| `highres_optical_harbin` | 哈尔滨 | 3（RGB） | 2560×2560 | 0.5 m |
| `highres_sar_haidian` | 海淀 | 1（VV） | 427×427 | ~3 m |
| `highres_sar_harbin` | 哈尔滨 | — | 无数据 | — |

### 2.3 训练阶段

- **阶段一**：`s2` + `s1` + `landsat` + `worldcover`（target）。
- **阶段二**：在阶段一 checkpoint 基础上，加入 `highres_optical_haidian`、`highres_optical_harbin`、`highres_sar_haidian`。

---

## 3. 模型架构

### 3.1 整体流程

```text
低分辨率时序 (B, T_obs, C, H, W)
  → per-sensor stem bank  (B, T_obs, stem_dim, H, W)
  → STPEncoder             (B, T_obs, H/2, W/2, precision_dim)
  → MonthlyEmbeddingModule (B, T_month, H/2, W/2, embed_dim)
  → EmbeddingUpsampleHead  (B, T_month, H, W, embed_dim)

高分辨率源（原生分辨率）
  → NativeResolutionHighResEncoder（每源独立）
  → (B, embed_dim, H, W)

逐月 base embedding + 高分辨率特征
  → AvailabilityAwareFusion
  → VMFBottleneck（单位球面，可学习 log_kappa）
  → 逐月 Decoder heads
```

### 3.2 关键模块

#### `VMFBottleneck`

- `proj`: 1×1 conv 投影到 `embed_dim`。
- `log_kappa`: 可学习参数，初始 `log(10.0)`。
- 训练：先 L2 归一化，加切空间高斯噪声 `noise ~ N(0, 1/exp(log_kappa))`，再归一化。
- 推理：仅 L2 归一化。

#### `MonthlyEmbeddingModule`

- 输入：`feats (B, T_obs, H, W, C)`、`timestamps (B, T_obs)`、`mask (B, T_obs)`。
- 将时间戳分配到月度 bin（`YYYYMM`）。
- 每月 bin 内加权平均；缺失月份用可学习 `missing_token`。
- 输出：`monthly_feats (B, T_month, H, W, embed_dim)`、`monthly_mask (B, T_month)`。

#### `NativeResolutionHighResEncoder`

- 每源独立，输入通道数按 sensor_channels 配置。
- 结构：3 层 stride=2 卷积 + GroupNorm + GELU，最后 adaptive avg pool 到 `(H, W)`。
- 输出通道数 = `embed_dim`。

#### Decoder

- 阶段一：先支持 `(B, T, D, H, W)` → `(B, T, C_out, H, W)`。
- 阶段二：再引入 sensor metadata + 相对时间码条件（implicit decoder）。

### 3.3 输出形状

```python
@dataclass
class AEFOutput:
    embedding_map: torch.Tensor       # (B, T_month, D, H, W)
    embedding: torch.Tensor           # (B, T_month, D)
    reconstructions: dict[str, torch.Tensor]  # (B, T_month, C_out, H, W)
```

---

## 4. 数据流

### 4.1 月度窗口采样

- `T_month`：阶段一取 17（2025-01 至 2026-05）。
- 每个 batch：
  1. 随机/顺序选择 patch。
  2. 对每源，将观测按 `YYYYMM` 分配到 17 个月度 bin。
  3. 缺失 bin 用 0 帧 + mask=0 表示；模型内部 `missing_token` 处理。
- collate 沿 batch 维度堆叠，时间维度固定为 17。

### 4.2 `prepare_batch`

- 输入：collate 后的 `source_frames`、`source_masks`、`timestamps`。
- 输出：
  - `source_frames`: 低分辨率源 `(B, T_month, C, H, W)`
  - `highres_frames`: 高分辨率源 `(B, C, H, W)`（阶段二启用）
  - `timestamps`: 全局月度索引 `(B, T_month)`
  - `targets`: 各 head 的月度真值
  - `target_masks`: 月度有效掩码

### 4.3 归一化

- 每源使用预计算 `mean/std` 做 z-score。
- WorldCover 不标准化。

---

## 5. 损失函数

```text
total = recon_loss + uniformity_loss
```

### 5.1 逐月 masked reconstruction

- 连续源：`MSE` 或 `L1`，按 `target_mask` 加权。
- 分类源：`CrossEntropy`，ignore_index=0。

### 5.2 Batch uniformity

- 对 `embedding (B, T_month, D)` 做 L2 归一化。
- 展平为 `(B*T_month, D)`。
- 最大化 pairwise squared distance（即最小化余弦相似度）。

### 5.3 权重

| head | loss_type | weight |
|---|---|---|
| s2_recon | continuous | 1.0 |
| s1_recon | continuous | 1.0 |
| landsat_recon | continuous | 1.0 |
| worldcover | categorical | 0.5 |
| highres_optical_haidian_recon | continuous | 1.0（阶段二） |
| highres_optical_harbin_recon | continuous | 1.0（阶段二） |
| highres_sar_haidian_recon | continuous | 1.0（阶段二） |

---

## 6. 训练超参数

| 参数 | 值 |
|---|---|
| epochs | 100（阶段一） |
| lr | 1.0e-4 |
| weight_decay | 0.05 |
| optimizer | AdamW |
| scheduler | cosine with warmup |
| warmup_epochs | 5 |
| batch_size | 4 per GPU |
| gradient_accumulation_steps | 1（OOM 时调 2/4） |
| save_every | 10 epochs |
| eval_every | 10 epochs |
| num_workers | 8 |
| pin_memory | True |
| AMP | True |
| gradient_checkpointing | True |
| seed | 42 |

---

## 7. WANDB 监控

- API key：通过环境变量 `WANDB_API_KEY` 注入，**不进入 git**。
- 记录内容：
  - `train/loss_total`、`train/loss_recon`、`train/loss_uniformity`
  - 各源 `train/recon_{head}`
  - `train/lr`
  - `val/loss_total`（每 10 epoch）
  - GPU/NPU 利用率、显存占用
- 项目名：`xuannv-embedding-stage1`

---

## 8. Checkpoint 策略

- 每 10 epoch 保存：`epoch_{N}.pt`。
- 仅保留：
  - `best.pt`：验证 loss 最低的权重。
  - 最近 3 个 epoch 权重：`epoch_{N}.pt`、`epoch_{N-10}.pt`、`epoch_{N-20}.pt`。
- 旧的 epoch 文件自动删除。

---

## 9. 评估标准

- `pytest tests/` 全绿。
- 100-step smoke test：不 OOM，total loss 下降。
- 阶段一完成 100 epoch：
  - total loss 稳定收敛；
  - 各源重建 loss 不再发散；
  - best checkpoint 已保存。

---

## 10. 文件改动清单

| 文件 | 改动 |
|---|---|
| `AGENTS.md` | 允许 VMF bottleneck 使用 L2 归一化 |
| `configs/base.yaml` | 改为阶段一/二模板，拆分高分辨率源 |
| `configs/haidian_128.yaml` | 海淀专用配置 |
| `configs/harbin_128.yaml` | 哈尔滨专用配置 |
| `src/xuannv_embedding/models/bottleneck.py` | 可学习 `log_kappa` |
| `src/xuannv_embedding/models/blocks.py` | 新增 `MonthlyEmbeddingModule` |
| `src/xuannv_embedding/models/sensor_encoders.py` | 新增 `NativeResolutionHighResEncoder` |
| `src/xuannv_embedding/models/highres_fusion.py` | 移除插值，要求同尺寸输入 |
| `src/xuannv_embedding/models/model.py` | 逐月输出、独立高分辨率源、新 bottleneck |
| `src/xuannv_embedding/models/decoders.py` | 支持时间维度 |
| `src/xuannv_embedding/data/dataset.py` | 月度窗口采样 |
| `src/xuannv_embedding/data/collate.py` | 月度时间对齐 |
| `src/xuannv_embedding/training/batch_preparation.py` | 月度 target / highres 处理 |
| `src/xuannv_embedding/training/losses.py` | 逐月 masked loss |
| `src/xuannv_embedding/training/trainer.py` | WANDB、best/latest-3 checkpoint |
| `src/xuannv_embedding/training/optimizer.py` | cosine scheduler |
| `scripts/train/train.py` | 6 卡 DDP 启动脚本 |

---

## 11. 已知未实现（阶段二或未来）

- Teacher-student video pretraining。
- Text alignment（geotagged Wikipedia / GBIF）。
- 隐式 decoder 的 sensor metadata + 相对时间码条件。
- Cross-scale contrastive loss（MSAD/JEPA）。
