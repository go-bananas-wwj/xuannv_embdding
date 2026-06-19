# AEF 数据与模型更新设计

> 范围：在哈尔滨 + 海淀全量数据上完成最小规模验证训练所需的数据预处理与模型架构改造。

## 1. 背景与问题

当前代码已实现一个基于 Space-Time-Precision（STP）编码器的 AEF（Angular Embedding Field）模型，但存在以下与用户最新决策不一致的假设：

1. **vMF bottleneck 强制 L2 归一化**，将嵌入约束到单位球面；用户明确要求移除 L2 归一化。
2. **TemporalSummarizer 把整段时序池化为单个 embedding**，导致月度变化信息丢失；用户要求输出逐月 embedding。
3. **高分辨率数据被假设为与低分辨率同尺寸**，并在 stem 后通过 `AvailabilityAwareFusion` 直接相加；用户要求以原生分辨率作为独立模态处理。
4. **S1 / Landsat / 高分辨率 patch 尚未生成**，当前 manifest 只包含 S2。
5. **训练目标仍为时序平均后的单帧重建**，需要改为逐月重建。
6. 需要适配 6× Ascend 910B NPU，优化显存与速度。

## 2. 目标

- 完成哈尔滨、海淀两地的 S1、Landsat、高分辨率光学/ SAR patch 生成。
- 更新 manifest，使其能描述每个 patch 的逐源、逐月可用性。
- 改造模型：
  - 移除 L2 归一化 bottleneck。
  - 用逐月 embedding 替代时序池化。
  - 高分辨率数据按原生分辨率作为独立模态参与融合。
- 在哈尔滨 + 海淀全部数据上跑通最小规模训练（单模型，多 NPU）。
- 保留可配置的 no-data 值（默认 0，支持用户后续改为 -1）。

## 3. 设计选项与决策

### 3.1 时序摘要方式

| 选项 | 说明 | 优点 | 缺点 | 决策 |
|------|------|------|------|------|
| A. 全局池化为单 embedding | 当前实现 | 简单、显存小 | 丢失月度变化，无法逐月重建 | 否 |
| B. 逐月 embedding 网格 | 将观测聚合到固定月度网格，缺失月份用可学习 token 填充 | 保留月度动态，支持逐月重建 | 显存随月份数线性增长 | **是** |
| C. Transformer 逐月解码 | 用跨注意力从整段时序生成每个月 | 表达力强 | 计算量大，超参多 | 否（先验证 B） |

### 3.2 高分辨率数据处理方式

| 选项 | 说明 | 优点 | 缺点 | 决策 |
|------|------|------|------|------|
| A. 下采样到 10m 作为额外通道 | 与低分辨率同尺寸 | 实现简单 | 丢失原生分辨率信息 | 否 |
| B. 原生分辨率独立模态 + availability-aware 融合 | 高分辨率独立编码后投影到低分辨率网格再融合 | 保留高分辨率语义，覆盖不全时自动退化为低分辨率 | 需要处理不同尺寸输入 | **是** |
| C. 仅作为 super-resolution 监督目标 | 高分辨率用于监督低分辨率embedding上采样 | 直接利用高分辨率信息 | 高分辨率覆盖稀疏，无法监督无高分辨率地区 | 否 |

### 3.3 Bottleneck 归一化

| 选项 | 说明 | 优点 | 缺点 | 决策 |
|------|------|------|------|------|
| A. 保留 vMF + L2 归一化 | 当前实现 | 与 AEF 原始论文一致 | 用户明确反对 | 否 |
| B. 无约束 learned bottleneck | 1×1 conv + 可选 GroupNorm，不强制单位范数 | 灵活，可直接用 MSE/cosine 损失 | 失去单位球面约束 | **是** |
| C. 可学习尺度归一化 | LayerNorm + 可学习 per-channel scale | 兼顾方向与幅度 | 增加超参 | 备选 |

**决策**：采用 B。后续若发现需要角度约束，可在损失中加入 cosine similarity 项，而不在 bottleneck 中强制 L2 归一化。

## 4. 数据层改造

### 4.1 生成 S1 / Landsat 128×128 patch

- 复用现有 `scripts/data/preprocess.py` 的切 patch 流程。
- S1 使用 10m 参考分辨率（原生已接近 10m），Landsat 从 30m 重采样到 10m。
- 与 S2 使用同一网格（1280m × 1280m），保证地理对齐。
- 输出目录：
  - `processed/harbin/patches/s1/`
  - `processed/harbin/patches/landsat/`
  - `processed/haidian/patches/s1/`
  - `processed/haidian/patches/landsat/`
- 记录每个 patch 的均值/标准差到 `statistics/<region>/`。

### 4.2 生成高分辨率 patch

- 高分辨率数据源：
  - 哈尔滨 DOM（0.5m 光学，2025-12、2026-05）
  - 海淀 PlanetScene mosaic（约 3m 光学）
  - 海淀 SAR（来自 ModelScope，原生分辨率以 ORG 文件为准）
- 每个高分辨率 patch 对应一个低分辨率 patch 的地理范围（1280m × 1280m）。
- 像素尺寸按原生分辨率决定：
  - 0.5m → 2560×2560
  - 3m → 约 427×427
- 为控制显存，高分辨率编码器将使用大步幅卷积 + adaptive pooling 把输入压缩到与低分辨率特征相同的空间网格。
- 输出目录：
  - `processed/harbin/patches/highres_optical/`
  - `processed/harbin/patches/highres_sar/`（如有）
  - `processed/haidian/patches/highres_optical/`
  - `processed/haidian/patches/highres_sar/`

### 4.3 Manifest 更新

当前 `manifest.json` 以 patch 为单位，仅记录 S2 帧数。需要扩展为：

```json
{
  "patch_id": "harbin_000001",
  "region": "harbin",
  "crs": "EPSG:32652",
  "bounds": [...],
  "sources": {
    "s2": {"months": [202501, 202502, ...], "count": 17},
    "s1": {"months": [202501, ...], "count": 15},
    "landsat": {"months": [202501, ...], "count": 8},
    "highres_optical": {"months": [202512, 202605], "count": 2},
    "highres_sar": {"months": [], "count": 0}
  }
}
```

- `months` 为整数列表（YYYYMM）。
- 采样时根据 `sources` 的 availability 构造月度训练样本。
- 生成脚本：`scripts/data/generate_manifest.py`。

### 4.4 No-data 值

- **当前默认**：输入数据缺失填充为 `0`，标签 `0` 视为 nodata/背景。
- **设计**：保留默认 `0`，并在配置中暴露 `nodata_value` 字段。
- 若用户后续决定改为 `-1`，只需修改配置并重新标准化即可，无需改模型代码。
- 推荐暂时不改 `-1`：S2/S1/Landsat 原始 fill value 通常为 0，改为 -1 会与无符号整型冲突且需要额外偏移。

## 5. 模型架构改造

### 5.1 整体流程（改造后）

```
多源时序输入 (B, T, C, H, W)
  → per-sensor stem bank (保持 B,T,Stem,H,W)
  → STPEncoder (输出 B, T, H/2, W/2, precision_dim)
  → MonthlyEmbeddingModule (聚合到 B, T_m, H/2, W/2, embed_dim)
  → EmbeddingUpsampleHead (逐月上采样到 B, T_m, H, W, embed_dim)
  → 高分辨率独立编码器 (native resolution → B, C, H, W)
  → AvailabilityAwareFusion (逐月 base + highres)
  → LearnedBottleneck (无 L2 归一化)
  → 逐月 Decoder heads (B, T_m, C_out, H, W)
```

### 5.2 移除 L2 归一化：LearnedBottleneck

新增文件：`src/xuannv_embedding/models/bottleneck.py`（替换现有 VMFBottleneck）。

```python
class LearnedBottleneck(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        num_groups = 8 if out_dim % 8 == 0 else out_dim
        self.norm = nn.GroupNorm(num_groups, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))
```

- 不调用 `F.normalize`。
- 训练时允许模型学习embedding的幅度。

### 5.3 逐月 embedding：MonthlyEmbeddingModule

新增文件：`src/xuannv_embedding/models/blocks.py` 中新增 `MonthlyEmbeddingModule`。

输入：
- `feats`: `(B, T_obs, H, W, C)`，来自 STPEncoder。
- `timestamps`: `(B, T_obs)`，每个观测的月份整数（YYYYMM 或相对月索引）。
- `mask`: `(B, T_obs)`，时间有效掩码。

输出：
- `monthly_feats`: `(B, T_month, H, W, embed_dim)`。
- `monthly_mask`: `(B, T_month)`，哪些月份有观测。

实现要点：
1. 根据输入的时间范围构造统一月度网格 `month_indices`，长度 `T_month`。
2. 使用 `nn.Embedding` 或正弦编码为每个月份生成位置编码。
3. 对每个观测，按 `timestamp` 分配到对应月份 bin，使用 mask 加权平均。
4. 缺失月份用可学习参数 `missing_token`（形状 `(1, 1, 1, 1, embed_dim)`）填充。
5. 最终通过 `nn.Linear(C, embed_dim)` 投影到 embedding 维度。

### 5.4 高分辨率原生分辨率处理

改造 `src/xuannv_embedding/models/model.py`：

1. 高分辨率 source 不再假设与低分辨率同尺寸。
2. `SensorEncoderBank` 为高分辨率 source 配置合适的 `spatial_stride`（大步幅下采样），或使用新的 `NativeResolutionHighResEncoder`：
   - 若干 `Conv2d + GroupNorm + GELU` 层，stride=2 交替下采样。
   - 最后使用 `F.adaptive_avg_pool2d` 投影到 `(H, W)`（即低分辨率 patch 尺寸）。
3. `AvailabilityAwareFusion` 保持不变，但 `highres_feat` 已经是 `(B, embed_dim, H, W)`。
4. 融合后的 `base_feat` 形状为 `(B, embed_dim, H, W)`，复制到每个月份，得到 `(B, T_month, embed_dim, H, W)`。

### 5.5 逐月解码器头

改造 `src/xuannv_embedding/models/decoders.py`：

- `ContinuousDecoder` 和 `CategoricalDecoder` 当前接收 `(B, C, H, W)`，输出 `(B, C_out, H, W)`。
- 扩展为接收 `(B, T, C, H, W)`，输出 `(B, T, C_out, H, W)`。
- 实现方式：将 `B*T` 视为 batch 维度，调用现有卷积头，再 reshape 回 `(B, T, ...)`。

### 5.6 AEFModel 输出

更新 `AEFOutput`：

```python
@dataclass
class AEFOutput:
    embedding_map: torch.Tensor       # (B, T_month, D, H, W)
    embedding: torch.Tensor           # (B, T_month, D)
    reconstructions: dict[str, torch.Tensor]  # 每个 head: (B, T_month, C_out, H, W)
```

## 6. 训练流程改造

### 6.1 采样策略

- 从 manifest 中按 patch 采样，并选择时间窗口长度 `W`（默认 12 个月）。
- 每次迭代：
  1. 随机选择一个起始月份 `m_start`。
  2. 截取 `[m_start, m_start + W)` 的月度网格。
  3. 对每个 source，只取落在窗口内的观测；缺失月份由模型内部 `missing_token` 填充。
- 最小规模验证：使用全部 17 个月（2025-01 至 2026-05）作为单个窗口，不截断。
- 高分辨率数据只有部分月份，在窗口内缺失时同样走 `missing_token` 路径。

### 6.2 Loss 改造

- 重建损失改为逐月计算，并对每个 source 的可用性 mask 加权。
- 连续模态（S2/S1/Landsat 反射率/后向散射）：使用 masked MSE。
- 分类模态（LULC）：使用 masked CrossEntropy。
- 可选：在 embedding 上加入 temporal consistency regularization（相邻月份 embedding 的 L2 距离惩罚），但最小验证阶段不加。

### 6.3 多 NPU 优化

- 使用 `torch_npu` + `torch.nn.parallel.DistributedDataParallel`。
- 在 STP 块中使用 gradient checkpointing（`torch.utils.checkpoint`）。
- 开启 AMP（`torch_npu.amp` / `torch.cuda.amp` 兼容接口）。
- 配置较大的 `num_workers`、pin_memory，减少 CPU 瓶颈。
- 若 batch size 过小，使用梯度累积。
- 高分辨率分支由于输入尺寸大，默认不参加所有样本，只在可用时前向，避免全 batch 显存峰值。

## 7. 文件改动清单

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `src/xuannv_embedding/models/bottleneck.py` | 重写 | `VMFBottleneck` → `LearnedBottleneck`，移除 L2 归一化 |
| `src/xuannv_embedding/models/blocks.py` | 新增 + 修改 | 新增 `MonthlyEmbeddingModule`；保留 STP 相关块 |
| `src/xuannv_embedding/models/model.py` | 修改 | 逐月 embedding、高分辨率原生分辨率、新 bottleneck |
| `src/xuannv_embedding/models/decoders.py` | 修改 | 支持 `(B, T, C, H, W)` 输入输出 |
| `src/xuannv_embedding/models/highres_fusion.py` | 可能修改 | 处理不同尺寸 highres feat，最终对齐到 base grid |
| `src/xuannv_embedding/models/sensor_encoders.py` | 可能新增 | `NativeResolutionHighResEncoder` |
| `src/xuannv_embedding/training/batch_preparation.py` | 修改 | 生成月度 target、处理 highres native size |
| `src/xuannv_embedding/training/loss.py` | 修改 | 逐月 masked loss |
| `src/xuannv_embedding/training/trainer.py` | 修改 | 多 NPU DDP、AMP、gradient checkpointing |
| `src/xuannv_embedding/data/dataset.py` | 可能修改 | 月度窗口采样 |
| `src/xuannv_embedding/config.py` | 修改 | 新增 `nodata_value`、`temporal_window`、`use_amp` 等 |
| `scripts/data/generate_manifest.py` | 修改 | 支持多源逐月记录 |
| `scripts/data/preprocess.py` | 修改/调用 | 完成 S1/Landsat/高分辨率切 patch |
| `configs/base.yaml` | 修改 | 更新 head 配置、窗口、优化参数 |
| `configs/harbin_128.yaml` / `configs/haidian_128.yaml` | 修改 | 更新 source 列表 |
| `tests/test_model.py` | 修改 | 验证逐月输出形状、无 L2 归一化 |
| `tests/test_batch_preparation.py` | 新增/修改 | 验证月度 target 构造 |
| `docs/data_layout.md` | 更新 | 记录新增 source 目录 |
| `docs/data_inventory.md` | 更新 | 自动生成 |

## 8. 成功标准

- S1 / Landsat / 高分辨率 patch 生成完成，manifest 正确记录逐月可用性。
- `pytest tests/test_model.py` 通过，且输出 `embedding_map` 形状为 `(B, T_month, D, H, W)`。
- `pytest tests/test_batch_preparation.py` 通过，target 形状为 `(B, T_month, C_out, H, W)`。
- 在哈尔滨 + 海淀全部数据上启动训练，前 100 个 step 不 OOM、loss 下降。
- 单模型同时见过哈尔滨与海淀样本。

## 9. 待用户确认项

1. **No-data 值**：默认保持 `0`，是否改为 `-1`？
2. **时间窗口长度**：默认 `W=12` 个月；最小验证是否直接用全部 17 个月？
3. **高分辨率 SAR**：海淀 SAR 数据是否也要切为高分辨率 patch？哈尔滨是否有高分辨率 SAR？

## 10. 决策记录

| 日期 | 事项 | 决策 |
|------|------|------|
| 2026-06-17 | 高分辨率融合策略 | 选 B：原生分辨率独立模态 + availability-aware 融合 |
| 2026-06-17 | 时序摘要 | 选 B：逐月 embedding 网格 |
| 2026-06-17 | Bottleneck | 选 B：无约束 learned bottleneck，移除 L2 归一化 |
| 2026-06-17 | 训练范围 | 哈尔滨 + 海淀全部数据，单模型最小规模验证 |
| 2026-06-17 | 空间分辨率/网格 | 低分辨率 10m，patch 128×128；高分辨率按原生分辨率切 patch |
