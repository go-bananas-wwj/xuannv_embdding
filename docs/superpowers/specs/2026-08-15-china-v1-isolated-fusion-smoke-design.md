# China V1 隔离三路融合冒烟测试设计

## 1. 目标

在不修改、不复用写目录、不占用生产 NPU 的前提下，用 4 个全国 2020—2021 patch 快速验证完整三路融合链路：

```text
真实季度 S1/S2
      ↓
隔离的极小季度基座
      ├── 模拟年度 AEF 旁路
      └── 模拟年度 2 m 旁路
      ↓
零初始化、有界、可关闭的残差融合
      ↓
8 季度 × 64 维 × 128 × 128 embedding
```

本次只验证工程合同、隔离性和 NPU 可运行性，不验证真实模型精度，不把任何模拟数据用于正式训练或下游评测。

## 2. 已确认的范围

- 真实输入：4 个 patch 的 2020—2021 月度 Sentinel-2 10 波段和 Sentinel-1 2 波段。
- 季度定义：每年 Q1—Q4，每季度聚合三个月；两年合计 8 个季度。
- 模拟 AEF：从真实年度 S2 生成、空间对齐、确定性的 64 维单位向量。
- 模拟 2 m：从真实年度 S2 RGB 生成、空间对齐、确定性的 `640×640` 年度影像。
- 输出：`[4, 8, 64, 128, 128]`，FP16 冒烟产物和可审计 manifest。
- 运行：CPU 合同测试后，仅使用一张空闲物理 NPU 2。
- 训练：只执行受控的最小反向传播，不做正式训练。
- 不接入 OlmoEarth，不下载真实 AEF，不等待真实 2 m 数据。

## 3. 非目标

以下内容明确不在本次冒烟范围：

- 不比较 TESSERA、OlmoEarth、AEF 或玄女的精度；
- 不声称模拟 AEF 具有官方 AEF 语义；
- 不声称模拟 2 m 具有真实 2 m 空间信息；
- 不修改 `src/xuannv_embedding/` 的现有生产模型；
- 不修改海淀、哈尔滨或全国已有实验配置；
- 不使用 NPU 0/1，不使用 6 卡 DDP；
- 不导出正式 China V1 产品；
- 不清理或覆盖任何已有缓存、checkpoint、日志和环境。

## 4. 隔离边界

### 4.1 代码隔离

```text
worktree: /root/workspace/xuannv/.worktrees/codex-china-v1-fusion-smoke
branch:   codex/china-v1-fusion-smoke
base:     4cf032a
```

所有新增代码、配置和测试只在该 worktree 中开发。主工作区 `/root/workspace/xuannv` 的未提交文件不读取、不暂存、不修改。

### 4.2 运行隔离

固定沙箱根目录：

```text
/data/xuannv_embedding/sandboxes/china_v1_fusion_smoke_20260815/
├── env/                 # 轻量 venv；只读复用当前 torch/torch_npu site-packages
├── cache/               # 4 patch 的只读源数据索引与局部缓存
├── synthetic/
│   ├── aef/
│   └── highres_2m/
├── outputs/
│   ├── base/
│   ├── base_aef/
│   ├── base_highres/
│   └── full/
├── checkpoints/
├── logs/
├── manifests/
├── tmp/
└── .xuannv_isolated_smoke
```

环境采用 `venv --system-site-packages`，目的是不复制或升级大型 Ascend 软件栈，同时确保本次任务无法向原环境安装包。启动时固定：

```text
PYTHONNOUSERSITE=1
PYTHONPATH=/root/workspace/xuannv/.worktrees/codex-china-v1-fusion-smoke/src
ASCEND_RT_VISIBLE_DEVICES=2
WANDB_MODE=disabled
```

每次运行都记录 Python executable、`xuannv_embedding.__file__`、Git commit、torch、torch_npu、CANN 和驱动版本。

### 4.3 原始数据只读

原始数据只从以下目录读取：

```text
/data2/china_xuannv_embedding/data
```

代码不得在 `/data2/china_xuannv_embedding/data` 内创建索引、锁、临时文件或完成标记。所有衍生内容都写入沙箱。

### 4.4 路径防护

运行器必须在写文件前执行 realpath 校验：

- 输出路径必须是固定沙箱根目录的子目录；
- 沙箱根目录必须含 `.xuannv_isolated_smoke` sentinel；
- 明确拒绝 `/root/workspace/xuannv`、`/data/xuannv_embedding/processed`、`/data/xuannv_embedding/outputs`、`/data/xuannv_embedding/embeddings` 和 `/data2/china_xuannv_embedding/data` 作为写目录；
- 固定 `max_patches=4`、`years=[2020, 2021]`、`max_optimizer_steps=2`；
- 任何路径或规模越界均 fail closed。

本次不实现自动删除。沙箱保留供审计；后续如需清理，必须另行确认并由 sentinel 保护的专用脚本执行。

## 5. 数据选择和契约

### 5.1 Patch 选择

从本地 24 个月 S2 与 S1 ZIP 索引中，选择按 `patch_id` 排序后最先满足以下条件的 4 个 patch：

- 2020、2021 均有 24 个月 S2；
- 2020、2021 均有 24 个月 S1；
- S2 shape 为 `[10,128,128]`；
- S1 shape 为 `[2,128,128]`；
- CRS、transform 和 bounds 在所有月份一致；
- 抽样读取无损坏、NaN 泛滥或全零影像。

选择结果写入 `manifests/patch_selection.json`，包含每个 ZIP entry、CRC、shape、CRS 和 transform。不得每次运行重新随机选 patch。

### 5.2 真实季度输入

每个 patch-year 形成：

```text
s2:        float32 [4,3,10,128,128]
s1:        float32 [4,3, 2,128,128]
valid_s2:  bool    [4,3, 1,128,128]
valid_s1:  bool    [4,3, 1,128,128]
year:      int
patch_id:  string
```

S2 按明确的 reflectance scale 转 float32。S1 原始顺序为 VH/VV，适配器重排成 VV/VH，并使用带 epsilon 的 dB 转换。所有数值变换和分位数写入 manifest。

### 5.3 模拟 AEF

模拟 AEF 只验证年度 side branch 合同：

1. 对一年 12 个月 S2 做 valid-mask 年度均值；
2. 使用固定 seed 生成并保存一个 `10→64` bias-free 投影矩阵；
3. 对每个 10 m 像素投影到 64 维；
4. 对 64 维做单位化；
5. 无效像素单独保存 mask，不用零值伪装有效向量。

输出：

```text
aef:       float32 [64,128,128]
aef_valid: bool    [ 1,128,128]
```

每个文件及 registry 必须包含：

```json
{
  "synthetic": true,
  "synthetic_kind": "annual_s2_fixed_projection",
  "allowed_use": "smoke_test_only",
  "formal_training_allowed": false,
  "formal_evaluation_allowed": false
}
```

正式数据加载器必须拒绝 `synthetic=true` 的 registry；冒烟加载器则要求该字段必须为 true，避免两条路径混用。

### 5.4 模拟 2 m

模拟 2 m 只验证 `640×640 → 128×128` 的高分旁路：

1. 取年度 S2 RGB 的 valid-mask 中位数；
2. 从 `128×128` 以 5 倍上采样到 `640×640`；
3. 使用固定参数加入轻微 unsharp edge 和固定 seed 的低幅纹理；
4. 截断到合法 reflectance 范围；
5. 保存原始有效 mask 上采样结果。

输出：

```text
highres:       float32 [3,640,640]
highres_valid: bool    [1,640,640]
```

registry 必须包含与模拟 AEF 相同的用途限制，并额外记录：

```json
{
  "synthetic_kind": "annual_s2_rgb_5x_deterministic_texture",
  "claimed_native_gsd_m": null,
  "model_input_gsd_m": 2,
  "contains_real_2m_information": false
}
```

## 6. 隔离冒烟模型

模型只放在独立实验包中，不注册到现有 `AEFModel` factory：

```text
experiments/china_v1_fusion_smoke/
├── data.py
├── synthetic.py
├── model.py
├── runner.py
├── registry.py
└── safety.py
```

### 6.1 极小季度基座

季度 S2/S1 分别经过小型原生 PyTorch stem，再按有效月份聚合并映射到：

```text
base: [B,4,64,128,128]
```

它不使用 `einops`，不复用或改动现有玄女生产模型。权重使用固定 seed 初始化，在融合冒烟中冻结。

### 6.2 AEF 旁路

```text
[B,64,128,128]
→ bias-free 1×1，64→16
→ depthwise 3×3
→ 1×1，16→64
→ aef_delta [B,64,128,128]
```

年度 delta 只在运行时 expand 到四季度，不重复保存四份。

### 6.3 2 m 旁路

```text
[B,3,640,640]
→ kernel=5, stride=5 的抗混叠 stem
→ [B,32,128,128]
→ 轻量 residual adapter
→ highres_delta [B,64,128,128]
```

这条路径必须真实处理 `640×640` 输入，不能在进入模型前直接缩成 `128×128`。

### 6.4 可撤销融合

```text
z = base
z = z + use_aef     * tanh(aef_gate)     * aef_delta
z = z + use_highres * tanh(highres_gate) * highres_delta
z = vMF(z)
```

- 两个 gate 相互独立并零初始化；
- `use_aef`、`use_highres` 是显式布尔开关；
- 关闭分支时直接跳过计算，不传全零假输入；
- 输出为单位范数 64 维向量。

## 7. 冒烟矩阵

固定运行四组：

| 组 | Base | 模拟 AEF | 模拟 2 m |
|---|---:|---:|---:|
| `base` | 是 | 否 | 否 |
| `base_aef` | 是 | 是 | 否 |
| `base_highres` | 是 | 否 | 是 |
| `full` | 是 | 是 | 是 |

执行顺序：

1. CPU 合同测试使用缩小 tensor，验证 shape、mask、registry 和路径防护；
2. NPU 2 上执行 4 patch 全尺寸前向；
3. gate 为零时验证 `full == base`，使用 FP32 容差；
4. 在专用 gradient-smoke 模式把 gate 临时设为 0.1，执行最多 2 个 optimizer step；
5. 验证 AEF adapter、2 m adapter、gate 和输出头都有有限的非零梯度；
6. 保存 checkpoint，重新加载并复算；
7. 导出四组 embedding、manifest 和摘要；
8. 使用相同 seed 再运行一次，比较 checksum 或预设数值容差。

临时把 gate 设为 0.1 只用于证明反向链路，不作为生产初始化或训练结论。

## 8. 输出合同

每组输出一个 Zarr v2 shard：

```text
embedding:    float16 [4,8,64,128,128]
valid:        bool    [4,8, 1,128,128]
patch_id:     string  [4]
period:       string  [8]  # 2020Q1...2021Q4
```

同时写出：

- `run_manifest.json`：代码、环境、设备、输入和模拟数据 provenance；
- `metrics.json`：shape、NaN/Inf、范数、梯度、延迟、峰值 HBM；
- `path_audit.json`：本次创建或修改的全部文件；
- `reproducibility.json`：两次运行的一致性结果；
- `smoke_checkpoint.pt`：只用于 reload 测试；
- `SUCCESS`：仅在所有验证通过后写入。

所有正式用途相关字段必须保持 false。

## 9. 测试设计

### 9.1 单元测试

- 路径守卫接受固定 sandbox，拒绝生产目录和目录穿越；
- patch selector 稳定选择同一组完整 patch；
- 模拟 AEF 同 seed 可重复、单位范数、用途字段完整；
- 模拟 2 m 同 seed 可重复、shape 正确、明确不含真实 2 m 信息；
- 正式 registry validator 拒绝 synthetic registry；
- 冒烟 registry validator 拒绝未标 synthetic 的模拟输入；
- 四种分支组合 shape 一致；
- 零 gate 与 base 恒等；
- 任一分支缺失时显式跳过；
- vMF 输出无 NaN/Inf 且单位范数；
- checkpoint reload 输出一致。

### 9.2 本地数据合同测试

- 4 个 patch 的 24 月 S2/S1 ZIP entry 均可读；
- 跨月份 CRS、transform、shape 一致；
- S1 顺序和 dB 变换统计在合理范围；
- 原始目录运行前后文件清单不变。

### 9.3 NPU 集成测试

- `ASCEND_RT_VISIBLE_DEVICES=2` 后只出现一个逻辑设备；
- 运行前确认物理 NPU 2 无其他进程；
- 全尺寸四组前向通过；
- gradient-smoke 通过；
- checkpoint reload 与重复运行通过；
- 峰值 HBM、延迟和 CPU fallback 被记录。

## 10. 成功与失败条件

### 成功条件

- 输出 shape 精确为 `[4,8,64,128,128]`；
- 所有有效像素无 NaN/Inf，vMF 范数在容差内；
- 零 gate 时 full 与 base 一致；
- gradient-smoke 中两个旁路均存在有限非零梯度；
- 任一旁路关闭或缺失时仍能运行；
- checkpoint 保存、恢复、复算一致；
- 固定 seed 重跑一致；
- 只使用物理 NPU 2；
- 原始数据目录没有任何写入；
- 所有新运行产物只存在固定 sandbox；
- 所有 synthetic 产物被正式 validator 拒绝。

### 立即停止条件

- NPU 2 被其他任务占用；
- 解析后的输出路径不在 sandbox；
- 原始目录出现写操作；
- patch 数据的 CRS/transform 不一致；
- 任一模拟 registry 缺少 synthetic 用途限制；
- 正式 validator 接受模拟数据；
- 出现 NaN/Inf、不可恢复的 NPU 错误或 checkpoint 不一致。

失败时保留日志和 partial 产物用于诊断，但不得写 `SUCCESS`。

## 11. 预计资源和时间

| 项目 | 预算 |
|---|---:|
| NPU | 1×Ascend 910B4，物理卡 2 |
| CPU | 默认 2 worker；当前 OCR 高负载时降为 0—1 worker |
| RAM | 小于 32 GB |
| 沙箱磁盘 | 目标小于 5 GB |
| CPU 测试 | 5—15 分钟 |
| 数据抽取与模拟输入 | 15—60 分钟，取决于 ZIP 解压和 CPU 背景负载 |
| NPU 全矩阵 | 10—30 分钟 |
| 实现、测试和排障 | 0.5—1 个工作日 |

## 12. 后续升级边界

本冒烟通过后，下一阶段才允许逐项替换：

1. 用真实官方 AEF 2020/2021 替换模拟 AEF；
2. 用真实年度 2 m 数据替换模拟高分；
3. 用通过 4 patch 算子闸门的 OlmoEarth/TESSERA 替换极小季度基座；
4. 扩到 200 patch 做效果消融。

每次只替换一个输入或基座，并重复相同四组矩阵。冒烟代码不直接并入玄女生产模型；只有真实数据和真实基座通过 200/2k/10k 闸门后，才另行设计生产集成。
