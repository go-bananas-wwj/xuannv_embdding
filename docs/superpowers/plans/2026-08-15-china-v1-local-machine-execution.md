# China V1 本机落地执行 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. 每完成一个 Task，必须立即 commit 并 push；不得把多个 Task 攒成一次提交。

**Goal:** 在当前这台 8×Ascend 910B4 服务器上，用 2020—2021 年全国月度遥感、年度 AEF、年度高分 2 米一张图，快速产出 8 个“年度完成后回溯生成”的季度 64 维 embedding；先做可信的 200 patch 版本，再通过 2,000、10,000、约 62,000 patch 三道闸门扩展。

**Architecture:** 冻结 OlmoEarth/TESSERA 类基础特征，只把它作为季度动态基座；AEF 年度向量与 2 米年度结构走两个独立、低维、可关闭的 side branch，通过零初始化有界门控注入。最终始终输出 `8×64×128×128`，空间含义仍是 10 m，不宣称 2 m embedding。年度信息允许回看全年，但产品必须标注为 `annual-retrospective-quarterly`，不是实时季度产品。

**Tech Stack:** Python 3.11、PyTorch/torch_npu、CANN 9.0.0、6×Ascend 910B4（仅 NPU 2—7）、Zarr v2、Numcodecs、Rasterio、pytest、HCCL。

---

## 0. 先给结论：这台机器应该怎么用

这台机器**可以完成**：

1. 200 patch 的接口、精度和方法选型；
2. 2,000 patch 的可信对比；
3. 10,000 patch 的 24—48 小时稳定性演练；
4. 约 62,000 patch 的训练样本特征生成和融合训练；
5. 小区域、代表区和训练集范围的 8 季度 embedding 导出。

这台机器**不适合单机完成**全国约 574 万 patch 的最终产品：全国 8 季度 int8 净数据约 49 TB，双副本约 110—140 TB，本机两个数据盘当前合计只剩约 4 TB。全国生产要另配 60 TB 以上单副本存储，稳妥配置为 120 TB 以上；若要求 10—15 天完成，还应使用 16—32 张同构 NPU。

最重要的四个现实约束是：

- NPU 0、1 正被长期服务占用，各用了约 57 GB HBM，绝不能抢占；本项目只能使用 NPU 2—7。
- `/workspace` 只剩约 4.4 GB，禁止在这里创建环境、缓存、checkpoint 或日志。
- 当前有一个外部 OCR 任务占用约 170 个 CPU 核；在它结束前，不应启动大规模 TIFF 解码、ZIP 随机读取或 Zarr 重打包。
- 当前 2020—2021 数据不是 OlmoEarth 原生输入：S2 只有 10 波段，缺 B01/B09，也没有 SCL、场景 ID 和准确时间；S1 是线性功率且顺序为 VH/VV；因此不能“直接套模型”。

### 2026-08-15 本机快照

| 项目 | 实测值 | 对方案的影响 |
|---|---|---|
| CPU | aarch64 Kunpeng-920，192 个物理核，4 socket、8 NUMA node | 解压能力强，但必须做 NUMA/并发控制 |
| 内存 | 1.0 TiB，审计时约 932 GiB 可用 | 足够做年度 patch 拼接和较大预取 |
| `/dev/shm` | 500 GiB | 可用于 DataLoader 共享内存，但不能把它当持久缓存 |
| memlock | 64 MiB | pin memory/HCCL 若报错，需先降预取或由管理员提高限制 |
| NPU | 8×Ascend 910B4-1，每张 64 GB HBM，HCCS 全连接 | 其中 6 张空闲，适合单机 6 卡训练/提取 |
| 当前 Python | 3.11.15 | 满足项目 Python 要求 |
| 当前 torch | `2.6.0+cpu` + `torch_npu 2.6.0.post5` | 玄女当前可用，但与 Olmo minimal 依赖不完全一致 |
| CANN/驱动 | CANN 9.0.0，npu-smi 26.0.rc1 | 新环境必须与这套驱动做 4 patch 验证 |
| 数据库 | Zarr 2.18.7、Numcodecs 0.15.1、Rasterio 1.4.4 | 现有代码走 Zarr v2，禁止无测试升级到 Zarr v3 |
| NumPy | 2.4.6 | 与 pyproject 中 `<2` 声明不一致，需记录真实环境而非相信声明 |

当前 CPU/NPU 亲和关系如下；只有外部 OCR 任务结束后才建议按此 pin worker：

| NPU | 建议 CPU 核 |
|---|---|
| 2、3 | 96—119 |
| 4、5 | 0—23 |
| 6、7 | 48—71 |
| I/O、QA、调度 | 24—47、72—95、120—143、168—191 |

当前还存在三类外部负载：NPU 0/1 上的 Qwen2.5-14B vLLM、占用约 170 核的 OCR、`/data2` 上的 ModelScope 下载和旧全国物化。实施者不得终止这些进程，只能通过资源闸门等待或降低本项目并发。

### 本机固定资源分配

| 用途 | 固定选择 | 原因 |
|---|---|---|
| 训练/推理 NPU | `2,3,4,5,6,7` | 0/1 已被 vLLM 占用 |
| 原始 2020—2021 数据 | `/data2/china_xuannv_embedding/data`，只读 | 已下载约 584 GB，避免重复 |
| 新环境 | `/data/wwj_torch21/conda/envs/olmoearth_npu` | `/workspace` 已满 |
| 小中型缓存 | `/data/xuannv_embedding/cache/china_v1_retrospective_2020_2021` | 与 `/data2` 源数据分盘读写 |
| 训练输出 | `/data/xuannv_embedding/outputs/china_v1_retrospective_2020_2021` | 便于 checkpoint 与日志管理 |
| 日志 | `/data/xuannv_embedding/logs/china_v1_retrospective_2020_2021` | 不污染代码盘 |
| 全量 int8 结果 | `/data2/xuannv_embedding/china_v1_retrospective_2020_2021/embeddings` | 仅在下载结束并重新核算空间后启用 |
| 代码与文档 | `/root/workspace/xuannv` | 只保存小文件并进入 Git |

磁盘硬闸门：

- `/workspace` 不产生任何实验大文件；
- `/data` 始终保留至少 300 GB；
- `/data2` 始终保留至少 500 GB；
- 每次从 200 → 2k → 10k → 62k 前重新执行磁盘审计；
- 未通过下一阶段前，禁止同时长期保存 Tiny、Small、FP16 最终结果和重复原始缓存。

---

## 1. 产品定义：先把“季度”说清楚

本项目的季度结果定义为：

> 在某一年结束后，允许利用这一整年的 AEF、2 米一张图和 12 个月上下文，回溯生成该年 Q1—Q4 的 embedding。季度动态主要来自该季度三个月的 S1/S2；年度 AEF 与年度 2 米图提供全年共享的语义和空间结构。

因此：

- 2020Q1 可以利用 2020 年全年信息，但不能利用 2021 年信息；
- 2021Q1 可以利用 2021 年全年信息，但不能利用 2020/2021 以外的信息；
- 结果不是“季度结束当天即可发布”的 causal 产品；
- 年度 2 米图不能被描述成每季度都有一次 2 米观测；
- 输出仍是 10 m 网格的 64 维 embedding，2 米图只帮助边界和纹理更清楚。

若以后需要实时季度版，应另建 `causal-quarterly` 产品线，不能把两种口径混在一个模型或指标表里。

---

## 2. 总路线：两条线并行，避免等数据时停工

### A 线：立即能跑的 raw baseline

使用现有 10 波段 S2、2 波段 S1 和有效月份 mask，先建立一个不依赖 OlmoEarth、AEF、高分数据的季度基线 B0。它的目的不是最终精度，而是尽快验证：

- patch ID、年份、季度是否正确；
- ZIP/Zarr 数据读取是否稳定；
- 输出 `8×64×128×128` 是否正确；
- 下游 building/road/water 的评测链是否能跑通；
- 后续 side branch 到底带来多少真实提升。

### B 线：把数据修成 OlmoEarth 可用

不能给缺失的 B01/B09 补零。对 200 patch 先重新查询原始 STAC/数据源，构造完整的 S2 12 波段与云掩膜，记录准确时间、场景 ID；S1 做 VV/VH 重排和 dB 转换。4 patch 跑通 OlmoEarth 后，再对 200 patch 比较 Tiny P8、Tiny P4、Small P4。

两条线在 200 patch 阶段汇合：只有 OlmoEarth 明显优于 B0，且速度/稳定性可接受，才进入 2k；否则 B0 仍是可交付版本，OlmoEarth 修复继续迭代，不阻塞整个项目。

---

## Task 1：冻结本机快照、设备和目录规则

**目的：** 让以后任何实验都能回答“用的是哪份代码、哪个 Python、哪六张卡、当时磁盘还剩多少”。

**Files:**

- Create: `scripts/qa/audit_china_v1_runtime.py`
- Create: `tests/test_audit_china_v1_runtime.py`
- Modify: `docs/data_layout.md`

### 实施步骤

1. 先写测试，要求审计 JSON 至少包含：Git commit、当前分支、Python 路径、`xuannv_embedding.__file__`、torch/torch_npu/CANN/driver 版本、8 张 NPU 的进程和 HBM、CPU/NUMA、RAM、`/workspace`/`/data`/`/data2` 空间。
2. 实现审计脚本，默认只读，不清理、不杀进程、不修改环境。
3. 脚本检查以下条件并给出明确红/黄/绿状态：
   - NPU 2—7 是否空闲；
   - NPU 0/1 是否被错误放入可见设备；
   - `/workspace` 是否被用作输出目录；
   - `/data` 是否少于 300 GB；
   - `/data2` 是否少于 500 GB；
   - `xuannv_embedding` 是否从当前 repo 的 `src/` 导入。
4. 把这台机器的目录分工写入 `docs/data_layout.md`。

**Run:**

```bash
export PYTHONPATH=/root/workspace/xuannv/src:/root/workspace/xuannv/downstreams
python -m pytest tests/test_audit_china_v1_runtime.py -q
python scripts/qa/audit_china_v1_runtime.py \
  --output /data/xuannv_embedding/logs/china_v1_retrospective_2020_2021/runtime_snapshot.json
```

**通过标准：** 审计文件明确显示代码来自当前分支，NPU 2—7 可用，输出路径都不在 `/workspace`。

**常见坑与处理：**

- **坑：导入到了旧 worktree。** 本机当前直接 `import xuannv_embedding` 会命中 `.worktrees/feat-v1.2-national-crossmodal`。处理：所有启动器显式设置绝对 `PYTHONPATH`，并把模块真实路径写进日志。
- **坑：脚本悄悄清理其他任务。** 审计脚本必须只读；发现 NPU/CPU 被占用时只退出并报告。
- **坑：看到 8 张卡就默认全用。** 当前可用的只有 2—7；0/1 是在线服务资源。

**Commit:** `docs: add China V1 local runtime audit and storage contract`

---

## Task 2：新增本项目专用的安全启动器

**目的：** 避免现有 `launch_6card.sh` 默认拿到 NPU 0—5，与在线服务撞车。

**Files:**

- Create: `scripts/train/launch_china_v1_quarterly_6card.sh`
- Create: `scripts/features/launch_china_v1_feature_workers.sh`
- Create: `tests/test_china_v1_launchers.py`

### 实施步骤

1. 训练启动器固定加载 CANN：

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=2,3,4,5,6,7
export PYTHONPATH=/root/workspace/xuannv/src:/root/workspace/xuannv/downstreams
```

2. 训练使用 6-rank HCCL；特征提取则使用 6 个互相独立的 shard worker，不用 DDP，不做 all-reduce。
3. 每次启动前自动运行 Task 1 的审计；任何一张 2—7 卡出现未知进程时 fail closed。
4. 每个 worker 固定一个输出 shard，写临时目录，校验后原子封口；不得六个进程写同一个 Zarr group。
5. 日志开头打印 Git commit、Python 路径、包版本、可见卡和配置 checksum。

**Run:**

```bash
python -m pytest tests/test_china_v1_launchers.py -q
bash scripts/features/launch_china_v1_feature_workers.sh --dry-run
bash scripts/train/launch_china_v1_quarterly_6card.sh --dry-run
```

**通过标准：** dry-run 只能显示物理 NPU 2—7；测试应明确拒绝 `0,1,2,3,4,5`。

**常见坑与处理：**

- **坑：逻辑卡 0 实际对应物理卡 2。** 日志同时记录可见列表和每个 rank 的物理设备映射。
- **坑：六卡提特征反而更慢。** 独立特征任务不应使用 DDP；在 200 patch 阶段对比 1/2/6 worker。
- **坑：HCCL/锁页失败。** 当前 `max locked memory` 只有 64 MB；若出现 pin-memory/HCCL 错误，先降低预取和 pinned memory，再由管理员提高 memlock，不在训练脚本里绕过错误。

**Commit:** `feat: add device-safe China V1 launchers for NPU 2-7`

---

## Task 3：建立真实的数据清单，不再把“约 62k”当成完整矩阵

**目的：** 对每个 patch、每个月、每个传感器明确“有/无/坏”，所有缺失都通过 mask 表达。

**Files:**

- Create: `src/xuannv_embedding/data/china_archive.py`
- Create: `scripts/data/inventory_china_v1_2020_2021.py`
- Create: `tests/test_china_archive.py`
- Create: `tests/test_china_v1_inventory.py`
- Create: `docs/data/china_v1_2020_2021_inventory.md`

### 已知事实

- S2 24 个月 ZIP 约 371 GB，单月 patch 数并不完全相同；
- S1 24 个月约 163 GB，波动更大；
- Landsat 24 个月约 26 GB；
- S2 24 月交集约 61,431；
- S1 24 月交集约 56,321；
- S2+S1 全月共同可用约 56,038；
- 所有 72 个“来源×月份”都存在的 patch 只有约 54,997；
- 所以不能把 README 的“62,000”理解成所有 patch 72 份文件都齐全。

### 实施步骤

1. 为外层 ZIP 建只读索引，不解压成数百万小文件。
2. 生成 `patch_id × 24 month × sensor` 的布尔可用矩阵、文件大小、shape、CRS、transform、dtype 和读取错误。
3. 记录 union、intersection、每月缺失率，不把缺月 patch 直接删掉；模型用 `valid_month_count` 与像素 mask。
4. 以空间分层方式选 200 patch：覆盖东中西、南北、城市/农田/水体/山地、云多/云少、缺月/完整月。
5. 同时生成固定的 2k、10k 和约 62k registry，后续规模只增加，不能每次重新抽样导致指标不可比。

**Run:**

```bash
python -m pytest tests/test_china_archive.py tests/test_china_v1_inventory.py -q
python scripts/data/inventory_china_v1_2020_2021.py \
  --source-root /data2/china_xuannv_embedding/data \
  --output-root /data/xuannv_embedding/cache/china_v1_retrospective_2020_2021/registries
```

**通过标准：** 每个训练样本都有明确月份 mask；随机抽 100 个 patch 的索引与 ZIP 内文件一致；损坏或缺失文件不会被当成全零影像。

**常见坑与处理：**

- **坑：ZIP 外层是 store，就以为读取不费 CPU。** 内部 TIFF 本身是 DEFLATE；随机读取仍会解压，六 worker 可能先把 CPU 打满。
- **坑：先全部解压。** 会产生大量小文件并快速耗尽 inode/空间；先 benchmark 直接 ZIP 与年度 patch Zarr。
- **坑：只保留 54,997 个全量样本。** OlmoEarth 支持缺月 mask；不应为了整齐丢掉所有非满月 patch，但应在评测中单列完整/缺月表现。

**Commit:** `data: add 2020-2021 national archive inventory and availability registry`

---

## Task 4：做 200 patch I/O 基准，决定是否重打包 Zarr

**目的：** 用实测而不是直觉决定数据格式。

**Files:**

- Create: `scripts/bench/benchmark_china_v1_archive_io.py`
- Create: `scripts/data/repack_china_v1_patch_year_zarr.py`
- Create: `tests/test_china_v1_patch_year_zarr.py`
- Create: `docs/benchmarks/china_v1_local_io_200.md`

### 实施步骤

1. 当前外部 OCR 任务仍占约 170 CPU 核时，只运行 1—2 worker 的小基准；不要直接重打包 62k。
2. 比较三种方案：
   - 直接从月度 ZIP 随机读；
   - 按月顺序读、在内存拼为 patch-year；
   - 把 200 patch 重打包为按 patch-year 分 shard 的 Zarr v2。
3. 每种方案记录：P50/P95 读取延迟、CPU 使用、内存、磁盘吞吐、NPU 等待比例。
4. 只有 Zarr 让端到端吞吐提升至少 25%，才为 2k/10k/62k 扩展重打包。
5. 推荐每 shard 32—64 个 patch-year，禁止每季度一个 `.pt` 小文件。

**Run:**

```bash
python -m pytest tests/test_china_v1_patch_year_zarr.py -q
python scripts/bench/benchmark_china_v1_archive_io.py \
  --registry /data/xuannv_embedding/cache/china_v1_retrospective_2020_2021/registries/pilot_200.json \
  --workers 1 2 4
```

**通过标准：** 选定的数据格式有端到端证据；不会因为六卡同时随机解压而让 NPU 大量空等。

**常见坑与处理：**

- **坑：只测磁盘 MB/s。** 真正需要看的是“完成一个 patch 两年输入”的秒数和 NPU 空闲比例。
- **坑：OCR 结束前得出 CPU 上限。** 当前 CPU 被外部任务占满，基准必须标注背景负载；OCR 结束后复测。
- **坑：在 `/data2` 同盘读写。** 原始数据从 `/data2` 读，试验缓存优先写 `/data`，降低单盘争抢。

**Commit:** `perf: benchmark and select China V1 patch-year storage layout`

---

## Task 5：先跑不依赖外部模型的 B0 季度基线

**目的：** 在 OlmoEarth 数据修复、AEF 申请和 2 米挂载期间，先拿到一个可评测版本。

**Files:**

- Create: `src/xuannv_embedding/data/quarterly_raw_dataset.py`
- Create: `src/xuannv_embedding/models/quarterly_raw_baseline.py`
- Create: `scripts/train/train_quarterly_raw_baseline.py`
- Create: `tests/test_quarterly_raw_dataset.py`
- Create: `tests/test_quarterly_raw_baseline.py`
- Create: `configs/national/china_v1_quarterly_raw_200_sanity_20260815.yaml`

### 实施步骤

1. 每个样本是一个 `patch-year`，每年四季度，每季度由三个月组成。
2. S2 10 波段按真实缩放处理；S1 按真实数据定义处理，但 B0 与 Olmo 分支的预处理配置必须分开记录。
3. Landsat 只有 6 波段、43×43；B0 可以做“有/无 Landsat”消融，但首个 Olmo 版本不使用 Landsat。
4. 输出四季度 `64×128×128`，2020/2021 两个 patch-year 合并后为八季度。
5. 用相同空间 fold 和相同 building/road/water 标签建立下游基线。

**Run:**

```bash
export PYTHONPATH=/root/workspace/xuannv/src:/root/workspace/xuannv/downstreams
python -m pytest tests/test_quarterly_raw_dataset.py tests/test_quarterly_raw_baseline.py -q
python scripts/train/train_quarterly_raw_baseline.py \
  --config configs/national/china_v1_quarterly_raw_200_sanity_20260815.yaml \
  --device npu:2
```

**通过标准：** 200 patch 训练、恢复、导出、下游 probe 全链路通过；输出无 NaN，季度与年份顺序正确。

**常见坑与处理：**

- **坑：把 B0 当最终模型。** B0 是管线和指标锚点；它不等价于 OlmoEarth。
- **坑：随机切 patch 泄漏。** 必须使用空间 fold，同一邻近区域不能跨 train/test。
- **坑：季度被年度上下文压平。** B0 不使用年度分支，正好用于后续判断 AEF/高分是否抹掉季度变化。

**Commit:** `feat: add 200-patch quarterly raw baseline`

---

## Task 6：把 200 patch 修成 OlmoEarth 可用数据

**目的：** 不伪造缺失波段，不用错误尺度，把接口风险先缩到 200 patch。

**Files:**

- Create: `scripts/data/prepare_china_v1_olmo_pilot.py`
- Create: `src/xuannv_embedding/data/olmo_input_adapter.py`
- Create: `tests/test_olmo_input_adapter.py`
- Create: `docs/data/china_v1_olmo_input_contract.md`

### 当前数据必须修复的内容

| 数据 | 当前状态 | 正确处理 |
|---|---|---|
| S2 | 10 波段，缺 B01/B09 | 从原始数据源重取 12 波段，不能补零 |
| S2 云 | 无 SCL/云 mask | 重取 SCL/QA 或重新做可追溯云掩膜 |
| S2 时间 | TIFF 无 scene ID/精确时间 | 重查 STAC，保存 scene ID、timestamp |
| S1 | 顺序 VH/VV，值是线性功率 | 重排为 VV/VH，做 `10*log10(max(x, eps))` |
| S1 轨道 | 无 orbit metadata | pilot 至少记录可查到的轨道/产品信息 |
| Landsat | 6 波段，不是官方 11 波段 | Olmo V0 排除；以后完整重建再接入 |

### 实施步骤

1. 以 200 patch 的 UTM bounds 和年月重新查询原始 STAC，重建完整 12-band S2 月度输入。
2. 不追求与旧 TIFF 恰好同一 scene；记录重建规则，并把“旧月度最佳影像”和“新 Olmo 月度输入”作为两个数据版本。
3. S2 做波段顺序、缩放、有效 mask 测试；S1 做顺序、dB 数值范围、NaN/负无穷测试。
4. 写入 pilot Zarr；每条记录包含来源 URI、scene ID、时间、CRS、transform、波段顺序和预处理版本。
5. 随机可视化 20 patch×12 月，人工检查云、错位、全零和明显色彩异常。

**Run:**

```bash
python -m pytest tests/test_olmo_input_adapter.py -q
python scripts/data/prepare_china_v1_olmo_pilot.py \
  --registry /data/xuannv_embedding/cache/china_v1_retrospective_2020_2021/registries/pilot_200.json \
  --output-root /data/xuannv_embedding/cache/china_v1_retrospective_2020_2021/olmo_ready/pilot_200
```

**通过标准：** 200 patch 中每个有效月份都有正确的 12-band S2、2-band S1、mask 和时间；缺失明确标 mask，不用零值伪装有效观测。

**常见坑与处理：**

- **坑：补零能跑但语义是错的。** B01/B09 是真实通道；补零会产生训练时从未见过的域偏移。
- **坑：S1 直接输入线性功率。** 官方归一化通常假设 dB；尺度错会让整个 backbone 表征失真。
- **坑：只看 RGB。** 必须检查每个波段、mask、数值分位数和时间字段。
- **坑：200 patch 修复成功就假设 62k 可取。** 2k 阶段要单独统计数据源覆盖率和下载配额。

**Commit:** `data: build traceable OlmoEarth-ready 200-patch inputs`

---

## Task 7：建立隔离的 OlmoEarth 昇腾环境，先过 4 patch 算子闸门

**目的：** 不污染现有 torch 2.6 生产环境，并尽早发现 Ascend 不支持的算子。

**Files:**

- Create: `environments/olmoearth_feature_extractor/README.md`
- Create: `environments/olmoearth_feature_extractor/requirements.lock`
- Create: `environments/olmoearth_feature_extractor/model_revision.json`
- Create: `scripts/features/precompute_olmo_quarterly_features.py`
- Create: `tests/test_olmo_quarterly_feature_contract.py`
- Create: `tests/npu/test_olmoearth_4patch_npu.py`

### 实施步骤

1. 在 `/data/wwj_torch21/conda/envs/olmoearth_npu` 建独立环境；不修改当前 `/data/wwj_torch21/conda/envs/torch26`。
2. 固定 OlmoEarth minimal 代码 revision、权重 hash、torch/torch_npu/CANN 版本。
3. 项目核心代码禁止 `einops`；适配器用 PyTorch 原生 reshape/permute 实现等价操作。
4. 每个年份一次 T=12 前向，保留月度 token，再按 1—3、4—6、7—9、10—12 月做 mask-aware pooling；禁止使用默认全年一次 pooling 后复制成四季度。
5. 先测 4 patch：Tiny P8/P4、Small P4，FP32/BF16；记录峰值 HBM、延迟、CPU fallback、NaN、重复运行确定性、CPU/NPU 余弦。
6. 只有 4 patch 通过，才对 200 patch 启动 6 个独立 worker。

**Run:**

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=2
/data/wwj_torch21/conda/envs/olmoearth_npu/bin/python \
  -m pytest tests/npu/test_olmoearth_4patch_npu.py -q
```

**通过标准：** 权重无 missing/unexpected keys；4 patch 全部无 NaN；BF16/FP32 相似度达到预设容差；没有不可接受的 CPU fallback；单 patch 两年延迟 `tau` 被真实记录。

**常见坑与处理：**

- **坑：直接升级现有环境。** 当前项目实际是 torch 2.6+torch_npu 2.6，而 Olmo minimal 依赖更新；原地升级可能让整个海淀生产线失效。
- **坑：pyproject 与实际环境不一致。** 当前 pyproject 仍写 torch<2.2、numpy<2，但实际为 torch 2.6、numpy 2.4；不要在本 Task 顺手大范围改依赖，先锁定独立提取环境。
- **坑：SDPA、bicubic、RoPE 某一步回 CPU。** 4 patch 必须用 profiler 看算子位置，不能只看最终能否输出。
- **坑：默认时间池化。** 如果先把全年池成一个向量再复制四次，季度产品从根上就是假的。

**Commit:** `feat: add isolated OlmoEarth NPU extractor and 4-patch gate`

---

## Task 8：200 patch 选基座，并冻结季度特征缓存

**目的：** 在速度、效果和缓存空间之间选一个主基座，不凭模型大小做决定。

**Files:**

- Create: `scripts/features/seal_quarterly_feature_cache.py`
- Create: `scripts/qa/audit_quarterly_feature_cache.py`
- Create: `tests/test_quarterly_feature_cache.py`
- Create: `docs/benchmarks/china_v1_olmo_selection_200.md`

### 比较组

1. B0 raw baseline；
2. Olmo Tiny P8；
3. Olmo Tiny P4；
4. Olmo Small P4；
5. T=3 独立季度输入，仅作消融；
6. T=12 年度回溯、按季度池化，作为正式语义。

### 评测

- building/road/water 5/10/50-shot 的 F1、AP、AUC；
- boundary F1、跨区表现；
- 季度变化保真、稳定区误报、云雾鲁棒性；
- `tau`、峰值 HBM、缓存 GB、六 worker 的有效利用率；
- P4 overlap seam 分数。

**Run:**

```bash
python -m pytest tests/test_quarterly_feature_cache.py -q
bash scripts/features/launch_china_v1_feature_workers.sh \
  --registry /data/xuannv_embedding/cache/china_v1_retrospective_2020_2021/registries/pilot_200.json \
  --models tiny_p8 tiny_p4 small_p4
python scripts/features/seal_quarterly_feature_cache.py \
  --root /data/xuannv_embedding/cache/china_v1_retrospective_2020_2021/backbone/pilot_200
```

**通过标准：** 默认优先选择 Tiny P4；只有 Small P4 在同协议下带来稳定且有意义的提升，才承担约 2 倍缓存与更长推理成本。任何候选若比 B0 核心任务平均下降或季度变化被压平，则不进入 2k。

**常见坑与处理：**

- **坑：Small 一定更好。** 需用下游指标和跨区指标决定。
- **坑：只看均值吞吐。** 要记录 P50/P95、失败重试和写盘时间。
- **坑：保存所有月 token。** 正式缓存只保留 8 季度特征和有效月计数；原始月 token 只在小型调试集保存。

**空间预算：** 62k 的 Tiny P4 BF16 季度缓存约 182 GiB，预留 195—220 GB；Small P4 约 363 GiB，预留 390—440 GB。首版只允许一个主缓存长期保留。

**Commit:** `experiment: select and seal 200-patch quarterly backbone cache`

---

## Task 9：取得并校验 2020/2021 年度 AEF 缓存

**目的：** 把 AEF 当年度教师和低秩语义旁路，不当成可直接复制的最终答案。

**Files:**

- Create: `scripts/data/export_aef_annual_2020_2021.py`
- Create: `src/xuannv_embedding/data/aef_annual_cache.py`
- Create: `tests/test_aef_annual_cache.py`
- Create: `docs/data/china_v1_aef_2020_2021_contract.md`

### 本机现状

本地 `aef-labels.zip` 只有 evaluation 数据和 training coordinates，**没有** 2020/2021 全国 AEF64。现有官方 AEF 缓存是海淀/哈尔滨 2025 年，不能拿来代替。

### 实施步骤

1. 先只取 200 patch 的 2020/2021 官方 AEF 窗口；GCS provider-pays 需要 billing project，或通过 Earth Engine 导出。
2. 保存官方版本字段、年份、源 URI、checksum、CRS、transform、许可与 attribution。
3. 正确处理 int8：`-128` 是 NoData；有效值按官方公式解量化，空间插值后重新单位化。
4. 每年只产生一个 `64×128×128` 年度向量，不复制存四份；训练时只做 view/expand。
5. 完成 200 → 2k → 10k 后再取 62k；62k 两年 int8 约 130 GB。

**Run:**

```bash
python -m pytest tests/test_aef_annual_cache.py -q
python scripts/data/export_aef_annual_2020_2021.py \
  --registry /data/xuannv_embedding/cache/china_v1_retrospective_2020_2021/registries/pilot_200.json \
  --years 2020 2021 \
  --output-root /data/xuannv_embedding/cache/china_v1_retrospective_2020_2021/aef
```

**通过标准：** 年份不串、网格误差小于 0.5 像素、NoData 不参与插值、有效向量单位范数统计正确。

**常见坑与处理：**

- **坑：把 `aef-labels.zip` 当 embedding。** 它不是年度稠密向量。
- **坑：对 int8 直接线性缩放。** 官方是带符号的非线性解量化，并需处理 `-128`。
- **坑：AEF 可见时又用它做 teacher target。** 这会形成抄答案捷径；teacher loss 只能在 raw-only view 上计算。
- **阻塞处理：** 如果 billing/Earth Engine 暂未开通，B0、Olmo 与高分分支继续推进；AEF 组不允许用假数据占位。

**Commit:** `data: add verified 2020-2021 annual AEF patch cache`

---

## Task 10：登记用户的 2 米一张图，并生成 D32 年度结构缓存

**目的：** 用 2 米影像补边界、纹理和小目标，不让供应商色彩或接缝成为语义。

**Files:**

- Create: `scripts/data/inventory_china_v1_highres.py`
- Create: `scripts/data/precompute_annual_highres_structure.py`
- Create: `src/xuannv_embedding/data/highres_structure_cache.py`
- Create: `tests/test_highres_structure_cache.py`
- Create: `docs/data/china_v1_highres_2020_2021_contract.md`

### 本机现状

项目路径中没有发现明确登记的“全国 2020/2021 高分 2 米”数据。现有海淀高分是 2025—2026 年，ModelScope 正下载的是 2023 年 1 米静态资产，都不能替代用户所说的两年度 2 米数据。因此本 Task 的第一个输入是**用户挂载或提供本机路径**，不是重新下载数据。

### 实施步骤

1. 只登记路径、许可、年份、传感器、正射方式、波段、GSD、nodata、云影、镶嵌接缝和 acquisition date；不复制原始大图。
2. 先抽 12 patch 检查 CRS、分辨率和半像素偏移，再扩到 200。
3. 每个 1280 m patch 读取 `640×640` 的 2 m 数据，计算 10 m 网格上的 D32 结构特征：均值/方差、x/y 梯度、边缘密度、纹理、有效率、接缝强度、配准置信度等。
4. 每个年份独立缓存；如果只有单年度 2 米图，不能同时广播给 2020 和 2021。
5. 62k 两年 D32 FP16 约 130 GB；不优先保存 D64 或全量原始 patch copy。

**Run:**

```bash
python -m pytest tests/test_highres_structure_cache.py -q
export HIGHRES_SOURCE_ROOT=/data2/china_highres_2m_2020_2021
python scripts/data/inventory_china_v1_highres.py \
  --source-root "${HIGHRES_SOURCE_ROOT}" \
  --output /data/xuannv_embedding/cache/china_v1_retrospective_2020_2021/registries/highres.json
python scripts/data/precompute_annual_highres_structure.py \
  --registry /data/xuannv_embedding/cache/china_v1_retrospective_2020_2021/registries/pilot_200.json \
  --highres-registry /data/xuannv_embedding/cache/china_v1_retrospective_2020_2021/registries/highres.json
```

**通过标准：** 稳定边缘配准残差不超过 5 m；年份、传感器和许可可追溯；接缝和云影有 mask；D32 能恢复局部边界但不能单独高精度预测年份/供应商。

**常见坑与处理：**

- **坑：高分图看着清楚就一定有用。** 镶嵌接缝、锐化、匀色很容易被模型当语义。
- **坑：直接重建 RGB。** 容易学供应商颜色；首版学习边缘和局部关系，不要求复原原图。
- **坑：先双线性缩到 10 m。** 会在模型学习前丢失 2 米结构；应先在 2 米网格提取结构，再聚合到 10 m。
- **阻塞处理：** 高分未挂载时不阻塞 B0/Olmo/AEF；只跳过 `+HR` 和 full fusion 组。

**Commit:** `data: add traceable annual 2m structure cache pipeline`

---

## Task 11：实现可撤销的季度融合模型

**目的：** 保持动态基座为主，AEF 和高分只能提供受限增量；关闭旁路时必须退化回 base-only。

**Files:**

- Create: `src/xuannv_embedding/data/quarterly_feature_registry.py`
- Create: `src/xuannv_embedding/data/quarterly_feature_dataset.py`
- Create: `src/xuannv_embedding/data/quarterly_collate.py`
- Create: `src/xuannv_embedding/models/quarterly_fusion.py`
- Create: `src/xuannv_embedding/training/quarterly_losses.py`
- Create: `scripts/train/train_quarterly_fusion.py`
- Create: `tests/test_quarterly_feature_registry.py`
- Create: `tests/test_quarterly_feature_dataset.py`
- Create: `tests/test_quarterly_fusion_model.py`
- Create: `tests/test_quarterly_losses.py`
- Create: `configs/national/china_v1_quarterly_fusion_200_sanity_20260815.yaml`

### 模型结构

```text
冻结季度基座 D(y,q)
        │
        ├── 年度 AEF 64D → bias-free 1×1 压到 16D ─┐
        │                                            ├→ StaticMixer → 有界门控残差
        └── 年度 2m D32 → 轻量结构 adapter ─────────┘
                                                     ↓
                                          64D vMF，128×128
```

具体约束：

- AEF 禁止存在 `64→64→最终输出` 的直通路；先压到 16 维。
- AEF gate 和高分 gate 独立，初值严格为 0，使用有界 `tanh`。
- 初始 side residual 范数不超过 base 特征的 25%，消融 10%/25%/50%。
- batch 固定包含 raw-only、raw+AEF、raw+HR、full 四种 view，各约 25%。
- 关闭 AEF/HR 时明确跳过分支，不能只传一张全零图。
- 输出始终 `B×4×64×128×128`；vMF 后单位范数。

### 损失

1. 季度原始观测重建为主损失，权重 1.0。
2. AEF teacher 只作用于 raw-only 四季度的年度聚合，cosine loss 初始权重 0.05—0.10。
3. 高分使用边缘和局部关系损失，不重建供应商 RGB。
4. 动态保真：融合后季度差分至少保留 raw-only 80% 的变化幅度。
5. 前 10% step 不启用 AEF 融合；10%—30% 线性引入；最后 20% 教师权重减半。

**Run:**

```bash
python -m pytest \
  tests/test_quarterly_feature_registry.py \
  tests/test_quarterly_feature_dataset.py \
  tests/test_quarterly_fusion_model.py \
  tests/test_quarterly_losses.py -q
python scripts/train/train_quarterly_fusion.py \
  --config configs/national/china_v1_quarterly_fusion_200_sanity_20260815.yaml \
  --device npu:2
```

**通过标准：** 零 gate 输出与 base-only 在容差内一致；任意一个年度分支缺失仍可运行；分支顺序不影响输出；关闭旁路可恢复 B0/Olmo base 表现。

**常见坑与处理：**

- **坑：AEF 一接入，训练 loss 很快下降。** 这往往是抄 teacher；检查 raw shuffle、AEF shuffle 和 gate 范数。
- **坑：四季度越来越像。** 监控同位置季度 cosine、差分幅度和 change AUROC；低于 80% 动态保真即退回更小 gate。
- **坑：concat 后交给大 MLP。** 它不可解释也难回退；首版必须使用分支独立、零初始化残差。

**Commit:** `feat: add reversible annual-side quarterly fusion model`

---

## Task 12：200 patch 四组消融，决定是否进入 2k

**目的：** 用同一协议回答“旁路到底有没有用”。

**Files:**

- Create: `configs/national/china_v1_quarterly_base_200_20260815.yaml`
- Create: `configs/national/china_v1_quarterly_base_aef_200_20260815.yaml`
- Create: `configs/national/china_v1_quarterly_base_hr_200_20260815.yaml`
- Create: `configs/national/china_v1_quarterly_full_200_20260815.yaml`
- Create: `scripts/eval/evaluate_china_v1_quarterly.py`
- Create: `docs/reports/china_v1_quarterly_200_gate.md`

### 四组必须同时做

| 组 | 输入 | 回答的问题 |
|---|---|---|
| Base | 季度基座 | 基础效果 |
| Base+AEF | 季度基座+年度 AEF | 年度语义是否有效 |
| Base+HR | 季度基座+年度 2 m 结构 | 边界与小目标是否改善 |
| Full | 三者 | 增量是否互补 |

所有组必须使用同样标签、空间 fold、shot、阈值选择和随机种子。报告 full-label 与 5/10/50-shot，不能混写。

### 进入 2k 的门槛

- 三个核心任务 5/10-shot 平均 F1 相对 base 提升至少 2 个百分点；
- 任一核心任务下降不超过 1 个百分点；
- raw-only dropout 表现相对原 base 下降不超过 1 个点；
- 融合后季度变化幅度不低于 base 的 80%；
- 稳定区没有明显接缝/云影伪变化；
- 两个随机种子方向一致。

**常见坑与处理：**

- **坑：200 patch 指标波动大。** 它是淘汰明显错误方案，不用于宣布全国领先；必须看两个种子和可视化。
- **坑：只汇报 Full。** 没有四组消融就不能知道提升来自哪里。
- **坑：只看平均 F1。** 道路和边界很容易被平均值掩盖，要单列每任务和 boundary F1。

**Commit:** `experiment: report 200-patch China V1 quarterly fusion gate`

---

## Task 13：2,000 patch 可信结果门

**目的：** 验证效果不是 200 patch 偶然，并校准真实吞吐和缓存增长。

**Files:**

- Create: `configs/national/china_v1_quarterly_fusion_2000_20260815.yaml`
- Create: `docs/reports/china_v1_quarterly_2000_gate.md`

### 实施步骤

1. 扩展 Olmo-ready 数据、AEF、HR 三类缓存到同一固定 2k registry。
2. 特征生成中每个 shard 单独 checksum、done marker 和失败清单。
3. 重做 Base/Base+AEF/Base+HR/Full，至少 3 个随机种子。
4. 增加跨区域、气候带、城市/农村、完整月/缺月分组。
5. 用 2k 实测得到：单 patch 两年 `tau`、六 worker 效率、每 TB 实际增长、融合训练 step/s。

**通过标准：** 200 patch 的主要提升在 2k 仍成立；没有明显地域性失败；缓存能中断恢复；实测资源在本机磁盘预算内。

**常见坑与处理：**

- **坑：只扩大训练集，不扩大验证地域。** 2k 必须提高空间多样性。
- **坑：中途改 preprocessing。** 任一处理变化都产生新 registry hash，不得覆盖旧 cache。
- **坑：六 worker 同时写盘拖垮下载任务。** ModelScope 下载和旧物化仍运行时限并发或排队执行。

**Commit:** `experiment: validate 2000-patch quarterly fusion representation gate`

---

## Task 14：10,000 patch 系统门

**目的：** 在正式 62k 前验证 24—48 小时连续运行、断点恢复和磁盘生命周期。

**Files:**

- Create: `configs/national/china_v1_quarterly_fusion_10000_20260815.yaml`
- Create: `scripts/qa/verify_china_v1_resume.py`
- Create: `docs/reports/china_v1_quarterly_10000_system_gate.md`

### 实施步骤

1. 使用与 62k 完全相同的目录、shard 大小、manifest 和命名。
2. 主动终止一个 feature worker，确认重启只重算 partial shard。
3. 主动从 checkpoint 恢复融合训练，检查 global_step、optimizer、AMP scaler、RNG、sampler 和 registry hash。
4. 连续跑 24—48 小时，记录 NPU 利用率、HBM、CPU、I/O、失败 patch、P95 延迟。
5. 随机复算 1% patch，检查 checksum 或数值容差一致性。
6. 比较 1/2/6 卡训练，融合头较轻，6 卡未必比 2 卡高效。

**通过标准：** 无数据丢失；恢复后指标和 loss 连续；NPU 无长期空转；失败率可解释；磁盘剩余量满足 62k 计划。

**常见坑与处理：**

- **坑：能恢复 checkpoint，但数据顺序变了。** sampler/RNG/registry hash 都必须进 checkpoint。
- **坑：partial shard 被当完成。** 只有 checksum 与完整性审计通过后才能写 done marker。
- **坑：用峰值吞吐估算 62k。** 使用 24—48 小时的 P50/P95 与有效利用率。

**Commit:** `experiment: validate 10000-patch quarterly fusion system gate`

---

## Task 15：约 62,000 patch 正式缓存与训练

**目的：** 在本机磁盘可承受范围内完成训练规模版本，而不是直接做全国产品。

**Files:**

- Create: `configs/national/china_v1_quarterly_fusion_62000_20260815.yaml`
- Create: `scripts/production/export_china_v1_quarterly_embedding.py`
- Create: `docs/production/china_v1_quarterly_model_card.md`
- Modify: `docs/checkpoint_registry.md`
- Modify: `docs/data_inventory.md`
- Modify: `CHANGELOG.md`

### 62k 本机空间预算

| 项目 | 预算 |
|---|---:|
| Tiny P4 八季度 BF16 缓存 | 195—220 GB |
| 两年 AEF int8 cache | 约 130 GB |
| 两年 HR D32 FP16 cache | 约 130 GB |
| raw-quarter/Zarr 工作缓存 | 80—150 GB，以 10k 实测回标 |
| checkpoint、日志、QA、临时 shard | 50—100 GB |
| 62k 最终 8 季度 int8 embedding | 约 520 GB |
| 合计增量 | 约 1.1—1.25 TB |

这在当前 `/data` 约 1.4 TB、`/data2` 约 2.6 TB 空闲下理论可行，但两个盘仍有外部下载和旧任务写入。因此 62k 启动前必须重新审计，并给上述目录做明确空间预留。

### 实施步骤

1. 主缓存只保留最终选中的一个 backbone；Small/Tiny 失败候选和过期 partial cache 清单化后再做可恢复清理。
2. 六卡独立特征分片，每卡约 10,334 patch；每 shard 完成立即封口和校验。
3. 融合训练按固定优化步和 early stop，不照搬海淀 800 epoch。
4. 最终优先导出 int8；FP16 只保留固定评测子集，避免额外约 1.04 TB。
5. 模型卡明确 annual-retrospective 语义、数据年份、AEF/Olmo 许可、缺月策略、side branch 是否可用。

**时间公式：**

单卡完成一个 patch“两年 Olmo 前向 + 八季度池化 + 写盘”的实测秒数记为 `tau`，六卡有效率按 `0.8`：

```text
总小时 = patch数 × tau / (6 × 3600 × 0.8)
```

| tau | 62k 特征生成 |
|---:|---:|
| 30 秒 | 约 4.5 天 |
| 60 秒 | 约 9.0 天 |
| 120 秒 | 约 17.9 天 |

Tiny P4 先按 30—60 秒规划，即 4.5—9 天；加数据修复、缓存、融合训练和 QA，正式 62k 阶段预算 8—15 天。

**通过标准：** 62k cache registry 完整；固定代表区和下游指标通过；checkpoint 可恢复；int8 与 FP16 评测子集差异在容差内；磁盘未越过水位。

**常见坑与处理：**

- **坑：训练很快就等于全国制图很快。** 62k 是训练样本；全国约 574 万 patch 是另一个数量级。
- **坑：全量同时留 FP16 和 int8。** 本机不应这样做；FP16 只留评测子集。
- **坑：旧缓存舍不得删。** 每个阶段必须按 registry 和 checksum 管理缓存生命周期，但删除前要生成清单并确认可重建。

**Commit:** `experiment: train and register China V1 quarterly fusion on national samples`

---

## 3. 排期：结合当前机器负载的现实版本

### 理想条件

前提：OCR 高 CPU 任务结束、ModelScope 下载稳定或完成、AEF 访问已开通、用户的 2020/2021 高分路径已挂载。

| 阶段 | 墙钟时间 | 交付物 |
|---|---:|---|
| 本机审计、registry、I/O 基准 | 1—2 个工作日 | 可复现环境和 200 patch 清单 |
| B0 基线与 Olmo 4 patch | 2—3 个工作日 | raw baseline、算子可行性 |
| 修复 200 数据、三模型选型 | 3—5 个工作日 | 选定 Tiny/Small 与 tau |
| AEF/2m 200 cache + 融合消融 | 2—4 个工作日 | Base/+AEF/+HR/Full 报告 |
| **200 patch 可跑可信版** | **8—12 个工作日** | 可展示、可评测的 V0 |
| 2k 表示门 | 再 3—5 天 | 可信精度结论 |
| 10k 系统门 | 再 4—7 天 | 62k 排期依据 |
| 62k 正式阶段 | 再 8—15 天 | 训练规模 V1 |
| **从现在到 62k V1** | **约 5—8 周** | 视数据修复和外部访问而定 |

### 以当前负载立即开工

- 今天可以做：Task 1—3 的代码、清单和低并发抽检，B0 代码框架，隔离环境准备。
- 暂不应做：大规模 ZIP 解压、全量 Zarr 重打包、6 worker CPU 密集数据修复。
- 当前 OCR 若持续占满 CPU，整体增加约 1—3 天；若高分路径或 AEF 访问迟迟不到，只交付 Base/Olmo 版本，旁路组顺延，但项目不应整体停住。

---

## 4. 每道闸门的停止/回退规则

### 4 patch 停止条件

- Olmo 权重无法无误加载；
- 核心算子长期 CPU fallback；
- BF16 出现 NaN 或与 FP32 差异不可接受；
- 单 patch HBM 超限或延迟明显不可生产。

回退：先用 Tiny P8；仍失败则交付 B0，并单独解决 Olmo Ascend 适配。

### 200 patch 停止条件

- Olmo 不优于 B0，或优势小于其数据修复成本；
- AEF/高分加入后季度变化幅度低于 base 的 80%；
- 接缝、云影成为明显变化热点；
- 关闭旁路不能恢复 base-only 性能。

回退：缩小 gate、去掉有害 branch，或只交付 Base+单一有效 branch。

### 2k 停止条件

- 提升只存在于某一小区域；
- 缺月 patch 明显失效且覆盖率无法接受；
- 数据源重取配额无法扩展到 10k；
- 缓存版本不可追溯或恢复不稳定。

回退：调整采样/数据源策略，不进入 10k。

### 10k 停止条件

- 24 小时内出现未处理的数据损坏、HCCL 不可恢复或磁盘越水位；
- 6 卡大量空转且 I/O 无法改善；
- 主动 kill 后不能精确恢复。

回退：降低并发、调整 shard/Zarr、比较 1/2/6 卡，再重做 10k 门。

### 62k 停止条件

- `/data` 低于 300 GB 或 `/data2` 低于 500 GB；
- NPU 0/1 被错误加入任务；
- 代码、权重、registry hash 与 10k 验证版本不一致；
- 下游核心任务较 2k/10k 出现不可解释退化。

---

## 5. 最容易被忽略的十个坑

1. **当前 10-band S2 不能直接喂官方 OlmoEarth。** 缺通道补零是最危险的“能跑但错”。
2. **S1 当前是 VH/VV 线性功率。** 官方常用 VV/VH dB，顺序和尺度都要改。
3. **AEF 本地 ZIP 不是年度 embedding。** 2020/2021 必须单独获取。
4. **2 米数据尚未在项目中登记。** 用户有数据不等于流水线知道路径、年份和许可。
5. **现有 6 卡启动脚本会撞 NPU 0/1。** 新启动器必须显式限定 2—7。
6. **Python 当前可能导入旧 worktree。** 每次运行都要记录 `xuannv_embedding.__file__`。
7. **`/workspace` 已 99% 满。** 一次环境安装就可能把系统盘写爆。
8. **当前 CPU 被 OCR 抢满。** 现在测出的低吞吐不代表硬件上限，也不适合立即重打包全量数据。
9. **年度 side 信息会抹平季度变化。** 必须有 raw-only view、独立 dropout 和动态差分保真损失。
10. **62k 训练完成不等于全国结果完成。** 全国产品要 49 TB 以上输出和更大集群。

---

## 6. 实施时的 Git 与验证纪律

每个 Task 都遵循：

1. 先写失败测试；
2. 做最小实现；
3. 运行该 Task 的定向测试；
4. 运行受影响的旧测试，尤其是 `tests/test_model.py`、`tests/test_dataset.py`、`tests/test_training.py`；
5. 记录一次实际命令和结果；
6. 只 stage 本 Task 文件，不碰用户已有未提交改动；
7. 立即 commit；
8. 立即 push 到当前分支；
9. 向用户汇报完成文件、测试、commit 和下一步。

现有海淀生产模型、`models/model.py`、旧月度 Dataset 与生产配置首版保持不动。季度融合用独立路径实现，等 10k 门通过后再考虑 model/data factory 的公共集成。

---

## 7. 最终推荐决策

本机首版选择应是：

> **Tiny P4 年度 T=12 回溯季度基座 + AEF 低秩年度语义旁路 + 2 米 D32 年度结构旁路 + 零初始化有界门控 + raw-only 动态保真。**

但这个选择是“默认候选”，不是预先宣布的赢家。真正顺序是：

```text
先做 B0
  ↓
修好 12-band S2 / S1 dB
  ↓
4 patch 判断 Olmo 能不能在 910B4 正确跑
  ↓
200 patch 决定 Tiny P8 / Tiny P4 / Small P4
  ↓
AEF 和 2m 分别做可关闭增量
  ↓
2k 证明效果，10k 证明系统，62k 才正式训练
```

这样最快拿到一版，也最不容易因为某一个外部数据、模型环境或 side branch 出错而把整个项目拖死。

---

## 8. 与总方案的关系

本计划是本机执行版，负责设备、磁盘、环境、数据可用性和逐级闸门。模型原理、年度回溯语义和通用实验设计见：

- `docs/superpowers/plans/2026-08-15-china-v1-retrospective-quarterly-fusion.md`

若两份计划出现冲突，以本机实测闸门和本计划的设备/路径限制为准；如果将来迁移到新集群，应重新生成机器执行计划，而不是照抄这里的 NPU 编号和磁盘路径。
