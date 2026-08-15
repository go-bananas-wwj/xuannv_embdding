# China V1 隔离三路融合冒烟证据报告（2026-08-15）

## 主结论

**工程冒烟结论：PASS。精度、语义质量与生产可用性：NOT EVALUATED。**

当前 corrected seal 同时满足以下条件：`SUCCESS` 存在且是 current sandbox 最晚写入的
证据，fresh `verify_success` 通过，四组 Zarr 与所有强制 evidence 均通过封口校验，323 个
digest roots 展开为 319 个唯一叶文件，322 个 path-audit 条目全部被覆盖（uncovered=0）。
因此，本报告仅判定以下工程合同通过：真实季度 S1/S2 的只读加载、四种分支组合的单卡
前向、受控反向、零门控可撤销性、checkpoint 恢复、确定性复算、FP16 Zarr 导出和隔离封口。

> **必须保留的用途边界：年度 AEF 与所谓“2 m”旁路输入均为 synthetic。** 模拟 AEF
> 不具备官方 AEF 语义；模拟高分影像由 Sentinel-2 确定性上采样得到，不含真实 2 m
> 信息。本次结果不支持任何精度、2 m 质量、全国生产能力、真实 AEF 语义、正式训练或
> 正式评测结论。所有 evidence 均声明 `formal_training_allowed=false`、
> `formal_evaluation_allowed=false`，且 accuracy conclusion 不被允许。

## 定义与范围

本次 `PASS` 是 **engineering PASS**，定义为计划中的 shape、finite、门控、梯度、恢复、
复现、设备映射、路径隔离与 evidence closure 全部通过；它不是模型效果 PASS。测试使用
真实的 2020—2021 月度 Sentinel-1/Sentinel-2，但旁路数据和极小季度融合模型只服务于
smoke contract，不是 China V1 产品或现有 `AEFModel` 的生产集成。

| 范围项 | 实际值 |
| --- | --- |
| Patch IDs | `parent_32643:310:3383`；`parent_32643:311:3390`；`parent_32643:312:3346`；`parent_32643:312:3406` |
| 时间轴 | `2020Q1`、`2020Q2`、`2020Q3`、`2020Q4`、`2021Q1`、`2021Q2`、`2021Q3`、`2021Q4` |
| 真实源数据 | S1/S2，4 patches × 2 years = 8 patch-years；每个 patch 48 个 month references |
| 旁路数据 | 年度 AEF：synthetic；年度高分：synthetic，`contains_real_2m_information=false` |
| 分组 | `base`、`base_aef`、`base_highres`、`full` |
| 运行约束 | 单张物理 NPU 2，进程内唯一逻辑设备 `npu:0`；最多 2 个受控 optimizer steps |

## 方法与证据

### 数据选择与只读源审计

`patch_selection.json` 固定了 4 个唯一 patch、192 个 ZIP member/CRC references 和 48 个
唯一 source archives。prepare 前后快照均覆盖 48 个 archives、55 个有界相关目录和 253
个 direct entries；两份 JSON 字节完全一致，`source_unchanged=true`，并明确记录
`hashing_performed=false`。因此这里的“unchanged”严格表示受监测的类型、大小和纳秒级
mtime 清单未变，不扩大解释为对数百 GB 源 ZIP 做了内容哈希。

### 运行环境与设备映射

| 项目 | Sealed provenance |
| --- | --- |
| 执行 Git commit | `0ef6e6b513ef4811a14a89e50e63bd8ea050c8a4` |
| Python | 3.11.15；fixed sandbox venv interpreter |
| PyTorch / torch-npu | 2.6.0+cpu / 2.6.0.post5 |
| CANN | 9.0.0，root `/usr/local/Ascend/cann-9.0.0` |
| Driver | 26.0.rc1 |
| NPU mapping | `/dev/davinci2`（physical 2）→ only visible `npu:0`（logical 0） |
| Device | `Ascend910B4-1`；logical device count = 1 |
| CPU fallback | `manifests/cpu_contract.json`，SHA-256 `2bfc300c7d3ce98393ef3f11e46221ce4a4364a27e4f0dcbf9218262323cae22` |

### 四组持久化 Zarr 合同

下表来自封口后独立重开。Logical bytes 是未压缩数组大小：64 MiB embedding + 0.5 MiB
valid mask；on-disk bytes 是 Zarr 目录当前实际占用，不应与 logical bytes 混用。

| Group | Embedding | Valid | Finite | Logical bytes | On-disk bytes |
| --- | --- | --- | --- | ---: | ---: |
| `base` | FP16 `[4,8,64,128,128]` | bool `[4,8,1,128,128]` | true | 67,633,152 (64.5 MiB) | 47,935,246 (45.71 MiB) |
| `base_aef` | FP16 `[4,8,64,128,128]` | bool `[4,8,1,128,128]` | true | 67,633,152 (64.5 MiB) | 47,935,250 (45.71 MiB) |
| `base_highres` | FP16 `[4,8,64,128,128]` | bool `[4,8,1,128,128]` | true | 67,633,152 (64.5 MiB) | 47,935,254 (45.71 MiB) |
| `full` | FP16 `[4,8,64,128,128]` | bool `[4,8,1,128,128]` | true | 67,633,152 (64.5 MiB) | 47,935,246 (45.71 MiB) |

四组的 patch axis 与季度 axis 完全相同，且所有 embedding 值均有限。

### 融合、反向、恢复与复现

| Gate / recovery check | 结果 |
| --- | ---: |
| Full/Base zero-gate max absolute error | `0.0` |
| AEF adapter gradient L1（gate override 0.1） | `0.2702299631200731` |
| High-resolution adapter gradient L1 | `0.4636795846745372` |
| High-resolution stem gradient L1 | `0.15332698542624712` |
| Output projection gradient L1 | `25.318119764328003` |
| Checkpoint reload max absolute error | `0.0` |
| Deterministic rerun | `matches=true`，max absolute error `0.0`，seed `20260815` |
| Checkpoint SHA-256 | `4960b82b90fb89645099c46649fd6a0ffc722aab8561688e80d7978d7b1a27f9` |

所有记录的 gradient L1 均为有限非零值。`gate_override=0.1` 仅用于证明反向链路连通，
不是训练结果，也不改变零初始化的生产假设。

### vMF 范数：FP32 模型输出与重开 FP16 产物

| Evidence | Source dtype | Computation dtype | Min | Median | Max |
| --- | --- | --- | ---: | ---: | ---: |
| 导出前模型输出（四组相同） | float32 | float32 | 0.9999996424 | 1.0000000000 | 1.0000003576 |
| 重开持久化 Zarr（四组相同） | float16 | float32 | 0.9998238087 | 0.9999999404 | 1.0001806021 |

FP16 行是在封口后重开实际 Zarr，再以 FP32 累加复算；量化后的轻微偏差与导出前 FP32
检查明确分开，未将二者混报。

### 单次延迟与 NPU 峰值内存

| Group | One-shot latency (s) | Peak allocated | Peak reserved |
| --- | ---: | ---: | ---: |
| `base` | 5.260446 | 695,546,880 B = 663.33 MiB = 0.648 GiB | 1,025,507,328 B = 978.00 MiB = 0.955 GiB |
| `base_aef` | 0.653409 | 1,165,312,512 B = 1,111.33 MiB = 1.085 GiB | 1,434,451,968 B = 1,368.00 MiB = 1.336 GiB |
| `base_highres` | 0.666613 | 1,187,988,992 B = 1,132.95 MiB = 1.106 GiB | 1,434,451,968 B = 1,368.00 MiB = 1.336 GiB |
| `full` | 0.007950 | 1,255,098,880 B = 1,196.96 MiB = 1.169 GiB | 1,434,451,968 B = 1,368.00 MiB = 1.336 GiB |

这些是固定执行顺序下每组各一次的 smoke observability 值。首组 `base` 包含 cold-start /
编译影响，后续组还受缓存和共享计算路径影响；**禁止对四个 latency 做横向性能比较，也不
据此推断分支速度。** 本报告不绘制柱图：仅有 4 个 one-shot 分组且首组 cold-start，图形
会放大不可比差异并造成误导。峰值 allocated/reserved 在每组同步、清缓存并重置 peak
statistics 后记录，但仍不是正式容量或性能 benchmark。

### Seal closure、沙箱与主工作区

| Evidence | 实际值 |
| --- | --- |
| SUCCESS | `sealed_at_utc=2026-08-15T12:17:44.025102Z`；current sandbox 最新写入 |
| Combined SHA-256 | `408993be3d507d4f604fd05a26a671026419c9975374ba403a7612b32860efab` |
| Digest closure | 323 roots；319 unique leaf files；322 audit entries；uncovered = 0 |
| Path audit | stage `finalize-seal`；history `npu-smoke → finalize-seal`；`.partial` = 0 |
| Sandbox size | 292,698,261 bytes = 279.14 MiB = 0.273 GiB，低于 5 GiB stop limit |
| 主工作区 | commit `4cf032a6215fa9c573c998a0ea6daa005b48fb96`，branch `v3-semantic-64d`，ahead/behind `0/0`；既有 2 modified + 5 untracked 状态未变 |

第一次 seal 已被复核否决并作为诊断材料保留在
`attempts/rejected_20260815_first_seal/`。该归档不属于 current evidence、不参与 corrected
seal 的结论，也不能替代本报告所引用的 current `SUCCESS`。

## 稳健性、回归与限制

Fresh read-only `verify_success` 已通过，并重新得到同一 combined SHA-256。Task 8 在未
设置 NPU opt-in 环境变量、未触发 NPU 的前提下运行指定全回归：`194 passed, 1 skipped in
46.36s`。唯一 skip 是刻意要求显式 opt-in 的 NPU integration test；其物理 NPU 2 运行
证据已由 corrected Task 7 seal 提供。`ruff check experiments/china_v1_fusion_smoke
tests/isolated_smoke` 返回 `All checks passed!`，`git diff --check` 退出码为 0。

主要限制如下：

1. AEF 和高分旁路均为 synthetic；没有真实 AEF 语义或真实 2 m 空间细节。
2. 只有 4 个 patch、8 个季度和一个固定 seed；没有地理代表性、统计功效或全国泛化证据。
3. 只执行工程连通所需的受控反向；没有损失收敛、精度、下游任务或生产稳定性评测。
4. latency 是固定顺序的一次测量，不能用于性能排序；峰值内存也不是长期训练容量结论。
5. 源数据“unchanged”基于受监测目录与 archive 的 size/mtime 快照，不是源 ZIP 内容哈希。
6. Task 6 留存两个非阻塞 minor：archive/header 使用至多 8 个有界进程；48 个 references
   均有 member/CRC，但 grid header 证据只做代表性记录。二者不改变本次 sealed smoke
   判定，但在扩容前应重新评估。

## 下一步与待回答问题

在不扩大本报告结论的前提下，下一阶段应逐项替换且每次只改变一个因素：

1. 用真实官方 AEF 2020/2021 替换 synthetic AEF，重新验证同一四组合同。
2. 用真实年度 2 m 数据替换 synthetic 上采样输入，并单独验证数据来源、GSD 与配准质量。
3. 在 4 patch 算子闸门后，再用真实季度基座替换极小 smoke base。
4. 工程合同再次通过后，另行 preregister 200 patch 的效果消融、精度指标和统计协议。
5. 在任何生产推进前回答：真实数据许可与 registry 如何封存、2 m/10 m 配准容差如何定义、
   cold/warm 性能如何重复测量、以及哪一组下游标签与 split 才能支持正式精度结论。

当前可交付的是可审计的工程 PASS；accuracy remains not evaluated。
