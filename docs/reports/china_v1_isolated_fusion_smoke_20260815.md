# China V1 隔离三路融合工程冒烟证据报告（2026-08-15）

## 主结论

**Synthetic engineering smoke：PASS。精度、真实 AEF 语义、真实 2 m 质量、正式训练与
正式评测：均未执行，也不得从本报告推断。**

最终 corrected seal 在物理 NPU 2 上完成一次固定前台运行，并通过 fresh
`verify_success`。它证明的范围仅包括：真实季度 S1/S2 的只读装载、synthetic 年度 AEF
与 synthetic 高分旁路的工程连通、四组前向、受控反向、零门控恒等、checkpoint
恢复、确定性复算、FP16 Zarr 导出及隔离证据封存。

> 年度 AEF 是由一份固定的全局 64×10 投影生成的 synthetic fixture，不是官方 AEF；
> “2 m”旁路是 patch-year-specific synthetic texture，不含真实 2 m 遥感信息。极小融合
> 模型也不是生产 `AEFModel` 或正式训练配置。全部 evidence 均声明
> `synthetic=true`、`formal_training_allowed=false`、
> `formal_evaluation_allowed=false`，且不允许 accuracy conclusion。

## 固定范围

| 范围项 | 实际值 |
| --- | --- |
| Patch IDs | `parent_32643:310:3383`；`parent_32643:311:3390`；`parent_32643:312:3346`；`parent_32643:312:3406` |
| 时间轴 | `2020Q1` 至 `2021Q4`，共 8 个季度 |
| 真实输入 | S1/S2，4 patches × 2 years；48 个唯一 source archives |
| Synthetic 旁路 | 8 个年度 AEF cache、8 个年度 high-resolution cache；AEF 固定全局投影 1 份 |
| 组合 | `base`、`base_aef`、`base_highres`、`full` |
| 设备 | 物理 NPU 2 → 唯一可见逻辑设备 `npu:0`；无 torchrun、nohup 或后台任务 |
| 运行代码 | `f36a4de49e263badea4a86b429fddbc3006d5953` |

## 隔离与安全合同

最终修复波 `86095fb`、`966737c`、`f36a4de`（及测试跟进 `f15a933`）补齐了以下合同：

- bootstrap 只接受固定 worktree 与固定 sandbox；在任何 mutation 前用 `lstat` 检查既有
  ancestor/root，并以排他、拒绝 symlink 的方式创建 sentinel；
- launcher 以前台 pipeline 运行，使用排他、no-clobber、no-follow 的安全日志写入器；
  `READY_TO_SEAL`、`TEE_COMPLETE`、最终日志与 `SUCCESS` 形成显式状态链；
- 所有可写入口在 `SUCCESS` 后 fail closed；finalizer 和 verifier 对重复调用、symlink、
  未声明路径、语义篡改及跨文件不一致 fail closed；
- 模型严格要求 4 quarters、每季度 3 months、S2 10 通道、S1 2 通道、单通道 bool mask、
  AEF 64 通道和精确 5H×5W high-resolution 输入；
- gate override 使用 straight-through 路径：前向门控值为 0.1，同时保留真实 gate 参数
  梯度；NPU evidence 强制两项 gate gradient 和所有分支梯度均为有限非零；
- FP32 vMF 容差为 `1e-5`，重开 FP16 Zarr 容差为 `5e-4`，并拒绝零向量与范数平方为零；
- 在 archive/cache 装载完成后、`set_device` 之前再次检查物理 NPU 2 空闲；每组输出先
  复制到 CPU 并释放 NPU `FusionOutput`，再读取下一组 HBM baseline；
- prepare 的不可变 preliminary audit、运行/配置/Git/选择/registry/cache/checkpoint/输出
  轴与 checksum 均在最终 semantic evidence graph 中交叉绑定。

宿主机无 `fuser`，因此设备空闲判断采用 fail-closed `/proc/<pid>/fd` 字符设备扫描。
最终 provenance 恰好记录两次成功空闲检查：`launcher_preload` 与
`post_load_pre_set_device`。运行期间物理设备 2 映射为唯一逻辑 `npu:0`，设备名为
`Ascend910B4-1`。

### 封口后非 NPU hardening

最终安全复审的 finalizer/verifier 与日志两路发现已由 `cbda7be` 关闭：

- `seal_success` 在完整 `SUCCESS.tmp` 已写入、仅原子 replace 中断时，可以在下一次调用
  中恢复；恢复前会重新校验临时文件的常规文件状态、精确 schema、synthetic/formal-use
  policy 及其 combined SHA-256 与当前 evidence 的一致性。残缺、伪造或语义不符的临时
  文件保持原样并 fail closed，不会被盲删、盲覆盖或升级为 `SUCCESS`；
- preliminary audit stages 必须精确为
  `prepare → cpu-contract → npu-smoke`，final audit stages 必须精确再追加
  `finalize-seal`；只保留末两阶段或插入额外阶段都会被拒绝；
- safe tee 对 launcher log fd 和 stdout destination 都采用 write-all 循环；合法短写继续
  写剩余字节，0、负数、非整数或超过 remaining length 的返回值全部 fail closed。

`cbda7be` 只修改 finalizer/verifier、安全日志复制器及其 CPU 回归测试，没有启动 NPU，
也没有改写 current sandbox evidence。物理 NPU 2 的唯一运行与所有数值 provenance 仍绑定
`f36a4de49e263badea4a86b429fddbc3006d5953`；current seal 已在收紧后的 verifier 下 fresh
复验通过，因此无需为这次非 NPU hardening 重跑物理设备。

后续 Critical TOCTOU 复审指出，`cbda7be` 虽然验证了 `SUCCESS.tmp` 内容，却仍在验证后按
路径执行 replace；攻击者可在两者之间替换临时路径，或在发布前抢先创建 `SUCCESS`。
非 NPU commit `13efe98` 进一步关闭该窗口：

- 通过固定 sandbox directory fd，以 `O_EXCL`、`O_NOFOLLOW` 创建临时文件，或以
  `O_NOFOLLOW` 只读恢复它；所有 schema/digest 校验都从同一个已打开 fd 读取；
- `fstat` 要求该 fd 始终指向单链接常规文件，并在发布后把目标的 device/inode 与已验证
  fd 精确比较，从而检测临时路径 swap；
- 使用 hard-link no-replace 原语发布 `SUCCESS`，并在每个目录状态转换后 `fsync`；并发
  winner 已存在时保留 winner、拒绝覆盖；
- 三类竞态回归分别覆盖：临时文件 open 前插入外部 symlink、验证后发布前替换临时路径、
  以及并发 `SUCCESS` winner 抢先发布。

`13efe98` 同样只涉及 finalizer 与 CPU 安全测试；未运行 NPU、未改写 current evidence。
收紧后的 fresh verifier 继续接受 `f36a4de` seal，因此物理运行 provenance 与结论边界均
不变。

## 数据只读证据

`patch_selection.json` 固定 4 个唯一 patch、192 个 ZIP member/CRC references 与 48 个
唯一 source archives。prepare 前后快照字节完全相同，SHA-256 均为
`540beb5bd1cdda2e746a5d0c8a6bddda67336324789d0b611db0aabb2a17349e`；快照覆盖 48 个
archives、55 个有界相关目录和 253 个 direct entries，`source_unchanged=true`。

这里的“未变化”只表示被监测路径的类型、大小和纳秒级 mtime 清单一致；证据明确记录
`hashing_performed=false`，没有把该结论扩大为源 ZIP 内容哈希。

## 四组持久化输出

封口校验会重开实际 Zarr。四组均满足：embedding 为 FP16
`[4,8,64,128,128]`，chunks 为 `[1,1,64,128,128]`；valid mask 为 bool
`[4,8,1,128,128]`；值有限，patch/period axes 精确一致。

| Group | On-disk bytes | One-shot latency (s) |
| --- | ---: | ---: |
| `base` | 47,935,246 | 5.495883 |
| `base_aef` | 47,935,250 | 0.640632 |
| `base_highres` | 47,935,254 | 0.661814 |
| `full` | 47,935,246 | 0.008031 |

首组包含 cold-start/编译影响，后续组受缓存与共享路径影响。这四个 one-shot latency 仅
用于 smoke observability，不能横向排序，也不是性能 benchmark。

## 融合、梯度、恢复与复现

| 检查 | 结果 |
| --- | ---: |
| Full/Base zero-gate max absolute error | `0.0` |
| AEF adapter gradient L1 | `0.28279454447329044` |
| AEF gate gradient L1 | `0.032597001641988754` |
| High-resolution adapter gradient L1 | `0.46360756270587444` |
| High-resolution gate gradient L1 | `0.006756966933608055` |
| High-resolution stem gradient L1 | `0.1532890168018639` |
| Output projection gradient L1 | `25.314753890037537` |
| Checkpoint reload max absolute error | `0.0` |
| Deterministic rerun | `matches=true`，max absolute error `0.0`，seed `20260815` |
| Checkpoint SHA-256 | `459e94f2ecd5b7753f6a343068c0fe8fdc2aeac8cd4e8bacb335855a98d763be` |

上述梯度仅证明本次 synthetic smoke 反向链路连通，不代表训练收敛、参数质量或可用精度。

## vMF 单位范数

四组统计相同。导出前 FP32 与重开持久化 FP16 Zarr 分开计算和报告：

| Evidence | Source / compute dtype | Min | Median | Max |
| --- | --- | ---: | ---: | ---: |
| 导出前模型输出 | float32 / float32 | 0.9999996424 | 1.0000000000 | 1.0000003576 |
| 重开持久化 Zarr | float16 / float32 | 0.9998238087 | 0.9999999404 | 1.0001806021 |

## NPU 峰值内存

四组 baseline 完全相同：allocated `156,573,696` bytes，reserved `178,257,920` bytes，
说明前一组 NPU output 已释放后才开始下一组测量。

| Group | Peak allocated | Allocated delta | Peak reserved | Reserved delta |
| --- | ---: | ---: | ---: | ---: |
| `base` | 695,546,880 | 538,973,184 | 1,025,507,328 | 847,249,408 |
| `base_aef` | 896,875,008 | 740,301,312 | 1,161,822,208 | 983,564,288 |
| `base_highres` | 919,551,488 | 762,977,792 | 1,161,822,208 | 983,564,288 |
| `full` | 986,661,376 | 830,087,680 | 1,161,822,208 | 983,564,288 |

这些是单次 smoke 的峰值 observability，不是长期训练容量结论。

## Final seal 与 evidence closure

| Evidence | 最终值 |
| --- | --- |
| `SUCCESS` | `sealed_at_utc=2026-08-15T13:59:25.492105Z`；current evidence 最后写入 |
| Combined SHA-256 | `45ec55838b74b09d48be96bf510e1e35c8368ff996ea6953f86414f0ba1944f9` |
| Final path audit | SHA-256 `41550dbea9269fef3eb486d450f3ad68192cf8102105fbd8602bc3a481420c01` |
| Audit stages | `prepare → cpu-contract → npu-smoke → finalize-seal` |
| Evidence closure | 350 validated roots；321 unique leaf files；350 audit entries；`attempts/` entries = 0 |
| Foreground log | 40,925 bytes；SHA-256 `cede5a20a3d81dd13a668510082c1ce9b8c9f3941607b9d67a89e983fdab8517` |
| READY / TEE | `READY_TO_SEAL` SHA-256 `7367ef804a5ca4062b9b20759a9490a17d31dd1d35515c351afc61421d9691ff`；`TEE_COMPLETE` 正确绑定 READY 与日志 |
| Partial outputs | 0 |
| Sandbox footprint | 587,926,789 bytes（含 diagnostic archives），低于 5 GiB stop limit |

`verify_success` 不只重算 combined digest，还重新解释 checkpoint、state keys、registry、
cache、投影、source snapshots、CPU/NPU metrics、四组 axes、门控、重开范数和状态链之间的
语义关系。因此，一份单独自洽但与其他 evidence 不一致的 JSON 也不能通过封口验证。

## 两次 rejected attempts 与归档纠正

两份 rejected attempt 都只作诊断，不属于 current evidence，也不能引用为成功结果：

1. `attempts/rejected_20260815_first_seal/`：因未独立证明 tee 完成、digest 未覆盖全部
   audit-created evidence、缺少完整 HBM/runtime provenance，并混报 FP32/FP16 vMF 范数而
   被否决。
2. `attempts/rejected_pre_f36a4de/`：因 bootstrap/log symlink 风险、gate override 梯度
   断连、vMF 与跨 artifact 语义校验不足、final audit 重试不安全，以及 occupancy/HBM
   生命周期不完整而被否决。其小证据在 `evidence/`，replaceable outputs 在
   `replaceable/`。

第二份归档最初被放在固定沙箱同级目录，复核后已原子纠正到固定沙箱的
`attempts/rejected_pre_f36a4de/`。归档位置稳定后，含归档的旧 root audit 被保存到
`audit_reset/`，当前 prepare/CPU/NPU/finalize audit 重新生成；最终 350 条 current audit
中没有 `attempts/` 前缀。该纠正没有把历史材料混入 current seal。

## 回归、限制与延期项

最终非 NPU 回归、静态检查、shell 语法、diff 检查和 fresh `verify_success` 的精确结果见
Task 8 报告；最终计数为 `255 passed, 1 skipped`，封口后的收紧版 read-only 复验通过。

仍需保留的限制和延期项：

1. 没有真实 AEF、真实 2 m、正式 AEFModel、正式损失训练或下游效果评测。
2. 只有 4 个 patch、8 个季度、一个固定 seed；没有全国代表性或统计功效。
3. 48 个 references 均记录 member/CRC，但 grid shape/CRS/transform 仅做代表性 header
   记录；尚未扩展为 192 个 grid headers 与全量统计。
4. archive/header 检查仍可能创建多个有界进程池（上限 8）；四 patch smoke 可接受，
   扩容前需改为共享/复用池并重新评估资源上限。
5. 源数据保护是 size/mtime 快照，不是内容哈希。

因此下一阶段应先用真实官方 AEF 和真实年度 2 m 数据分别替换 synthetic 旁路，补齐
全量 grid/statistics 与 archive-pool 设计，再单独制定正式训练和同协议下游评测。当前
唯一可交付结论是：**隔离 synthetic engineering smoke PASS；accuracy remains not
evaluated。**
