# Task 7 Report: Final Physical-NPU-2 Synthetic Engineering Smoke

## Result

Task 7 的最终 corrected physical-NPU run 为 **synthetic engineering smoke PASS**。该结果
只证明固定小模型、真实 S1/S2 与 synthetic AEF/high-resolution 旁路的工程合同；不证明
模型精度、真实 AEF 语义、真实 2 m 质量、正式训练、正式评测或全国生产能力。

最终运行绑定 Git commit
`f36a4de49e263badea4a86b429fddbc3006d5953`，在物理 NPU 2 上以前台方式执行一次，映射
为唯一可见逻辑设备 `npu:0`。Fresh `verify_success` 通过，最终 `SUCCESS` combined
SHA-256 为
`45ec55838b74b09d48be96bf510e1e35c8368ff996ea6953f86414f0ba1944f9`。

## Final review fix wave

本轮最终审查通过 TDD 完成三个实现提交及一个测试跟进，均已推送：

- `86095fb fix: harden isolated smoke bootstrap and logging`
- `966737c fix: enforce fusion gradients and unit norm contracts`
- `f36a4de fix: close isolated smoke evidence semantics`
- `f15a933 test: accept semantic tamper rejection`

修复范围：

- 固定 root 的 ancestor/root/sentinel 在任何 mutation 前用 `lstat` 校验；sentinel 与日志
  均排他创建并拒绝 symlink/no-clobber；
- launcher 使用自有安全 tee helper，在前台 pipeline 成功后才写 `TEE_COMPLETE`，随后
  finalizer 独立复验并最后写 `SUCCESS`；
- `SUCCESS` 后全部写入口 fail closed；preliminary audit 不可变，final audit retry-safe；
- 模型输入严格限定 4 quarters、3 months、S2 10/S1 2 channels、bool 1-channel masks、
  AEF 64 channels 和 high-resolution 5H×5W；
- gate override 改为 straight-through：前向使用 0.1，又保留 gate parameter gradient；
  两项 gate gradient 及所有融合分支 gradient 必须有限且非零；
- vMF FP32/重开 FP16 容差分别收紧为 `1e-5`/`5e-4`，并拒绝零向量；
- synthetic AEF 使用一份跨 patch-year 固定的 64×10 投影；high-resolution texture 保持
  patch-year-specific；smoke extra 显式包含 Zarr/numcodecs；
- archive/cache 装载后、`set_device` 前再次确认物理 NPU 2 空闲；每组 output 转 CPU 并
  释放 NPU 引用后，才读取下一组 HBM baseline；
- final verifier 重算并交叉绑定 checkpoint/model/state keys、run/config/Git/selection、
  registries/caches/projection、48 source archives、CPU 四组、NPU 四组、axes、gate
  gradients、持久化 norms、READY/TEE/log 与 final audit。

## Rejected attempts

两次旧 seal 均已明确否决，只保留作诊断材料：

1. `attempts/rejected_20260815_first_seal/`：缺少独立 tee completion、完整 digest
   coverage、峰值 HBM/runtime provenance，并混淆 FP32 与持久化 FP16 norm。
2. `attempts/rejected_pre_f36a4de/`：存在 bootstrap/log symlink 风险、gate override
   梯度断连、vMF/semantic validation 不足、final audit retry 问题，以及 occupancy/HBM
   生命周期缺口。其 `evidence/` 与 `replaceable/` 都不是 current evidence。

第二份归档最初位于固定沙箱同级，随后按复核意见原子移入固定沙箱。含归档路径的旧
root audits 已保存在 `attempts/rejected_pre_f36a4de/audit_reset/`；当前 audit 在归档位置
稳定后重新生成，350 条 current entries 中 `attempts/` 前缀计数为 0。

## Final foreground execution

宿主机没有 `fuser`，launcher 使用 fail-closed `/proc/<pid>/fd` 字符设备扫描。独立 guard
确认物理设备 `/dev/davinci2` 空闲。执行的唯一正式 launcher command 为：

```bash
bash scripts/smoke/run_china_v1_isolated_fusion_smoke.sh
```

没有使用 `torchrun`、`nohup`、后台任务或 kill。运行 provenance 记录：

- physical NPU 2 → logical `npu:0`，logical device count 1；
- device `Ascend910B4-1`；CANN 9.0.0；driver 26.0.rc1；
- Python 3.11.15；PyTorch 2.6.0+cpu；torch-npu 2.6.0.post5；
- occupancy checks 恰为 `launcher_preload` 与 `post_load_pre_set_device`，均
  `idle=true`；
- source before/after snapshots 字节完全一致，48 archives、55 directories、253 direct
  entries，`source_unchanged=true`。

## Outputs and numerical contracts

四组 `base`、`base_aef`、`base_highres`、`full` 均重开验证：embedding FP16
`[4,8,64,128,128]`、chunks `[1,1,64,128,128]`、valid bool
`[4,8,1,128,128]`、finite，并共享精确的 4 patch IDs 与 `2020Q1..2021Q4` axes。

| Group | Latency (s) | Peak allocated | Allocated delta | Peak reserved | Reserved delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base` | 5.495883 | 695,546,880 | 538,973,184 | 1,025,507,328 | 847,249,408 |
| `base_aef` | 0.640632 | 896,875,008 | 740,301,312 | 1,161,822,208 | 983,564,288 |
| `base_highres` | 0.661814 | 919,551,488 | 762,977,792 | 1,161,822,208 | 983,564,288 |
| `full` | 0.008031 | 986,661,376 | 830,087,680 | 1,161,822,208 | 983,564,288 |

每组 baseline 均为 allocated `156,573,696`、reserved `178,257,920` bytes。Latency 与峰值
仅为 one-shot smoke observability，禁止解释为性能排序或训练容量 benchmark。

其他合同：

- zero-gate Full/Base max absolute error：`0.0`；
- checkpoint reload max absolute error：`0.0`；
- deterministic rerun：seed `20260815`，`matches=true`，max error `0.0`；
- checkpoint SHA-256：
  `459e94f2ecd5b7753f6a343068c0fe8fdc2aeac8cd4e8bacb335855a98d763be`；
- gradient L1：AEF adapter `0.28279454447329044`、AEF gate
  `0.032597001641988754`、high-resolution adapter `0.46360756270587444`、
  high-resolution gate `0.006756966933608055`、stem `0.1532890168018639`、output
  projection `25.314753890037537`；
- pre-export FP32 vMF min/median/max：
  `0.9999996423721313 / 1.0 / 1.0000003576278687`；
- reopened FP16 Zarr vMF min/median/max：
  `0.999823808670044 / 0.9999999403953552 / 1.0001806020736694`。

## Final seal

- sealed at `2026-08-15T13:59:25.492105Z`；
- path audit stages 精确为
  `prepare → cpu-contract → npu-smoke → finalize-seal`；
- 350 validated roots、321 unique leaf files、350 audit entries、0 个 `attempts/` current
  entries、0 个 `.partial`；
- final audit SHA-256：
  `41550dbea9269fef3eb486d450f3ad68192cf8102105fbd8602bc3a481420c01`；
- foreground log 40,925 bytes，SHA-256
  `cede5a20a3d81dd13a668510082c1ce9b8c9f3941607b9d67a89e983fdab8517`；
- `READY_TO_SEAL` SHA-256
  `7367ef804a5ca4062b9b20759a9490a17d31dd1d35515c351afc61421d9691ff`，并被
  `TEE_COMPLETE` 精确绑定；
- sandbox 总体积 587,926,789 bytes（含两份 diagnostic archives），低于 5 GiB。

## Deferred work

以下重项有意留到生产规模设计，不属于本次 PASS：

- 192 个 grid headers 与全量 statistics；当前每个 reference 有 member/CRC，但
  shape/CRS/transform 只保存代表性 header；
- archive/header worker pool 共享化；当前最多可创建 8 个有界 pool，四 patch smoke 可
  接受，扩容前必须重构并复测；
- 源 ZIP 内容哈希；当前只做 size/mtime 清单；
- 真实 AEF、真实 2 m、正式 `AEFModel` 训练和同协议下游精度评测。

Task 7 的结论到此止于 synthetic engineering contract，不得扩写为模型效果结论。
