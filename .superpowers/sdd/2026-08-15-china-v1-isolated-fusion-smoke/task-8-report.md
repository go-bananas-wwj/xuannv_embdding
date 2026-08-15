# Task 8 Report: Final Evidence Report, Regression, and Handoff

## Result

Task 8 已把最终 `f36a4de` physical-NPU-2 seal 与最终审查修复波同步到仓库文档。唯一结论
为 **synthetic engineering smoke PASS**；accuracy、真实 AEF 语义、真实 2 m 信息、
正式 `AEFModel` 训练、正式下游评测与全国生产能力均未评估、未授权。

主报告只消费 current corrected seal。两份 rejected attempts 均被列为 diagnostic archives，
并从 current audit、digest closure 与结论中排除。交付仍为 repository Markdown；未创建网页
或 latency 图，因为四个 one-shot 值包含 cold-start/cache 差异，不具备横向性能可比性。

## Final evidence inventory

Fresh read-only inventory 建立了以下事实：

- 运行 commit：`f36a4de49e263badea4a86b429fddbc3006d5953`；
- `SUCCESS` combined SHA-256：
  `45ec55838b74b09d48be96bf510e1e35c8368ff996ea6953f86414f0ba1944f9`；
- 350 validated roots、321 unique leaf files、350 current audit entries、0 个
  `attempts/` audit entries、0 个 `.partial`；
- audit stages：`prepare → cpu-contract → npu-smoke → finalize-seal`；
- 四组重开后均为有限 FP16 `[4,8,64,128,128]`，bool valid mask
  `[4,8,1,128,128]`，patch/period axes 一致；
- exact zero-gate、checkpoint reload、deterministic rerun errors 均为 `0.0`；AEF/highres
  adapters、两项 gate、highres stem 与 output projection gradients 均为有限非零；
- 导出前 FP32 与重开 FP16 Zarr vMF 范数分别复算并满足各自阈值；
- 四组 HBM baselines 完全相同，且各自记录 peak allocated/reserved 与 delta；
- source snapshots 字节一致，覆盖 48 archives、55 directories、253 direct entries；
- provenance 记录两次物理 NPU 2 idle check、唯一逻辑 `npu:0`、CANN/driver/
  Python/torch/torch-npu 与模块路径；
- 最终 sandbox 587,926,789 bytes，含两份被排除的 diagnostic archives，低于 5 GiB。

## Rejected evidence handling

- `attempts/rejected_20260815_first_seal/` 是第一次 Task 7 seal 的小型诊断证据；
- `attempts/rejected_pre_f36a4de/` 保存最终 review 前旧 seal 的 `evidence/` 与
  `replaceable/`，以及归档位置纠正前的 `audit_reset/` 记录；
- 第二份归档由沙箱同级位置原子纠正到固定沙箱内。位置稳定后重新运行 prepare 与 CPU
  contract，再执行唯一一次最终 physical-NPU launcher；current audit 不含任一归档路径。

两份归档都不得作为成功证据、训练证据或精度证据。

## Repository deliverables

- `docs/reports/china_v1_isolated_fusion_smoke_20260815.md`：最终 answer-first 技术报告；
- `.superpowers/sdd/2026-08-15-china-v1-isolated-fusion-smoke/task-7-report.md`：最终 NPU
  执行、修复与 seal 细节；
- `.superpowers/sdd/2026-08-15-china-v1-isolated-fusion-smoke/task-8-report.md`：最终回归与
  handoff；
- `.superpowers/sdd/2026-08-15-china-v1-isolated-fusion-smoke/progress.md`：SDD ledger；
- `CHANGELOG.md`：最终修复波、final seal、rejected attempts 和结论边界。

## Verification

最终回归未设置 NPU opt-in，也未触发第二次 NPU 运行。结果如下：

```text
pytest: 239 passed, 1 skipped in 61.44s
Ruff: All checks passed!
Black: 23 files would be left unchanged.
launcher shell syntax: PASS
git diff --check: PASS
fresh verify_success: PASS
```

唯一 skip 是显式 opt-in 的 NPU integration test；本轮没有以回归命令再次占用 NPU，物理
NPU 2 证据来自 Task 7 唯一一次最终前台运行。Fresh verifier 重新检查了当前 seal 的 digest
与 semantic graph，而不是只读取 `SUCCESS` 文本。

## Explicitly deferred

本次未实现 192 个 grid headers/全量 statistics，也未把多个有界 archive/header pools
重构为共享池；source protection 仍为 size/mtime 清单而非内容哈希。这些项目在扩大 patch
数量或进入生产设计前必须关闭。真实 AEF、真实 2 m、正式训练与正式评测属于后续独立
阶段，不能由本次 synthetic smoke 替代。
