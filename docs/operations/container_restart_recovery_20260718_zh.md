# 容器重启前后台任务恢复手册（2026-07-18）

## 1. 暂停状态

暂停时间：2026-07-18 10:32 UTC。

- 论文底座训练已停止，NPU 0-5 已释放。
- 全国 China V1 数据获取的 3 个 worker 已停止，心跳不再更新。
- 已完成 checkpoint、训练日志、Zarr partial、质量记录和 STAC 目录均保留。
- 汇报网站的 8001/8002 服务与本次恢复无关，容器重启后可按需另行启动。

## 2. 论文训练恢复

### 2.1 当前进度

- 队列总数：40。
- 完整完成：12，均有 `epoch_800.pt`。
- 队列状态文件：`/data/xuannv_embedding/logs/paper_registered_20260716/status.tsv`。
- 输出根目录：`/data/xuannv_embedding/outputs/paper_registered_20260716/`。
- 日志根目录：`/data/xuannv_embedding/logs/paper_registered_20260716/`。

中断任务及可恢复点：

| 任务 | 停止前最后完成 Epoch | 恢复 checkpoint |
|---|---:|---|
| `paper_registered_no_highres_path_150_fold2_20260716` | 388 | `epoch_200.pt` |
| `paper_registered_no_highres_path_150_fold3_20260716` | 480 | `epoch_400.pt` |
| `paper_registered_no_highres_path_150_fold4_20260716` | 383 | `epoch_200.pt` |

调度器会跳过已有 `epoch_800.pt` 的 12 组任务，并自动从每个非空输出目录中编号最大的 `epoch_*.pt` 恢复 optimizer、scheduler 和 epoch。中断后尚未到下一个保存点的轮次会重新计算，这是预期行为。

调度器默认还会对 Ascend/HCCL 瞬时超时进行最多 3 次自动重试，每次等待 60 秒并重新读取最新编号 checkpoint。可通过 `MAX_JOB_RETRIES` 和 `RETRY_DELAY_SECONDS` 环境变量调整，但正式运行应在状态文件中保留实际值与重试记录。

### 2.2 训练 watchdog 与滚动恢复

训练器默认每 20 个 epoch 原子覆盖一次 `recovery.pt`；可通过
`XUANNV_RECOVERY_SAVE_EVERY` 调整。调度器会在 `recovery.pt`、`epoch_*.pt`
和 `best.pt` 中按原子写入时间选择最新状态，因此首次 200 epoch 保存点前发生故障也可以恢复。

独立 watchdog 每 30 秒扫描一次 `torchrun`。某个实验启动超过 10 分钟且训练日志连续
10 分钟没有更新时，watchdog 对对应 torchrun 发送 SIGTERM，由所属队列在 90 秒后重试，
不再等待 HCCL 默认约 30 分钟超时：

```bash
tmux new-session -d -s paper_training_watchdog \
  'cd /root/workspace/xuannv && while true; do \
   PYTHONPATH=/root/workspace/xuannv/src \
   /data/wwj_torch21/conda/envs/torch26/bin/python \
   scripts/train/watch_training_jobs.py \
     --log-root /data/xuannv_embedding/logs/paper_registered_20260716 \
     --heartbeat /data/xuannv_embedding/logs/paper_registered_20260716/watchdog_heartbeat.json \
     --events /data/xuannv_embedding/logs/paper_registered_20260716/watchdog_events.jsonl \
     --stall-seconds 600 --startup-grace-seconds 600 --poll-seconds 30; \
   sleep 30; done'
```

查看状态：

```bash
cat /data/xuannv_embedding/logs/paper_registered_20260716/watchdog_heartbeat.json
tail -20 /data/xuannv_embedding/logs/paper_registered_20260716/watchdog_events.jsonl
```

### 2.3 恢复命令

```bash
cd /root/workspace/xuannv
git pull --rebase
tmux new-session -d -s paper_registered_20260716 \
  'cd /root/workspace/xuannv && bash scripts/experiments/run_registered_paper_queue.sh'
```

恢复后检查：

```bash
tail -30 /data/xuannv_embedding/logs/paper_registered_20260716/status.tsv
npu-smi info
```

状态文件应出现三条 `resume` 记录，并分别指向上表 checkpoint。禁止手工删除已完成目录或把 `best.pt` 当作续训 checkpoint。

## 3. China V1 数据获取恢复

> 2026-07-19 更新：正式生产根目录已切换为
> `/data2/xuannv_embedding/china_v1/shards/full_pc_v2_20260719/`。旧目录
> `full_pc_20260717/` 只保留早期故障 partial，不计入当前进度，也不再恢复写入。

### 3.1 当前进度与问题

- 数据根目录：`/data2/xuannv_embedding/china_v1/`。
- 当前占用约 7 GB；`/data2` 尚余约 3.2 TB。
- 正式流水线：60,500 个采样点，13 个月（2025-04 至 2026-04）。
- 正式输出目录中已有 13 个 partial shard，尚无通过完整校验的正式 shard。
- 已写入约 1,000 条 patch 质量记录；另外 6 个完整 `shard_report.json` 属于 smoke/pilot，不计入正式进度。
- 主要阻塞是远端 COG 的 403/206、过期签名与 TIFF tile 短读。partial 和 `done` 矩阵会保留已成功读取的内容，watchdog 重启时重新签名并继续。

暂停时三个活动 partial 为：

- worker 0：`china_v1_full_006.zarr.partial`
- worker 1：`china_v1_full_215.zarr.partial`
- worker 2：`china_v1_full_410.zarr.partial`

### 3.2 恢复前清理失效锁

只删除以下三个 `.lock` 目录；不要删除 `.zarr.partial`：

```bash
rm -rf \
  /data2/xuannv_embedding/china_v1/shards/full_pc_20260717/china_v1_full_006.zarr.partial.lock \
  /data2/xuannv_embedding/china_v1/shards/full_pc_20260717/china_v1_full_215.zarr.partial.lock \
  /data2/xuannv_embedding/china_v1/shards/full_pc_20260717/china_v1_full_410.zarr.partial.lock
```

### 3.3 启动三个 watchdog

在 `/root/workspace/xuannv` 执行：

```bash
cd /root/workspace/xuannv
mkdir -p /data2/xuannv_embedding/china_v1/watchdog/full_pc_20260717

for worker in 0 1 2; do
  tmux new-session -d -s "china_v1_worker_${worker}" \
    "cd /root/workspace/xuannv && \
     bash scripts/data/run_china_v1_direct.sh \
     python scripts/data/run_china_v1_watchdog.py \
       --heartbeat /data2/xuannv_embedding/china_v1/watchdog/full_pc_v2_20260719/worker_${worker}.heartbeat.json \
       --log /data2/xuannv_embedding/china_v1/logs/full_pc_v2_20260719/worker_${worker}.log \
       --state /data2/xuannv_embedding/china_v1/watchdog/full_pc_v2_20260719/worker_${worker}.state.json \
       --stall-seconds 900 --restart-delay 30 --max-restarts 100 -- \
       python scripts/data/materialize_china_v1_worker.py \
         --points-dir /data2/xuannv_embedding/china_v1/atlas/full_60500_spatial \
         --prefix china_v1_full \
         --catalog-root /data2/xuannv_embedding/china_v1/stac_catalogs \
         --output-root /data2/xuannv_embedding/china_v1/shards/full_pc_v2_20260719 \
         --months 2025-04 2025-05 2025-06 2025-07 2025-08 2025-09 2025-10 2025-11 2025-12 2026-01 2026-02 2026-03 2026-04 \
         --worker-index ${worker} --worker-count 3 \
         --max-clean-scenes 3 --strategy scene \
         --asset-workers 1 --asset-cache-size 128 --passes 3"
done
```

### 3.4 恢复后检查

```bash
tmux ls
for f in /data2/xuannv_embedding/china_v1/watchdog/full_pc_20260717/*.heartbeat.json; do
  stat -c '%y %n' "$f"
  cat "$f"
done
tail -30 /data2/xuannv_embedding/china_v1/logs/full_pc_20260717/worker_0.log
```

三个 heartbeat 应每隔数秒到数分钟继续更新。若持续出现 403，先看对应 `worker_*.state.json` 是否在重启并刷新签名，不要删除 partial 重下。

单个 COG 候选景读取失败、但同一 patch 已由其他清晰景形成有效合成时，质量记录会保留失败景信息，但不会再阻止 shard 完成。只有远端错误导致该 patch 完全没有有效像素时才保留为 `retryable_error`。正式全国任务使用 900 秒心跳阈值，避免一个跨境 COG 批量窗口读取超过 180 秒时被误杀。

`--max-clean-scenes 3` 表示每个 patch、每个数据源、每个月最多合成 3 景通过本地质量筛选的影像。scene-first 实现必须按 patch 计数，不能把该上限错误应用成“每景最多处理 3 个 patch”。

逐 patch 的 `done` 矩阵会立即保留成功的数据源/月组合，后续恢复只重试失败 patch。STAC 几何与真实 COG 边界不一致产生的 `WindowError: Intersection is empty` 属于永久不覆盖，记录为无有效像素并通过 availability mask 表达，不作无限网络重试。修复后的端到端 smoke 位于 `/data2/xuannv_embedding/china_v1/smoke/fix_20260719_0428.zarr`，已通过 1 patch、1 月、S2/S1/Landsat 三源原子落盘与形状校验。

`materialize_china_v1_worker.py` 已安装 SIGTERM 处理器。watchdog 因心跳超时终止子进程时，Python 会先展开 shard materializer 的 `finally` 并释放当前 `.lock`，从而允许下一个子进程继续同一个 partial。若使用本修复前的进程产生了 stale lock，必须先停止全部 China V1 worker，确认 heartbeat 不再更新，再仅删除 `.zarr.partial.lock` 空目录；不得在 worker 活跃时批量清锁。

## 4. 重启后验收清单

1. `npu-smi info` 中 NPU 0-5 出现三组双卡训练进程。
2. 训练 `status.tsv` 出现 12 条 `skip_complete` 和 3 条 `resume`。
3. 三个 China V1 tmux 会话存在，三个 heartbeat 持续更新。
4. `/data2` 可用空间保持在 3 TB 以上。
5. 不存在新的 OOM、NaN、Zarr fingerprint mismatch 或锁占用异常。
