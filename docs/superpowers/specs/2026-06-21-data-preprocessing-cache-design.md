# 训练数据预处理缓存设计

## 背景

v1.1 AEF 蒸馏训练在 6×NPU 上运行时，数据加载成为瓶颈：

- 每个 epoch 都要从 TIFF 读取、归一化、做月度 binning，以及把 worldcover 复制到 17 个月。
- 744 个 patch 的原始 TIFF 约 105GB，但预处理后的张量只有约 31GB。
- 当前环境内存 1TB、NVMe SSD 剩余 1.2TB，完全支撑把预处理结果持久化到磁盘并在内存中热读。

## 目标

把 `MonthlyEmbeddingDataset.__getitem__` 的输出按 patch 缓存到 NVMe SSD，使得：

1. 第一次 epoch 之后，后续 epoch 跳过 TIFF 解码和预处理。
2. 缓存持久化，训练重启后仍然有效。
3. DDP 多卡训练时不会写冲突。
4. 配置或统计量变化时自动失效重建。

## 设计

### 1. 缓存内容

缓存每个 patch 对应的 `__getitem__` 输出：

```python
{
    "patch_id": str,
    "source_frames": Dict[str, Tensor],
    "source_masks": Dict[str, Tensor],
    "timestamps": Dict[str, Tensor],
    "teacher_embedding_map": Tensor,
}
```

不缓存 `prepare_batch` 的输出，因为 target_heads 配置可能变化。

### 2. 缓存路径

```
/data/xuannv_embedding/cache/preprocessed_128/{region}/{patch_id}.pt
```

其中 `preprocessed_128` 包含 patch_size（128），方便以后换分辨率时隔离。

### 3. 惰性缓存（训练时自动建）

在 `MonthlyEmbeddingDataset.__init__` 中接收可选参数 `cache_dir: Path | None`。

在 `__getitem__` 中：

1. 若 `cache_dir` 为 `None`，走原流程。
2. 构造缓存文件路径 `cache_file = cache_dir / region / f"{patch_id}.pt"`。
3. 若文件存在，直接 `torch.load(cache_file, map_location="cpu", weights_only=False)` 并返回。
4. 若不存在，走原流程生成样本，然后 `torch.save(sample, cache_file)`。

为保证目录存在，保存前调用 `cache_file.parent.mkdir(parents=True, exist_ok=True)`。

### 4. 预生成脚本

新增 `scripts/data/preprocess_cache.py`：

- 接收 `--config` 参数，读取 `cfg.data.cache_dir`、`cfg.data.manifest_path` 等。
- 实例化 `MonthlyEmbeddingDataset`，单进程遍历所有 patch，调用 `dataset[i]` 触发缓存生成。
- 可选地通过 `--num-workers` 开启多进程预生成（使用 `DataLoader(num_workers=N)` 或 `ProcessPoolExecutor`）。
- 预生成前检查并创建 `cache_meta.json`。

### 5. 缓存失效

在 `cache_dir` 根目录写入 `cache_meta.json`，记录影响预处理结果的 key：

```json
{
  "patch_size": 128,
  "num_months": 17,
  "sources": ["s2", "s1", "landsat", "worldcover", "highres_optical_harbin", "highres_optical", "highres_sar"],
  "statistics_hash": "sha256-of-all-stats-files",
  "teacher_embedding_root": "/data/xuannv_embedding/embeddings/aef_teacher_2025_annual",
  "version": "1.0"
}
```

`MonthlyEmbeddingDataset` 初始化时：

1. 若 `cache_meta.json` 不存在，写一份新的。
2. 若存在，对比当前配置生成的 meta；不一致则清空整个 `cache_dir` 并重建。

### 6. DDP 安全性

- 训练时使用 `DistributedSampler`，每个 rank 分到不重叠的 patch 索引，因此不会同时写同一个 `.pt` 文件。
- 预生成脚本建议单进程或每个 worker 按 patch 索引分区，避免写冲突。
- 即使两个进程同时写一个文件，也可先写到临时文件再原子重命名：
  ```python
  tmp = cache_file.with_suffix(".tmp")
  torch.save(sample, tmp)
  tmp.replace(cache_file)
  ```

### 7. 配置接入

在 `DataConfig` 中新增字段：

```python
cache_dir: Path | None = None  # 默认关闭
```

在 `v1.1_distill_long_stable_50ep.yaml` 中启用：

```yaml
data:
  cache_dir: /data/xuannv_embedding/cache/preprocessed_128
```

`build_dataloader` 和 `_build_loader` 把 `cache_dir` 传给 `MonthlyEmbeddingDataset`。

### 8. 异常与回退

- `torch.load` 失败时，删除损坏的缓存文件并走原流程重新生成。
- 写缓存失败时，打印 warning 并继续训练，不影响本次迭代。

## 预期收益

- 第一次 epoch：构建缓存，可能略慢（约 10~20 分钟）。
- 第二次及以后：每个 epoch 数据加载时间预计降至原来的 1/3 ~ 1/5。
- 50 epoch 总训练时间有望从约 6 小时降到 4 小时左右。

## 风险与回退

| 风险 | 缓解措施 |
|------|----------|
| 第一次 epoch 构建缓存时崩溃 | 原子写 + 损坏文件自动重生成 |
| 缓存占用 31GB 磁盘 | `/data` 剩余 1.2TB，可接受 |
| 统计量或配置变更后读旧缓存 | cache_meta.json 自动失效重建 |
| DDP 多进程写冲突 | DistributedSampler 分区 + 原子写 |

## 测试计划

1. 在单卡/少量 patch 上启用缓存，验证输出与原流程逐张量一致。
2. 跑一个 2 epoch 的 smoke 测试，确认第二次 epoch 明显快于第一次。
3. 验证 DDP 6 卡训练能稳定跑完 3~5 个 epoch。
4. 验证修改统计量后缓存自动失效重建。

## 未选方案

- **全部加载到内存**：速度最快，但 31GB 常驻内存；且需要改 DataLoader 为从内存索引，实现更复杂。磁盘缓存已足够快。
- **LMDB/TFRecord**：对 744 个 patch 这种小规模数据过度设计，维护成本高。
- **不缓存直接调 num_workers**：受 /dev/shm 8GB 限制，不够稳定。
