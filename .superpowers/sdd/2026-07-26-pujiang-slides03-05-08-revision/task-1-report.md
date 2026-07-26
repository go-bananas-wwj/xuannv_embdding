# Task 1 实施报告：严格 PU + Query 特征源比较

## 修改文件

- `scripts/eval/run_pu_query_sparse_eval.py`
  - 增加三种特征来源的严格同协议比较。
  - 传统 42 通道特征直接复用强下游基线的 `fixed_highres_feature_map` 和
    `load_manifest`，未复制栅格读取或归一化规则。
- `tests/test_pu_query_sparse_comparison.py`
  - 新增特征规格、确定性支持选择、合成同协议比较测试。

## 接口

- `FeatureSpec(name, kind, root, month, channels)` 是冻结的数据类。
- `feature_specs(args)` 返回：
  - `xuannv`: `embedding_map.pt`、`202604`、64 通道。
  - `aef`: `embedding_map.pt`、`202512`、64 通道。
  - `traditional`: `fixed_highres_feature_map`、`202604`、42 通道。
- `load_feature(spec, patch_id)` 统一加载三种特征并校验 `C,H,W` 及通道数。
- `run_comparison(args)` 对每个任务只选一次 3 个支持多边形，并将同一支持
  patch ID 和同一测试 patch ID 用于全部三种特征；输出 `results.json`。

## 测试

执行：

```bash
python -m pytest tests/test_pu_query_sparse_comparison.py -q
```

结果：`3 passed in 10.19s`。

## Commit

`1f6c758 feat: compare sparse PU query feature sources`（已推送至
`v3-semantic-64d`）。

## 风险

- 未运行真实数据的完整评测，按要求只执行了合成数据聚焦测试。
- 传统特征路径通过导入既有强基线函数复用；测试验证了统一接口和比较协议，
  但未读取 `/data/xuannv_embedding/` 的真实 manifest 或栅格。
