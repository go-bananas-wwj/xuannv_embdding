# Task 1 实施报告：严格 PU + Query 特征源比较

## 修改文件

- `scripts/eval/run_pu_query_sparse_eval.py`
  - 增加三种特征来源的严格同协议比较。
  - 传统 42 通道特征直接复用强下游基线的 `fixed_highres_feature_map` 和
    `load_manifest`，未复制栅格读取或归一化规则。
  - 传统 `FeatureSpec` 显式携带 `data_root`；真实传入 `load_manifest`，不再
    从 manifest 所在目录推断数据根目录。
  - 删除未调用的 `make_visual`、`plot_summary` 及其仅供旧绘图路径使用的辅助
    函数；比较图由 Task 2 单独负责。
- `tests/test_pu_query_sparse_comparison.py`
  - 新增特征规格、确定性支持选择、真实临时 `.pt` / manifest 集成测试和同协议
    比较入口 JSON 测试。

## 接口

- `FeatureSpec(name, kind, root, month, channels, data_root=None)` 是冻结的数据类。
- `feature_specs(args)` 返回：
  - `xuannv`: `embedding_map.pt`、`202604`、64 通道。
  - `aef`: `embedding_map.pt`、`202512`、64 通道。
  - `traditional`: `fixed_highres_feature_map`、`202604`、42 通道。
- `load_feature(spec, patch_id)` 统一加载三种特征并校验 `C,H,W` 及通道数；传统
  分支以 spec 中的 `data_root` 加载 manifest。
- `run_comparison(args)` 对每个任务只选一次 3 个支持多边形，并将同一支持
  patch ID 和同一测试 patch ID 用于全部三种特征；输出 `results.json`。

## 测试

执行：

```bash
python -m pytest tests/test_pu_query_sparse_comparison.py -q
```

结果：`4 passed in 9.49s`。

集成测试会写入临时真实 `embedding_map.pt`，由 `load_feature` 直接读取；同时
使用真实 manifest 文件和不同于 manifest 父目录的 `data_root`，以 mock 的
`fixed_highres_feature_map` 断言 `load_manifest` 生成的栅格路径确实以配置的
数据根目录为基准。比较入口使用这些 fixture 并验证 `results.json`。

## Commit

`1f6c758 feat: compare sparse PU query feature sources` 和
`5595de1 fix: honor PU query traditional data root`（均已推送至
`v3-semantic-64d`）。

## 风险

- 未运行真实数据的完整评测，按要求只执行了合成数据聚焦测试。
- 传统特征路径通过导入既有强基线函数复用；临时 fixture 已验证 manifest 与
  `data_root` 传递，但未读取 `/data/xuannv_embedding/` 的真实数据盘栅格。
