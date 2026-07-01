# OSM辅助每patch阈值+后处理诊断结果

说明：该结果把 OSM 标注作为阈值选择与对照参考，因此属于 oracle/诊断图层，不是纯模型预测结果。后处理仅使用 WorldCover、OSM水体、OSM绿地等外部先验抑制明显误报。

|任务|Raw F1|每patch阈值 F1|OSM辅助后处理 F1|Raw Precision|后处理 Precision|Raw Recall|后处理 Recall|
|---|---:|---:|---:|---:|---:|---:|---:|
|建筑物提取|0.4191|0.3862|0.4323|0.3209|0.3143|0.6039|0.6921|
|道路提取|0.3985|0.4091|0.4208|0.2932|0.3487|0.6220|0.5307|

建议：API 可以增加 `postprocessed` 或 `osm_assisted_debug` 图层，但不要替换 `raw_prediction`，并在文档中说明该层使用外部 OSM/地类先验。
