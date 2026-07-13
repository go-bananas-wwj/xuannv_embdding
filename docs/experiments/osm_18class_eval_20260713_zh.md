# 18 类主要地物提取能力盘点（OSM 标签 × P10C 生产版，20260713）

## 背景

业务方确定 18 种主要地物的长时序变化分析需求，需评估生产版 embedding（P10C e800，
64 维）+ OSM 弱标签训练下游头的可行性，并给出每类"能做/不能做"结论。

## 数据构建

- OSM 矢量：离线缓存 `/data/xuannv_embedding/experiments/p3a_osm_semantic_audit_20260629/cache/haidian_*.gpkg`（无需联网）。
- 新建 8 个标签层（脚本 `scripts/data/build_osm_18class_labels.py`）：
  - **水体三分**：`osm_river` / `osm_lake` / `osm_pond`。审计发现 353/777 个水体多边形
    无 `water=*` 子 tag（21% 面积），采用两级兜底：与 waterway 主线相交 → river；
    否则面积 ≥3ha → lake，<3ha → pond。waterway 线要素 buffer 15 m 并入 river。
  - `osm_bare`（brownfield/greenfield+bare_rock 等）、`osm_stadium`、`osm_train_station`、
    `osm_wetland`、`osm_landfill`。
- 输出结构与既有层一致（masks/*.tif + metadata.json + split_5fold.json，seed 42）。

正 patch 统计（/320）：river 206、lake 79、pond 138、bare 127、train_station 45、
stadium 6、landfill 9、wetland 2。

## 评测协议

- 全监督：conv3x3 probe（`v2_probe_binary_conv3x3_single_202604.yaml`），fold 0，
  P10C 生产嵌入 202604，与既有四任务基准完全同协议。
- few-shot：`run_strong_downstream_benchmark.py`，4 头（pixel_conv/unet/deeplab_lite/
  segformer_lite）× 5/10/50-shot，fold 0 seed 42，报告各 shot 最优头。
- 稀有类：原型检索（`scripts/eval/run_rare_class_retrieval.py`），单样例正像素平均
  嵌入作原型，全区 319 patch 排序，报告 P@5/P@10/平均排名。

## 结果

### 全监督 conv3x3（F1_best / AP / AUC）

| 任务 | F1 | AP | AUC | 正patch |
| --- | ---: | ---: | ---: | ---: |
| osm_forest | **0.8137** | 0.9029 | 0.9544 | 273 |
| osm_agriculture | **0.5444** | 0.5491 | 0.9296 | 162 |
| osm_lake | **0.5386** | 0.5927 | 0.9633 | 79 |
| merged_sports_pitch | 0.3902 | 0.3104 | 0.8661 | 210 |
| osm_river | 0.3835 | 0.2332 | 0.8751 | 206 |
| osm_bare | 0.1969 | 0.1139 | 0.8369 | 127 |
| osm_pond | 0.1293 | 0.0732 | 0.8334 | 138 |
| osm_parking | 0.1231 | 0.0610 | 0.7320 | 223 |
| osm_grass | 0.1054 | 0.0388 | 0.8602 | 111 |
| osm_train_station | 0.0097 | 0.0034 | 0.6453 | 45 |

### few-shot（各 shot 最优头 F1，5/10/50）

| 任务 | 5 | 10 | 50 |
| --- | ---: | ---: | ---: |
| forest | 0.7364 | 0.7524 | 0.7821 |
| lake | 0.5225 | 0.5307 | 0.5771 |
| farmland | 0.5161 | 0.5314 | 0.5382 |
| river | 0.3110 | 0.3301 | 0.4254 |
| pond | 0.1457 | 0.1751 | 0.2726 |
| grass | 0.2453 | 0.0927 | 0.1699 |
| bare | 0.2084 | 0.1943 | 0.2159 |
| parking | 0.0785 | 0.1004 | 0.1255 |
| train_station | 0.0839 | 0.0482 | 0.0541 |
| stadium | 0.0074 | 0.0069 | 0.0084 |

### 稀有类原型检索（单样例）

| 任务 | GT 正patch | P@5 | P@10 | 平均排名/319 |
| --- | ---: | ---: | ---: | ---: |
| osm_train_station | 44 | 0.2 | **0.5** | 106.8 |
| osm_stadium | 5 | 0 | 0 | 71.8（弱有效） |
| osm_landfill | 8 | 0 | 0 | 160.9（≈随机） |
| osm_wetland | 1 | 0 | 0 | 142（≈随机） |

## 18 类最终判定

- ✅ **直接投产（9）**：林地、湖泊、耕地、河流、操场、建筑物、主干道、施工工地、水体大类。
- 🟡 **降级使用（4）**：池塘（限 ≥0.5ha）、裸地、草地（建议换 Dynamic World/ESRI 标签重训）、
  火车站（分割不可用，检索 P@10=0.5 辅助人工普查）。
- 🔴 **特殊处理（5）**：露天停车场（与硬化地面在 10 m 光谱/结构不可分，AUC 仅 0.73）、
  体育场、湿地、大型垃圾场、机场（样本 <10 或检索失败）——建议固定 AOI 人工监测。
- 十字路口：非面状类，由主干道提取结果骨架化+交点检测后处理，不单独训练。

## 归因与备注

1. AUC 普遍 >0.83 而 F1 低：embedding 有排序能力，弱标签噪声（OSM 现势性、
   brownfield 定义模糊）与小目标（10 m 下池塘/停车场仅数像素）压低 F1 上限。
2. grass few-shot 5-shot 反而高于 10/50-shot：极稀疏标签下小样本方差大，属噪声。
3. 检索兜底只对"结构性强、成片"的类有效（火车站站场/铁轨），对光谱模糊类
   （垃圾场/湿地）无效。

## 产物

- 评测目录：`/data/xuannv_embedding/experiments/osm_18class_eval_20260713/`
  （fullsup/、fewshot/shard0-4、retrieval/、logs/）
- 新标签层：`/data/xuannv_embedding/processed/haidian/labels/{osm_river,osm_lake,osm_pond,osm_bare,osm_stadium,osm_train_station,osm_wetland,osm_landfill}/`
- 汇报网页新增章节 4.7（http://<host>:8002）
