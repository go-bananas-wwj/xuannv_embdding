# China V1 全国高分一张图与开放地球模型：来源记录

调研日期：2026-08-14。本文是报告的来源与口径说明，不替代正式的数据授权或法律意见。

## 项目内部证据

- `docs/agent_memory/haidian_embedding_project_memory.md`：海淀生产版、few-shot 规范与历史结论。
- `docs/plans/china_v1_data_preparation_plan.md`：China V1 的 62,000 个训练 patch、2020–2021 八季度方案。
- `docs/experiments/v4a_distill_from_scratch_report.md`：DINOv3 蒸馏从头训练的收益与表征退化风险。
- `docs/experiments/v5_p10c_distill_finetune_report.md`：在 P10C 上低学习率追加蒸馏失败的记录。
- `src/xuannv_embedding/training/losses.py`、`src/xuannv_embedding/data/dataset.py`：当前单教师离线特征蒸馏接口。

## 全国 2 米影像与许可

- [2021 年中国 2 米/16 米一张图系列发布](https://www.noda.ac.cn/rsgs/news/showNewsById?id=6694ccca4782da475b5c8ffa)：国家对地观测科学数据中心面向科研人员发布，系列总量 19.67 TB、8,037 景；下载可用不等于机器学习派生权和模型再发布权已经明确。
- [GF-6 官方参数](https://www.cnsa.gov.cn/n6758823/n6758842/c6801760/content.html)：2 米全色、8 米多光谱。报告据此区分“2 米全色/融合色彩”与“原生 2 米多光谱”。
- [资源三号官方参数](https://www.cnsa.gov.cn/n6758823/n6758838/c6771322/content.html)：正视全色约 2.1 米、多光谱约 5.8 米。
- [自然资源卫星遥感云服务平台](https://sasclouds.com/) 与 [NODA 影像检索](https://www.noda.ac.cn/datasharing/imageFilmSearch)：可查询/申请的官方入口；具体用途、分发范围和数据级别需逐单确认。
- [四维地球产品说明](https://siweiearth.com/sw-nav/helpDoc?file=%E4%BA%A7%E5%93%81%E8%AF%B4%E6%98%8E%2F%E5%9B%9B%E7%BB%B4%E5%9C%B0%E7%90%83%E5%B9%B3%E5%8F%B0%E4%BA%A7%E5%93%81%E4%BD%93%E7%B3%BB.md&title=%E5%9B%9B%E7%BB%B4%E5%9C%B0%E7%90%83%E5%B9%B3%E5%8F%B0%E4%BA%A7%E5%93%81%E4%BD%93%E7%B3%BB)：全国 2 米/0.8 米商业底图候选。
- [吉林一号高分参数](https://www.jl1.cn/chanpin_view.aspx?id=4248)：0.5 米全色、2 米多光谱，是原生 2 米多光谱商业候选之一。
- [地图管理条例](https://www.gov.cn/zhengce/zhengceku/2015-12/14/content_10403.htm)：公开地图和测绘成果发布需要专项合规审查。本报告不作法律结论。

## 开放地球表征模型

### TESSERA

- [TESSERA 官方仓库](https://github.com/ucam-eo/tessera)：代码 MIT；模型权重和 embedding 为 CC0；年度、10 米、S1+S2 的逐像素时序表征。
- [GeoTESSERA](https://github.com/ucam-eo/geotessera)：v1.1 已可用，v2 全域预计算仍在逐步发布。
- [TESSERA v2 论文](https://arxiv.org/abs/2607.03949)：student 为 Matryoshka 表征，16/32/64/128 维可截取；报告采用 v2-M64 作为首选 POC。

### Clay

- [Clay 官方文档](https://clay-foundation.github.io/model/) 与 [官方仓库](https://github.com/Clay-foundation/model)：当前文档指向 v1.5；代码和权重 Apache-2.0；支持按波长、GSD、地点、时间处理多种影像。
- [Clay v1.5 规格](https://clay-foundation.github.io/model/release-notes/specification.html)：用于核对模型体量、1024 维表示与 patch 特征。
- [Clay 分割微调教程](https://clay-foundation.github.io/model/finetune/segment.html)：冻结特征、轻量下游头和全量微调参考。

### OlmoEarth

- [OlmoEarth 官方仓库](https://github.com/allenai/olmoearth_pretrain)：v1.2；S1、S2、Landsat 多模态多时序，提供 embedding、分割微调与预训练流程。
- [OlmoEarth v1.2 技术报告](https://allenai.org/papers/olmoearth-v1-2)：模型改进与版本口径。
- [OlmoEarth Artifact License](https://github.com/allenai/olmoearth_pretrain/blob/main/LICENSE)：允许派生和蒸馏，但限制军事/国防/情报/监控警务及油气、采矿、毁林等开采用途；派生分发需要传递限制，除非获得 Ai2 书面许可。
- [OlmoEarth embedding 流程](https://github.com/allenai/rslearn/blob/master/docs/examples/OlmoEarthEmbeddings.md) 与 [微调流程](https://github.com/allenai/rslearn/blob/master/docs/examples/FinetuneOlmoEarth.md)：工程参考。

## 关键估算口径

- 中国陆域按 960 万平方公里近似。
- China V1 训练集合按 62,000 个 1,280 米 × 1,280 米 patch。
- 全国无重叠 1,280 米网格约 586 万个 patch；边界、海岛、重叠与空瓦片未计。
- 存储均采用十进制 TB，除非另有说明；未计索引、掩膜、压缩差异、临时空间和副本。
- TESSERA 全国 H100 等价工期来自论文的全球量级按陆地面积比例粗估，不可直接等价为 Ascend NPU 天数。

## 调研边界

- “OllamaEarth”没有找到可信的同名遥感基础模型；本报告按用户可能指的 **OlmoEarth** 调研。Ollama 本身是本地大语言模型运行工具。
- 数据许可、模型许可证传导和测绘安全结论必须由采购、法务和合规负责人基于具体合同复核。
- 公开模型的中国域收益尚未在本项目同协议下实测；报告把所有收益判断都写成待验证假设，并设置 Go/No-Go 门槛。
