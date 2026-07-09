# XuannvEarth: Toward China-Scale Monthly Geospatial Embeddings for Robust Mapping and Change Detection

**Special Session:** SS095 Open Source GeoAI Practice with ModelScope

**Authors:** Weijie Wu^1, Xinyi Fan^2, and Long Zhao^1

**Affiliations:** ^1 Aerospace Information Research Institute, Chinese Academy of Sciences; ^2 Institute of Geographic Sciences and Natural Resources Research, Chinese Academy of Sciences

**Corresponding Author:** Long Zhao, zhaolong@aircas.ac.cn

**Abstract**

China's urban and regional governance increasingly requires monthly, fine-grained maps of buildings, roads, water bodies, green spaces, construction activities, and land-cover changes. However, operational remote sensing is rarely clean: optical observations are often affected by clouds, haze, snow, seasonal appearance shifts, and missing acquisitions, while dense task-specific annotations are expensive to update. We present XuannvEarth, a monthly multimodal geospatial embedding framework that converts heterogeneous Earth observation data into reusable dense representations for robust mapping and change detection. A central idea is to generate the embedding once and reuse it across many downstream tasks by attaching simple linear, MLP, convolutional, or lightweight segmentation heads, instead of retraining a separate heavy model for every target. The current system is validated in two representative regions: Haidian District, Beijing, for fine-grained urban semantic mapping, and Harbin New Area for bitemporal land-change detection under imperfect observations. XuannvEarth integrates Sentinel-2 optical imagery, Sentinel-1 SAR, Landsat observations, high-resolution optical imagery, high-resolution SAR imagery, and OpenStreetMap-derived weak semantic labels to produce 10 m, 64-dimensional monthly embedding maps.

XuannvEarth is trained to preserve multi-sensor appearance, high-resolution spatial detail, monthly surface state, and weak urban semantics in a unified representation. Its objective uses reconstruction-and-weak-supervision learning with masked-source training and context-aware patch cropping, encouraging the embedding to remain informative when one data source is missing or when optical imagery is partially contaminated by clouds or haze. After pre-training, the encoder is frozen and evaluated under a unified downstream protocol, where train/validation/test splits, validation-selected thresholds, and label budgets are kept consistent.

Experiments show that the same generated embeddings support both semantic mapping and change detection. In Haidian, under a 5-shot setting, XuannvEarth is compared with raw 2026-04 multisource-feature baselines built from stacked Sentinel-2, Sentinel-1, Landsat, high-resolution optical/SAR features and candidate heads including Conv3x3, MLP, U-Net, and DeepLab-lite. It improves F1 by 9.8%, 18.3%, and 41.5% on building, road, and water extraction, respectively. In a 50-shot six-task benchmark, XuannvEarth with lightweight or spatial heads achieves an average F1 of 0.552 and outperforms raw-feature supervised U-Net, DeepLab-lite, or SegFormer-lite baselines on five of six urban mapping tasks. In Harbin New Area, five-fold bitemporal evaluations reach AUC values of 0.962, 0.945, and 0.995 for building change, farmland change, and rubbish or land-surface change, outperforming the AEF annual embedding baseline on these tasks and demonstrating robustness under cloud and source-availability issues.

The project is designed as an open GeoAI workflow rather than a closed single-task model. We will release trained weights, monthly embeddings, training configurations, data documentation, downstream evaluation scripts, visualization examples, and reproducible ModelScope notebooks. Looking forward, we aim to scale XuannvEarth from regional validation to a China-oriented monthly geospatial embedding foundation, enabling robust sparse-label mapping, query-by-example retrieval, and land-change monitoring across diverse Chinese landscapes.

**Index Terms**--geospatial embeddings, remote sensing foundation models, sparse-label mapping, urban GeoAI, multimodal representation learning, ModelScope
