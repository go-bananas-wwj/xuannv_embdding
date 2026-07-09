# Xuannv: A Monthly Multimodal Geospatial Foundation Embedding for Scalable Sparse-Label Urban Mapping

**Special Session:** SS095 Open Source GeoAI Practice with ModelScope

**Authors:** Weijie Wu, Xinyi Fan, and Long Zhao

**Affiliations:** University of Chinese Academy of Sciences; Aerospace Information Research Institute, Chinese Academy of Sciences

**Corresponding Author:** Weijie Wu, wuweijie25@mails.ucas.ac.cn

**Abstract**

Urban remote-sensing applications increasingly require frequently updated, fine-grained maps of buildings, roads, water bodies, green spaces, schools, sports facilities, and other urban objects. However, conventional task-specific models depend on dense annotations, are costly to update, and rarely transfer across mapping targets. Inspired by recent geospatial foundation model practice, we present Xuannv, a monthly multimodal geospatial foundation embedding framework for scalable sparse-label urban mapping. Rather than treating each target as an isolated segmentation problem, Xuannv converts heterogeneous Earth observation data into reusable dense embeddings that can support many downstream tasks with lightweight heads and limited labels. We instantiate and validate the framework over Haidian District, Beijing, using 320 spatial patches and six monthly observations from December 2025 to May 2026. The training data integrate Sentinel-2 optical imagery, Sentinel-1 SAR, Landsat observations, high-resolution optical imagery, high-resolution SAR imagery, and OpenStreetMap-derived weak semantic labels. The model produces 10 m, 64-dimensional dense embedding maps that can be reused by different downstream heads.

Xuannv is trained to preserve multi-sensor appearance, high-resolution spatial detail, temporal surface state, and weak urban semantics in a unified representation. Its objective combines multi-source reconstruction, high-resolution reconstruction, OSM weak semantic supervision, hard reconstruction with masked inputs, and context-aware patch cropping to improve spatial continuity across neighboring tiles. After pre-training, the encoder is frozen and evaluated under a unified downstream protocol. Lightweight linear, MLP, and spatial heads are trained with sparse labels for building extraction, road extraction, water detection, urban semantic mapping, and query-by-example retrieval. All downstream comparisons use the same train/validation/test split, validation-selected thresholds, and identical label budgets to reduce protocol-driven bias.

Experiments show that Xuannv is especially effective when labels are limited. In the 5-shot setting, compared with strong raw-feature baselines trained directly on multi-source imagery, Xuannv improves F1 by 9.8%, 18.3%, and 41.5% on building, road, and water extraction, respectively. In the 50-shot six-task benchmark covering buildings, roads, water, park/green space, education areas, and sports pitches, Xuannv achieves an average F1 of 0.552 and outperforms raw-feature supervised models on five of six tasks. Even simple MLP heads obtain usable performance on several categories, indicating that the learned embeddings encode transferable urban semantics rather than only task-specific cues.

The project is designed as an open GeoAI practice workflow rather than a closed single-task model. We will release trained weights, monthly embeddings, training configurations, data documentation, downstream evaluation scripts, visualization examples, and reproducible ModelScope notebooks. These resources are intended to let users inspect the learned embedding space, reproduce the reported metrics, and quickly build new sparse-label mapping heads for additional urban categories. By converting complex multi-source remote-sensing data into reusable monthly embeddings, Xuannv provides a practical path for rapid urban mapping with limited manual labels and offers a reproducible foundation for future regional geospatial representation learning.

**Index Terms**--geospatial embeddings, remote sensing foundation models, sparse-label mapping, urban GeoAI, multimodal representation learning, ModelScope
