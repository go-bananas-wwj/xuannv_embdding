# Learning Monthly Multimodal Earth Observation Embedding Fields for Label-Efficient Urban Mapping

> Target venue: *Journal of Remote Sensing*, Special Issue on Foundation Models based Multimodal Earth Observation Data Fusion and Applications  
> Article type: Research Article  
> Status: evidence-aware manuscript framework; bracketed fields must be replaced only after the corresponding registered experiment is complete  
> Target length: 8,000-10,000 words; abstract <=250 words; main-text figures and tables combined <=10

## One-Sentence Paper Claim

XuannvEarth learns reusable 10 m, 64-dimensional monthly embedding fields from multimodal Earth observation data and sparse high-resolution observations, enabling label-efficient urban mapping with frozen features and lightweight task heads under spatially independent evaluation.

This claim deliberately does **not** assert that XuannvEarth is a global foundation model, that it universally outperforms annual embeddings, or that monthly modeling improves change detection until the corresponding experiments are complete.

## Candidate Titles

1. **Learning Monthly Multimodal Earth Observation Embedding Fields for Label-Efficient Urban Mapping**
2. **XuannvEarth: City-Scale Monthly Multimodal Embeddings for Label-Efficient Urban Mapping**
3. **Reusable Monthly Earth Observation Embeddings from Multisensor Reconstruction and Weak Geographic Semantics**

Title 1 is preferred because it states the scientific object and evaluation goal without overstating model scale.

## Abstract Skeleton (Maximum 250 Words)

**Background.** Dense geospatial embeddings can amortize the cost of repeated urban mapping, but existing products are commonly optimized at annual cadence or global scale, while monthly city-scale mapping must cope with clouds, missing observations, sensor noise, and limited labels.

**Gap.** It remains unclear whether a compact monthly representation learned from heterogeneous observations can preserve local spatial detail and reduce downstream annotation requirements under spatially independent evaluation.

**Method.** We present XuannvEarth, a multimodal encoder that converts six months of Sentinel-2, Sentinel-1, Landsat, and sparse high-resolution optical and SAR observations into a 64-channel embedding field at 10 m spacing. Training combines sensor reconstruction under structured input corruption, high-resolution detail reconstruction, embedding-distribution regularization, and cleaned OpenStreetMap-derived weak semantic objectives.

**Results.** Across spatial five-fold evaluation and [three] random seeds, frozen XuannvEarth embeddings with identical lightweight heads achieve [RESULT: mean +/- standard deviation for the primary few-shot tasks] relative to raw multisensor features and annual embedding baselines. Scaling, ablation, cloud-quality, temporal-input, and cross-city experiments show [RESULT: only insert conclusions supported beyond uncertainty].

**Significance.** These results establish [CLAIM PENDING] and clarify the conditions under which regional monthly embedding fields provide practical label efficiency for repeated urban mapping.

## 1. Introduction

### 1.1 Operational motivation

- Urban mapping is not a single segmentation task. The same region is repeatedly queried for buildings, roads, water, vegetation, land use, facilities, and change.
- Training a separate high-capacity model from raw imagery for every new class repeats feature learning and annotation work.
- A reusable embedding field changes the interface: compute the representation once, then fit a small task head from sparse labels.

### 1.2 Scientific gap

- Global and annual geospatial embeddings demonstrate broad transfer, but an annual representation can suppress month-scale state changes.
- Monthly modeling has less temporal redundancy and is more exposed to clouds, haze, missing modalities, SAR speckle, and acquisition misalignment.
- City-scale training offers dense local observations and high-resolution sources, but creates risks of geographic overfitting, weak-label leakage, and transductive evaluation.
- Therefore, the key question is not whether a regional model is larger than a global model, but whether a compact monthly field can improve label efficiency while preserving spatial detail and surviving strict spatial holdout.

### 1.3 Research questions

**RQ1. Representation utility.** Do frozen monthly embeddings support diverse urban mapping tasks with linear or shallow convolutional heads?

**RQ2. Label efficiency.** At matched 5-, 10-, and 50-shot budgets, how do the embeddings compare with raw multisensor features and external geospatial representations?

**RQ3. Training recipe.** What are the separate contributions of data scale, OSM weak semantics, hard-negative sampling, high-resolution reconstruction, and structured input corruption?

**RQ4. Robustness and transfer.** How does performance vary with cloud quality, missing modalities, temporal context, and a second city?

### 1.4 Contributions

1. We formulate a city-scale monthly embedding field that retains a dense 128 x 128 grid of 64-dimensional features for every 1.28 km x 1.28 km patch.
2. We develop a multimodal training recipe that combines reconstruction under structured input corruption, sparse high-resolution detail supervision, and cleaned OSM-derived weak semantic regularization without using the final manually annotated test labels.
3. We introduce a leakage-aware evaluation protocol with spatial five-fold splits, a one-patch buffer, validation-only threshold selection, matched downstream heads, and repeated few-shot trials.
4. We provide controlled scaling, ablation, robustness, temporal, and cross-city analyses, including negative results that separate embedding diversity from downstream utility.

### 1.5 Introduction closing paragraph

End the Introduction with a concise roadmap and the one-sentence claim. Do not list implementation details here.

## 2. Related Work

### 2.1 Geospatial embedding fields and Earth observation foundation models

- Annual embedding products and sparse-label mapping.
- EO foundation encoders and multimodal representations.
- Distinguish reusable dense fields from task-specific segmentation networks.

### 2.2 Multimodal and temporal Earth observation learning

- Optical-SAR fusion, missing-modality learning, and temporal aggregation.
- Monthly observations as a distinct operating regime from annual compositing.

### 2.3 Masked reconstruction and representation regularization

- Masked autoencoding and modality/source dropout.
- Cross-modal reconstruction and feature-distribution regularization.
- State clearly that XuannvEarth corrupts model inputs and reconstructs valid targets; do not call the objective a masked-pixel-only loss unless verified.

### 2.4 Weak geographic supervision

- OSM-derived labels as noisy, incomplete semantic priors.
- Risks of label incompleteness, spatial bias, and benchmark leakage.
- Position the work as weakly supervised multimodal representation learning, not purely self-supervised learning.

### 2.5 Positioning statement

The novelty is the evaluated system-level combination for a monthly regional embedding field, not a claim that each individual loss or sensor fusion component is new.

## 3. Materials and Methods

### 3.1 Experimental and Technical Design

This must be the first Methods subsection to match the journal guidance.

Describe Figure 1 from left to right:

1. Six monthly observations are assembled from Sentinel-2, Sentinel-1, and Landsat.
2. Sparse high-resolution optical and SAR observations provide detail targets where available.
3. Cloud/validity masks and source availability control valid supervision.
4. Structured corruption drops selected modalities, months, and spatial blocks from the inputs.
5. A multimodal spatiotemporal encoder produces one dense 64-dimensional field per month.
6. Reconstruction, weak semantic, and distribution regularization objectives train the encoder.
7. At evaluation time, the encoder is frozen and identical lightweight heads are fitted from matched labels.

State two evaluation settings explicitly:

- **Spatially independent scientific setting:** test and validation blocks never enter embedding training or checkpoint selection.
- **Full-region transductive product setting:** the production P10C model uses all 320 Haidian patches and is reported only as a deployment-oriented case study.

### 3.2 Study areas and spatial units

- Haidian District: 320 georeferenced patches, each 1.28 km x 1.28 km.
- Harbin study area: [VERIFY number and geographic description before writing].
- Define coordinate reference system, tiling origin, spatial coverage, urban/rural composition, and date range.
- Explain the geographic five-fold partition, adjacent validation band, and one-patch exclusion buffer.

### 3.3 Multisource observations and quality control

Create a compact table covering source, channels, native resolution, temporal availability, preprocessing, quality masks, and training role.

- Sentinel-2: 12 channels.
- Sentinel-1: 2 channels.
- Landsat: 7 channels.
- High-resolution optical: 3 channels, sparsely observed.
- High-resolution SAR: 1 channel, sparsely observed.
- Period used by the current six-month configuration: December 2025 through May 2026.

Document cloud screening, valid-pixel masks, geometric alignment checks, quality-based compositing, normalization statistics, and how the 30 April observation is assigned. Do not imply that every high-resolution source is independently observed in every month.

### 3.4 OSM-derived weak semantic data

- Explain tag extraction, removal of invalid geometry, rasterization, class merging, and positive-coverage audit.
- Broad land-cover-style supervision is stored under a historical `worldcover` configuration alias but is derived from the cleaned OSM taxonomy; use the scientific name in the manuscript.
- The fine semantic probe contains 13 OSM-derived categories in the production recipe.
- Explain incompleteness: unlabeled pixels are not automatically reliable negatives.
- Specify which evaluation labels are independent, which are OSM-derived, and which results are therefore diagnostic rather than leakage-free evidence.

### 3.5 XuannvEarth architecture

Use a notation table before equations.

- Sensor-specific stems map each source to a shared feature space.
- The spatiotemporal processor uses six blocks, eight attention heads, 512-dimensional spatial, 256-dimensional temporal, and 128-dimensional precision components in the registered P10C configuration.
- Temporal fusion uses a gated sum with full temporal attention.
- The output is a 64 x 128 x 128 embedding tensor for each month.
- High-resolution features are fused into the embedding path when available.

Insert a code-audited architecture diagram and parameter count. Do not describe the production model as a von Mises-Fisher bottleneck unless the selected paper checkpoint actually enables unit-sphere normalization.

### 3.6 Training objectives

Use one total objective rather than presenting every term as an unrelated novelty:

\[
\mathcal{L}=\mathcal{L}_{\mathrm{recon}}+
\lambda_{\mathrm{sem}}\mathcal{L}_{\mathrm{sem}}+
\lambda_{\mathrm{uni}}\mathcal{L}_{\mathrm{uni}}.
\]

Define multimodal reconstruction as a weighted sum:

\[
\mathcal{L}_{\mathrm{recon}}=
\sum_{s \in \mathcal{S}} \lambda_s
\mathcal{L}_s(\hat{\mathbf{x}}_s,\mathbf{x}_s;\mathbf{m}_s),
\]

where validity mask \(\mathbf{m}_s\) excludes unavailable or invalid targets. The registered production weights are 0.80 (Sentinel-2), 0.25 (Sentinel-1), 0.45 (Landsat), 0.90 (high-resolution optical), 0.35 (high-resolution SAR), and 0.45 for the coarse OSM semantic target. Verify the exact per-source loss form in code before converting this outline to manuscript prose.

The fine OSM semantic objective uses a weighted binary classification loss with task-specific positive weights and a 2% hard-negative subset weighted by 0.35 after warm-up. The embedding-distribution regularizer has weight 0.06 after warm-up. The final paper must provide the exact implemented formulas and sampling domains.

### 3.7 Structured input corruption

Describe three mechanisms and their probabilities:

- Modality dropout: 0.18 for Sentinel-2, 0.35 for Sentinel-1, and 0.35 for Landsat.
- Month dropout: probability 0.65, with up to four months dropped per sample.
- Spatial block corruption: probability 0.65, block size 16, corruption ratio 0.32.

Clarify that the corruption is applied to inputs while supervision is evaluated on valid targets. Include a no-corruption ablation.

### 3.8 Optimization and checkpointing

- 800 epochs; Adam-family optimizer [VERIFY exact implementation].
- Learning rate 2 x 10^-6; weight decay 0.05; 30 warm-up epochs.
- Mixed precision, gradient checkpointing, and gradient accumulation of two.
- Checkpoints evaluated every 20 epochs and saved every 200 epochs.
- Define the checkpoint selection metric using validation data only. Do not call epoch 800 “best” without downstream-independent evidence.

### 3.9 Frozen-feature downstream protocol

- Primary heads: linear and shallow 3 x 3 convolutional probe; stronger heads are secondary capacity analyses.
- 5-shot means five positive training patches plus a matched number of negative patches. Define 10- and 50-shot analogously.
- Use identical labels, spatial split, shot samples, head architecture, optimizer, seed, and training budget for XuannvEarth, AEF, DINO-family, and raw-feature baselines.
- Select decision thresholds on validation predictions only; report test F1 at that fixed threshold.
- Keep per-image min-max visualization out of quantitative evaluation because it destroys calibration.

### 3.10 Statistical analysis

- Five spatial folds x three downstream seeds.
- Report mean, standard deviation, and spatial-patch paired 95% confidence intervals.
- Primary metrics: F1 at validation-selected threshold and average precision.
- Secondary metrics: IoU and ROC-AUC; explain why ROC-AUC can be optimistic under severe class imbalance.
- Report effect sizes and paired differences, not only independent aggregate means.
- Predefine handling of folds with no positive pixels and missing class coverage.

## 4. Results

### 4.1 Main spatially independent benchmark

Answer RQ1 with Table 2 and Figure 3. Report frozen-feature performance for buildings, roads, and water first; extended OSM-derived tasks are secondary. Separate independent-label results from OSM-derived diagnostic results.

**Required result placeholder:** `[E3/E6: 5-fold x 3-seed mean +/- SD, paired CI, validation-threshold F1, AP, IoU, AUC]`.

### 4.2 Label-efficiency curves

Answer RQ2 with 5-, 10-, and 50-shot curves. Compare identical heads and annotation budgets. Report absolute difference and relative improvement with uncertainty.

**Required result placeholder:** `[E3/E6: label-efficiency curves and spatially paired significance]`.

### 4.3 Data scaling and recipe ablation

Answer RQ3 using the strictly nested 40/80/160 training subsets and clean-from-scratch ablations:

- no OSM supervision;
- coarse OSM only;
- fine semantic probe without hard negatives;
- full recipe;
- no high-resolution detail injection;
- no structured input corruption.

Old overlapping or contaminated-initialization runs must not enter this causal table.

### 4.4 Temporal context and observation quality

Compare 1-, 3-, and 6-month input, temporal pooling, and month-order shuffling. Stratify by clear-pixel rate and source availability. Only after this section may the paper claim an advantage from monthly temporal modeling.

### 4.5 Cross-city reproducibility and transfer

Separate two questions:

- Training the same recipe from scratch in Harbin tests recipe reproducibility.
- Applying the Haidian encoder directly to Harbin tests geographic transfer.

Do not call the first setting zero-shot transfer.

### 4.6 Representation diagnostics and negative results

- Effective rank, per-channel variance, class separation, retrieval precision, and patch-boundary continuity.
- Explain that a higher effective rank is not sufficient for better few-shot utility.
- Include failed distillation or overly complex downstream-head results only when they illuminate the representation-design trade-off.

### 4.7 Full-region product case study

Show the P10C embedding mosaic and downstream maps over all 320 patches. Label this section explicitly as a transductive deployment case study; do not merge its numbers with spatially independent evidence.

## 5. Discussion

### 5.1 What the representation learns

Interpret the relationship between boundary detail, weak semantics, temporal context, and downstream label efficiency. Tie each interpretation to an ablation or diagnostic.

### 5.2 Why monthly regional embeddings can be useful

Discuss repeated mapping, reusable storage, fast head fitting, and local sensor availability. Avoid unsupported storage-ratio claims; compare exact tensor sizes and preprocessing costs.

### 5.3 What the comparison with annual embeddings means

Discuss cadence, training scale, geographic scope, and input differences. A matched downstream protocol makes probe performance comparable, but does not make the upstream models equivalent.

### 5.4 Limitations

- Primary development in one city and six months.
- OSM incompleteness, label bias, and partial task overlap.
- Sparse high-resolution observations and imperfect temporal correspondence.
- Remaining cloud, registration, and patch-boundary artifacts.
- Regional model scale and uncertain out-of-region transfer.
- Computational cost and environmental footprint to be quantified.

### 5.5 Practical implications and future work

- More cities and years, independent labels, overlap-crop inference, uncertainty maps, and retrieval-assisted annotation.
- Avoid promising global generalization without a corresponding training and evaluation program.

## 6. Conclusion

One paragraph only:

1. Restate the problem.
2. State the framework.
3. State the strongest supported quantitative result.
4. State the practical implication and one honest boundary.

## Declarations Required Before Submission

### Acknowledgments

`[AUTHORS TO COMPLETE]`

### Funding

`[AUTHORS TO COMPLETE]`

### Author Contributions

Use CRediT roles. `[AUTHORS TO COMPLETE]`

### Competing Interests

`[AUTHORS TO COMPLETE]`

### Data and Materials Availability

- Code repository and immutable release tag.
- ModelScope checkpoint, embeddings, downstream heads, and checksums.
- Download/preprocessing instructions for redistributable source data.
- Split manifests, label provenance, and experiment configurations.
- Explicit restrictions for imagery that cannot be redistributed.

### Prior Publication and Related Papers

Disclose the APGARSS abstract and explain the additional methods, experiments, and analysis in this full paper.

## Main-Text Figure and Table Budget

| No. | Type | Purpose |
|---|---|---|
| 1 | Figure | Experimental and technical design overview |
| 2 | Figure | Study area, monthly multisource observations, masks, and OSM weak labels |
| 3 | Table | Data sources, architecture, and training specification |
| 4 | Table | Main spatial 5-fold x 3-seed benchmark |
| 5 | Figure | Label-efficiency curves and representative qualitative maps |
| 6 | Figure | Strict scaling and causal ablation results |
| 7 | Figure | Temporal context, cloud quality, and missing-source robustness |
| 8 | Figure | Cross-city reproducibility and zero-shot transfer |
| 9 | Figure | Representation diagnostics, retrieval, and negative-result analysis |
| 10 | Figure | Full-region 320-patch product case study and comparison mosaic |

All additional head comparisons, per-fold tables, extended OSM categories, hyperparameters, and extra mosaics belong in Supplementary Materials.

## Manuscript Integrity Rules

1. Never replace a result placeholder with a number lacking a traceable JSON/CSV artifact.
2. Never use test labels for threshold selection, checkpoint selection, or ablation choices.
3. Never merge transductive P10C results with spatially independent paper results.
4. Never describe OSM-supervised training as fully self-supervised.
5. Never infer temporal sensitivity from PCA appearance alone.
6. Never infer semantic quality from effective rank alone.
7. Report unfavorable baselines and failed ablations when they answer a registered research question.
8. Cite primary papers and official repositories; verify every DOI and numerical claim.

