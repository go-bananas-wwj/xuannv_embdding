# XuannvEarth: Spatial Evaluation of Monthly-Indexed Multimodal Urban Embeddings

> Target venue: *Journal of Remote Sensing*, Special Issue on Foundation Models based Multimodal Earth Observation Data Fusion and Applications  
> Article type: Research Article  
> Status: evidence-aware manuscript framework; bracketed fields must be replaced only after the corresponding registered experiment is complete  
> Target length: 8,000-10,000 words; abstract <=250 words; main-text figures and tables combined <=10

## Authors and Affiliations

`[AUTHOR NAMES, ORCIDS, AFFILIATIONS, CORRESPONDING AUTHOR, AND EMAIL TO COMPLETE]`

## One-Sentence Paper Claim

XuannvEarth produces candidate reusable 10 m, 64-dimensional monthly-indexed embedding fields from multimodal Earth observation data and aggregated finer-resolution observations, and we test spatial task transfer across crossed encoder-supervision, downstream-task-relation, and label-source conditions.

This claim deliberately does **not** assert that XuannvEarth is a global foundation model, that it universally outperforms annual embeddings, or that monthly modeling improves change detection until the corresponding experiments are complete.

## Candidate Titles

1. **XuannvEarth: Spatial Evaluation of Monthly-Indexed Multimodal Urban Embeddings**
2. **Learning Monthly Multimodal Earth Observation Embedding Fields for Labeled-Patch-Efficient Urban Mapping**
3. **Reusable Monthly Earth Observation Embeddings from Multisensor Reconstruction and Weak Geographic Semantics**

Title 1 is preferred because it states the scientific object and evaluation design without presupposing superiority. “Labeled-patch-efficient” may move into the title only if spatial five-fold held-out-category or no-OSM results with independent labels support it. Broader “annotation-efficient” wording additionally requires matched annotation time, valid labeled pixels, positive pixels, and object counts.

## Abstract Skeleton (One Unstructured Paragraph; Maximum 250 Words)

Dense geospatial embeddings can amortize repeated urban mapping, but monthly city-scale representations must cope with clouds, missing observations, sensor noise, weak labels, and limited independent annotations. Existing annual or globally pretrained products provide strong transfer references, yet they do not by themselves establish whether a compact regional representation can preserve month-indexed spatial detail, reduce labeled-patch requirements, and remain useful on geographically held-out urban areas. We present XuannvEarth, which converts six months of Sentinel-2, Sentinel-1, and Landsat observations plus temporally aggregated finer-resolution optical and synthetic aperture radar context into six 10 m, 64-channel fields using valid-target reconstruction under structured input corruption, a hyperspherical bottleneck inspired by the von Mises-Fisher distribution, embedding regularization, and OpenStreetMap-derived auxiliary weak supervision. Under a preregistered spatial five-fold protocol, we compare frozen features through the same 64-channel shallow convolutional probe, validation-selected thresholds, and paired labeled-patch budgets across crossed conditions for encoder OSM supervision, downstream OSM overlap, and OSM-derived versus independent labels; the verified benchmark, scaling, ablation, observation-quality, temporal-context, and cross-city results will populate [PRIMARY RESULTS]. The completed evidence will support only [VERIFIED, CLAIM-BOUNDED SIGNIFICANCE], conditional on regional training scale, upstream supervision asymmetry, one-encoder-seed uncertainty where applicable, and [PRIMARY LIMITATION].

## 1. Introduction

### 1.1 Operational motivation

- Urban mapping is not a single segmentation task. The same region is repeatedly queried for buildings, roads, water, vegetation, land use, facilities, and change.
- Training a separate high-capacity model from raw imagery for every new class repeats feature learning and annotation work.
- A reusable embedding field changes the interface: compute the representation once, then fit a small task head from sparse labels.

### 1.2 Scientific gap

- Global and annual geospatial embeddings demonstrate broad transfer, but an annual representation can suppress month-scale state changes.
- Monthly modeling has less temporal redundancy and is more exposed to clouds, haze, missing modalities, SAR speckle, and acquisition misalignment.
- City-scale training offers dense local observations and finer-resolution sources, but creates risks of geographic overfitting, weak-label leakage, and transductive evaluation.
- Therefore, the key question is not whether a regional model is larger than a global model, but whether a compact monthly field can improve labeled-patch efficiency while preserving spatial detail and surviving strict spatial holdout.

### 1.3 Research questions

**RQ1. Representation utility.** Do frozen monthly-indexed embeddings support diverse urban mapping tasks with a preregistered shallow convolutional head?

**RQ2. Labeled-patch efficiency.** At matched 5-, 10-, and feasible 50-shot budgets, how do the embeddings compare with raw multisensor features and external geospatial representations across encoder OSM supervision, downstream OSM overlap, and OSM-derived versus independent label sources?

**RQ3. Training recipe.** What performance associations are observed for data scale, OSM auxiliary supervision, the hardest-2%-negative loss term, the combined finer-resolution-source pathway, and structured input corruption; which associations remain stable across at least three independently initialized encoders; and can a 2 x 2 experiment separate finer-resolution input fusion from reconstruction supervision? The OSM-dependent patch sampler remains fixed in the hardest-negative-loss ablation.

**RQ4. Robustness and transfer.** How does performance vary with cloud quality, missing modalities, temporal context, and a second city?

### 1.4 Contributions

1. We formulate a city-scale monthly-indexed embedding field that retains a dense 128 x 128 grid of 64-dimensional features for every 1.28 km x 1.28 km patch.
2. We develop a multimodal training recipe that combines reconstruction under structured input corruption, 10 m-grid supervision derived from finer-resolution sources, and cleaned OSM-derived auxiliary weak supervision without using the final manually annotated test labels.
3. We introduce a leakage-aware evaluation protocol with spatial five-fold splits, a one-patch buffer, validation-only threshold selection, a preregistered matched downstream head, and repeated few-shot trials.
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
- State clearly that XuannvEarth corrupts model inputs and reconstructs all originally valid targets; this is not a masked-pixel-only loss.

### 2.4 OSM-derived auxiliary weak supervision

- OSM-derived labels as noisy, incomplete semantic priors.
- Risks of label incompleteness, spatial bias, and benchmark leakage.
- Use **OSM-derived auxiliary weak supervision** for the data and training paradigm, and **OSM auxiliary loss** for its implemented loss term. Do not call this direct supervision a mere prior.
- Position the work as weakly supervised multimodal representation learning, not purely self-supervised learning.

### 2.5 Positioning statement

The novelty is the evaluated system-level combination for a monthly regional embedding field, not a claim that each individual loss or sensor fusion component is new.

## 3. Materials and Methods

### 3.1 Experimental and Technical Design

This must be the first Methods subsection to match the journal guidance.

Describe Figure 1 from left to right:

1. Six monthly observations are assembled from Sentinel-2, Sentinel-1, and Landsat.
2. Sparse finer-resolution optical and SAR observations provide date-specific reconstruction targets; their input features are aggregated across available dates before fusion.
3. Cloud/validity masks and source availability control valid supervision.
4. Structured corruption drops selected modalities, months, and spatial blocks from the inputs.
5. A multimodal spatiotemporal encoder with full cross-month attention produces six monthly-indexed dense 64-dimensional fields. These are not causal, current-month-only encodings.
6. Reconstruction, weak semantic, and distribution regularization objectives train the encoder.
7. At evaluation time, the encoder is frozen and identical lightweight heads are fitted from matched labels.

State two evaluation settings explicitly:

- **Spatially independent scientific setting:** test and validation blocks never enter encoder fitting. The registered validation band alone selects the encoder checkpoint and downstream threshold; the test band remains untouched until final evaluation.
- **Full-region transductive product setting:** the production P10C model uses all 320 Haidian patches and is reported only as a deployment-oriented case study.

### 3.2 Study areas and spatial units

- Haidian District: 320 georeferenced patches, each 1.28 km x 1.28 km.
- Harbin study area: [VERIFY number and geographic description before writing].
- Define coordinate reference system, tiling origin, spatial coverage, urban/rural composition, and date range.
- Explain the geographic five-fold partition, adjacent validation band, and one-patch exclusion buffer.

### 3.3 Multisource observations and quality control

Use Figure 2 to introduce the study grid, representative multisource observations, validity/cloud masks, OSM weak labels, and the full-region PCA context before any result figure is cited.

Create a compact table covering source, channels, native resolution, temporal availability, preprocessing, quality masks, and training role.

- Sentinel-2: 12 channels.
- Sentinel-1: 2 channels.
- Landsat: 7 channels.
- Finer-resolution optical: 3 channels, sparsely observed; approximately 3 m source pixels are resampled to the 10 m embedding grid, and input features are temporally aggregated and reused across monthly slots.
- Finer-resolution SAR: 1 channel, sparsely observed; source pixels are resampled to the 10 m embedding grid, and input features are temporally aggregated and reused across monthly slots.
- Period used by the current six-month configuration: December 2025 through May 2026.

Document cloud screening, valid-pixel masks, geometric alignment checks, quality-based compositing, normalization statistics, and how the 30 April observation is assigned. Do not imply that every finer-resolution source is independently observed in every month.

### 3.4 OSM-derived weak semantic data

- Explain tag extraction, removal of invalid geometry, rasterization, class merging, and positive-coverage audit.
- Record the authoritative OSM snapshot/extract timestamp, provider/version, download URL or replication sequence, raw checksum, processing commit, and its relationship to the December 2025-May 2026 imagery window. If the snapshot contains later edits, describe it as noncausal spatial weak supervision and forbid prospective or “available at image date” claims; a causal mapping experiment requires an OSM cutoff available by the claimed date.
- Broad land-cover-style supervision is stored under a historical `worldcover` configuration alias but is derived from the cleaned OSM taxonomy; use the scientific name in the manuscript.
- The fine semantic probe contains 13 OSM-derived categories in the production recipe.
- Training patch sampling is also OSM-dependent: for each task, the sampler records whether a patch contains any positive pixel, sums the configured task weights, applies `1 + 2.5 x score`, and caps the with-replacement sampling weight at 5.0. This distribution shift must be disclosed separately from pixel-level hard-negative mining.
- For P10C and every scientific fold, scaling run, and ablation with OSM-weighted sampling enabled, persist and checksum the per-patch weights plus minimum, maximum, mean, clipped fraction, weighted-patch count, rank sampler seeds, and sampled-ID histogram. Current P10C log inspection suggests near-saturation, so no sampling-effect claim is allowed until that audit is traceable and the same gate passes for every compared run.
- Explain incompleteness: unlabeled pixels are not automatically reliable negatives.
- Specify which evaluation labels are independent, which are OSM-derived, and which results are therefore diagnostic rather than leakage-free evidence.

### 3.5 XuannvEarth architecture

Use a notation table before equations.

- Each temporal source uses an independent 3 x 3 convolution, GroupNorm, and ReLU stem to produce 32 channels; availability-aware gated summation combines the sources.
- The spatiotemporal processor uses six blocks and eight attention heads. Its spatial path is 8 x 8 x 512, temporal path is 16 x 16 x 256, and precision path is 128 x 128 x 128 in the registered P10C implementation.
- Temporal fusion uses a gated sum with full temporal attention.
- A 1 x 1 projection followed by L2 normalization gives a unit-norm 64 x 128 x 128 embedding tensor for each of six monthly slots.
- During training, isotropic Gaussian noise with learned scale `1/exp(log_kappa)` is added before a second projection to the unit sphere. Describe this as a **vMF-style hyperspherical bottleneck**, not exact vMF likelihood training.
- Finer-resolution features are fused before the bottleneck when available. The same temporally aggregated finer-resolution feature is repeated across monthly slots.

Insert a code-audited architecture diagram and parameter count. Explain that full temporal attention allows a monthly-indexed output to use observations from other months.

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

where validity mask \(\mathbf{m}_s\) excludes unavailable or invalid targets. The registered production weights are 0.80 (Sentinel-2), 0.25 (Sentinel-1), 0.45 (Landsat), 0.90 (finer-resolution-source optical resampled to 10 m), 0.35 (finer-resolution-source SAR resampled to 10 m), and 0.45 for the coarse OSM semantic target. Continuous targets use channel-averaged L1 over valid pixels. The categorical target uses masked cross-entropy with class 0 ignored; this static coarse OSM target is copied to all six monthly slots.

The fine OSM semantic objective supervises only the final monthly slot (May 2026) through 13 independent 1 x 1 linear probes. Each task combines positive-weighted BCE, Dice loss, and an additional hardest-2%-negative BCE term. The hard-negative term is added on top of BCE that already contains all negative pixels. The task-averaged semantic weight ramps linearly to 0.14 over 80 epochs; the hard-negative multiplier ramps to 0.35 over 120 epochs. The within-rank uniformity term ramps to 0.06 over 60 epochs and operates on spatially pooled monthly vectors. Covariance, patch discrimination, temporal contrast, supervised change, distillation, prototype, boundary, and latent reconstruction terms are inactive in P10C and must not be listed as trained objectives.

### 3.7 Structured input corruption

Describe three mechanisms and their probabilities:

- Modality dropout: 0.18 for Sentinel-2, 0.35 for Sentinel-1, and 0.35 for Landsat.
- Month dropout: probability 0.65, with up to four months dropped per sample.
- Spatial block corruption: probability 0.65, block size 16, corruption ratio 0.32.

Clarify that the corruption is applied to inputs while supervision is evaluated on valid targets. Include a no-corruption ablation.

### 3.8 Optimization and checkpointing

- 800 epochs with AdamW using the PyTorch default betas (0.9, 0.999).
- Learning rate 2 x 10^-6; weight decay 0.05; 30 linear warm-up epochs followed by cosine decay.
- The recorded production run used one node with two NPUs, batch size 3 per NPU, gradient accumulation of two, and nominal effective global batch size 12, with mixed precision and gradient checkpointing. Because `drop_last=true` leaves a final residual accumulation, also report the exact updates and processed instances per epoch from the audited training log.
- Checkpoints evaluated every 20 epochs and saved every 200 epochs.
- The recorded production lineage is at least `P7A best -> P8A best -> P9A epoch 400 -> P9B epoch 800 -> P10C epoch 800`. P10C loaded P9B model weights, then trained for 800 P10C epochs with a newly initialized optimizer and scheduler; it is not a from-scratch scientific replicate.
- Under the old checkpoint implementation, `best.pt` compared the aggregate validation objective only at the 200/400/600/800 save points and resolved to epoch 600. Earlier 20-epoch validation records contain a lower aggregate objective, so epoch 600 is not the global validation minimum; reconstruction-only minima are not the checkpoint criterion. Report this legacy selection bug without inserting exact values until the source log and parsed audit table are checksummed. The production artifact uses epoch 800, and clean paper models use the corrected validation-only selection implementation.

### 3.9 Frozen-feature downstream protocol

- Preregister one shallow 3 x 3 convolutional probe as the primary matched head. Linear, MLP, UNet, and DeepLab-style heads are secondary capacity analyses and cannot be selected by test performance.
- The exact few-shot sampling unit, feasibility rule, normalization, temporal information budget, threshold grid, and confidence-interval estimator are fixed in `registered_evaluation_protocol.md` before any main-paper result is computed.
- Use identical labels, spatial split, shot samples, head architecture, optimizer, seed, and training budget for XuannvEarth, AEF, DINO-family, and raw-feature baselines.
- Select decision thresholds on validation predictions only; report test F1 at that fixed threshold.
- Keep per-image min-max visualization out of quantitative evaluation because it destroys calibration.

### 3.10 Statistical analysis

- Five spatial folds x three downstream seeds. If the embedding encoder itself is trained only once per fold, state this as “one encoder seed x three probe/few-shot seeds,” not three independent pretraining seeds.
- Report mean, descriptive standard deviation, and paired 95% confidence intervals from a hierarchical spatial bootstrap. Pixels are not independent replicates.
- Primary metrics: F1 at validation-selected threshold and average precision.
- Secondary metrics: IoU and ROC-AUC; explain why ROC-AUC can be optimistic under severe class imbalance.
- Report effect sizes and paired differences, not only independent aggregate means. Use Holm correction if multiple confirmatory p-values are presented.
- Predefine handling of folds with no positive pixels and missing class coverage.

## 4. Results

### 4.1 Main spatially independent benchmark

Answer RQ1 with the main benchmark table and qualitative mapping Figure 3. Lead with held-out-category, no-OSM, and independent-label evidence. Report buildings, roads, and water as the operational OSM-overlap stratum, with the upstream supervision asymmetry stated in the table header and caption.

**Required result placeholder:** `[E3/E6: five folds x one frozen encoder checkpoint per fold x three downstream shot/probe seeds; mean +/- SD, paired CI, validation-threshold F1, AP, IoU, AUC]`.

### 4.2 Labeled-patch-efficiency curves

Answer RQ2 with Figure 4 using 5- and 10-shot curves as the confirmatory budgets. The feasible 50-shot result is secondary and appears only for exact-budget task-fold cells. Compare identical heads and labeled-patch budgets. Report absolute difference and relative improvement with uncertainty. Call this “labeled-patch efficiency” unless annotation time, valid labeled pixels, positive pixels, and object counts have also been compared. A broader annotation-efficiency claim requires held-out-category or no-OSM evidence with independent labels plus those cost measures; OSM-assisted overlap tasks alone support only conditional labeled-patch efficiency.

**Required result placeholder:** `[E3/E6: labeled-patch-efficiency curves and spatially paired uncertainty]`.

### 4.3 Data scaling and recipe ablation

Answer RQ3 with Figure 5 using strictly nested 40/80/150 training subsets within every fold and clean-from-scratch ablations. Treat one-encoder-seed results as exploratory associations; reserve causal or stable-recipe language for comparisons repeated with at least three independently initialized encoders:

- no OSM supervision;
- coarse OSM only;
- fine semantic probe without hard negatives;
- full recipe;
- no finer-resolution-source pathway (combined input and reconstruction removal);
- no structured input corruption.

The combined finer-resolution-pathway ablation cannot distinguish input-fusion effects from reconstruction-supervision effects. A two-factor ablation is required before making either narrower causal claim. The previous 160-patch pilot is valid only for fold 0 because other folds contain fewer than 160 eligible training patches. Old overlapping or contaminated-initialization runs must not enter this causal table.

### 4.4 Temporal context and observation quality

Answer the temporal and observation-quality part of RQ4 with Figure 6 by comparing separately trained and information-matched 1-, 3-, and 6-month models, temporal pooling, and month-order controls. Test-time deletion or shuffling of a six-month model is only a distribution-shift robustness test, not causal evidence for temporal modeling. Stratify by clear-pixel rate and source availability. Only after this section may the paper claim an advantage from monthly temporal modeling.

### 4.5 Cross-city reproducibility and transfer

Answer the cross-city part of RQ4 with Figure 7 by separating two questions:

- Training the same recipe from scratch in Harbin tests recipe reproducibility.
- Applying the Haidian encoder directly to Harbin tests geographic transfer.

Do not call the first setting zero-shot transfer.

### 4.6 Representation diagnostics and negative results

- Effective rank, per-channel variance, class separation, retrieval precision, and patch-boundary continuity.
- Explain that a higher effective rank is not sufficient for better few-shot utility.
- Include failed distillation or overly complex downstream-head results only when they illuminate the representation-design trade-off.

### 4.7 Full-region product case study

Use Figure 8 to show the P10C embedding mosaic, downstream maps over all 320 patches, and retrieval use case. Label this section explicitly as a transductive deployment case study; do not merge its numbers with spatially independent evidence.

## 5. Discussion

### 5.1 What the representation learns

Interpret the relationship between boundary detail, weak semantics, temporal context, and downstream labeled-patch efficiency. Tie each interpretation to an ablation or diagnostic.

### 5.2 Why monthly regional embeddings can be useful

Discuss repeated mapping, reusable storage, fast head fitting, and local sensor availability. Avoid unsupported storage-ratio claims; compare exact tensor sizes and preprocessing costs.

### 5.3 What the comparison with annual embeddings means

Discuss cadence, training scale, geographic scope, and input differences. A matched downstream protocol makes probe performance comparable, but does not make the upstream models equivalent.

### 5.4 Limitations

- Primary development in one city and six months.
- OSM incompleteness, label bias, and partial task overlap.
- Sparse finer-resolution observations and imperfect temporal correspondence.
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

`[AUTHORS TO COMPLETE]` Include a factual disclosure of generative-AI assistance used for manuscript organization, language editing, or figure preparation, and repeat the required disclosure in the cover letter. AI tools are not authors and all scientific claims remain the authors’ responsibility.

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

Before submission, resolve every item in `prior_publication_audit.md`. Determine whether the APGARSS abstract was merely submitted, accepted, posted online, assigned a DOI, or formally published. Provide the text and publication status to the JRS editor when required, cite it if public, and explain the substantial additions in this full paper. Obtain written editorial confirmation if the prior-publication status is ambiguous.

### Informed Consent

`Not applicable`, unless human-subject annotation procedures trigger institutional requirements.

### Animal Care and Use

`Not applicable.`

### Personal Communications and Unpublished Data

`None`, or list and document permission for every cited communication.

### License Selection

`[AUTHORS TO SELECT THE JOURNAL LICENSE]`

### Materials Sharing

State which code, configurations, model weights, embeddings, split manifests, and label artifacts are shared and identify restrictions on third-party imagery.

### Third-Party Image Reuse

Record the source, license, and permission status of every reused image, map, logo, and basemap; otherwise state that no third-party figure is reused.

### Publication of Accepted Version

`[AUTHORS TO CONFIRM THE JOURNAL POLICY OPTION]`

## References

Use one numbered reference list for the main text and Supplementary Materials. Cite primary papers and official repositories, include verified DOIs where available, and never retain an uncited or machine-invented entry. `[REFERENCE LIBRARY PENDING]`

## Supplementary Materials

- Supplementary Methods: preprocessing, architecture, objective details, compute, and full hyperparameters.
- Supplementary Tables: per-fold/per-seed results, all metrics, threshold distributions, and extended categories.
- Supplementary Figures: all qualitative cases, calibration, additional months, and representation diagnostics.
- Supplementary Data: immutable split/subset/shot manifests, label provenance, checksums, and artifact registry.

## Cover Letter Checklist

- State explicitly that the manuscript is intended for the special issue “Foundation Models based Multimodal Earth Observation Data Fusion and Applications.”
- Describe the manuscript’s fit to multimodal EO representation learning and urban applications without calling XuannvEarth a global foundation model.
- Disclose AI assistance consistently with the manuscript Acknowledgments.
- Disclose the APGARSS abstract and attach or cite the related text when required.
- Confirm that the work is original, is not under review elsewhere, and satisfies all data and image permissions.

## Main-Text Figure and Table Budget

| Label | Purpose |
|---|---|
| Figure 1 | Experimental and technical design overview |
| Figure 2 | Study area, multisource observations, masks, OSM weak labels, and global PCA context |
| Table 1 | Data sources, architecture, information budgets, and training specification |
| Table 2 | Main spatial benchmark with encoder supervision, task relation, and label source reported separately |
| Figure 3 | Representative qualitative maps and failure cases |
| Figure 4 | Labeled-patch-efficiency curves |
| Figure 5 | Strict 40/80/150 scaling and registered ablations |
| Figure 6 | Temporal context, cloud quality, and missing-source robustness |
| Figure 7 | Cross-city recipe reproducibility and direct geographic transfer |
| Figure 8 | Full-region 320-patch transductive product case study and retrieval use case |

Representation-rank diagnostics, all additional head comparisons, per-fold tables, extended OSM categories, hyperparameters, and extra mosaics belong in Supplementary Materials.

## Manuscript Integrity Rules

1. Never replace a result placeholder with a number lacking a traceable JSON/CSV artifact.
2. Never use test labels for threshold selection, checkpoint selection, or ablation choices.
3. Never merge transductive P10C results with spatially independent paper results.
4. Never describe OSM-supervised training as fully self-supervised.
5. Never infer temporal sensitivity from PCA appearance alone.
6. Never infer semantic quality from effective rank alone.
7. Report unfavorable baselines and failed ablations when they answer a registered research question.
8. Cite primary papers and official repositories; verify every DOI and numerical claim.
9. Do not generate a main-paper number with a script that cannot consume the registered spatial split and shot manifests.
10. Do not silently reduce a shot budget; report the setting as infeasible and omit it from paired aggregation.
