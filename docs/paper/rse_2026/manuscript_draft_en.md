# XuannvEarth: Reusable monthly geospatial embeddings from heterogeneous Earth observations for urban mapping

> **Working RSE manuscript.** Bracketed slots must be filled only with metrics, tables, and figures
> admitted by `docs/paper/rse_2026/evidence_ledger.md`; they are not claims.

## Title page

**Weijie Wu**\(^1\), **Xinyi Fan**\(^2\), and **Long Zhao**\(^1\)

\(^1\) Aerospace Information Research Institute, Chinese Academy of Sciences, Beijing, China<br>
\(^2\) Institute of Geographic Sciences and Natural Resources Research, Chinese Academy of Sciences, Beijing, China<br>
Corresponding author: Long Zhao (`zhaolong@aircas.ac.cn`)

> Author order, affiliations, ORCIDs, corresponding-author details, funding, and CRediT roles
> are provisional and require author confirmation before submission.

## Abstract

Urban land-surface mapping is recurrent: buildings, roads, water, vegetation, and public
facilities must be updated repeatedly, yet task-specific segmentation models repeatedly consume
labels and computation. We present XuannvEarth, a city-scale framework that encodes six monthly
slots of heterogeneous Earth observations into frozen, dense 64-dimensional embedding fields on a
10 m grid. The encoder combines sensor-specific stems, availability-aware spatiotemporal fusion,
higher-resolution feature pathways, a hyperspherical bottleneck, valid-target reconstruction under
structured source corruption, and OpenStreetMap-derived auxiliary semantics. We evaluate whether a
frozen field can be read by the same lightweight spatial probe under geographically separated
folds and fixed labelled-patch budgets. **[Insert the strongest verified, evidence-ledger-admitted
V5 result, comparison, uncertainty, and limitation here.]** The study is limited to a single
urban region and OSM-overlapping diagnostic readouts do not establish ontology-independent
semantic transfer.

**Keywords:** geospatial embedding; multimodal Earth observation; time series; weak supervision;
few-shot mapping; urban remote sensing.

## 1. Introduction

Urban mapping is not a one-time prediction problem. Municipal and scientific workflows repeatedly
request maps of buildings, roads, water, green space, facilities, and land-use proxies as new
imagery arrives. Training a separate high-capacity segmentation model for every category and date
duplicates upstream feature learning and ties each request to a new annotation campaign. Dense
geospatial embedding products offer a different interface: an encoder is trained once, frozen, and
read by small task-specific heads for subsequent mapping questions.

Recent Earth-observation representation learners motivate reusable spatial features across sensors,
regions, and tasks. A monthly urban setting is materially different from an annual composite or a
globally aggregated product: a monthly slot can contain cloud and haze, missing optical
acquisitions, radar speckle, sparse higher-resolution imagery, and residual cross-sensor
registration error. It is therefore insufficient to show only an attractive embedding mosaic or
an in-region map. The representation must be evaluated with geographic separation, matched
downstream readers, and a threshold protocol that never observes test labels.

XuannvEarth produces monthly-indexed dense geospatial embeddings from heterogeneous observations.
Each output is associated with one target month but may use six-month context through full temporal
attention; it is not a causal, month-local representation. For each 1.28 km by 1.28 km patch, the
model produces a 128 by 128 map of 64-dimensional unit-norm vectors, corresponding to a 10 m grid.
The representation is learned from multi-sensor reconstruction, a merged OSM land-cover target,
fine OSM-derived auxiliary probes, structured source corruption, and batch-uniformity regularisation.

This paper asks a bounded question: under geographically separated folds, can a frozen monthly
embedding field support label-efficient urban mapping through the same lightweight reader and fixed
labelled-patch budgets? The fail-closed protocol binds the encoder checkpoint, exported feature
files, frozen shot schedules, validation-only thresholds, and downstream artifacts. OSM-overlapping
labels are treated as OSM-assisted diagnostic readouts, not evidence of ontology-independent
transfer.

The contributions are threefold. First, we specify a reusable dense monthly representation for
heterogeneous urban Earth observations. Second, we provide a provenance-bound spatial evaluation
protocol that separates weak-supervision diagnostics from independent-transfer claims. Third, we
report only conditions supported by released, admitted evidence and state the limitations of
regional scale, imperfect weak labels, and sparse higher-resolution observations.

## 2. Related work

### 2.1 Dense geospatial representations and EO foundation models

Large-scale Earth-observation encoders and embedding products motivate a stable feature interface
rather than a separate end-to-end model per task (Brown et al., 2025; Herzog et al., 2025;
Szwarcman et al., 2026). Temporal and multispectral masked pre-training has also shown that a
single encoder can support classification and segmentation readouts after downstream adaptation
(Cong et al., 2022). XuannvEarth follows this interface at city scale and monthly cadence. It is
not a global foundation model: its training extent, observation history, and label sources are
narrower than those of globally trained products. Accordingly, the paper does not infer global or
cross-region transfer from a single-city spatial-holdout study.

### 2.2 Multimodal and temporal Earth observation learning

Optical, radar, and Landsat observations offer complementary strengths but differ in noise,
availability, radiometry, and spatial resolution. Multimodal temporal models therefore require
availability-aware fusion and explicit treatment of missing sources. Masked reconstruction offers a
general self-supervised precedent for learning from partially observed visual inputs (He et al.,
2022), while SatMAE explicitly studies temporal and spectral positional encoding with both
consistent and independently masked observations (Cong et al., 2022). XuannvEarth uses
independent sensor stems and full temporal attention, while structured source corruption exposes
the encoder to missing modalities, months, and spatial blocks during training. This design targets
robustness; it does not by itself prove a causal contribution until matched ablations are admitted.

### 2.3 Reconstruction and weak geographic supervision

Masked reconstruction can encourage a representation to encode information shared across inputs,
but reconstruction targets must respect invalid pixels and source availability. OpenStreetMap is a
collaboratively maintained geographic database (Haklay and Weber, 2008); it can add weak semantic
structure while being incomplete, temporally uncertain, ontology-overlapping with downstream
labels, and spatially heterogeneous in quality (Barron et al., 2014). XuannvEarth therefore treats
OSM as auxiliary weak supervision and separates OSM-assisted
diagnostic evaluation from independent-label evidence.

### 2.4 Label-efficient and spatially valid evaluation

EO foundation-model evaluations commonly contrast limited-label adaptation with task-specific
learners, including linear-probe and segmentation settings (Dionelis et al., 2024; Szwarcman et
al., 2026). Such comparisons are only interpretable when the representation, support budget,
reader capacity, split, and decision rule are held fixed. They are particularly sensitive in spatial
data because nearby pixels and neighbouring patches are correlated (Roberts et al., 2017). We
therefore use
geographically separated folds, fixed support-patch schedules, training-only feature
standardisation, and validation-only threshold selection. These safeguards constrain the scope of
our conclusions; they do not replace independent labels or a cross-city replication.

## 3. Materials and methods

### 3.1 Study area and spatial units

The study area is Haidian District, Beijing, China. The registered protocol partitions 320
georeferenced patches, each covering 1.28 km by 1.28 km, into geographically separated folds with
train, validation, test, and a one-complete-block-wide buffer at each test edge (complete blocks
are 2 by 2 patches). The
observation window spans December 2025
through May 2026. A full-region P10C product is retained separately as a transductive qualitative
case study and is not used as spatial-generalisation evidence.

**[Insert Figure 1: study area, 320-patch layout, fold allocation, and representative observations
only after provenance admission.]**

### 3.2 Observations and auxiliary labels

Temporal inputs are Sentinel-2 optical imagery with 12 channels (Drusch et al., 2012), Sentinel-1
SAR with two channels (Torres et al., 2012), and Landsat with seven channels. The model also
receives availability-masked higher-resolution optical imagery with three channels and
higher-resolution SAR imagery with one channel.
Higher-resolution observations are aggregated from available acquisitions and fused into every
target-month output; they are not assumed to be unique monthly acquisitions.

The target-only categorical layer is an OSM-derived merged land-cover raster stored under the
legacy manifest key `worldcover`; it is removed before encoder input. Its 11 IDs comprise ignored
background (0) plus residential, commercial, industrial, agriculture, green, recreation,
construction, water, building, and transport. The configured decoder has 11 output channels and
the loss ignores class 0. Fine auxiliary semantic supervision separately uses 13 cleaned OSM
layers: building, major road, minor road, rail, water, green space, agriculture, residential,
commercial, industrial, construction, path/walk, and playground. OSM masks are not contemporaneous
ground truth for monthly change claims. Native resolutions, acquisition coverage, quality masks,
mosaicking, reprojection, alignment statistics, normalisation, source missingness, and OSM
provenance will be reported only from their corresponding quality-control audits.

**[Insert Table 1: source, channels, native resolution, coverage, preprocessing, quality control,
and role after audit admission.]**

### 3.3 XuannvEarth encoder

Each temporal sensor enters an independent 3 by 3 convolution, GroupNorm, and ReLU stem with 32
output channels. A learned source-aware gated-sum fusion combines available temporal stems. The
space-time-precision encoder has six blocks and eight attention heads. Its spatial, temporal, and
precision paths have dimensions 8 by 8 by 512, 16 by 16 by 256, and 128 by 128 by 128. Each block
performs spatial self-attention, per-pixel temporal self-attention with sinusoidal time encoding,
local precision convolutions, and cross-scale exchange.

Higher-resolution optical and SAR features are resized to the output grid, encoded by three 3 by 3
convolutional layers with 32, 64, and 64 channels, and fused before the bottleneck. A learned 1 by
1 projection feeds a vMF-style hyperspherical bottleneck. The implementation L2-normalises each
embedding vector and, during training, adds isotropic Gaussian noise scaled by a learned
concentration before re-normalisation. Thus, *vMF-style* denotes an implemented approximation, not
a claim of exact vMF sampling.

The encoder returns six dense maps per sample, one for each target month, with shape 64 by 128 by
128. Per-month scene embeddings are spatial means followed by L2 normalisation. Continuous targets
are decoded by two-layer 1 by 1 convolutional heads; categorical targets use one 1 by 1
convolution.

**[Insert Figure 2: sensor stems, spatiotemporal processor, high-resolution pathway, bottleneck,
and frozen downstream readout.]**

### 3.4 Objective and structured source corruption

Continuous reconstructions use masked L1 loss; the merged OSM land-cover target uses masked
cross-entropy with background ID 0 ignored. The registered V5 weights are 0.80 for Sentinel-2,
0.25 for Sentinel-1, 0.45 for Landsat, 0.45 for merged OSM land cover, 0.90 for higher-resolution
optical, and 0.35 for higher-resolution SAR. Fine OSM semantic-probe loss ramps to 0.14 over 80
epochs. Its hard-negative component uses a ratio of 0.02, weight 0.35, and 120-epoch warmup. The
batch-uniformity term uses temperature 2.0 and ramps to 0.06 over 60 epochs. Temporal-
endpoint, temporal-contrast, and supervised-change terms are disabled.

Corruption is applied during training after targets are prepared. Sentinel-2, Sentinel-1, and
Landsat are dropped independently with probabilities 0.18, 0.35, and 0.35. With probability 0.65,
one to four months are zeroed across temporal inputs. With probability 0.65, 16 by 16 spatial
blocks are selected and each is dropped with probability 0.32; this also masks higher-resolution
inputs. Availability masks are retained during configured corruptions, and reconstruction is
evaluated on originally valid targets.

The registered configuration uses AdamW with learning rate 2e-6, weight decay 0.05, 30 epochs of
linear warmup, and cosine decay over 800 epochs. It uses batch size 3 per process, gradient
accumulation of six, mixed precision, gradient checkpointing, validation every 20 epochs, and a
checkpoint interval of 200 epochs. Realised wall-clock time, effective global batch size, and
energy use will be reported only from final run records.

**[Insert Table 2: model configuration, losses, schedules, and realised training resources.]**

### 3.5 Registered downstream evaluation

The primary reader is a frozen-feature 64-channel Conv3x3 segmentation probe. For each task, fold,
and probe seed, the probe is trained from scratch for 80 epochs using AdamW (learning rate 0.001,
weight decay 0.0001), cosine learning-rate decay, batch size eight, and pixelwise
binary-cross-entropy logits loss with a training-set class-balance weight equal to the
negative-to-positive pixel ratio, clipped at 50. No probe epoch is
selected on test data: the final epoch is evaluated once. Feature standardisation is fit on the
support patches only and then applied unchanged to validation and test patches.

For each fold, the support set contains exactly five or ten deterministic, nested, mixed-class
labeled patches. A valid support patch contains at least 64 foreground and 64 background pixels;
where the requested budget cannot be met, the frozen shot schedule records that budget as
unavailable rather than silently downsampling it. Probabilities are computed with a sigmoid. The decision threshold is selected
on pooled validation pixels by exhaustive F1 search from 0.001 to 0.999 in increments of 0.001 and
is fixed once for the held-out test fold; ties select the greatest threshold. The protocol reports F1, average precision, IoU,
ROC-AUC, precision, and recall. Aggregated comparisons will use paired spatial-block uncertainty,
not independent-pixel intervals. Every representation within a comparison cell uses the same
spatial split, label source, shot schedule, reader, optimiser budget, random-seed policy, and
validation-selected threshold. A five-fold, three-probe-seed design is registered for the
`full_150` V5 family.

The primary V5 task family is OSM-assisted and ontology-overlapping. It can diagnose whether frozen
fields are readable under spatial holdout, but it cannot establish ontology-independent semantic
transfer. Independent-label evidence must be sealed and reported separately. The manuscript policy
prohibits preliminary outputs from entering tables or figures; this policy is enforced through the
evidence-ledger admission workflow, whose release-admission consumer must complete before any V5
result is used.

### 3.6 Annual-AEF contextual comparator

We pre-register a contextual comparison using the official AlphaEarth Foundations (AEF)
annual-2025 embedding for the same 320 Haidian patches. This comparator is intentionally classified as a *contextual*,
time-inequivalent comparison: it contrasts an annual 2025 AEF product with the XuannvEarth
2026-04 monthly product. It does not match observation dates or sensor inputs, and it does not
control for or establish pretraining extent, geographic overlap, or weak-supervision history. The
planned AEF export is label-free; its official annual index, source COGs, patch coverage, output
index, and valid-pixel masks must be hash-locked before probe training.

For each of the three tasks, five folds, two support budgets, and three probe seeds, we apply the
identical frozen Conv3x3 reader, support-only standardisation, optimizer schedule, deterministic
support schedule, validation-only threshold rule, and held-out test metrics described above. The
90-cell AEF matrix is therefore useful for a controlled spatial-readout context, but not for
claims of matched temporal information, independent transfer, or global superiority. Tables,
captions, and discussion retain this qualifier wherever the comparator appears.

### 3.7 Registered assets, aggregation, and reproducibility

The registered protocol is identified as `rse_v5_registered_20260726`. Its immutable five-fold
split is `haidian_spatial_5fold_complete2x2_v5_seed42.json`
(SHA-256 `9a6d98d6d6456ce0a791ef74ac725e4d360e7d4e0a17c9cb5edfceb850093c0b`), and its nested,
label-free 40/80/150-patch subset registry is
`haidian_paper_subsets_40_80_150_complete2x2_v5_seed42.json`
(SHA-256 `5477d4a3c0ca9d1dbbf343fe04ae675725479a60cafa3be67ee29fa1db500c93`). The five
self-contained encoder configurations follow the naming pattern
`paper_registered_v5_full_150_fold{0..4}_20260726.yaml`; the archive will bind their byte hashes,
queue-sealed checkpoint hashes, export file indices, and the V5 export registry to the reported
cells. All preprocessing statistics are estimated only from the full upstream training manifest
of the corresponding fold, never from its validation, test, or buffer patches. The normalization
registry is `haidian_paper_v5_normalization_statistics.json`
(SHA-256 `ba1fb10bc105a29286750367dff2ab45e02f89c32f69725ab89da994d0c56929`); it records each
training-manifest and statistics-audit hash. The archive will include all five audits and the
source-specific mean and standard-deviation arrays.

For each task and labelled-patch budget, Table 3 will report the arithmetic mean and sample standard
deviation over the three registered schedule/probe-seed means, where each condition mean averages
the five held-out spatial folds. Each registered condition selects the support schedule and
deterministically sets the probe initialisation. The table will retain all 15 fold-by-condition
values in the supplement, but will not treat them as 15 independent encoder pretraining repetitions. Comparative
intervals for F1, IoU, precision, and recall will use a registered 10,000-resample hierarchical
paired bootstrap (random seed 20260725): spatial folds, complete 2 by 2 geographic test clusters,
and registered schedule/probe-seed conditions are resampled in that order. The interval does not quantify uncertainty
from an independently repeated encoder pretraining run.
AP and ROC-AUC will be reported as point estimates with the same fold/seed aggregation but are not
assigned this confusion-matrix bootstrap interval. The bootstrap compares only paired result cells
with the same task, budget, fold, seed, test-patch IDs, labels, split, and protocol provenance.

## 4. Results

### 4.1 Spatially held-out OSM-assisted diagnostic readout

**[Populate only after all five encoder folds, embedding exports, 90 registered probe jobs,
aggregation, spatial bootstrap, and release admission complete.]** Report this section as
“OSM-assisted, ontology-overlapping diagnostic readout.” Do not use it as independent transfer or
general semantic generalisation evidence.

**[Insert Table 3: admitted five-fold/three-seed metrics.]**<br>
**[Insert Figure 3: admitted held-out imagery, label, probability, and prediction examples.]**

### 4.2 Annual-AEF contextual comparison

**[Populate only after all 90 AEF probe cells, paired aggregation, bootstrap analysis, and
evidence-ledger admission complete.]** This section will report the annual-2025 AEF versus
monthly-2026-04 XuannvEarth comparison exclusively as a time-inequivalent, OSM-assisted spatial
readout. It will use the same 15 fold-seed observations per task and support budget as Section
4.1, retain the fixed caveat in every table and caption, and omit any interpretation as an
information-matched or independent-transfer benchmark.

**[Insert Table 4: admitted contextual AEF comparison with F1, IoU, AP, ROC-AUC, and paired
uncertainty where applicable.]**

### 4.3 Scale and recipe ablations

**[Omit unless every requested comparator is trained, evaluated, and admitted.]** Strictly nested
40/80/150 scale and matched recipe ablations can be reported only as registered foldwise
comparisons. One-encoder-seed ablations must be described as exploratory. A combined
higher-resolution pathway effect cannot be attributed separately to fusion or reconstruction
without factorial evidence.

### 4.4 Quality, temporal, and geographic robustness

**[Omit unless information-matched experiments are complete and admitted.]** A test-time source
deletion or month shuffling experiment is a distribution-shift diagnosis, not a causal temporal
ablation. Claims about temporal context require independently trained 1/3/6-month encoders with
matched information budgets. A second-city result requires clean-from-scratch training and the
same registered protocol.

### 4.5 Full-region product case study

**[Omit unless P10C provenance is bound and admitted.]** Full-region PCA mosaics, retrieval
examples, and product maps may be shown only as transductive qualitative illustrations. They cannot
support the spatial-generalisation result table or an independent-transfer claim.

## 5. Discussion

The central interpretation will be limited to admitted comparisons. If the registered diagnostics
show utility, they support the narrower claim that a frozen monthly field can be read by a
lightweight spatial probe in the measured OSM-assisted setting. They do not demonstrate a universal
land-surface ontology, transfer independent of OSM, or parity with globally trained products.

The design has several limitations. The present protocol is based on one urban region; spatial
holdout within Haidian is not second-city validation. OSM labels are incomplete, temporally
uncertain, and semantically related to downstream categories. Monthly observations have limited
temporal redundancy and can retain cloud, haze, source-missingness, radar speckle, and registration
artifacts. Sparse higher-resolution inputs are availability-masked and may be reused across target
months. Finally, a hyperspherical bottleneck does not guarantee uniformly distributed or linearly
separable semantic categories.

## 6. Conclusion

XuannvEarth is a reusable monthly embedding framework for heterogeneous urban Earth observations.
It produces a 64-dimensional dense field at a 10 m grid and is designed for frozen-feature,
label-efficient readout. **[Insert one verified and evidence-ledger-admitted conclusion after V5
release admission.]** Until then, the supported conclusion is methodological: provenance-bound
spatial evaluation is necessary to distinguish a useful reusable representation from a visually
plausible but unverified urban mapping product.

## Data and code availability

The archival release will contain: source code; exact self-contained encoder and probe
configurations; frozen spatial split, buffer, subset and shot-schedule manifests; normalisation
statistics; environment and command records; checkpoint/configuration hashes; embedding-export
file indices and provenance; frozen probe weights; per-patch validation and test predictions;
metrics, aggregation and spatial-bootstrap reports; and the release-admission registry that binds
each reported result to these artifacts. The release DOI, version, licence, repository URL, and
artifact manifest will be inserted here before submission. Checkpoints, embedding products, and
downstream artifacts will additionally be versioned on ModelScope, with the exact version and
content manifest cited here.

The OSM-derived training and readout masks can be released as derived rasters and documented
rasterisation rules where their source terms permit. Sentinel-1, Sentinel-2, and Landsat inputs
will be provided through reproducible acquisition/preprocessing instructions rather than
redistributed copies. Higher-resolution optical and SAR imagery are subject to third-party access
and redistribution terms and will not be redistributed; the archival release will state the source,
access condition, date window, preprocessing inputs, and a procedure for access requests. Any
artifact that cannot be publicly shared will be listed in the release manifest with its reason and
access route.

## Supplementary material

The supplementary material will contain: (S1) acquisition, quality-mask, mosaicking,
reprojection, alignment, source-availability, and OSM-rasterisation audits; (S2) the five-fold
split, buffer, training-subset, and downstream-shot manifests; (S3) fold-specific normalization
statistics and audits; (S4) self-contained encoder/probe configurations, training and inference
commands, environment records, and checkpoint/export provenance; (S5) per-fold/seed metrics,
validation thresholds, prediction-file schemas, and held-out qualitative examples; and (S6) the
release-admission records, aggregation output, and paired spatial-bootstrap reports. Exact archive
paths, version identifiers, and file hashes will be inserted when the final immutable release is
created.

## CRediT authorship contribution statement

**[Author-confirmed roles required before submission.]**

## Funding

**[Author-confirmed funding information required before submission.]**

## Declaration of competing interests

**[Author-confirmed declaration required before submission.]**

## Acknowledgements

**[Author-confirmed acknowledgements required before submission.]**

## Prior dissemination statement

An earlier APGARSS conference abstract described the project direction and preliminary internal
experiments. This manuscript is a substantially expanded journal study with a separate registered
spatial evaluation protocol, provenance-bound results, and a full methods and limitations analysis.
The authors will provide the exact conference citation and ensure compliance with final publisher
policy before submission.

## Declaration of generative AI and AI-assisted technologies in the writing process

**[Author-confirmed disclosure required immediately before references if applicable under the
current Elsevier policy.]**

## References

Barron, C., Neis, P., and Zipf, A., 2014. A Comprehensive Framework for Intrinsic OpenStreetMap
Quality Analysis. Transactions in GIS 18, 877-895. https://doi.org/10.1111/tgis.12073.

Brown, C.F., Kazmierski, M.R., Pasquarella, V.J., Rucklidge, W.J., Samsikova, M., Zhang, C.,
Shelhamer, E., Lahera, E., Wiles, O., Ilyushchenko, S., Gorelick, N., Zhang, L.L., Alj, S.,
Schechter, E., Askay, S., Guinan, O., Moore, R., Boukouvalas, A., and Kohli, P., 2025.
AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from
sparse label data. arXiv:2507.22291. https://doi.org/10.48550/arXiv.2507.22291.

Cong, Y., Khanna, S., Meng, C., Liu, P., Rozi, E., He, Y., Burke, M., Lobell, D.B., and Ermon,
S., 2022. SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery.
arXiv:2207.08051. https://doi.org/10.48550/arXiv.2207.08051.

Dionelis, N., Fibaek, C., Camilleri, L., Luyts, A., Bosmans, J., and Le Saux, B., 2024.
Evaluating and Benchmarking Foundation Models for Earth Observation and Geospatial AI.
arXiv:2406.18295. https://doi.org/10.48550/arXiv.2406.18295.

Drusch, M., Del Bello, U., Carlier, S., Colin, O., Fernandez, V., Gascon, F., Hoersch, B., Isola,
C., Laberinti, P., Martimort, P., Meygret, A., Spoto, F., Sy, O., Marchese, F., and Bargellini,
P., 2012. Sentinel-2: ESA's Optical High-Resolution Mission for GMES Operational Services. Remote
Sensing of Environment 120, 25-36. https://doi.org/10.1016/j.rse.2011.11.026.

Haklay, M., and Weber, P., 2008. OpenStreetMap: User-Generated Street Maps. IEEE Pervasive
Computing 7, 12-18. https://doi.org/10.1109/MPRV.2008.80.

He, K., Chen, X., Xie, S., Li, Y., Dollar, P., and Girshick, R., 2022. Masked Autoencoders Are
Scalable Vision Learners. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition, 16000-16009. https://doi.org/10.1109/CVPR52688.2022.01553.

Herzog, H., Bastani, F., Zhang, Y., Tseng, G., Redmon, J., Sablon, H., Park, R., Morrison, J.,
Buraczynski, A., Farley, K., Hansen, J., Howe, A., Johnson, P.A., Otterlee, M., Schmitt, T.,
Pitelka, H., Daspit, S., Ratner, R., Wilhelm, C., Wood, S., Jacobi, M., Kerner, H., Shelhamer, E.,
Farhadi, A., Krishna, R., and Beukema, P., 2025. OlmoEarth: Stable Latent Image Modeling for
Multimodal Earth Observation. arXiv:2511.13655. https://doi.org/10.48550/arXiv.2511.13655.

Roberts, D.R., Bahn, V., Ciuti, S., Boyce, M.S., Elith, J., Guillera-Arroita, G., Hauenstein, S.,
Lahoz-Monfort, J.J., Schröder, B., Thuiller, W., Warton, D.I., Wintle, B.A., Hartig, F., and
Dormann, C.F., 2017. Cross-validation strategies for data with temporal, spatial, hierarchical,
or phylogenetic structure. Ecography 40, 913-929. https://doi.org/10.1111/ecog.02881.

Szwarcman, D., Roy, S., Fraccaro, P., Gislason, T.E., Blumenstiel, B., Ghosal, R., de Oliveira,
P.H., de Sousa Almeida, J.L., Sedona, R., Kang, Y., Chakraborty, S., Wang, S., Gomes, C., Kumar,
A., Gaur, V., Truong, M., Godwin, D., Khallaghi, S., Lee, H., Hsu, C.-Y., Akbari Asanjan, A.,
Mujeci, B., Shidham, D., Balogun, R.O., Kolluru, V., Keenan, T., Arevalo, P., Li, W., Alemohammad,
H., Olofsson, P., Mayer, T., Hain, C., Kennedy, R., Zadrozny, B., Bell, D., Cavallaro, G., Watson,
C., Maskey, M., Ramachandran, R., and Moreno, J.B., 2026. Prithvi-EO-2.0: A Versatile
Multitemporal Foundation Model for Earth Observation Applications. IEEE Transactions on
Geoscience and Remote Sensing 64, 4400120. https://doi.org/10.1109/TGRS.2025.3642610.

Torres, R., Snoeij, P., Geudtner, D., Bibby, D., Davidson, M., Attema, E., Potin, P., Rommen, B.,
Floury, N., Brown, M., Traver, I.N., Deghaye, P., Duesmann, B., Rosich, B., Miranda, N., Bruno,
C., L'Abbate, M., Croci, R., Pietropaolo, A., Huchler, M., and Rostan, F., 2012. GMES Sentinel-1
mission. Remote Sensing of Environment 120, 9-24. https://doi.org/10.1016/j.rse.2011.05.028.
