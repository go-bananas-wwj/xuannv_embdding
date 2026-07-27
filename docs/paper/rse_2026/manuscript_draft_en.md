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
folds and fixed labelled-patch budgets. The protocol separates upstream encoder training from
spatially held-out downstream readout, freezes the support-patch schedule before probe fitting, and
selects operating thresholds using validation pixels only. **[Insert the strongest verified, evidence-ledger-admitted
V5 result, comparison, uncertainty, and limitation here.]** The study is limited to a single
urban region and OSM-overlapping diagnostic readouts do not establish ontology-independent
semantic transfer.

**Keywords:** geospatial embedding; multimodal Earth observation; time series; weak supervision;
few-shot mapping; urban remote sensing.

## Highlights

- A monthly dense embedding interface for urban mapping readouts
- Heterogeneous EO observations are fused with availability awareness
- Structured corruption targets missing sources and monthly observations
- Spatially held-out probes use frozen supports and validation-only thresholds
- OSM-overlapping results are scoped as weak-label diagnostic evidence

> These draft highlights are intentionally result-free. Elsevier guidance describes highlights as
> three to five short bullet points and limits each bullet to 85 characters; final wording must be
> rechecked against the current RSE Guide for Authors before submission.

## 1. Introduction

Urban mapping is not a one-time prediction problem. Municipal and scientific workflows repeatedly
request maps of buildings, roads, water, green space, facilities, and land-use proxies as new
imagery arrives. Training a separate high-capacity segmentation model for every category and date
duplicates upstream feature learning and ties each request to a new annotation campaign. Dense
geospatial embedding products offer a different interface: an encoder is trained once, frozen, and
read by small task-specific heads for subsequent mapping questions.

This interface is valuable only if it changes the downstream trade-off. A representation that is
visually smooth but cannot be read with a modest support set does not reduce the practical effort of
making a new map. Conversely, a representation need not outperform a fully supervised, task-specific
network trained on abundant labels to be useful: its intended operating point is repeated, diverse
mapping requests for which the feature extractor is already available and labels are scarce. The
relevant question is therefore one of reusable, label-efficient readout rather than universal
dominance over every raw-image segmentation pipeline.

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

The word *monthly* consequently describes the index and intended product cadence of the output, not
an assertion that all information is available within that calendar month. This distinction matters
for interpretation: the representation can use nearby-month context to compensate for missing or
degraded acquisitions, but a result at one target month is not evidence of causal monitoring or
instantaneous change detection. We make this information window explicit in the method and reserve
temporal-benefit claims for independently trained, information-matched ablations.

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

The remainder of the paper first situates this design among reusable EO representations and
weakly supervised geographic learning (Section 2). Section 3 specifies the data, architecture,
training objective, and registered evaluation protocol. Section 4 is deliberately structured so
that tables and figures can be populated only after their underlying artifacts pass release
admission. Sections 5 and 6 interpret the resulting evidence within the stated geographic,
temporal, and label-provenance boundaries.

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

This distinction also affects the role of external products. A globally trained embedding can be a
useful contextual comparator because it exposes a common dense-feature interface to the same
downstream reader. It is not automatically an information-matched baseline: cadence, observation
period, input sensors, pretraining geography, and the treatment of weak labels can differ. We
therefore distinguish controlled *readout* comparisons from claims about upstream model scale or
general transfer, and retain that distinction in every comparator caption and interpretation.

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

The central difficulty is that missingness is not exchangeable across sources. Optical data may be
obscured by cloud or haze, SAR remains available but is affected by speckle and different scattering
physics, and higher-resolution acquisitions may be sparse or temporally misaligned with the target
month. Treating absent values as ordinary observations can lead a reconstruction objective to model
acquisition artifacts rather than land surface. XuannvEarth therefore carries source-availability
information alongside the observations and evaluates reconstruction only where the original target
is valid. The construction is a robustness-oriented training mechanism, not an imputation product.

### 2.3 Reconstruction and weak geographic supervision

Masked reconstruction can encourage a representation to encode information shared across inputs,
but reconstruction targets must respect invalid pixels and source availability. OpenStreetMap is a
collaboratively maintained geographic database (Haklay and Weber, 2008); it can add weak semantic
structure while being incomplete, temporally uncertain, ontology-overlapping with downstream
labels, and spatially heterogeneous in quality (Barron et al., 2014). XuannvEarth therefore treats
OSM as auxiliary weak supervision and separates OSM-assisted
diagnostic evaluation from independent-label evidence.

This separation is especially important for dense urban tasks. A building, road, or water mask
derived from the same broad OSM ontology used to guide representation learning may be a useful
diagnostic of whether weak semantic structure is readable, but it cannot on its own demonstrate
that the encoder discovered an unseen ontology. In addition, locally retained OSM caches are not a
versioned historical snapshot aligned with the 2025--2026 observations. We thus neither treat them
as contemporaneous monthly ground truth nor use them to substantiate change claims.

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

For imbalanced segmentation, threshold selection is a substantive part of the method rather than a
visualization detail. Reporting only ranking metrics can obscure a poor operating point, while
choosing a threshold after inspecting test masks leaks information into the final score. We report
both threshold-free metrics (AP and ROC-AUC) and thresholded metrics (F1, IoU, precision, and
recall), freeze the validation-selected threshold before test evaluation, and retain probabilities
and binary masks for audit. Spatially clustered uncertainty is likewise preferred to independent
pixel intervals because the latter can be artificially narrow in contiguous imagery.

## 3. Materials and methods

### 3.1 Study area and spatial units

The study area is Haidian District, Beijing, China, and contains 320 georeferenced patches, each
covering 1.28 km by 1.28 km. The registered V5 spatial protocol uses the 272 patches contained in
68 complete 2 by 2 blocks; the remaining 48 boundary patches are explicitly excluded from that
fold registry. The 272-patch subset is divided into geographically separated train, validation,
test, and one-complete-block-wide buffer regions at each test edge. The observation window spans December 2025
through May 2026. A full-region P10C product is retained separately as a transductive qualitative
case study and is not used as spatial-generalisation evidence.

The patch is the unit of spatial sampling, support-set accounting, and held-out evaluation within
the registered 272-patch subset. This
choice avoids an apparently large sample size created by treating strongly correlated pixels as
independent geographic observations. The buffer is applied at the complete-block level so that a
test patch is not immediately adjacent to upstream training patches. The registered split is fixed
before downstream fitting; any visual inspection used for data quality control is kept separate
from test-label-informed model selection. Haidian provides heterogeneous dense urban fabric,
transport corridors, water bodies, green areas, and construction-related surfaces, but this local
diversity should not be interpreted as geographic representativeness of other cities.

**[Insert Figure 1: 320-patch study layout, 272-patch complete-block V5 subset, excluded boundary
patches, fold allocation, and representative observations only after provenance admission.]**

### 3.2 Observations and auxiliary labels

Temporal inputs are Sentinel-2 optical imagery with 12 channels (Drusch et al., 2012), Sentinel-1
SAR with two channels (Torres et al., 2012), and Landsat with seven channels. The model also
receives availability-masked higher-resolution optical imagery with three channels and
higher-resolution SAR imagery with one channel.
Higher-resolution observations are aggregated from available acquisitions and fused into every
target-month output; they are not assumed to be unique monthly acquisitions.

All sources are transformed to a common patch grid before model input, while their validity and
availability are retained explicitly. Source quality control is therefore a first-class part of the
data definition: observations that are absent or invalid are represented by their availability
state rather than silently converted to nominal reflectance or backscatter values. Monthly
composition, cloud and invalid-pixel screening, reprojection, and cross-source alignment are
recorded by source-specific audits. The manuscript will report audited coverage and alignment
statistics, rather than inferring data quality from a small number of visually selected examples.

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

The label design has three deliberately separate roles. First, the merged categorical raster
provides broad, target-only land-surface structure during encoder training. Second, the cleaned
fine layers provide auxiliary semantic-probe supervision and hard-negative sampling. Third,
binary building, road, and water masks provide downstream diagnostic readouts. The latter overlap
with the broad OSM ontology and must not be described as independent labels. In particular, the
locally retained OSM material is re-rasterisable but lacks a versioned historical snapshot tied to
the observation window; it is consequently unsuitable as contemporaneous ground truth for monthly
change or for independent-transfer claims.

**[Insert Table 1: source, channels, native resolution, coverage, preprocessing, quality control,
and role after audit admission.]**

### 3.3 XuannvEarth encoder

Each temporal sensor enters an independent 3 by 3 convolution, GroupNorm, and ReLU stem with 32
output channels. A learned source-aware gated-sum fusion combines available temporal stems. The
space-time-precision encoder has six blocks and eight attention heads. Its spatial, temporal, and
precision paths have dimensions 8 by 8 by 512, 16 by 16 by 256, and 128 by 128 by 128. Each block
performs spatial self-attention, per-pixel temporal self-attention with sinusoidal time encoding,
local precision convolutions, and cross-scale exchange.

More formally, let \(x_{m,s}\) denote the observation for month \(m\) and source \(s\), and let
\(a_{m,s}\) denote its availability mask. Sensor-specific stems map observations to features
\(h_{m,s}\), and a source-aware fusion operator combines only available sources to obtain a
month-conditioned feature \(h_m\). The spatiotemporal processor maps the full sequence
\(\{h_m, a_{m,s}\}_{m=1}^{6}\) to target-month maps \(z_m \in \mathbb{R}^{128\times128\times64}\).
Thus, an unavailable source can influence the fusion through its availability state but does not
contribute an invented observation. This notation also makes clear that target month \(m\) may
attend to other months in the six-slot window.

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

The dense output preserves a spatially addressable interface: a downstream head receives one
64-dimensional vector per 10 m cell, rather than a patch-level descriptor that would require
reconstructing boundaries from a single global vector. Unit-norm embeddings make angular
relationships well defined for retrieval and regularisation, but do not themselves guarantee that
individual land-cover classes form disjoint clusters. We therefore evaluate the representation by
held-out downstream readout rather than by PCA appearance or an intrinsic-dimension estimate alone.

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

For a continuous target \(y_t\) with original validity mask \(v_t\), the reconstruction term is
\(\mathcal{L}_{t}^{cont}=\|v_t\odot(\hat y_t-y_t)\|_1/(\sum v_t+\epsilon)\). For a categorical
target, the corresponding term is masked cross-entropy over non-background valid cells. The total
training objective is the weighted sum of continuous and categorical reconstruction terms, fine
semantic-probe loss, hard-negative loss, and batch-uniformity regularisation. Corruption changes
the inputs supplied to the encoder but not the definition of \(v_t\); a target is never made
"correct" merely because its source was synthetically dropped. This distinction prevents the
training loss from rewarding reconstruction of invalid or unavailable imagery.

The registered configuration uses AdamW with learning rate 2e-6, weight decay 0.05, 30 epochs of
linear warmup, and cosine decay over 800 epochs. It uses batch size 3 per process, gradient
accumulation of six, mixed precision, gradient checkpointing, validation every 20 epochs, and a
checkpoint interval of 200 epochs. Realised wall-clock time, effective global batch size, and
energy use will be reported only from final run records.

The recipe is intentionally reported as a configuration rather than as a claim that every loss
term is independently beneficial. The high-resolution optical weight prioritises local visual
detail, whereas SAR-related reconstruction weights are lower because their noise statistics and
interpretation differ. The contribution of any one weight, the hard-negative term, or the
high-resolution pathway requires an admitted matched ablation. Until then, the configuration
documents the trained system but does not establish a mechanism of improvement.

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

The term *shot* refers to a labelled patch, not to an independently sampled pixel. Mixed-class
support patches make the class-balance calculation well posed for pixelwise segmentation, while
the spatial test fold remains unseen by both the encoder and the probe. The 5- and 10-shot
schedules are nested so that increasing the nominal label budget adds support patches rather than
replacing the previous support set. Every unavailable schedule cell is preserved as unavailable in
the result registry; it is never silently substituted with a smaller budget. This ensures that
label-efficiency comparisons refer to the same unit of annotation effort.

The primary V5 task family is OSM-assisted and ontology-overlapping. It can diagnose whether frozen
fields are readable under spatial holdout, but it cannot establish ontology-independent semantic
transfer. Independent-label evidence must be sealed and reported separately. The manuscript policy
prohibits preliminary outputs from entering tables or figures; this policy is enforced through the
evidence-ledger admission workflow, whose release-admission consumer must complete before any V5
result is used.

We distinguish three levels of evidence throughout the study. A registered artifact records a
protocol-compliant computation but is still preliminary until release admission binds its hashes
and reports. An admitted OSM-assisted result supports only an OSM-overlapping diagnostic claim.
An independent-transfer claim additionally requires a frozen external or adjudicated label release
whose ontology and access history are separately documented. This hierarchy avoids converting an
engineering run record into a stronger scientific conclusion than its labels permit.

### 3.6 Annual-AEF contextual comparator

We pre-register a contextual comparison using the official AlphaEarth Foundations (AEF)
annual-2025 embedding with label-free coverage of all 320 Haidian patches and evaluation on the
same 272-patch V5 complete-block subset. This comparator is intentionally classified as a *contextual*,
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

The contextual AEF comparison consequently answers a narrow implementation question: given a
fixed spatial reader and support schedule, how do the two available feature products behave on the
same downstream folds? It does not answer whether one representation has intrinsically better
temporal information, whether either product has seen related upstream geography, or whether a
monthly product should dominate an annual product for all applications. These questions require
additional information-matched experiments and are kept outside the stated comparison.

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

Reproducibility is treated as a chain rather than a collection of filenames. Before a result is
reported, its record must identify the frozen split and support manifests, encoder checkpoint and
configuration, embedding export index, probe configuration, normalisation artifact, validation
threshold, per-patch predictions, metrics, aggregation report, and release-admission record. The
chain is designed to make a failed or altered link visible. It is not a substitute for independent
replication with new pretraining seeds or new cities, both of which remain future evidence needs.

## 4. Results

### Planned tables and figures

**Table 1. Observation sources and preprocessing.** Rows will list source, channels, native and
working resolution, temporal coverage, quality screening, monthly compositing rule, alignment
procedure, and encoder role. Values are populated only from source QA audits.

**Table 2. Registered encoder and optimisation configuration.** Rows will list encoder dimensions,
target losses, corruption schedules, optimiser, effective batch size, stopping/checkpoint policy,
and realised computational resources. The current V5 configuration is described in Section 3.4;
realised values remain blank until the registered runs finish.

**Table 3. Spatially held-out OSM-assisted readout.** Columns will show task, labelled-patch
budget, F1, IoU, AP, ROC-AUC, precision, recall, paired uncertainty, and the evidence-admission
identifier. Rows remain blank until all five folds and three registered reader conditions are
complete.

**Table 4. Contextual annual-AEF comparison.** This table will use the same columns as Table 3,
plus a paired difference. Its caption will state that AEF annual-2025 and XuannvEarth monthly-2026
inputs are time-inequivalent and that the OSM-assisted readout is not independent-transfer evidence.

**Figure 1. Study design and data provenance.** Map the 320 regional patches, identify the 272
complete-block V5 subset and 48 excluded boundary patches, then show geographic folds and buffers,
one quality-controlled multi-source observation sheet, and the audit chain from imagery to the
registered manifest.

**Figure 2. XuannvEarth learning and readout pipeline.** Show sensor-specific inputs, validity
masks, temporal fusion, high-resolution pathways, bottleneck, reconstruction/weak-semantic heads,
and the frozen Conv3x3 downstream reader.

**Figure 3. Held-out readout examples.** For each admitted task, show high-resolution imagery,
reference mask, probability, and validation-thresholded prediction for the same held-out patch.
Every panel will identify the task, support budget, fold, and evidence-admission identifier.

**Figure 4. Aggregate support-budget and readout comparison.** Plot only admitted fold/seed
aggregates with spatial-block uncertainty. Any contextual AEF panel will retain its
time-inequivalence qualifier and will not be described as a labelled-patch-efficiency comparison.

**Table S1. Data and label-provenance audit.** Report the source-specific quality-control records,
valid-pixel and source-availability summaries, alignment checks, OSM provenance limitations, and
the precise status of each label family. This table makes clear which labels are auxiliary,
OSM-overlapping, temporally mismatched, or independently adjudicated.

**Table S2. Per-fold and per-seed readout records.** Report support-patch identifiers, foreground
and background pixel counts, validation-selected thresholds, test-patch identifiers, and all six
metrics for every admitted cell. The main text will show aggregates; the supplement will retain the
cells required to audit their construction.

**Figure S1. Data-quality sheets.** Show randomly selected, pre-specified patches across the full
input window with source availability and validity masks. The figure is a quality-control audit,
not a curated performance illustration.

### 4.1 Spatially held-out OSM-assisted diagnostic readout

**[Populate only after all five encoder folds, embedding exports, 90 registered probe jobs,
aggregation, spatial bootstrap, and release admission complete.]** Report this section as
“OSM-assisted, ontology-overlapping diagnostic readout.” Do not use it as independent transfer or
general semantic generalisation evidence.

The completed section will first state the exact encoder family, checkpoint-selection rule, label
provenance, support budget, and number of admitted fold/seed cells. It will then report all primary
metrics together, rather than selecting a single favorable metric. Qualitative panels will be chosen
by a pre-specified sampling rule from held-out patches and will include both probability maps and
the validation-thresholded binary prediction. Errors such as boundary displacement, omission, and
false positives will be described alongside successful examples.

**[Insert Table 3: admitted five-fold/three-seed metrics.]**<br>
**[Insert Figure 3: admitted held-out imagery, label, probability, and prediction examples.]**

### 4.2 Annual-AEF contextual comparison

**[Populate only after all 90 AEF probe cells, paired aggregation, bootstrap analysis, and
evidence-ledger admission complete.]** This section will report the annual-2025 AEF versus
monthly-2026-04 XuannvEarth comparison exclusively as a time-inequivalent, OSM-assisted spatial
readout. It will use the same 15 fold-seed observations per task and support budget as Section
4.1, retain the fixed caveat in every table and caption, and omit any interpretation as an
information-matched or independent-transfer benchmark.

The reported paired difference will be interpreted only at the feature-readout level. Any apparent
advantage or disadvantage must be accompanied by the corresponding uncertainty estimate and by the
time-inequivalence qualifier. The section will not aggregate this comparison with raw-image or
other foundation-model baselines unless their input window, support schedule, split, reader, and
threshold policy are documented as matched.

**[Insert Table 4: admitted contextual AEF comparison with F1, IoU, AP, ROC-AUC, and paired
uncertainty where applicable.]**

### 4.3 Scale and recipe ablations

**[Omit unless every requested comparator is trained, evaluated, and admitted.]** Strictly nested
40/80/150 scale and matched recipe ablations can be reported only as registered foldwise
comparisons. One-encoder-seed ablations must be described as exploratory. A combined
higher-resolution pathway effect cannot be attributed separately to fusion or reconstruction
without factorial evidence.

For data scale, the horizontal axis will be the registered number of upstream training patches and
the vertical axis will contain the same held-out metric used in the primary analysis. The narrative
will describe the curve as an association under one encoder initialisation per fold, not an estimate
of a universal scaling law. For recipe comparisons, each row will state exactly what was removed
and what remained: for example, the historical `no_osm_150` setting removes both merged OSM land
cover and fine OSM probes, so it cannot isolate the fine-probe contribution.

### 4.4 Quality, temporal, and geographic robustness

**[Omit unless information-matched experiments are complete and admitted.]** A test-time source
deletion or month shuffling experiment is a distribution-shift diagnosis, not a causal temporal
ablation. Claims about temporal context require independently trained 1/3/6-month encoders with
matched information budgets. A second-city result requires clean-from-scratch training and the
same registered protocol.

When completed, robustness analyses will stratify results with a predeclared quality or availability
definition and retain the same held-out geography. A source-deletion result will be called a stress
test of an already trained representation. It will not be relabeled as proof that temporal context
or a modality was causally learned, because the test-time intervention changes the input
distribution without retraining the encoder.

### 4.5 Full-region product case study

**[Omit unless P10C provenance is bound and admitted.]** Full-region PCA mosaics, retrieval
examples, and product maps may be shown only as transductive qualitative illustrations. They cannot
support the spatial-generalisation result table or an independent-transfer claim.

If retained, this section will explain how the full-region output differs from the inductive study:
P10C has access to the complete regional training geography and is therefore useful for inspecting
map continuity, retrieval behavior, and operational workflow, but not for estimating performance
on unseen regions. PCA colours will be described as a visualization of a projection, not as class
labels or quantitative evidence of semantic separation.

## 5. Discussion

The central interpretation will be limited to admitted comparisons. If the registered diagnostics
show utility, they support the narrower claim that a frozen monthly field can be read by a
lightweight spatial probe in the measured OSM-assisted setting. They do not demonstrate a universal
land-surface ontology, transfer independent of OSM, or parity with globally trained products.

The proposed interface is most naturally interpreted as an amortisation of upstream learning. Once
an embedding field is exported, a new mapping request can begin from a common spatial feature
representation rather than from a separate multisource ingestion and large end-to-end model. This
does not remove the need for labels, calibration, or domain review. Instead, it shifts the
downstream question toward whether a small, auditable reader can turn a fixed feature field into a
useful map under a stated support budget. The value of this framing is strongest where many
categories or repeated mapping questions share the same observation archive.

The protocol also clarifies a tension in weakly supervised geographic learning. OSM-derived
semantics can be scientifically useful because they introduce broad spatial structure without
manual labels for every cell. At the same time, they make downstream tasks with related ontologies
easier to read and thus limit the independence of the evidence. Treating these results as
diagnostic rather than hiding the overlap permits an ablation question to be asked honestly: how
does the representation behave when particular weak semantic ingredients are removed? It does not
make an OSM-overlapping result equivalent to an independently adjudicated benchmark.

Several extensions are required before broad claims can be made. A second-city clean-from-scratch
replication would assess whether the training recipe is reproducible beyond Haidian. Independently
created building, road, and water labels, selected before model comparison and accompanied by
adjudication records, are needed for ontology-independent transfer evidence. Information-matched
1-, 3-, and 6-month encoders are needed to identify the contribution of temporal context. Finally,
factorial high-resolution experiments and repeated encoder initialisations would separate recipe
effects from training variation. These are not cosmetic additions: each changes the type of claim
the study can support.

The design has several limitations. The present protocol is based on one urban region; spatial
holdout within Haidian is not second-city validation. OSM labels are incomplete, temporally
uncertain, and semantically related to downstream categories. Monthly observations have limited
temporal redundancy and can retain cloud, haze, source-missingness, radar speckle, and registration
artifacts. Sparse higher-resolution inputs are availability-masked and may be reused across target
months. Finally, a hyperspherical bottleneck does not guarantee uniformly distributed or linearly
separable semantic categories.

The reporting discipline has a practical cost: it postpones attractive numerical claims until all
folds, artifacts, and release records are complete. We consider this cost preferable to reporting a
convenient subset of folds, a threshold tuned on test labels, or a qualitative full-region map as
if it were independent validation. In this study, the evidence ledger is therefore part of the
scientific method, not merely a software release checklist.

## 6. Conclusion

XuannvEarth is a reusable monthly embedding framework for heterogeneous urban Earth observations.
It produces a 64-dimensional dense field at a 10 m grid and is designed for frozen-feature,
label-efficient readout. **[Insert one verified and evidence-ledger-admitted conclusion after V5
release admission.]** Until then, the supported conclusion is methodological: provenance-bound
spatial evaluation is necessary to distinguish a useful reusable representation from a visually
plausible but unverified urban mapping product.

Its final empirical contribution will be defined by the released V5 evidence rather than by the
existence of a registered training run. The paper therefore presents the representation, protocol,
and limitations in sufficient detail for the final admitted results to be interpreted narrowly and
reproduced. The intended outcome is a reusable urban mapping interface whose claims grow only as
its geographically valid, label-provenance-aware evidence grows.

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

The release will also document software and hardware environments sufficiently to rerun each
stage: preprocessing, encoder training, embedding export, probe fitting, aggregation, and report
generation. Public code will exclude credentials and third-party imagery. Where a computation
depends on a restricted asset, the artifact manifest will name the dependency and state whether it
can be reproduced from public inputs, requested under a license, or only inspected through a
derived release product.

The OSM-derived training and readout masks can be released as derived rasters and documented
rasterisation rules where their source terms permit. Sentinel-1, Sentinel-2, and Landsat inputs
will be provided through reproducible acquisition/preprocessing instructions rather than
redistributed copies. Higher-resolution optical and SAR imagery are subject to third-party access
and redistribution terms and will not be redistributed; the archival release will state the source,
access condition, date window, preprocessing inputs, and a procedure for access requests. Any
artifact that cannot be publicly shared will be listed in the release manifest with its reason and
access route.

## Submission-package files

The submission package will contain the main manuscript, a separate title page if requested by the
submission system, a highlights file, figure files with captions, supplementary material, and a
graphical abstract only if required by the current RSE Guide for Authors. The graphical abstract
will communicate the input-to-embedding-to-lightweight-readout workflow without displaying
unadmitted quantitative claims. The final package will also include a data-availability statement
that distinguishes public code and derived artifacts from third-party higher-resolution imagery.

Before submission, the package will be checked against the current journal instructions for word
limits, figure specifications, reference style, data-availability wording, declarations, and any
requirements for graphical abstracts or highlights. This working manuscript deliberately keeps
author-dependent declarations separate so that project collaborators can review the technical text
without unintentionally asserting funding, authorship, or access conditions that have not been
confirmed.

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
