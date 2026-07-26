# XuannvEarth: Reusable Monthly Geospatial Embeddings from Heterogeneous Earth Observations for Urban Mapping

> Target journal: *Remote Sensing of Environment* (RSE), Original Research Article.
>
> This is the sole English manuscript outline for the RSE submission. The former JRS outline
> is retained only as historical planning material and must not be used for RSE wording,
> declarations, or reference style.

## Front matter

**Authors, affiliations, ORCIDs, corresponding author, funding, CRediT roles, competing
interests, acknowledgements, and data/code availability:** author-supplied before submission.

**Reference style:** author-year citations and an alphabetised reference list, consistent with
the RSE/Elsevier journal instructions current at submission time.

## Abstract (maximum 250 words; one paragraph)

Urban land-surface information must be updated repeatedly for different mapping tasks, while
task-specific segmentation models repeatedly consume labels and compute. We introduce
XuannvEarth, a city-scale framework that encodes six monthly slots of heterogeneous Earth
observations into frozen, dense 64-dimensional embedding fields at a 10 m grid. The encoder
uses sensor-specific stems, spatiotemporal fusion, a vMF-style hyperspherical bottleneck,
valid-target reconstruction under structured source corruption, and OpenStreetMap-derived
auxiliary weak semantics. The study evaluates whether these fields can be read by the same
lightweight segmentation head under geographically separated folds and fixed labelled-patch
budgets. **Insert only verified registered metrics here.** The abstract must state the spatial
protocol, the comparison family, the strongest verified result, and the principal limitation;
it must not use preliminary, transductive, or test-selected results.

**Keywords:** geospatial embedding; multimodal fusion; Earth observation time series;
few-shot learning; urban mapping; representation learning.

## 1. Introduction

Urban mapping is a recurring rather than a single-task problem: buildings, roads, water,
vegetation, facilities, and land use are repeatedly queried at changing dates. Training a new
high-capacity model from raw imagery for every query duplicates representation learning and
label demand. Dense geospatial embedding products offer a different interface: the expensive
upstream encoder is trained once, then a small reader is fitted for each new task.

This promise is harder to test for monthly urban mapping than for annual composites. A monthly
slot may contain cloud, haze, missing optical observations, radar speckle, sparse higher
resolution acquisitions, and imperfect cross-sensor alignment. It is therefore insufficient to
show an attractive embedding visualisation or a full-region map. The scientific question is
whether a frozen monthly-indexed field remains useful when test areas are geographically held
out, the downstream protocol is matched, and thresholds are selected without observing test
labels.

XuannvEarth addresses this setting with a multimodal representation learner that produces a
128 x 128 x 64 field for each 1.28 km x 1.28 km patch and target month. The output is
monthly-indexed, not month-local: full temporal attention may use observations from the other
months. The training recipe combines reconstruction of valid multi-sensor targets, sparse
higher-resolution pathways resampled to the 10 m grid, structured source corruption, and
cleaned OSM-derived auxiliary weak semantics. The paper treats every component as a hypothesis
to be tested, rather than as an automatic contribution.

The paper makes three bounded contributions. First, it specifies a reusable dense monthly
embedding field for heterogeneous urban Earth observations. Second, it defines a fail-closed
spatial evaluation protocol that binds encoder provenance, exported features, frozen shot
schedules, validation-only thresholds, and downstream artifacts. Third, it reports the
verified conditions under which the field is useful, together with the limits imposed by
regional scale, weak labels, and sparse higher-resolution observations. The exact claims and
their admissible evidence are maintained in `evidence_ledger.md`.

## 2. Related work

### 2.1 Dense geospatial representations and EO foundation models

Position XuannvEarth against annual embedding products, global EO encoders, and reusable
dense features. Distinguish a reusable feature field from a task-specific segmentation model.
Discuss AlphaEarth Foundations as a transfer reference without implying equivalent upstream
training scale, geographic coverage, cadence, or sensor inputs.

### 2.2 Multimodal temporal Earth observation learning

Review optical-radar fusion, time-series modelling, missing-modality learning, cloud-aware
processing, and spatial/temporal fusion. State why a six-month city-scale setting has less
temporal redundancy than annual compositing.

### 2.3 Reconstruction, corruption, and weak geographic supervision

Review masked reconstruction and source dropout. Explain that XuannvEarth corrupts inputs but
computes reconstruction on originally valid targets. Review OSM as noisy and incomplete weak
semantic supervision, including the risk that OSM-overlap evaluation can overstate transfer.

## 3. Materials and methods

### 3.1 Study area and spatial units

Describe Haidian District, the 320 georeferenced 1.28 km x 1.28 km patches, coordinate system,
land-cover diversity, and the December 2025--May 2026 period. State that full-region P10C
maps are a transductive product case study, whereas the scientific study uses spatial five-fold
training, validation, test, and one-patch buffer regions.

### 3.2 Observations, quality control, and OSM semantics

Table 1 must report Sentinel-2 (12 channels), Sentinel-1 (2), Landsat (7), higher-resolution
optical (3), and higher-resolution SAR (1): native resolution, monthly availability,
preprocessing, quality mask, and role. Report quality-screening, mosaicking, reprojection,
alignment checks, clear-pixel statistics, source-missingness statistics, and OSM snapshot
provenance. Higher-resolution inputs are resampled to the 10 m output grid and may be reused
across monthly slots; do not imply a unique high-resolution acquisition in every month.

### 3.3 XuannvEarth encoder

Each sensor first passes through an independent 3 x 3 convolution, GroupNorm, and ReLU stem
to 32 channels. An availability-aware spatiotemporal processor has six blocks, eight attention
heads, a gated temporal fusion pathway, and full temporal attention. The registered
configuration uses spatial, temporal, and precision paths of 8 x 8 x 512, 16 x 16 x 256, and
128 x 128 x 128, respectively. Finer-resolution features are fused before a 64-dimensional
vMF-style hyperspherical bottleneck. The output is a unit-norm 128 x 128 x 64 embedding map
for each of six target months.

### 3.4 Training objective and corruption

Define a single weighted objective comprising valid-pixel reconstruction, a categorical WorldCover
target, fine OSM semantic probes, hard-negative terms, and within-rank uniformity. The registered
V5 recipe gives weights 0.80, 0.25, 0.45, 0.90, 0.35, and 0.45 to Sentinel-2, Sentinel-1,
Landsat, higher-resolution optical, higher-resolution SAR, and WorldCover targets. The fine OSM
semantic-probe weight ramps to 0.14 and the uniformity weight ramps to 0.06. Input
corruption drops Sentinel-2/Sentinel-1/Landsat with probabilities 0.18/0.35/0.35, drops up to
four months with probability 0.65, and applies spatial blocks with probability 0.65. State
all masks, normalisation rules, optimiser settings, parameter count, training time, and energy
accounting in Table 2 or Supplementary Material.

### 3.5 Registered downstream evaluation

Freeze the encoder and use a preregistered 64-channel Conv3x3 probe as the primary reader.
Every representation compared within a table uses the same split, label source, shot schedule,
probe architecture, optimiser budget, seed policy, and validation-selected threshold. Report
F1, AP, IoU, ROC-AUC, precision, and recall; calculate uncertainty at the spatial-block level,
not from independent pixels. Treat OSM-overlap tasks as OSM-assisted evidence and keep
independent-label/held-out-ontology evidence separate.

## 4. Results

### 4.1 Spatially independent multi-task readout

Insert the registered five-fold, three-seed table only after every row is admitted by the
evidence ledger. Its section title, table captions, figure captions, and discussion must identify
it as an **OSM-assisted, ontology-overlapping diagnostic readout**, not as independent semantic
transfer evidence. Figure 1 should show representative held-out predictions with high-resolution
imagery, labels, probabilities, and masks; Figure 2 should show a compact task-by-budget
comparison with uncertainty. Keep independent-label and held-out-ontology results in a separate,
explicitly named analysis; do not combine their conclusions with the OSM-overlap table.

### 4.2 Data scale and recipe ablations

Include this section only after every scale or ablation row has passed the evidence-ledger
admission gate. Report strictly nested 40/80/150 training sets and matched no-OSM, coarse-OSM,
no-hard-negative, no-high-resolution-pathway, and no-masking ablations only when their registered
comparators and downstream results are complete. Describe one-encoder-seed ablations as
exploratory associations. Do not attribute the combined higher-resolution pathway result to input
fusion or reconstruction separately without the required 2 x 2 experiment. If these conditions
are not met at submission, omit this subsection rather than presenting planned experiments.

### 4.3 Quality, temporal, and geographic robustness

Include this section only for completed, information-matched 1/3/6-month, clear/cloudy,
missing-source, and second-city experiments whose results have also been verified and admitted
by the evidence ledger. A test-time deletion or shuffling study is a distribution-shift diagnosis,
not a causal temporal ablation. Omit any completed but preliminary robustness output from the
manuscript and supplementary evidence.

### 4.4 Full-region product case study

Show P10C maps, PCA mosaics, and retrieval examples only after their provenance is bound and
admitted in the evidence ledger. Present them separately as a transductive, qualitative product
case study; these visuals cannot enter the spatial-generalisation main table or support an
independent-transfer claim. If provenance admission remains pending, exclude them from both the
main manuscript and supplementary evidence.

## 5. Discussion

Interpret only verified differences between representations, data scale, and objectives.
Discuss why a monthly-indexed embedding could reduce repeated downstream feature learning,
while acknowledging weak-label overlap, local training scale, cloud/registration artifacts,
sparse higher-resolution data, and uncertain transfer outside the study area.

## 6. Conclusion

Restate the problem, the monthly dense embedding framework, the strongest verified and
evidence-ledger-admitted finding, the practical implication, and one boundary. If no finding has
completed admission, omit quantitative conclusions rather
than treating registration as evidence. Do not make a global-foundation-model or universal
superiority claim.

## Declarations and supplementary material

Include CRediT roles, funding, competing interests, acknowledgements, data/code availability,
third-party imagery restrictions, generative-AI disclosure if required, and a prior-publication
statement for the APGARSS abstract. Supplementary material should contain full protocol details,
quality statistics, provenance hashes, and negative results. Include task figures and ablation
tables only after their underlying results are verified and admitted by the evidence ledger; do
not use supplementary material to present preliminary results as manuscript evidence.

## References

Use only bibliographic records verified against the final cited source. Apply the RSE author-year
citation style consistently in the text and provide an alphabetised reference list here. Do not
carry numbered-reference instructions or journal-specific wording from the historical JRS outline
into this section.
