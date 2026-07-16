# Registered Evaluation Protocol for the JRS Manuscript

> Status: frozen design specification. No value may enter a main-paper table until every relevant implementation gate in Section 12 passes.

## 1. Scientific Settings

The paper reports three settings separately.

1. **Spatially independent representation learning.** Encoder training excludes each fold's validation band, test band, and one-patch buffer. This is the main scientific setting.
2. **External product-use comparison.** Frozen XuannvEarth, AEF, and DINO-family products are used as delivered, then evaluated with the same downstream labels and protocol. Different upstream information budgets are disclosed; this setting measures downstream usability, not causal superiority of the upstream recipe.
3. **Full-region transductive case study.** P10C has seen all 320 Haidian patches and OSM-derived supervision. It is excluded from the main benchmark, paired confidence intervals, and spatial-generalization claims.

## 2. OSM Supervision and Claim Tracks

XuannvEarth receives direct OSM-derived auxiliary weak supervision for 13 classes and OSM-dependent weighted patch sampling. Building, road, water, construction, green, and playground results therefore cannot by themselves establish task-agnostic labeled-patch efficiency against baselines that did not receive equivalent OSM supervision.

Results are divided into:

- **Track A: OSM-assisted task-overlap transfer.** Tasks overlap the 13 auxiliary classes. Claims are limited to labeled-patch efficiency conditional on prior OSM supervision.
- **Track B: held-out-category transfer.** The task taxonomy, ontology graph, synonyms, parent/child relations, label-generation code, and a spatial-overlap audit are frozen and hashed before evaluation. A category must be absent from both the 13 fine OSM tasks and the coarse OSM categorical target and must not be a renamed, parent/child, or geometrically near-equivalent target. This track supports broader representation claims only after the ontology audit passes.
- **Track C: no-OSM representation.** A clean no-OSM encoder is compared with raw and external representations. This isolates multisensor reconstruction from OSM supervision.
- **Track D: independent manual labels.** Labels are drawn without consulting OSM and are used to test whether observed trends survive a different annotation source. Independence of labels does not erase the upstream supervision asymmetry, so Track A wording still applies to overlapping classes.

## 3. Spatial Split and Frozen Training Subsets

- Master split: `configs/eval/haidian_spatial_5fold_buffer1_seed42.json`.
- Five continuous test bands; the next band is validation; a one-patch geographic buffer is excluded from training.
- Candidate training counts are 172, 157, 150, 176, and 151.
- Scaling uses exact, deterministic, label-free, spatially balanced `40 subset 80 subset 150` lists from `configs/eval/haidian_paper_subsets_40_80_150_seed42.json`.
- Every scaling and ablation model for a fold uses the same registered subset list. Lists may not be changed after viewing downstream results.
- The existing 160-patch fold-0 run is a pilot and cannot enter the five-fold scaling estimate.
- Encoder-side normalization, quality thresholds, compositing choices, source-selection rules, and other data-driven preprocessing are fitted or chosen using the encoder-training geography only. Globally fixed physical conversions and externally published algorithms may be reused, but their versions and parameters must be frozen before test access. Validation/test imagery may be transformed only by those frozen operations.

## 4. Independent Manual Annotation

Before annotation, freeze a label handbook containing class definitions, inclusion/exclusion examples, minimum mapping unit, boundary rules, unknown/occluded handling, and image dates.

- Annotators may inspect the registered optical/SAR reference stack but not OSM geometry, model predictions, embedding PCA, or baseline outputs.
- Two annotators independently label a stratified set of train, validation, and test patches; a third adjudication pass resolves disagreements.
- Report annotator agreement using class prevalence, object/region counts, pixel IoU, and boundary F1 before adjudication.
- Record annotation minutes, valid labeled pixels, positive pixels, and object counts so “shot” is not presented as the entire labeling-cost measure.
- Freeze raster labels, provenance, handbook version, and checksums before model comparison.

## 5. Few-Shot Sampling Unit

For a task, a **positive patch** contains at least 64 valid positive pixels (0.39% of a 128 x 128 grid); a **negative patch** contains zero annotated positive pixels. Unknown and ignored pixels do not count toward either criterion.

- A nominal `N`-shot run uses exactly `N` positive and `N` negative training patches, sampled without replacement from the training fold.
- The fixed seeds are 42, 43, and 44. For each seed and task, `5-shot subset 10-shot subset 50-shot` is strictly nested.
- The identical shot manifest is used by every representation in a paired comparison.
- If either class has fewer than `N` eligible patches, the task-fold-shot cell is `NA`; code must fail loudly and must not truncate with `min()`.
- The confirmatory budgets are 5 and 10 shots. The 50-shot result is secondary and is reported only for task-fold pairs where the exact budget is feasible.
- The validation and test patch sets never change across shot levels or seeds.
- Shot seed controls patch selection; probe initialization uses a separately recorded seed derived deterministically from `(fold, task, shot_seed)`.

## 6. Feature and Information-Budget Controls

All feature maps are aligned to the same 128 x 128 grid. Channel means and standard deviations are estimated from the downstream training fold only and then frozen.

### 6.1 Matched 64-D probe comparison

- XuannvEarth and AEF use their native 64 channels.
- DINO-family native features are projected to 64 dimensions by PCA fitted only on unlabeled pixels from the downstream training fold.
- Raw six-month features use the same December-May temporal window and available sources as XuannvEarth, then a 64-D PCA projection fitted only on the downstream training fold.
- All four inputs therefore enter the identical 64-channel probe. PCA fitting never sees validation/test pixels or labels.

This is the primary downstream-head fairness comparison. A DINO native-dimensional upper bound and raw native-channel strong heads are reported separately as capacity analyses.

### 6.2 Product-use comparison

Each product uses its native cadence and upstream information. Report feature dimension, spatial resolution, dates, sensors, upstream geographic exposure, and trainable head parameters. Do not attribute differences solely to representation learning.

### 6.3 Temporal leakage control

Because the April monthly-indexed XuannvEarth field can attend to May and aggregated finer-resolution context, the matched raw baseline receives the same six-month window. A causal “data available by April 30” experiment requires a separately trained causal/cutoff model and may not reuse the full-context April field.

## 7. Primary Downstream Head

The preregistered head is fixed for every 64-D representation:

1. `Conv2d(64, 128, kernel_size=3, padding=1, bias=False)`;
2. `BatchNorm2d(128)`, GELU, and dropout 0.1;
3. `Conv2d(128, 64, kernel_size=3, padding=1, bias=False)`;
4. `BatchNorm2d(64)`, GELU;
5. `Conv2d(64, 1, kernel_size=1)`.

Optimizer, learning-rate schedule, loss, batch construction, epoch count, and augmentation are identical across representations and frozen before testing. Any model-specific hyperparameter tuning receives the same validation-only search budget and is reported as secondary. Linear, MLP, UNet, DeepLab-lite, and SegFormer-lite results cannot replace the preregistered primary head after test inspection.

The primary probe uses a fixed epoch count and the **final epoch checkpoint**; no early stopping or validation-driven probe checkpoint selection is allowed. Validation predictions from that final checkpoint are used only for the registered decision-threshold selection.

## 8. Threshold Selection and Metrics

For each `(representation, task, fold, shot seed)`:

1. Pool all valid validation pixels after applying the frozen valid/ignore mask.
2. Evaluate thresholds `0.001, 0.002, ..., 0.999` on sigmoid probabilities.
3. Select the threshold maximizing pooled validation F1; if tied to machine precision, choose the largest threshold to favor fewer false positives.
4. Freeze that threshold and apply it once to the corresponding test predictions.
5. If validation contains no positive or no negative valid pixels, mark the cell invalid and do not use test labels to repair it.

Primary metrics are pooled-pixel test F1 at the validation-selected threshold and average precision. Secondary metrics are pooled IoU, precision, recall, and ROC-AUC. Oracle test F1 is an explicitly labeled supplementary upper bound only. Per-image min-max normalization is forbidden in metrics.

## 9. Seeds and Statistical Inference

- The main downstream analysis is explicitly `one frozen encoder checkpoint per fold x three shot/probe seeds`, not three independent encoder seeds.
- Confidence intervals therefore quantify spatial-test and downstream-sampling uncertainty conditional on the selected encoder initialization.
- Pretraining stability is a separate experiment. Any scaling or ablation trained with one encoder seed is described as exploratory; a confirmatory causal claim requires at least three independently initialized encoders for the compared conditions.
- For each test fold, compute metrics by pooling valid pixels. Also retain per-patch confusion counts and probabilities.
- Paired differences use identical folds, shots, seeds, labels, and test pixels.
- Use a 10,000-replicate paired hierarchical block bootstrap: resample the five folds with replacement, resample 2 x 2 geographic patch clusters within each selected fold, and resample the three downstream seeds with replacement. Preserve paired model predictions at every level.
- Report the point estimate, descriptive fold and seed standard deviations separately, and the percentile 95% confidence interval of the paired difference.
- If confirmatory p-values are reported across tasks, apply Holm correction. Do not treat pixels or the 15 fold-seed runs as independent geographic replicates.

## 10. Checkpoint and Ablation Rules

- Scientific models use `best.pt`, selected solely by the aggregate validation objective recorded during encoder training. Downstream labels and probe scores never select the encoder checkpoint.
- Epoch 800 is a separately identified deployment checkpoint and may appear in the transductive case study, not as a post hoc “best” scientific model.
- Clean ablations start from random initialization and share fold, subset, optimizer, update budget, seed policy, and checkpoint rule.
- The current no-finer-resolution ablation removes both the input-fusion path and reconstruction targets. It supports only a claim about the combined pathway.
- To separate input information from supervision, add a 2 x 2 design: fusion on/off crossed with reconstruction target on/off.
- Matched 1-, 3-, and 6-month temporal models must be trained for the causal temporal comparison. Test-time month deletion or shuffling of a six-month model is reported only as robustness under distribution shift.

## 11. Baseline Exposure Disclosure

For XuannvEarth, AEF, and DINO-family encoders, report upstream training geography, date range, input sensors, spatial resolution, and whether Haidian/test geography may have been seen. A shared downstream spatial split makes probe training comparable but does not make upstream exposure inductive. Wording must distinguish regional clean training, globally pretrained external products, and transductive P10C.

## 12. Implementation Gates

No main-paper number may be generated until all gates pass:

| Gate | Required condition | Current status |
|---|---|---|
| G1 | A fail-closed evaluation CLI verifies the registered split hash and records complete provenance | PENDING: current diagnostic entrypoints are explicitly `paper_eligible=false` |
| G2 | Immutable shot manifests include split/label/rule hashes, nested seeds, and exact-budget failure | PENDING: current diagnostic entrypoints are explicitly `paper_eligible=false` |
| G3 | Exact 40/80/150 registry is consumed and hash-checked by manifest/config/training entrypoints | PASS: registered manifests/configs generated; encoder-training preflight fails on hash or ID mismatch |
| G4 | Train-only normalization/PCA artifacts are fitted, saved, and hash-checked | PENDING |
| G5 | Six-month raw information-matched baseline is implemented | PENDING |
| G6 | Validation-threshold algorithm and paired block bootstrap match this document | PENDING |
| G7 | The full encoder initialization/checkpoint ancestry is hash-verified; every ancestor's training patches are disjoint from the fold's validation/test/buffer geography; validation-only selection is recorded | PENDING |
| G8 | Independent annotation handbook, labels, provenance, and agreement are frozen | PENDING |
| G9 | OSM-assisted, held-out-category, no-OSM, and manual-label tracks are not mixed | PENDING |
| G10 | A 2 x 2 finer-resolution fusion/reconstruction ablation is complete before making mechanism-specific claims | PENDING |
| G11 | Test labels are sealed; preregistered config hashes and a test-access ledger prevent iterative test-driven model selection | PENDING |
| G12 | A canonical result registry binds metrics to code, data, preprocessing, full checkpoint ancestry, predictions, and environment hashes, then anchors the registry outside rewriteable local history | PENDING |
| G13 | Track B ontology and geometric-overlap audit is frozen and hash-checked | PENDING |

Legacy fold-0, random-split, oracle-threshold, silently truncated, or task-selected-head results remain preliminary even if their scores are higher.

### 12.1 Fail-Closed Result Admission

- Every evaluation output and aggregate must carry a machine-readable `paper_eligible` field. Missing values are interpreted as `false`; the field alone is never sufficient for admission.
- Existing diagnostic benchmark entrypoints explicitly emit `paper_eligible=false`; their outputs may inform debugging and experiment planning but may not populate manuscript claims, figures, or tables.
- A future registered evaluation entrypoint may emit `paper_eligible=true` only after it verifies all gates relevant to that result, records immutable artifact hashes, and writes a gate-by-gate provenance record into a canonical append-only result registry committed to Git.
- Each registry entry binds the metric payload and per-patch predictions to the Git commit, dirty-tree state, complete self-contained configuration, split and shot manifests, raw/processed data manifests and checksums, preprocessing artifacts, label/ontology versions, the complete initialization/checkpoint ancestry and every ancestor's training-patch manifest, feature checksums, dependency lock or container digest, Python/PyTorch/torch_npu versions, device model, random seeds, and aggregation code. The registry entry itself receives a content hash; hand-edited or hash-mismatched entries are rejected.
- A fold-independent scientific encoder must start from random initialization unless an external generic checkpoint is preregistered and its upstream exposure is explicitly treated as external pretraining. A checkpoint descended from P10C, P9B, or any ancestor trained on that fold's validation/test/buffer patches is ineligible, even after additional clean-fold training.
- Preregistration and canonical result-registry snapshots use signed Git commits and signed annotated tags pushed to a protected remote, and their digests are deposited in an independent timestamped archive such as Zenodo or OSF. Later history rewrites or regenerated hash-consistent registries cannot replace the externally timestamped snapshot.
- Aggregation and figure scripts for the paper must accept only entries present in the canonical registry whose content hashes verify and whose `paper_eligible` value is exactly `true`.

### 12.2 Test-Set Sealing and Experiment Disclosure

- Before any confirmatory test run, freeze the research questions, task ontology, primary metrics, model/config hashes, baseline set, ablations, and figure/table plan in a signed, externally timestamped Git commit and tag.
- Test labels remain outside routine training/evaluation paths. A designated release command records every access in an append-only ledger containing user, time, commit, config hashes, and purpose.
- Validation may be used for the preregistered choices. Once confirmatory test results are released, no encoder, task definition, head, threshold rule, or primary analysis may be changed and re-labeled confirmatory without registering a new holdout or clearly marking the follow-up exploratory.
- The supplement reports all registered confirmatory runs, failures, exclusions, and deviations; selective retention of the best test run is forbidden.
