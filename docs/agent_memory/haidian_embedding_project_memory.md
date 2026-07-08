# Haidian Embedding Project Memory

This document captures durable project knowledge for future agents working on
Xuannv Haidian embedding training and evaluation. Read this before changing
training code, launching experiments, or interpreting downstream results.

## 1. North Star

The goal is not to build a separate segmentation model for each downstream
task. The goal is to build a reusable monthly geospatial embedding, following
the spirit of AEF-style geospatial embeddings:

- Generate a dense 128x128 embedding map for each 1280 m x 1280 m patch.
- Use multi-source Earth observation inputs to produce a compact semantic
  representation, currently 64 channels.
- Let downstream users label only a few examples and train lightweight heads
  for many tasks.
- Support building extraction, road extraction, water extraction, land-use
  classes, green space, schools, sports fields, construction sites, and future
  OSM-derived categories.
- Make the embedding useful for few-shot mapping, fast iteration, retrieval,
  change analysis, and multi-task reuse.

The embedding should be evaluated as a reusable representation. It does not
need to beat raw imagery plus a large UNet in every fully supervised setting.
Its key value is that a small amount of labels plus a simple head can produce
usable maps across many tasks.

## 2. AEF-Inspired Evaluation Philosophy

AEF-style evaluation emphasizes whether a pretrained embedding reduces labeling
and downstream training burden. For Xuannv Haidian, always separate these
settings:

- Full-label supervised setting: raw multi-source imagery plus UNet or
  DeepLab-lite is a strong baseline and may outperform embedding plus a light
  head.
- Few-shot setting: this is the key proof point for the embedding. Use 5-shot,
  10-shot, and 50-shot tests with the same labels, split, threshold selection,
  and candidate heads for both embedding and raw-image baselines.
- Lightweight head setting: linear, MLP, and small conv heads show whether the
  embedding already carries semantic structure.
- Strong downstream head setting: UNet and DeepLab-lite show the upper bound
  when more downstream modeling is allowed.

Never claim the embedding is better only from AUC. Always inspect F1, AP, AUC,
thresholded masks, probability maps, and false positives.

## 3. Current Best Evidence

The most important reports are:

- `docs/production/haidian_v1_leadership_summary_20260708.md`
- `docs/production/haidian_fewshot_raw_vs_xuannv_20260708.md`
- `docs/production/haidian_raw_image_baseline_comparison_20260708.md`

The report images are vendored under:

- `docs/production/assets/haidian_v1_20260708/`

Important conclusions:

- In full-label tests, raw 2026-04 multi-source imagery plus UNet or
  DeepLab-lite can match or exceed Xuannv embedding plus a light head.
- In few-shot tests, Xuannv embedding is the stronger story.
- In 5-shot and 10-shot tests, Xuannv embedding beats raw 2026-04 imagery on
  building, road, and water.
- In 50-shot tests, Xuannv still leads on building and road; water is roughly
  tied.
- The `conv3x3` downstream head is consistently strong and usually better than
  pure MLP heads.
- Pure MLP heads can compress probabilities into a narrow range and create
  misleading visualizations if thresholding/calibration is not checked.

Few-shot best F1 summary:

| Task | Shot | Xuannv Best F1 | Raw Best F1 | Relative Gain |
| --- | ---: | ---: | ---: | ---: |
| Building | 5 | 0.459 | 0.418 | +9.8% |
| Building | 10 | 0.454 | 0.439 | +3.5% |
| Building | 50 | 0.490 | 0.467 | +5.0% |
| Road | 5 | 0.474 | 0.401 | +18.3% |
| Road | 10 | 0.487 | 0.435 | +11.8% |
| Road | 50 | 0.517 | 0.501 | +3.3% |
| Water | 5 | 0.613 | 0.433 | +41.5% |
| Water | 10 | 0.612 | 0.487 | +25.6% |
| Water | 50 | 0.631 | 0.631 | roughly tied |

## 4. Data Lessons

Data quality has been the largest source of model quality problems.

- Monthly imagery is more sensitive to clouds and haze than annual AEF-style
  composites. Annual averaging can hide cloud artifacts; monthly embedding
  must explicitly handle them.
- Cloudy S2/Landsat inputs can make the embedding learn cloud reconstruction
  instead of land semantics.
- Always inspect S2, Landsat, S1, high-res optical, high-res SAR, masks, and
  labels for sampled patches before training.
- Data sources must be spatially aligned. Misalignment creates blurred
  embeddings, weak boundaries, and false positives.
- High-res optical around 2026-04-30 was treated as usable for May when needed.
- WorldCover is too coarse for many fine-grained downstream semantics in
  Haidian. It can be useful for broad context but should not be treated as
  detailed building/road labels.
- OSM weak labels are valuable, but they must be cleaned and merged into
  semantic groups. Similar tags such as multiple road types or building tags
  should map to a unified downstream category.
- Do not train the main embedding directly on the downstream manual labels used
  for final evaluation. This leaks task labels and weakens the claim of general
  reusable representation.
- OSM-derived weak semantics are acceptable as auxiliary weak supervision when
  framed as broad semantic priors, not task-specific target leakage.

Before training a new major version, generate visual sheets for one or two
random patches showing each month and each source exactly as the model sees
them.

## 5. Model and Training Lessons

Hard constraints:

- Use PyTorch plus `torch_npu`.
- Keep model output at 128x128 spatial resolution.
- Do not use `einops`.
- Do not use L2 normalization at the bottleneck by default. VMF bottleneck is
  the exception.
- Actual experiment YAML files must be self-contained and must not use `_base_`.

Training insights:

- High-res reconstruction matters because it preserves local boundaries and
  fine structure.
- S1 and SAR are noisy. Their reconstruction weights should generally be lower
  than high-res optical or semantic objectives.
- Hard reconstruction / modality dropout is useful, but too many losses can
  fight each other. Watch whether losses and downstream metrics improve
  together.
- Longer training helped reduce patch-to-patch PCA discontinuity in P10/P12
  style runs.
- Patch boundary artifacts can come from insufficient context. Context or
  overlap-crop inference is a plausible future upgrade.
- Effective embedding dimension is an important unresolved issue. If only a
  small number of dimensions carry information, 64 channels are underused and
  downstream categories may overlap.
- When PCA visualization looks smooth but downstream heads are weak, inspect
  class separation in embedding space, logits distribution, thresholds, and
  false positives. Do not trust PCA alone.

## 6. Downstream Evaluation Standard

For all serious comparisons:

- Use the same labels.
- Use the same train/val/test fold.
- Use the same shot protocol.
- Use the same threshold-selection method.
- Report F1, AP, and AUC.
- Show visualizations.
- Separate full-label and few-shot claims.

Required tasks for the core benchmark:

- Building extraction
- Road extraction
- Water extraction

Useful extended tasks:

- Park / green space
- Education / school / university, with overlapping OSM tags merged carefully
- Sports fields / playgrounds
- Construction site if labels are reliable
- Additional OSM categories after auditing positive coverage

Standard heads to compare:

- `wide_mlp`
- `deep_wide_mlp`
- `conv3x3`
- `unet`
- `deeplab_lite`

Raw-image baseline:

- Uses 2026-04 raw / derived feature stack.
- Current stack is 42 channels from Sentinel-2, Sentinel-1, Landsat, high-res
  optical, high-res SAR, indices, and local statistics.
- This is not the same input as embedding. It is a strong raw-feature baseline
  designed to ask whether embedding reduces downstream labeling/model burden.

Few-shot protocol:

- 5-shot means 5 positive training patches plus the same number of negative
  patches.
- 10-shot and 50-shot follow the same rule.
- Validation and test sets remain unchanged.
- Threshold is selected on validation and reported on test.

## 7. Visualization and Reporting Lessons

- Probability maps must be shown together with thresholded masks and GT.
- Red-on-white binary masks are easier to inspect for reports.
- Per-image min-max probability visualizations are useful for debugging, but
  they are not calibrated probabilities and must be labeled clearly.
- If probability values cluster in a tight range, inspect logits and calibration
  before concluding the model has failed.
- For leadership reports, keep the story simple:
  1. Full-label raw strong models are strong.
  2. Few-shot is where Xuannv embedding wins.
  3. The business value is less labeling, fast mapping, and multi-task reuse.
- Report images must be copied into the repo under `docs/production/assets/...`
  and referenced with relative paths. Do not reference `/data/...` paths in
  Markdown intended for GitHub, PDF, or user-facing reports.

## 8. Important Paths

Repository:

- `/root/workspace/xuannv`

Data root:

- `/data/xuannv_embedding`

Production reports:

- `docs/production/haidian_v1_leadership_summary_20260708.md`
- `docs/production/haidian_fewshot_raw_vs_xuannv_20260708.md`
- `docs/production/haidian_raw_image_baseline_comparison_20260708.md`

Report assets:

- `docs/production/assets/haidian_v1_20260708/`

Representative few-shot experiment output:

- `/data/xuannv_embedding/experiments/p12_context_eval_20260707/fewshot_raw_vs_xuannv_20260708/`

Representative raw full-label experiment output:

- `/data/xuannv_embedding/experiments/p12_context_eval_20260707/raw_202604_head_sweep_20260708_fixedenv/`

## 9. Operational Rules

- Commit and push after each substantive code, config, script, or document
  change. This project expects frequent Git synchronization.
- Do not commit data under `/data/xuannv_embedding`.
- Keep code, scripts, configs, and lightweight reports in the repository.
- Large experiment outputs belong under `/data/xuannv_embedding`.
- Be careful with untracked files from the user. Do not add unrelated files.
- Before using NPU training/evaluation, source the CANN environment, for
  example:

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
```

- If `torch_npu` errors mention missing `tbe`, the CANN environment is likely
  not loaded correctly.

## 10. Future Upgrade Directions

The next upgrades should focus on making the embedding itself stronger:

- Increase effective dimension utilization and reduce representation collapse.
- Improve category separation in embedding space.
- Reduce building false positives.
- Improve local boundary and texture encoding.
- Improve monthly cloud/haze robustness.
- Add context or overlap-crop inference to reduce patch boundary artifacts.
- Add more OSM-derived few-shot tasks after label audit.
- Evaluate cross-month generalization.
- Build a standard pipeline for: export embedding, train downstream heads,
  evaluate, visualize, and generate reports.

When future conversations become long, use this document as the stable memory
before making new training or evaluation decisions.
