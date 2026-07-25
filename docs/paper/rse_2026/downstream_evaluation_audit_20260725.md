# Registered Downstream Evaluation Audit

**Date:** 2026-07-25  
**Scope:** the 40 completed encoder runs in
`/data/xuannv_embedding/outputs/paper_registered_20260716`.

## Decision

All 40 registered encoder runs have both `best.pt` and `epoch_800.pt`.
The registered scientific analysis uses the validation-selected `best.pt` for
each corresponding fold.  `epoch_800.pt` is retained for reproducibility but
is not selected using downstream results.

The existing downstream entrypoints are diagnostic utilities.  They explicitly
write `paper_eligible=false` and must not be used for numerical results in the
RSE manuscript.  A new fail-closed runner is required before evaluating the
completed encoders.

## Verified Encoder Inventory

| Family | Folds | Scientific role |
| --- | --- | --- |
| `full_40`, `full_80`, `full_150` | 0--4 | Nested data-scale experiment |
| `no_osm_150` | 0--4 | Remove-all-semantic-supervision ablation (OSM and WorldCover) |
| `coarse_osm_only_150` | 0--4 | WorldCover-only coarse land-cover semantic ablation |
| `probe_nohardneg_150` | 0--4 | Hard-negative ablation |
| `no_highres_path_150` | 0--4 | Joint high-resolution path ablation |
| `no_masking_150` | 0--4 | Structured-masking ablation |

Every family is configured against the frozen spatial fold and deterministic
40/80/150 subset manifests.  This satisfies the encoder-side subset registry
requirement (G3) only.  It does not, by itself, make a downstream result
paper-eligible.

## Required Primary Evaluation Matrix

For each encoder, use only its matching evaluation fold.  The primary tasks
are building, road, and water; the primary head is the fixed 64-dimensional
Conv3x3 probe in the registered protocol.  Each task uses nested 5-shot and
10-shot patch manifests and downstream seeds 42, 43, and 44.  The full-label
probe is a supporting scaling/ablation analysis.

The primary few-shot matrix therefore contains
`40 encoders x 3 tasks x 2 shots x 3 probe seeds = 720` runs.  A full-label
supporting matrix adds 360 runs.  Each result must select its decision
threshold on validation probabilities only and apply it once to the held-out
test fold.

## Result-Admission Requirements

The implementation must fail closed unless it can verify and record:

1. the frozen split, encoder train/validation/test/buffer manifests, and their
   hashes;
2. exact, nested mixed-class support-patch budgets with no silent truncation;
3. train-fold-only feature normalization artifacts and hashes;
4. the fixed final-epoch Conv3x3 probe, validation threshold grid, and pooled
   test F1/AP/AUC/IoU/precision/recall;
5. checkpoint/configuration hashes, code revision, environment information,
   per-patch predictions, and an immutable result record.

The analysis also needs a paired spatial block bootstrap before reporting
comparative confidence intervals.  Independent manual labels, held-out
ontology controls, raw/DINO information-matched baselines, Harbin replication,
and temporal/cloud strata remain separate registered experiments.  They cannot
be inferred from the 40 completed encoder checkpoints.

## Protocol Amendment V2 (2026-07-25)

The initial V1 schedule required every few-shot segmentation task to provide
the same number of entirely foreground-free patches as foreground-containing
patches. This is infeasible for road extraction: every training patch in some
spatial folds contains roads, although each such dense mask still has abundant
background pixels. The probe uses pixelwise BCE and therefore does not require
an image-level negative patch for binary class supervision.

V2 defines a shot as one mixed-class labeled segmentation patch with at least
64 foreground and 64 background pixels. It is applied uniformly to building,
road, and water for every fold, seed, representation, and baseline. Schedules
remain deterministic and nested, record pixel counts, and fail as `NA` when an
exact budget is unavailable. V1 manifests and attempted `NA` road cells are
retained as superseded audit evidence; no V1 and V2 result is pooled.

The first V2 outputs remain explicitly preliminary (`paper_eligible=false`)
until all remaining admission gates are implemented and independently audited.

## Interpretation Guardrails

- OSM-derived building, road, and water probes overlap with encoder weak
  supervision.  These results measure label efficiency under OSM auxiliary
  semantics, not fully task-independent transfer.
- `no_osm_150` is not an OSM-only removal: it also removes WorldCover.  It may
  support a remove-all-semantic-supervision comparison only.  A pure OSM
  contribution claim requires a new run retaining WorldCover while disabling
  the OSM semantic probe and OSM-weighted sampling.
- Despite its historical name, `coarse_osm_only_150` contains no OSM probe; it
  is referred to as `WorldCover-only` in all manuscript text and figures.
- One encoder initialization exists per family and fold.  Fold and downstream
  seed variation quantify evaluation uncertainty; they are not independent
  pretraining repeats.
- The historical full-area P10C product remains a transductive application
  case.  It must not enter the spatially independent RSE main table.

## Next Implementation Step

Implement and test the registered fail-closed evaluator, then run a single
fold/task/seed smoke evaluation.  Only after its artifact manifest passes an
independent code review will the 720-run primary matrix be scheduled across
the available NPUs.
