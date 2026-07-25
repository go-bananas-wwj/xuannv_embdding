# XuannvEarth RSE Evidence Ledger

Status convention: only a row marked **verified** may support the Abstract, Results, or
Conclusion. A completed run that is still marked `paper_eligible=false` is a diagnostic,
not a paper result.

| Claim | Required evidence | Required statistical evidence | Status | Use in manuscript |
|---|---|---|---|---|
| Frozen monthly embeddings support buildings, roads, and water under spatial holdout | Five folds, three probe seeds, fixed Conv3x3 head, frozen registered split and validation threshold | F1, AP, IoU, ROC-AUC, precision and recall; paired spatial-block CI | pending | Main Results only after admission |
| A particular labelled-patch budget is more efficient than a matched baseline | Same fold, shot schedule, head, optimiser, threshold rule, and information window for every representation | Paired effect size and hierarchical spatial bootstrap CI | pending | Main Results only after admission |
| Training-set scale is associated with transfer quality | Strictly nested 40/80/150 subsets in every fold | Foldwise curves and uncertainty; one encoder seed described as exploratory | pending | Exploratory scaling subsection |
| OSM auxiliary semantics improve representation utility | Full recipe, no-OSM, and coarse-OSM-only models with matched clean training sets | Foldwise paired comparison; distinguish OSM-overlap labels from independent labels | pending | Ablation only; no general claim without independent labels |
| Hard-negative supervision improves transfer quality | Full recipe versus probe-no-hard-negative with identical OSM-dependent sampler | Foldwise paired comparison | pending | Ablation only |
| Fine-resolution pathway improves transfer quality | Full pathway versus no-pathway; a 2x2 is required to separate fusion from reconstruction | Foldwise paired comparison | pending | Combined-pathway statement only until 2x2 exists |
| Structured input corruption improves robustness | Full recipe versus no-masking and clear/cloudy or missing-source strata | Foldwise paired comparison within each stratum | pending | Robustness section |
| Temporal context benefits monthly mapping | Independently trained 1/3/6-month encoders with matched information budgets | Foldwise paired comparison | pending | Do not infer from test-time month deletion alone |
| The recipe reproduces in a second city | Harbin clean-from-scratch training and the same registered protocol | Independent city-level result; report separately from direct transfer | pending | Reproducibility subsection |
| The Haidian production product supports mapping and retrieval | P10C full-region exports, maps, and retrieval examples | Qualitative/transductive case study only | pending provenance binding | Case study or Supplementary Material |

## Current artifact gates

- Registered exports must have a six-shard canonical provenance record, a content index, and
  a Git-anchored registry entry.
- Downstream artifacts must pass `verify_artifact_registry_binding()` and retain
  `paper_eligible=true` before they enter a manuscript table or figure.
- The paper aggregator is intentionally fail-closed by default. `--allow-preliminary` output
  is for internal diagnosis only and must be labelled preliminary.
- OSM-overlap tasks are reported as OSM-assisted readout evidence. They do not by themselves
  establish ontology-independent semantic transfer.

## Manuscript decision rule

Before updating the Abstract, Results, Discussion, or Conclusion, record the exact artifact
path, registry entry hash, metric aggregation command, and statistical output for each claim
above. If any required component is unavailable, retain the claim as `pending` or move it to
limitations/future work.
