# XuannvEarth RSE Evidence Ledger

Status convention: only a row marked **verified** may support the Abstract, Results, or
Conclusion. Registered probe artifacts remain immutable preliminary records
(`paper_eligible=false`); they become admissible paper evidence only when an immutable
release-admission record binds their exact registry, metric, artifact, and report hashes.

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

- Every new registered downstream result records a `protocol_id`, `evidence_class`, and
  `label_independence_status` in its metric payload, artifact sidecar, and registry entry.
  Aggregation and paired bootstrap reject mixed protocol IDs.
- `v5_osm_assisted` is an **OSM-assisted spatially held-out readout** with
  `label_independence_status=osm_overlapping_not_independent`. It may appear only in explicitly
  marked diagnostic or weak-supervision reports. It must never be presented in an
  independent-transfer table or used as evidence of ontology-independent transfer.
- An independent-transfer claim remains pending a frozen external or adjudicated independent-label
  release and its separately sealed evidence record. This ledger entry is a protocol boundary, not
  a release-admission decision.
- Registered exports must have a six-shard canonical provenance record, a content index, and
  a Git-anchored registry entry.
- Downstream artifacts must pass `verify_artifact_registry_binding()` while retaining their
  original preliminary state. **Pending implementation and execution of the release-admission
  consumer**, a manuscript table or figure must not consume them. Once that gate exists, it will
  require a Git-anchored release-admission record whose selected registry, metric, artifact, and
  report hashes exactly match the sealed files. The admission overlay never rewrites probe
  artifacts.
- The paper aggregator is intentionally fail-closed by default. `--allow-preliminary` output
  is for internal diagnosis only and must be labelled preliminary.
- OSM-overlap tasks are reported as OSM-assisted readout evidence. They do not by themselves
  establish ontology-independent semantic transfer.
- The locally retained OSM caches are not versioned historical snapshots aligned with the
  2025--2026 imagery. See `osm_temporal_provenance_audit_20260725.md`; do not use OSM masks for
  contemporaneous ground-truth or monthly-change claims.

## Manuscript decision rule

Before updating the Abstract, Results, Discussion, or Conclusion, record the exact
release-admission path and hash, the versioned Zenodo DOI and DOI-bound registry URL, matching
`registry_sha256`, every selected registry-entry/artifact/`metrics_sha256`, every aggregate or
bootstrap `report_sha256`, the metric aggregation command, and the statistical output for each
claim above. If any required component is unavailable, retain the claim as `pending` or move it
to limitations/future work.
