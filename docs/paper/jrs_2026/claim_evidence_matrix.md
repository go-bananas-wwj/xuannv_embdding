# XuannvEarth Claim-Evidence Matrix

This matrix is the gatekeeper for manuscript claims. A claim enters the abstract or conclusion only when its required evidence is complete and traceable.

| ID | Candidate claim | Required evidence | Current status | Allowed wording now |
|---|---|---|---|---|
| C1 | Frozen XuannvEarth embeddings support multiple urban mapping tasks | Spatial 5-fold x 3-seed results for at least building, road, and water with fixed lightweight heads | Pending final E3/E6 | “We evaluate whether...” |
| C2 | XuannvEarth improves labeled-patch efficiency beyond prior OSM task supervision | Matched 5/10/feasible-50-shot curves on held-out categories or no-OSM encoders, with paired uncertainty; broader annotation-efficiency wording additionally requires time, pixel, and object-count costs | Preliminary fold-0 task-overlap evidence only | “We test labeled-patch efficiency in separate supervision tracks...” |
| C3 | The full recipe improves over reconstruction alone | Clean-from-scratch no-OSM, coarse-OSM, no-hard-negative, and full-recipe ablation repeated across at least three independent encoder seeds | Running/pending E2 | One-seed differences are exploratory associations; no directional causal claim |
| C4 | Finer-resolution input fusion or reconstruction supervision preserves boundaries | A 2 x 2 fusion on/off by reconstruction-target on/off experiment, boundary metric, and qualitative audit | Combined-path ablation is insufficient; 2 x 2 pending | State each mechanism as a design hypothesis |
| C5 | Structured input corruption improves robustness | No-corruption ablation and missing-source/cloud strata | Running/pending E2-Mask/E8 | State as design hypothesis |
| C6 | More embedding dimensions are effectively used | Effective rank, per-channel variance, covariance spectrum, and downstream association | Existing diagnostics are inconclusive | Do not claim |
| C7 | Higher effective rank improves downstream utility | Controlled experiments with uncertainty | Existing P14/V4 negative evidence contradicts simple claim | State that rank alone is insufficient |
| C8 | Monthly temporal context is beneficial | 1/3/6-month, temporal pooling, and month-shuffle comparison | Pending E7 | No advantage claim |
| C9 | Monthly embeddings are temporally sensitive | Independent change/retrieval tasks with dates and stable-area controls | Pending | “Designed to represent monthly state” |
| C10 | The recipe reproduces in another city | Harbin from-scratch training with matched evaluation | Training complete; evaluation pending/verify | “We test recipe reproducibility...” |
| C11 | The Haidian model transfers to another city | Direct Haidian-to-Harbin zero-shot evaluation | Pending E5-ZS | No transfer claim |
| C12 | XuannvEarth outperforms AEF | Same labels, spatial splits, heads, optimization, seeds, thresholds, and task definitions; report cadence/scale caveats | Preliminary fold-0 mixed result | “We compare under a matched probe protocol...” |
| C13 | XuannvEarth outperforms DINO-family features | Same downstream protocol and documented feature resolution | Preliminary mixed result | No global superiority claim |
| C14 | OSM weak semantics improve general representation | Clean ablation plus independent human/non-OSM labels and held-out categories | Pending E2/E4 | “OSM provides weak semantic regularization” |
| C15 | OSM-trained performance is free of label leakage | Independent labels and provenance audit | Not yet established | Explicitly separate OSM-derived diagnostics |
| C16 | P10C generalizes across Haidian | Embedding training excludes spatial test blocks | False for production P10C | Call it a transductive full-region case study |
| C17 | XuannvEarth is a foundation model | Large-scale, broad geographic and task evidence | Not established | Use “regional embedding framework,” not “foundation model” |
| C18 | XuannvEarth is more storage-efficient | Exact tensor precision, cadence, feature resolution, and baseline storage calculation | Not finalized | Report exact sizes only |
| C19 | Each monthly output uses only that month’s observations | Causal/month-local encoder implementation | False for P10C: full temporal attention and aggregated high-resolution context cross months | Call it “monthly-indexed,” not “current-month-only” |

## Evidence Provenance Requirements

Every quantitative manuscript entry must record:

- preregistration commit/tag and test-access-ledger entry;
- code commit, dirty-tree state, dependency lock/container digest, and hardware/software versions;
- experiment ID and immutable configuration path;
- raw and processed data manifests, preprocessing chain, and checksums;
- checkpoint path and checksum;
- feature artifact version;
- split, shot, label, and ontology manifests with checksums;
- downstream head configuration and seed;
- validation-selected threshold;
- raw per-patch predictions or sufficient statistics;
- aggregation script and output JSON/CSV;
- canonical result-registry entry and content hash;
- figure-generation script and source data.

## Result Labels

Use these labels in drafting notes:

- `[VERIFIED]`: artifact exists, protocol passes audit, and value has been independently recomputed.
- `[PRELIMINARY]`: useful diagnostic but not eligible for the main claim.
- `[PENDING]`: registered experiment not complete.
- `[INVALIDATED]`: overlap, contaminated initialization, test-driven threshold, or another protocol violation prevents inferential use.

Only `[VERIFIED]` results may appear without qualification in the abstract, main result statements, and conclusion.
