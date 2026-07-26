# Annual AEF Contextual-Comparator Protocol

## Scope and claim boundary

This protocol compares the registered Xuannv `full_150` monthly representation with the official
AlphaEarth Foundations (AEF) **annual 2025** embedding product. It is a downstream-protocol-aligned,
time-inequivalent, OSM-assisted contextual comparison. It is not a same-date comparison, a
matched-input comparison, a labelled-patch-efficiency comparison against an information-matched
baseline, an
independent-transfer experiment, or evidence that either model was geographically unseen during
pretraining.

The Xuannv candidate uses a 2026-04 monthly embedding; AEF uses one annual 2025 map. Every paper
table and caption must state this temporal asymmetry.

## Frozen source inventory

The source is the official `aef/v1/annual` 2025 release. The local cache currently contains two
64-channel int8, 10 m COGs in EPSG:32650. Their cache-byte SHA-256 values are:

| COG | SHA-256 |
| --- | --- |
| `x6oxy7iyy4d2hrngi-0000000000-0000008192.tiff` | `239f5230b3bc22395770ad3f5dda2290b95c4740edba63e18251b523d008603b` |
| `x7gr0ny5ao7h28kxr-0000008192-0000008192.tiff` | `c395929b3f9599101c5d7c6893bfdcdff659bb5bed1a319e9bec675dfa6cec2a` |

The old local directory `embeddings/aef_official_2025_annual` is diagnostic-only. It has no
versioned index snapshot, patch-to-COG map, individual-file index, deterministic source manifest,
or V5 protocol binding and must not enter paper results.

## Required sealed export

A new exporter must derive a fresh annual AEF map for every patch in the frozen V5 320-patch
manifest before any task/fold/shot labels are read. Its accepted inputs are restricted to the
Git-pinned 320-patch manifest, reference-raster inventory, official AEF index and COG cache; it
must reject label roots, tasks, folds, shots, and probe-result paths. It records the complete argv,
an input allowlist and SHA-256 values for each accepted input. It must fail on missing coverage, a source COG
read failure, an output whose shape is not `64 x 128 x 128`, or an output set that differs from the
manifest patch set. It must record:

1. official release identifier, STAC/GeoParquet index URI and immutable index bytes/hash;
2. source COG URI, local cache path and source bytes hash for each patch;
3. reference-grid CRS, transform, bounds and hash for each patch;
4. AEF int8 dequantization formula, nodata rule and non-learned bilinear reproject/resample rule;
5. exporter source revision, command, environment, output map hash, output file index hash, and
   an explicit `global_label_free: true` declaration;
6. a fixed output identifier `annual_2025`, never a fabricated monthly acquisition date.

For every patch, the source manifest must seal the ordered candidate COG URI/hash list, intersection
rule, selected source window or reprojection footprint, and either a fixed mosaic order or an
explicit multi-source rejection. It must also seal an output valid-pixel mask and its hash before
nodata values are replaced with zero. The contextual-comparator protocol requires full valid-pixel
coverage at every evaluated 128 by 128 output location; any missing AEF pixel rejects the export.
Thus no AEF-only zero-filled nodata value can enter feature standardisation, probe fitting, threshold
selection, or metric computation.

The source and output manifests must be immutable, content indexed and cited from a separate,
Git-tracked baseline registry. AEF must not be placed in the Xuannv checkpoint/config registry.
The comparison also requires a Git-pinned comparison matrix with a distinct comparison identifier,
`candidate_family=full_150`, `baseline_id=aef_annual_2025`, exact V5 split/320-patch-manifest/
statistics-registry hashes, tasks, shots, folds, seeds, reader and annual-versus-monthly time
statement. This comparison matrix is separate from the Xuannv encoder-family matrix.

## Matched downstream readout

The AEF reader must use the same V5 complete-2x2 spatial split, OSM label roots, deterministic
5/10-shot schedules, three probe seeds (42, 43, 44), frozen 64-channel Conv3x3 reader, optimizer
budget, support-only feature standardization, final-epoch selection, and validation-only
`0.001..0.999` threshold grid as Xuannv. It produces exactly 90 jobs:

`building/road/water x 5/10 shot x 5 folds x 3 probe seeds`.

The baseline runner must validate annual AEF provenance separately from Xuannv encoder provenance.
It must reject learned channel/spatial adapters, task-specific export, missing source metadata,
non-64-channel maps, mismatched patch support, mismatched schedules, or mismatched protocol hashes.
Each probe/result/artifact record must bind and revalidate: the comparison-matrix hash; spatial
split hash; 320-patch-manifest hash; statistics-registry hash; label-root tree hash; task/fold/shot/
seed; shot-manifest hash; test-patch-ID hash; AEF source-manifest and output-file-index hashes;
runner-source hash; threshold candidate record; prediction, probe, metric, and artifact hashes.
It must share the same artifact identity schema as the Xuannv reader so paired spatial bootstrap can
verify identical support and test provenance without accepting AEF as a Xuannv encoder family.
The metric payload additionally records `label_sha256`, `shot_manifest_label_sha256`,
`shot_manifest_split_sha256`, and test-patch-keyed `per_patch_confusion` with `tp`, `fp`, `fn`, and
`tn`, using the same schema as the Xuannv reader.

## Admission and reporting

AEF and Xuannv results remain preliminary until their separate export registries, all probe
artifacts, aggregate report, paired bootstrap, and release-admission records are sealed. The
baseline uses `protocol_id=aef_annual_2025_contextual`; Xuannv retains `protocol_id=v5_osm_assisted`.
A dedicated, Git-pinned comparison-level protocol record binds both protocol descriptors, both
export registries, the comparison-matrix hash, and their shared split/manifest/label/schedule
identity snapshot. Its admission record must additionally enumerate the ordered 90-cell matrix on
each side and, for every selected cell, bind `result_id`, metric hash, artifact hash, and
result-registry-record hash; it seals the canonical ordered-set hash for each side before any
comparison report is rendered. A cross-protocol paired-bootstrap and comparison-admission consumer
may accept only exactly this ordered protocol pair and must otherwise reject mixed protocol records.
It must not rely on the Xuannv-only V5 admission consumer. The consumer must also fail closed unless
it uses the registered 10,000-resample bootstrap, random seed 20260725, five spatial folds,
complete 2 by 2 test clusters, hierarchy `fold -> cluster -> probe seed`, and the committed
patch-metadata hash `17137cc379a874949a3eb460d48d8ad7c7698b63dbaaf5b4d3c1b40f84f69d97`.
The comparison matrix, bootstrap output, and admission record must carry the bootstrap input and
source-code hashes. A paired contextual comparison may report only the OSM-assisted evidence class. The
Figure/Table wording is:

> Annual AEF 2025 versus monthly Xuannv 2026-04 under an aligned spatial downstream protocol.

No comparison may claim matched temporal information, independent semantic transfer, or global
superiority. OSM caches are not versioned historical snapshots aligned to the 2025--2026 imagery;
therefore this comparison cannot use them as contemporaneous ground truth or monthly-change
evidence. The paper will disclose unknown AEF pretraining overlap with Haidian.
