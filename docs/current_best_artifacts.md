# Current Best Artifacts

This file records the paths that should be used by default after the
2026-07-03 cleanup.

## Current Model

P5H Haidian masked monthly embedding model:

```text
/data/xuannv_embedding/outputs/v2_p5h_haidian_olmo_masked_mim_full_20260702/best.pt
```

## External Baseline

AEF official 2025 annual embedding:

```text
/data/xuannv_embedding/embeddings/aef_official_2025_annual
```

## Benchmarks

P5H MLP probe benchmark:

```text
/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/p5h_full_haidian_mlp_probe_202604_20260703_014658
```

AEF pixel MLP probe benchmark:

```text
/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/aef_haidian_2025_pixel_mlp_probe_20260703
```

AEF vs P5H visualization:

```text
/data/xuannv_embedding/experiments/v2_202512_202605/benchmarks/aef_vs_p5h_visualizations_20260703
```

## Notes

- Generated P5H embeddings were deleted because old exports wrote too many
  months and consumed unnecessary space.
- Re-export embeddings only for the specific month and region needed by a
  downstream evaluation.
- Do not use old P1/P2/P3/P4 embedding paths. They were removed during cleanup.
