# V2 Manifest Validation: 202512-202605

Generated with:

```bash
python scripts/data/build_v2_manifest.py
```

Data outputs:

- Manifest:
  `/data/xuannv_embedding/processed/v2_haidian_harbin_202512_202605/manifest_v2_202512_202605.json`
- Metadata:
  `/data/xuannv_embedding/processed/v2_haidian_harbin_202512_202605/manifest_v2_202512_202605.meta.json`

Summary:

- Entries: `744`
- Missing files: `0`
- Regions: `haidian`, `harbin`
- Months requested: `202512`, `202601`, `202602`, `202603`, `202604`, `202605`

Known coverage gaps recorded in metadata:

- Haidian `highres_optical_haidian` covers `202512-202604`; `202605` is missing.
- Harbin `s2` covers `202512`; `202601-202605` are missing.
- Harbin `landsat` covers `202512`; `202601-202605` are missing.
- Harbin `highres_optical_harbin` covers `202512` and `202605`;
  `202601-202604` are missing.
- Harbin has no `highres_sar_haidian` equivalent in the current processed manifest.

High-resolution optical sources are intentionally region-specific in V2 because
their native spatial resolutions differ by region even though both currently have
3 channels. Keeping them separate prevents mixed-resolution highres assumptions
while retaining unified low-resolution source names.

These gaps are intentional availability-mask cases for V2. The manifest builder
does not synthesize missing observations.
