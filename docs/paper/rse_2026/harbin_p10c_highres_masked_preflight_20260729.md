# Harbin P10C high-resolution-masked export preflight

The diagnostic uses a new P10C-only embedding export and is not an interpretation of the
unmasked P10C or AEF embeddings.

The independent one-patch preflight passed on `harbin_patch_000000`:

- the physical high-resolution optical input was present before masking (`mask_sum_before=11083`)
  and unavailable during inference (`mask_sum_after=0`);
- the high-resolution SAR schema slot was physically absent and remains explicitly unavailable
  (`mask_sum_before=0`, `mask_sum_after=0`);
- the resulting April-2026 map passed finite-value and shape validation (`64×128×128`).

The exporter writes a per-patch availability audit and refuses sealing unless both
`highres_optical_haidian` and `highres_sar_haidian` have zero availability after masking. The
full 380-patch export uses the same sealed contract in
[the diagnostic configuration](../../configs/eval/harbin_p10c_highres_mask_diagnostic_20260729.json).
No downstream diagnostic cell may start until that new full export is sealed.
