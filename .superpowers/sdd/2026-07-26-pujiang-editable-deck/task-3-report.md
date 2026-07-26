# Task 3 Report: Pujiang Editable Slides 8-10

## Delivered

- Added `scripts/report/pujiang_editable_pages08_10.py` with native builders for
  pages 8, 9, and 10.
- Page 8 renders F1 and AUC as native horizontal-bar shapes and AP as editable
  table cells. Values follow the reviewed 3-polygon PU+Query comparison; Xuannv,
  AEF, and traditional features are red, blue, and gray respectively.
- Page 9 uses one independent PCA picture and four native evidence cards covering
  road advantage, water boundary, building boundary, and next validation steps.
- Page 10 uses five independent pictures for Harbin PCA, building, water, land-use /
  cover, and land-use conversion, with native captions and explanatory text.

## Verification

- `ruff check scripts/report/pujiang_editable_pages08_10.py
  scripts/report/build_pujiang_editable_deck.py tests/test_pujiang_editable_deck.py`
  passed.
- Focused Pujiang tests passed: 16 tests.
