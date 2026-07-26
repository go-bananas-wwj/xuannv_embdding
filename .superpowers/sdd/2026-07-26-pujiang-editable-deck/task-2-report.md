# Task 2 Report: Pujiang Editable Slides 4-7

## Delivered

- Added `scripts/report/pujiang_editable_pages04_07.py` with `build_page04`,
  `build_page05`, `build_page06`, and `build_page07`.
- Rebuilt all page text, cards, arrows, labels, and evidence boundaries as native
  PowerPoint objects.
- Page 5 crops each reviewed row asset into six separate raster pictures for each
  task, resulting in 18 independent picture objects; task names and column titles
  are native text.
- Page 7 embeds all five observation-timeline panels and the independent generation
  example as six separate picture objects. The first four crops are 275x262 pixels,
  the final crop is 276x262 pixels, and all omit the source strip's 34-pixel label
  band.
- Adjusted the page 5 task/footer text, page 6 card/boundary text, and page 7
  title/boundary text to fit their native text boxes without overlap. Page 6 now
  carries the native boundary label `概念流程示意（非真实系统界面）`.
- Follow-up quality repair: expanded the page 6 concept label to 2.55x0.24 inches
  and every 14-point workflow-card title to at least 0.22 inches high; each title
  now ends before its detail text box begins.
- Archived the platform screenshot and generation example in
  `docs/presentations/assets/pujiang_202607/editable_sources/`, with source paths
  recorded in that directory's README; the builder now uses these repository-local
  assets instead of `/data` paths.
- Generated and reopened the four-slide inspection deck at
  `docs/presentations/pujiang_202607/浦江交流_第04至07页_可编辑检查版_20260726.pptx`.

## Verification

- `ruff check scripts/report/pujiang_editable_pages04_07.py tests/test_pujiang_editable_deck.py`
  passed.
- `python -m pytest tests/test_pujiang_editable_deck.py -q` passed: 9 tests.
- Reopened inspection PPTX: pages 4-7 contain 35, 55, 35, and 35 shapes;
  picture counts are 3, 18, 1, and 6. No slide contains a full-slide picture.
- The re-opened page 7 picture blobs contain exactly four 275x262 crops and one
  276x262 crop; their boundary rows are not all black.
- The re-opened page 6 concept label and all five workflow-card title/detail pairs
  meet the size and non-overlap geometry checks.
