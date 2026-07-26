# Task 1 Report: Editable PPT primitives and structural tests

## Status

DONE

## Changes

- `scripts/report/pujiang_editable_common.py`
  - Added shared colors, fonts, 16:9 geometry, editable text/rectangle/header helpers,
    image contain/cover fitting, full-slide-picture detection, and relationship-aware
    first-slide cloning.
  - Reworked slide cloning to copy only slide content shape XML. Each embedded picture
    now creates an image part owned by the destination package from the source blob;
    layout, theme, notes, and other non-content package relationships are not copied.
  - Full-slide picture detection now checks coverage bounds, including negatively offset
    oversized pictures.
- `tests/test_pujiang_editable_deck.py`
  - Added structural tests for distinct text/picture shapes, geometry-based full-slide
    picture detection, and cloned picture relationships after reopening a PPTX.
  - Added real-slide regression coverage for slides 1-3: merged output reopens, ZIP
    members are unique, and every page preserves text plus the picture blob SHA multiset.

## Tests

```text
python -m ruff check scripts/report/pujiang_editable_common.py tests/test_pujiang_editable_deck.py
All checks passed!

python -m pytest tests/test_pujiang_editable_deck.py -q
4 passed in 1.94s

git diff --check
exit 0
```

## Commit

- `499a5f8 feat: add editable Pujiang PPT primitives` (pushed to `origin/v3-semantic-64d`)
- `9ed7258 fix: clone editable Pujiang slides safely` (pushed to `origin/v3-semantic-64d`)

## Concerns

- None.
