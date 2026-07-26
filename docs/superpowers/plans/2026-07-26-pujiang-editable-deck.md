# Pujiang Editable Deck Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the reviewed ten-slide Pujiang presentation as a genuinely editable PowerPoint rather than ten full-slide screenshots.

**Architecture:** Reuse the existing editable slide 1-3 PPTX files through a relationship-aware slide clone helper. Build slides 4-10 directly with `python-pptx` shapes, text boxes, tables, bars, and independent raster picture objects. A final builder assembles all ten slides and a structural test rejects any slide that collapses into one full-slide picture.

**Tech Stack:** Python 3.11, `python-pptx`, Pillow, pytest.

## Global Constraints

- Final output has exactly 10 slides at 16:9.
- No slide may consist of one full-slide screenshot.
- Text, cards, arrows, legends, tables, and metric bars are native editable PowerPoint elements.
- Remote-sensing imagery, PCA rasters, prediction rasters, and platform screenshots remain independent picture objects.
- Slide 3, 5, 8, and 9 content must match the already reviewed evidence.
- Every completed task is committed and pushed immediately.

---

### Task 1: Editable PPT primitives and structural tests

**Files:**
- Create: `scripts/report/pujiang_editable_common.py`
- Create: `tests/test_pujiang_editable_deck.py`

**Interfaces:**
- Produces: `new_blank_slide(prs)`, `add_header(slide, title, kicker)`, `add_text(...)`, `add_rect(...)`, `add_picture(...)`, `clone_first_slide(source, destination)`.

- [ ] Write a failing test that creates a deck and asserts helper-created text and picture shapes are separate.
- [ ] Write a failing test that defines a full-slide-picture detector using slide width and height.
- [ ] Implement shared colors, fonts, geometry helpers, image fitting, and relationship-aware cloning for pictures.
- [ ] Run `python -m pytest tests/test_pujiang_editable_deck.py -q`.
- [ ] Commit and push `feat: add editable Pujiang PPT primitives`.

### Task 2: Rebuild slides 4-7 as native elements

**Files:**
- Create: `scripts/report/pujiang_editable_pages04_07.py`
- Modify: `tests/test_pujiang_editable_deck.py`

**Interfaces:**
- Consumes: helpers from `pujiang_editable_common.py`.
- Produces: `build_page04(prs)`, `build_page05(prs)`, `build_page06(prs)`, `build_page07(prs)`.

- [ ] Add tests for required page titles and minimum editable shape counts.
- [ ] Build page 4 with independent platform screenshot, PCA, poster, and four editable function cards.
- [ ] Build page 5 with editable task/column labels and 18 independent raster panels cropped from the reviewed slide-row assets.
- [ ] Build page 6 with an independent workflow image and five editable workflow cards.
- [ ] Build page 7 with independently cropped observation panels, editable arrows, labels, and boundary statement.
- [ ] Run focused tests and generate a four-slide inspection deck.
- [ ] Commit and push `feat: rebuild Pujiang slides 4 to 7 as editable`.

### Task 3: Rebuild slides 8-10 as native elements

**Files:**
- Create: `scripts/report/pujiang_editable_pages08_10.py`
- Modify: `tests/test_pujiang_editable_deck.py`

**Interfaces:**
- Consumes: helpers from `pujiang_editable_common.py`.
- Produces: `build_page08(prs)`, `build_page09(prs)`, `build_page10(prs)`.

- [ ] Add tests for native F1/AUC bars, AP table text, slide 9 evidence boundary, and slide 10 independent image count.
- [ ] Build page 8 with native bars, legends, AP table, conclusion cards, and time-product boundary.
- [ ] Build page 9 with independent PCA image and editable road/water/building evidence cards.
- [ ] Build page 10 with independent Harbin PCA and four task images plus editable captions.
- [ ] Run focused tests and generate a three-slide inspection deck.
- [ ] Commit and push `feat: rebuild Pujiang slides 8 to 10 as editable`.

### Task 4: Assemble, validate, and review the ten-slide deck

**Files:**
- Create: `scripts/report/build_pujiang_editable_deck.py`
- Modify: `tests/test_pujiang_editable_deck.py`
- Create: `docs/presentations/pujiang_202607/玄女月度地理嵌入_浦江交流_10页可编辑版_20260726.pptx` (ignored binary artifact)

**Interfaces:**
- Consumes: existing editable slides 1-3 and page builders from Tasks 2-3.
- Produces: `build_deck(output: Path) -> Path`.

- [ ] Add tests for 10 slides, editable shape counts, key text, picture independence, and no full-slide image.
- [ ] Assemble slides 1-3 by cloning their existing editable PPTX content.
- [ ] Append native slides 4-10 and save the final file.
- [ ] Run `python -m pytest tests/test_pujiang_editable_deck.py tests/test_pujiang_slide03.py tests/test_pujiang_slides04_10.py -q`.
- [ ] Inspect slide shape inventory and render representative pages where rendering support is available.
- [ ] Request independent review of narrative, evidence, visual quality, and editability; fix until all checks pass.
- [ ] Commit and push `feat: deliver editable Pujiang presentation`.
