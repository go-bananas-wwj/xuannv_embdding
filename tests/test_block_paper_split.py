from __future__ import annotations

from scripts.data.build_block_paper_split import build_split


def test_block_split_uses_only_complete_blocks_and_disjoint_sets() -> None:
    records = [
        {"patch_id": f"p{row}{col}", "bounds": [col, row, col + 1, row + 1]}
        for row in range(10)
        for col in range(20)
    ] + [{"patch_id": "edge", "bounds": [20, 0, 21, 1]}]
    split = build_split(records, (1.0, 0.0))

    assert split["complete_block_count"] == 50
    assert split["excluded_boundary_patch_ids"] == ["edge"]
    for fold in split["folds"]:
        train, val, test, buffer = (set(fold[key]) for key in ("train", "val", "test", "buffer"))
        assert not ((train & val) | (train & test) | (train & buffer) | (val & test))
        assert len(test) % 4 == len(val) % 4 == len(buffer) % 4 == 0
