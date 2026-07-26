from __future__ import annotations

import pytest

from scripts.data.build_block_paper_split import build_complete_blocks, build_split


def test_block_split_uses_only_complete_blocks_and_disjoint_sets() -> None:
    records = [
        {"patch_id": f"p{row}_{col}", "bounds": [col, row, col + 1, row + 1]}
        for row in range(16)
        for col in range(20)
    ] + [{"patch_id": "edge", "bounds": [20, 0, 21, 1]}]
    split = build_split(records)

    assert split["complete_block_count"] == 80
    assert split["excluded_boundary_patch_ids"] == ["edge"]
    complete_blocks, _ = build_complete_blocks(records)
    complete_block_sets = [set(block["patch_ids"]) for block in complete_blocks]
    block_cell_by_patch = {
        patch_id: (block["row"], block["col"])
        for block in complete_blocks
        for patch_id in block["patch_ids"]
    }
    all_block_cells = set(block_cell_by_patch.values())
    all_test = set().union(*(set(fold["test"]) for fold in split["folds"]))
    assert sum(len(fold["test"]) for fold in split["folds"]) == len(all_test)
    cell_by_patch = {
        record["patch_id"]: (int(record["bounds"][1]), int(record["bounds"][0]))
        for record in records
        if record["patch_id"] != "edge"
    }
    for fold in split["folds"]:
        train, val, test, buffer = (set(fold[key]) for key in ("train", "val", "test", "buffer"))
        assert not (
            (train & val)
            | (train & test)
            | (train & buffer)
            | (val & test)
            | (val & buffer)
            | (test & buffer)
        )
        assert len(test) % 4 == len(val) % 4 == len(buffer) % 4 == 0
        assert len(test) == 40
        assert len(val) == 20
        assert not (val & (all_test - test))
        assert len(train) >= 150
        for group in (train, val, test, buffer):
            assert all(not (block & group) or block <= group for block in complete_block_sets)
        test_cells = {block_cell_by_patch[patch_id] for patch_id in test}
        buffer_cells = {block_cell_by_patch[patch_id] for patch_id in buffer}
        for test_row, test_col in test_cells:
            for row, col in all_block_cells:
                is_neighbor = max(abs(row - test_row), abs(col - test_col)) <= 2
                if (row, col) not in test_cells and is_neighbor:
                    assert (row, col) in buffer_cells
        for test_patch in test:
            for train_patch in train:
                test_row, test_col = cell_by_patch[test_patch]
                train_row, train_col = cell_by_patch[train_patch]
                assert max(abs(test_row - train_row), abs(test_col - train_col)) > 2


def test_block_split_requires_enough_complete_blocks() -> None:
    records = [
        {"patch_id": f"p{row}_{col}", "bounds": [col, row, col + 1, row + 1]}
        for row in range(8)
        for col in range(12)
    ]

    with pytest.raises(ValueError, match="at least 55 complete 2 x 2 blocks"):
        build_split(records)
