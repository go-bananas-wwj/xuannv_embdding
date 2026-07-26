#!/usr/bin/env python3
"""Build a five-fold Haidian split whose sets are unions of complete 2 x 2 patches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def build_complete_blocks(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Return global-lattice 2 x 2 blocks and boundary patches excluded from inference."""
    bounds = {str(record["patch_id"]): record["bounds"] for record in records}
    first = next(iter(bounds.values()))
    width, height = first[2] - first[0], first[3] - first[1]
    origin_x = min(value[0] for value in bounds.values())
    origin_y = min(value[1] for value in bounds.values())
    cells: dict[tuple[int, int], str] = {}
    for patch_id, (min_x, min_y, max_x, max_y) in bounds.items():
        if max_x - min_x != width or max_y - min_y != height:
            raise ValueError("Patch metadata does not define one regular grid")
        row, col = round((min_y - origin_y) / height), round((min_x - origin_x) / width)
        if (row, col) in cells:
            raise ValueError("Patch metadata has overlapping lattice cells")
        cells[(row, col)] = patch_id
    blocks = []
    used: set[str] = set()
    for row, col in sorted(cells):
        if row % 2 or col % 2:
            continue
        patch_ids = [cells.get((row + dr, col + dc)) for dr in range(2) for dc in range(2)]
        if all(patch_ids):
            blocks.append({"row": row, "col": col, "patch_ids": sorted(patch_ids)})
            used.update(patch_ids)
    return blocks, sorted(set(bounds) - used)


def build_split(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Create compact test clusters with a full 2D complete-block buffer."""
    blocks, excluded = build_complete_blocks(records)
    blocks.sort(key=lambda item: (item["row"], item["col"]))
    blocks_by_cell = {(block["row"], block["col"]): block for block in blocks}
    all_ids = {patch_id for block in blocks for patch_id in block["patch_ids"]}
    count = len(blocks)
    test_block_count, validation_block_count = 10, 5
    if count < 5 * test_block_count + validation_block_count:
        raise ValueError("Five-fold block split requires at least 55 complete 2 x 2 blocks")
    remaining_indices = set(range(count))
    selected_cells: set[tuple[int, int]] = set()
    test_groups: list[set[int]] = []
    for _fold in range(5):
        if selected_cells:
            anchor_index = max(
                remaining_indices,
                key=lambda index: (
                    min(
                        (blocks[index]["row"] - row) ** 2 + (blocks[index]["col"] - col) ** 2
                        for row, col in selected_cells
                    ),
                    blocks[index]["row"],
                    blocks[index]["col"],
                ),
            )
        else:
            anchor_index = min(
                remaining_indices, key=lambda index: (blocks[index]["row"], blocks[index]["col"])
            )
        anchor = (blocks[anchor_index]["row"], blocks[anchor_index]["col"])
        test_indices = set(
            sorted(
                remaining_indices,
                key=lambda index: (
                    (blocks[index]["row"] - anchor[0]) ** 2
                    + (blocks[index]["col"] - anchor[1]) ** 2,
                    blocks[index]["row"],
                    blocks[index]["col"],
                ),
            )[:test_block_count]
        )
        test_groups.append(test_indices)
        remaining_indices -= test_indices
        selected_cells.update(
            (blocks[index]["row"], blocks[index]["col"]) for index in test_indices
        )
    all_test_indices = set().union(*test_groups)
    folds = []
    for fold in range(5):
        test_indices = test_groups[fold]
        test_cells = {(blocks[index]["row"], blocks[index]["col"]) for index in test_indices}
        buffer_cells = {
            cell
            for cell in blocks_by_cell
            if cell not in test_cells
            and any(max(abs(cell[0] - test[0]), abs(cell[1] - test[1])) <= 2 for test in test_cells)
        }
        center_row = sum(row for row, _ in test_cells) / len(test_cells)
        center_col = sum(col for _, col in test_cells) / len(test_cells)
        candidates = [
            index
            for index, block in enumerate(blocks)
            if index not in all_test_indices
            and (block["row"], block["col"]) not in buffer_cells
        ]
        if len(candidates) < validation_block_count:
            raise ValueError(f"Fold {fold} cannot place a validation band outside the test buffer")
        val_indices = set(
            sorted(
                candidates,
                key=lambda index: (
                    (blocks[index]["row"] - center_row) ** 2
                    + (blocks[index]["col"] - center_col) ** 2,
                    blocks[index]["row"],
                    blocks[index]["col"],
                ),
            )[:validation_block_count]
        )
        groups = {
            "test": test_indices,
            "val": val_indices,
            "buffer": {
                next(
                    index
                    for index, block in enumerate(blocks)
                    if (block["row"], block["col"]) == cell
                )
                for cell in buffer_cells
            },
        }
        grouped_ids = {
            name: sorted(
                patch_id for index in indices for patch_id in blocks[index]["patch_ids"]
            )
            for name, indices in groups.items()
        }
        train = sorted(all_ids - set().union(*map(set, grouped_ids.values())))
        if len(train) < 150:
            raise ValueError(
                f"Fold {fold} leaves only {len(train)} training patches; need at least 150"
            )
        folds.append({"fold": fold, "train": train, **grouped_ids})
    return {
        "schema_version": 1,
        "strategy": "complete_2x2_blocks_compact_farthest_point_test_clusters",
        "validation_strategy": "nearest_non_test_blocks_outside_complete_buffer",
        "buffer": "one_complete_block_at_each_test_edge",
        "n_folds": 5,
        "test_blocks_per_fold": test_block_count,
        "validation_blocks_per_fold": validation_block_count,
        "complete_block_count": len(blocks),
        "excluded_boundary_patch_ids": excluded,
        "folds": folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patch-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.patch_metadata.read_text(encoding="utf-8"))
    result = build_split(records)
    first = records[0]["bounds"]
    result["patch_metadata_path"] = str(args.patch_metadata.resolve())
    result["patch_metadata_sha256"] = hashlib.sha256(args.patch_metadata.read_bytes()).hexdigest()
    result["lattice_origin"] = [
        min(record["bounds"][0] for record in records),
        min(record["bounds"][1] for record in records),
    ]
    result["patch_lattice_spacing"] = [first[2] - first[0], first[3] - first[1]]
    result["generation"] = {
        "command": "complete_2x2_blocks + compact_farthest_clusters + chebyshev_one_block_buffer",
        "seed": None,
        "seed_note": (
            "The v5_seed42 filename preserves paper experiment lineage; no random split is used."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
