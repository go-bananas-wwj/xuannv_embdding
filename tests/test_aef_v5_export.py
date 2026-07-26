from __future__ import annotations

import fcntl
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin


def _load_exporter():
    path = Path(__file__).parents[1] / "scripts/eval/export_aef_v5_embeddings.py"
    spec = importlib.util.spec_from_file_location("export_aef_v5_embeddings", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_raster(path: Path, data: np.ndarray, nodata: int | None = None) -> None:
    height, width = data.shape[-2:]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=data.shape[0],
        dtype=data.dtype,
        crs="EPSG:32650",
        transform=from_origin(500000, 4500000, 10, 10),
        nodata=nodata,
    ) as dst:
        dst.write(data)


def test_parser_rejects_label_and_probe_arguments() -> None:
    exporter = _load_exporter()
    parser = exporter.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--manifest", "manifest.json", "--label-root", "labels"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--manifest", "manifest.json", "--shot", "5"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--coverage-inventory",
                "coverage.json",
                "--output-root",
                "output",
                "--cog-cache-dir",
                "cache",
                "--cog-lock",
                "lock.json",
            ]
        )


def test_export_rejects_remote_index_uri_before_any_output_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exporter = _load_exporter()
    monkeypatch.setattr(exporter, "load_coverage_inventory", lambda _: ([], {}, "digest"))
    args = exporter.build_parser().parse_args(
        [
            "--coverage-inventory",
            str(tmp_path / "coverage.json"),
            "--output-root",
            str(tmp_path / "output"),
            "--aef-index-path",
            "s3://not-local/index.parquet",
            "--cog-cache-dir",
            str(tmp_path / "cache"),
            "--cog-lock",
            str(tmp_path / "lock.json"),
        ]
    )
    with pytest.raises(ValueError, match="local immutable"):
        exporter.export(args)
    assert not args.output_root.exists()


def test_coverage_inventory_requires_registered_320_patch_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exporter = _load_exporter()
    v5_manifest = tmp_path / "v5_manifest.json"
    v5_manifest.write_text(
        json.dumps([{"patch_id": f"patch_{index:06d}"} for index in range(320)]), encoding="utf-8"
    )
    monkeypatch.setattr(exporter, "V5_EVAL_MANIFEST", v5_manifest)
    monkeypatch.setattr(exporter, "V5_EVAL_MANIFEST_SHA256", exporter.sha256_file(v5_manifest))
    monkeypatch.setattr(
        exporter,
        "load_committed_json_snapshot",
        lambda path: (json.loads(Path(path).read_text(encoding="utf-8")), "test-digest"),
    )
    inventory = {
        "schema_version": 1,
        "kind": "aef_v5_coverage_inventory",
        "v5_manifest_sha256": exporter.V5_EVAL_MANIFEST_SHA256,
        "records": [
            {"patch_id": f"patch_{index:06d}", "reference_grid": {"path": "/frozen/ref.tif"}}
            for index in range(320)
        ],
    }
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inventory), encoding="utf-8")
    ids, records, digest = exporter.load_coverage_inventory(path)
    assert len(ids) == 320
    assert len(records) == 320
    assert digest == "test-digest"
    inventory["records"].pop()
    path.write_text(json.dumps(inventory), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 320"):
        exporter.load_coverage_inventory(path)


def test_read_requires_complete_valid_coverage(tmp_path: Path) -> None:
    exporter = _load_exporter()
    ref = tmp_path / "reference.tif"
    cog = tmp_path / "aef.tif"
    _write_raster(ref, np.zeros((1, 128, 128), dtype=np.uint16))
    values = np.full((64, 128, 128), 10, dtype=np.int8)
    values[:, 0, 0] = -128
    _write_raster(cog, values, nodata=-128)

    with pytest.raises(ValueError, match="complete valid coverage"):
        exporter.read_complete_aef_patch(cog, ref)


def test_read_rejects_non_64_channel_product(tmp_path: Path) -> None:
    exporter = _load_exporter()
    ref = tmp_path / "reference.tif"
    cog = tmp_path / "aef.tif"
    _write_raster(ref, np.zeros((1, 4, 4), dtype=np.uint16))
    _write_raster(cog, np.full((63, 4, 4), 10, dtype=np.int8), nodata=-128)

    with pytest.raises(ValueError, match="64 channels"):
        exporter.read_complete_aef_patch(cog, ref)


def test_read_rejects_source_without_declared_aef_nodata(tmp_path: Path) -> None:
    exporter = _load_exporter()
    ref = tmp_path / "reference.tif"
    cog = tmp_path / "aef.tif"
    _write_raster(ref, np.zeros((1, 128, 128), dtype=np.uint16))
    _write_raster(cog, np.full((64, 128, 128), 10, dtype=np.int8), nodata=None)
    with pytest.raises(ValueError, match="nodata must be declared"):
        exporter.read_complete_aef_patch(cog, ref)


def test_candidate_selection_rejects_multiple_cogs_without_mosaic_rule(tmp_path: Path) -> None:
    exporter = _load_exporter()
    ref = tmp_path / "reference.tif"
    _write_raster(ref, np.zeros((1, 4, 4), dtype=np.uint16))
    with pytest.raises(ValueError, match="multiple official AEF COGs"):
        exporter.select_single_cog(["s3://a.tif", "s3://b.tif"], ref)


def test_output_index_is_content_addressed_and_patch_complete(tmp_path: Path) -> None:
    exporter = _load_exporter()
    root = tmp_path / "export"
    for patch_id in ("000001", "000002"):
        path = root / "haidian" / patch_id
        path.mkdir(parents=True)
        torch.save(
            torch.zeros((64, 128, 128), dtype=torch.float32), path / "annual_2025_embedding_map.pt"
        )
    index = exporter.build_output_index(root, "haidian", {"000001", "000002"})
    assert index["file_count"] == 2
    assert len(index["index_sha256"]) == 64
    with pytest.raises(ValueError, match="patch set"):
        exporter.build_output_index(root, "haidian", {"000001", "000003"})


def test_locked_cog_snapshot_is_independent_of_cache_replacement(tmp_path: Path) -> None:
    exporter = _load_exporter()
    source = tmp_path / "cache.tiff"
    source.write_bytes(b"locked-aef-bytes")
    digest = exporter.sha256_file(source)
    snapshot = exporter.snapshot_locked_cog(source, digest, tmp_path / "staging")
    source.write_bytes(b"different-bytes")
    assert snapshot.read_bytes() == b"locked-aef-bytes"


def test_stable_flock_path_rejects_all_contenders_without_unlinking(tmp_path: Path) -> None:
    lock_path = tmp_path / ".published.lease"
    owner = lock_path.open("a+")
    contender = lock_path.open("a+")
    third = lock_path.open("a+")
    try:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert lock_path.exists()
        with pytest.raises(BlockingIOError):
            fcntl.flock(third.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        fcntl.flock(owner.fileno(), fcntl.LOCK_UN)
        owner.close()
        contender.close()
        third.close()


def test_mosaic_uses_ordered_first_valid_pixels_and_requires_full_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exporter = _load_exporter()
    first = torch.full((64, 128, 128), 1.0)
    second = torch.full((64, 128, 128), 2.0)
    first_valid = np.zeros((128, 128), dtype=bool)
    first_valid[:, :64] = True
    second_valid = np.ones((128, 128), dtype=bool)
    values = [(first, first_valid), (second, second_valid)]
    monkeypatch.setattr(exporter, "read_aef_patch_with_validity", lambda *_: values.pop(0))
    embedding, valid, selector = exporter.read_complete_aef_mosaic(
        [tmp_path / "a.tif", tmp_path / "b.tif"], tmp_path / "ref.tif"
    )
    assert bool(valid.all())
    assert float(embedding[0, 0, 0]) == 1.0
    assert float(embedding[0, 0, 127]) == 2.0
    assert selector[0, 0] == 0 and selector[0, 127] == 1

    monkeypatch.setattr(
        exporter,
        "read_aef_patch_with_validity",
        lambda *_: (torch.zeros((64, 128, 128)), np.zeros((128, 128), dtype=bool)),
    )
    with pytest.raises(ValueError, match="complete valid coverage"):
        exporter.read_complete_aef_mosaic([tmp_path / "empty.tif"], tmp_path / "ref.tif")


def test_failed_export_never_publishes_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exporter = _load_exporter()
    ids = [f"patch_{index:06d}" for index in range(320)]
    v5_manifest = tmp_path / "v5_manifest.json"
    v5_manifest.write_text(
        json.dumps([{"patch_id": patch_id} for patch_id in ids]), encoding="utf-8"
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "aef_v5_coverage_inventory",
                "v5_manifest_sha256": exporter.sha256_file(v5_manifest),
                "records": [
                    {"patch_id": patch_id, "reference_grid": {"path": "/frozen/ref.tif"}}
                    for patch_id in ids
                ],
            }
        ),
        encoding="utf-8",
    )
    source_index = tmp_path / "index.parquet"
    source_index.write_bytes(b"index")
    lock = tmp_path / "lock.json"
    lock.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(exporter, "V5_EVAL_MANIFEST", v5_manifest)
    monkeypatch.setattr(exporter, "V5_EVAL_MANIFEST_SHA256", exporter.sha256_file(v5_manifest))

    def load_test_json(path: Path) -> dict[str, object]:
        if Path(path) == lock:
            return {
                "schema_version": 1,
                "release": "aef/v1/annual",
                "year": 2025,
                "index_uri": "s3://official/index.parquet",
                "index_sha256": exporter.sha256_file(source_index),
                "assets": [{"filename": "bad.tif", "sha256": "0" * 64}],
            }
        return json.loads(Path(path).read_text(encoding="utf-8"))

    monkeypatch.setattr(
        exporter, "load_committed_json_snapshot", lambda path: (load_test_json(path), "test-digest")
    )
    monkeypatch.setattr(exporter, "verify_runtime_sources", lambda: {"git_revision": "test"})
    monkeypatch.setattr(
        exporter.gpd, "read_parquet", lambda _: SimpleNamespace(columns=[], empty=False, crs="x")
    )
    monkeypatch.setattr(exporter, "candidate_cogs", lambda *_: ["s3://bad.tif"])
    monkeypatch.setattr(exporter, "cache_cog", lambda *_: str(tmp_path / "missing.tif"))
    args = SimpleNamespace(
        coverage_inventory=inventory,
        output_root=tmp_path / "published",
        overwrite=False,
        aef_index_path=str(source_index),
        aef_index_uri="s3://official/index.parquet",
        cog_lock=lock,
        cog_cache_dir=tmp_path / "cache",
        region="haidian",
    )
    with pytest.raises(FileNotFoundError):
        exporter.export(args)
    assert not args.output_root.exists()
    assert not list(tmp_path.glob(".published.staging.*"))
