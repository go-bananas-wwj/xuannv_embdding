from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

from experiments.china_v1_fusion_smoke.export import (
    ExportError,
    export_group_zarr,
    load_smoke_checkpoint,
    save_smoke_checkpoint,
    seal_success,
)
from experiments.china_v1_fusion_smoke.safety import SafetyError

GROUPS = ("base", "base_aef", "base_highres", "full")
PATCH_IDS = ("patch-000", "patch-001", "patch-002", "patch-003")
PERIODS = tuple(f"{year}Q{quarter}" for year in (2020, 2021) for quarter in range(1, 5))


@pytest.fixture
def tmp_sandbox(tmp_path: Path) -> Path:
    """提供带隔离哨兵的最小可写 smoke 沙箱。"""
    sandbox = tmp_path / "china_v1_fusion_smoke_20260815"
    sandbox.mkdir()
    (sandbox / ".xuannv_isolated_smoke").touch()
    return sandbox


@pytest.fixture
def full_contract_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    """精确全尺寸合同张量，避免由被测代码推导预期 shape。"""
    return (
        torch.zeros((4, 8, 64, 128, 128), dtype=torch.float32),
        torch.ones((4, 8, 1, 128, 128), dtype=torch.bool),
    )


def _metadata_for(model: torch.nn.Module) -> dict[str, object]:
    return {
        "git_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "selected_patch_manifest_sha256": "c" * 64,
        "seed": 20260815,
        "model_class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "synthetic": True,
    }


def _write_evidence(sandbox: Path) -> list[Path]:
    evidence = []
    for name in (
        "run_manifest.json",
        "metrics.json",
        "path_audit.json",
        "reproducibility.json",
        "smoke_checkpoint.pt",
    ):
        path = sandbox / name
        path.write_text("{}", encoding="utf-8")
        evidence.append(path)
    return evidence


def _export_all_groups(
    sandbox: Path,
    embedding: torch.Tensor,
    valid: torch.Tensor,
) -> list[Path]:
    return [
        export_group_zarr(
            group,
            embedding,
            valid,
            PATCH_IDS,
            PERIODS,
            sandbox / "outputs" / group / "embedding.zarr",
            sandbox,
        )
        for group in GROUPS
    ]


def test_export_contract_and_checkpoint_reload(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """错误的 shape、dtype 或 synthetic 用途字段必须无法伪装成正式导出。"""
    embedding, valid = full_contract_tensors
    output = tmp_sandbox / "outputs" / "full" / "embedding.zarr"

    written = export_group_zarr("full", embedding, valid, PATCH_IDS, PERIODS, output, tmp_sandbox)

    group = zarr.open_group(str(written), mode="r")
    assert written == output
    assert not output.with_name("embedding.zarr.partial").exists()
    assert group["embedding"].shape == (4, 8, 64, 128, 128)
    assert group["embedding"].dtype == np.dtype("float16")
    assert group["embedding"].chunks == (1, 1, 64, 128, 128)
    assert group["embedding"].compressor.cname == "zstd"
    assert group["valid"].shape == (4, 8, 1, 128, 128)
    assert set(group.array_keys()) == {"embedding", "valid", "patch_id", "period"}
    assert group["patch_id"][:].tolist() == list(PATCH_IDS)
    assert group["period"][:].tolist() == list(PERIODS)
    assert group.attrs["synthetic"] is True
    assert group.attrs["formal_training_allowed"] is False
    assert group.attrs["formal_evaluation_allowed"] is False

    model = torch.nn.Linear(2, 1)
    checkpoint = tmp_sandbox / "smoke_checkpoint.pt"
    metadata = _metadata_for(model)
    checksum = save_smoke_checkpoint(model, checkpoint, metadata)
    with torch.no_grad():
        model.weight.zero_()
    load_smoke_checkpoint(model, checkpoint, metadata)
    assert checksum == sha256(checkpoint.read_bytes()).hexdigest()
    assert not torch.equal(model.weight, torch.zeros_like(model.weight))


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("git_commit", "d" * 40),
        ("config_sha256", "e" * 64),
        ("selected_patch_manifest_sha256", "f" * 64),
        ("seed", 1),
        ("model_class", "other.module.Model"),
        ("synthetic", False),
    ],
)
def test_checkpoint_reload_rejects_any_metadata_mismatch(
    tmp_sandbox: Path,
    field: str,
    replacement: object,
) -> None:
    """错配 provenance 时恢复权重会混淆实验，因此必须 fail closed。"""
    model = torch.nn.Linear(2, 1)
    metadata = _metadata_for(model)
    checkpoint = tmp_sandbox / "smoke_checkpoint.pt"
    save_smoke_checkpoint(model, checkpoint, metadata)
    expected = dict(metadata)
    expected[field] = replacement

    with pytest.raises(ExportError, match="metadata"):
        load_smoke_checkpoint(model, checkpoint, expected)


def test_checkpoint_save_requires_a_sentinel_sandbox(tmp_path: Path) -> None:
    """保存 checkpoint 也必须经由同一写路径守卫，不能成为沙箱逃逸口。"""
    model = torch.nn.Linear(2, 1)

    with pytest.raises(ExportError, match="sentinel sandbox"):
        save_smoke_checkpoint(model, tmp_path / "smoke_checkpoint.pt", _metadata_for(model))


def test_seal_success_requires_complete_sealed_false_formal_evidence(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """SUCCESS 只能在四组已封存且全部审计证据存在时写入。"""
    embedding, valid = full_contract_tensors
    groups = _export_all_groups(tmp_sandbox, embedding, valid)
    evidence = _write_evidence(tmp_sandbox)

    success = seal_success(tmp_sandbox, [*groups, *evidence])

    assert success == tmp_sandbox / "SUCCESS"
    payload = success.read_text(encoding="utf-8")
    assert "combined_sha256" in payload
    assert "utc" in payload


def test_seal_success_rejects_missing_manifest_partial_or_formal_use(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """缺证据、未封存 Zarr 或可正式使用标记都不得生成 SUCCESS。"""
    embedding, valid = full_contract_tensors
    groups = _export_all_groups(tmp_sandbox, embedding, valid)
    evidence = _write_evidence(tmp_sandbox)
    (tmp_sandbox / "run_manifest.json").unlink()
    with pytest.raises(ExportError, match="run_manifest"):
        seal_success(tmp_sandbox, [*groups, *evidence])
    assert not (tmp_sandbox / "SUCCESS").exists()

    (tmp_sandbox / "run_manifest.json").write_text("{}", encoding="utf-8")
    partial = groups[0].with_name("embedding.zarr.partial")
    partial.mkdir()
    with pytest.raises(ExportError, match="partial"):
        seal_success(tmp_sandbox, [*groups, *evidence])
    partial.rmdir()

    exported = zarr.open_group(str(groups[0]), mode="a")
    exported.attrs["formal_training_allowed"] = True
    with pytest.raises(ExportError, match="formal"):
        seal_success(tmp_sandbox, [*groups, *evidence])
    assert not (tmp_sandbox / "SUCCESS").exists()


def test_seal_success_rejects_required_outputs_outside_the_sandbox(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
    tmp_path: Path,
) -> None:
    """调用者注入沙箱外 required path 时不能绕过隔离路径守卫。"""
    embedding, valid = full_contract_tensors
    groups = _export_all_groups(tmp_sandbox, embedding, valid)
    evidence = _write_evidence(tmp_sandbox)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(SafetyError):
        seal_success(tmp_sandbox, [*groups, *evidence, outside])
