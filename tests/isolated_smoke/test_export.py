from __future__ import annotations

import os
import shutil
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
    ):
        path = sandbox / name
        path.write_text("{}", encoding="utf-8")
        evidence.append(path)
    model = torch.nn.Linear(2, 1)
    checkpoint = sandbox / "smoke_checkpoint.pt"
    save_smoke_checkpoint(model, checkpoint, _metadata_for(model))
    evidence.append(checkpoint)
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


def _complete_seal_inputs(
    sandbox: Path,
    tensors: tuple[torch.Tensor, torch.Tensor],
) -> tuple[list[Path], list[Path]]:
    embedding, valid = tensors
    return _export_all_groups(sandbox, embedding, valid), _write_evidence(sandbox)


def _unsafe_marker(marker: str) -> None:
    Path(marker).write_text("unsafe pickle executed", encoding="utf-8")


class _UnsafeLegacyPayload:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return (_unsafe_marker, (str(self.marker),))


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


@pytest.mark.parametrize(
    ("link_name", "is_live"),
    [
        ("smoke_checkpoint.pt", False),
        ("smoke_checkpoint.pt", True),
        ("smoke_checkpoint.pt.partial", False),
        ("smoke_checkpoint.pt.partial", True),
    ],
    ids=["target-broken", "target-live", "partial-broken", "partial-live"],
)
def test_checkpoint_save_rejects_symlinked_target_or_partial_without_touching_outside(
    tmp_sandbox: Path,
    tmp_path: Path,
    link_name: str,
    is_live: bool,
) -> None:
    """target 或 partial 的损坏/存活外部 symlink 都不得成为 checkpoint 写出跳板。"""
    model = torch.nn.Linear(2, 1)
    checkpoint = tmp_sandbox / "smoke_checkpoint.pt"
    outside = tmp_path / "outside-checkpoint.pt"
    if is_live:
        outside.write_bytes(b"outside must remain unchanged")
    os.symlink(outside, checkpoint.with_name(link_name))

    with pytest.raises(ExportError, match="symlink"):
        save_smoke_checkpoint(model, checkpoint, _metadata_for(model))

    assert checkpoint.is_symlink() if link_name == checkpoint.name else not checkpoint.exists()
    assert (
        outside.read_bytes() == b"outside must remain unchanged"
        if is_live
        else not outside.exists()
    )


def test_checkpoint_load_requires_sentinel_and_safe_weights_only_payload(
    tmp_sandbox: Path,
    tmp_path: Path,
) -> None:
    """加载不得越过 sandbox，也不能执行 legacy pickle 的任意代码。"""
    model = torch.nn.Linear(2, 1)
    metadata = _metadata_for(model)
    checkpoint = tmp_sandbox / "smoke_checkpoint.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": metadata,
            "legacy": _UnsafeLegacyPayload(tmp_path / "unsafe-marker"),
        },
        checkpoint,
    )
    with pytest.raises(ExportError):
        load_smoke_checkpoint(model, checkpoint, metadata)
    assert not (tmp_path / "unsafe-marker").exists()

    outside = tmp_path / "outside.pt"
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, outside)
    with pytest.raises(ExportError, match="sentinel sandbox"):
        load_smoke_checkpoint(model, outside, metadata)


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "shape"], ids=str)
def test_checkpoint_schema_failures_do_not_mutate_model(
    tmp_sandbox: Path,
    mutation: str,
) -> None:
    """严格恢复在键或 tensor 合同错误时必须不触碰现有模型参数。"""
    model = torch.nn.Linear(2, 1)
    metadata = _metadata_for(model)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    state_dict = dict(model.state_dict())
    if mutation == "missing":
        del state_dict["bias"]
    elif mutation == "unexpected":
        state_dict["unexpected"] = torch.zeros(1)
    else:
        state_dict["weight"] = torch.zeros((3, 2))
    checkpoint = tmp_sandbox / "smoke_checkpoint.pt"
    torch.save({"state_dict": state_dict, "metadata": metadata}, checkpoint)

    with pytest.raises(ExportError, match="state_dict"):
        load_smoke_checkpoint(model, checkpoint, metadata)

    after = model.state_dict()
    assert after.keys() == before.keys()
    for key in before:
        assert torch.equal(after[key], before[key])


def test_seal_success_requires_complete_sealed_false_formal_evidence(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """SUCCESS 只能在四组已封存且全部审计证据存在时写入。"""
    groups, evidence = _complete_seal_inputs(tmp_sandbox, full_contract_tensors)

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
    groups, evidence = _complete_seal_inputs(tmp_sandbox, full_contract_tensors)
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
    groups, evidence = _complete_seal_inputs(tmp_sandbox, full_contract_tensors)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(SafetyError):
        seal_success(tmp_sandbox, [*groups, *evidence, outside])


@pytest.mark.parametrize("kind", ["directory", "malformed_json", "checkpoint"], ids=str)
def test_seal_success_rejects_non_evidence_files(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
    kind: str,
) -> None:
    """空目录、坏 JSON 或坏 checkpoint 都不能满足封存证据合同。"""
    groups, evidence = _complete_seal_inputs(tmp_sandbox, full_contract_tensors)
    if kind == "directory":
        target = tmp_sandbox / "metrics.json"
        target.unlink()
        target.mkdir()
    elif kind == "malformed_json":
        (tmp_sandbox / "metrics.json").write_text("not-json", encoding="utf-8")
    else:
        torch.save({"not": "a smoke checkpoint"}, tmp_sandbox / "smoke_checkpoint.pt")

    with pytest.raises(ExportError):
        seal_success(tmp_sandbox, [*groups, *evidence])
    assert not (tmp_sandbox / "SUCCESS").exists()


@pytest.mark.parametrize("link_kind", ["group", "ancestor"], ids=str)
def test_seal_success_rejects_mandatory_zarr_symlink_to_outside(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
    tmp_path: Path,
    link_kind: str,
) -> None:
    """即使外部 target 是完整 Zarr，mandatory group symlink 也不得封存成功。"""
    groups, evidence = _complete_seal_inputs(tmp_sandbox, full_contract_tensors)
    if link_kind == "group":
        outside = tmp_path / "outside-full.zarr"
        shutil.copytree(groups[-1], outside)
        shutil.rmtree(groups[-1])
        os.symlink(outside, groups[-1], target_is_directory=True)
    else:
        outside = tmp_path / "outside-full-parent"
        outside.mkdir()
        shutil.copytree(groups[-1], outside / "embedding.zarr")
        shutil.rmtree(groups[-1].parent)
        os.symlink(outside, groups[-1].parent, target_is_directory=True)

    with pytest.raises(ExportError, match="symlink"):
        seal_success(tmp_sandbox, [*groups, *evidence])
    assert not (tmp_sandbox / "SUCCESS").exists()


@pytest.mark.parametrize("mutation", ["uncompressed", "extra", "wrong_identifier"], ids=str)
def test_seal_success_rejects_altered_zarr_contract(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
    mutation: str,
) -> None:
    """编解码器、array 集合和标识符任一被改写时均拒绝 SUCCESS。"""
    groups, evidence = _complete_seal_inputs(tmp_sandbox, full_contract_tensors)
    exported = zarr.open_group(str(groups[0]), mode="a")
    if mutation == "uncompressed":
        embedding = exported["embedding"][:]
        del exported["embedding"]
        exported.create_dataset(
            "embedding",
            data=embedding,
            chunks=(1, 1, 64, 128, 128),
            compressor=None,
        )
    elif mutation == "extra":
        exported.create_dataset("unexpected", data=np.array([1], dtype=np.int8))
    else:
        exported["patch_id"][0] = "wrong-000"

    with pytest.raises(ExportError):
        seal_success(tmp_sandbox, [*groups, *evidence])
    assert not (tmp_sandbox / "SUCCESS").exists()
