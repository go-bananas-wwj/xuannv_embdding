from __future__ import annotations

import json
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
    verify_success,
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
    embedding = torch.zeros((4, 8, 64, 128, 128), dtype=torch.float32)
    embedding[:, :, 0] = 1.0
    return embedding, torch.ones((4, 8, 1, 128, 128), dtype=torch.bool)


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
    from experiments.china_v1_fusion_smoke.model import IsolatedFusionSmokeModel

    selection = sandbox / "manifests" / "patch_selection.json"
    selection.parent.mkdir()
    selection.write_text(json.dumps({"patch_ids": list(PATCH_IDS)}) + "\n", encoding="utf-8")
    launcher_log = sandbox / "logs" / "npu_smoke.log"
    launcher_log.parent.mkdir()
    launcher_log.write_text("complete foreground runner output\n", encoding="utf-8")
    preliminary = sandbox / "manifests/preliminary_path_audit.json"
    preliminary.write_text(
        json.dumps(
            {
                "stage": "npu-smoke",
                "stages": ["prepare", "cpu-contract", "npu-smoke"],
                "sandbox_root": str(sandbox),
                "created_or_modified": ["manifests/preliminary_path_audit.json"],
                "formal_training_allowed": False,
                "formal_evaluation_allowed": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    ready = sandbox / "READY_TO_SEAL"
    ready.write_text(
        json.dumps(
            {
                "status": "npu_compute_complete",
                "git_commit": "a" * 40,
                "config_sha256": "b" * 64,
                "preliminary_path_audit": {
                    "path": "manifests/preliminary_path_audit.json",
                    "sha256": sha256(preliminary.read_bytes()).hexdigest(),
                },
                "source_unchanged": True,
                "synthetic": True,
                "formal_training_allowed": False,
                "formal_evaluation_allowed": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    tee_complete = sandbox / "TEE_COMPLETE"
    tee_complete.write_text(
        json.dumps(
            {
                "status": "tee_pipeline_complete",
                "git_commit": "a" * 40,
                "config_sha256": "b" * 64,
                "ready_to_seal_sha256": sha256(ready.read_bytes()).hexdigest(),
                "launcher_log": {
                    "path": "logs/npu_smoke.log",
                    "size": launcher_log.stat().st_size,
                    "mtime_ns": launcher_log.stat().st_mtime_ns,
                    "sha256": sha256(launcher_log.read_bytes()).hexdigest(),
                },
                "synthetic": True,
                "formal_training_allowed": False,
                "formal_evaluation_allowed": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    audit = sandbox / "path_audit.json"
    audit.write_text(
        json.dumps(
            {
                "stage": "finalize-seal",
                "stages": ["prepare", "cpu-contract", "npu-smoke", "finalize-seal"],
                "created_or_modified": [
                    "logs/npu_smoke.log",
                    "manifests/preliminary_path_audit.json",
                    "path_audit.json",
                    "READY_TO_SEAL",
                    "TEE_COMPLETE",
                    "SUCCESS",
                ],
                "declared_files": {
                    relative: {
                        "size": path.stat().st_size,
                        "mtime_ns": path.stat().st_mtime_ns,
                        "sha256": sha256(path.read_bytes()).hexdigest(),
                    }
                    for relative, path in {
                        "logs/npu_smoke.log": launcher_log,
                        "READY_TO_SEAL": ready,
                        "TEE_COMPLETE": tee_complete,
                    }.items()
                },
                "final_seal": {
                    "path": "SUCCESS",
                    "status": "expected_last_write",
                    "exists_when_audit_written": False,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    model = IsolatedFusionSmokeModel(embed_dim=64)
    checkpoint = sandbox / "smoke_checkpoint.pt"
    metadata = {
        "git_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "selected_patch_manifest_sha256": sha256(selection.read_bytes()).hexdigest(),
        "seed": 20260815,
        "model_class": ("experiments.china_v1_fusion_smoke.model.IsolatedFusionSmokeModel"),
        "synthetic": True,
        "formal_training_allowed": False,
        "formal_evaluation_allowed": False,
    }
    checkpoint_sha = save_smoke_checkpoint(model, checkpoint, metadata)
    _add_prepared_evidence_declared_by_audit(sandbox)
    (sandbox / "metrics.json").write_text(
        json.dumps(
            {
                "groups": _valid_group_metrics(),
                "zero_gate_max_abs_error": 0.0,
                "gradient_l1": {
                    "aef_adapter": 1.0,
                    "aef_gate": 1.0,
                    "highres_stem": 1.0,
                    "highres_adapter": 1.0,
                    "highres_gate": 1.0,
                    "output_projection": 1.0,
                },
                "checkpoint_reload_max_abs_error": 0.0,
                "accuracy_conclusion_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    (sandbox / "run_manifest.json").write_text(
        json.dumps(_valid_run_manifest(sandbox)), encoding="utf-8"
    )
    (sandbox / "reproducibility.json").write_text(
        json.dumps(
            {
                "fixed_seed": 20260815,
                "max_abs_error": 0.0,
                "matches": True,
                "checkpoint_sha256": checkpoint_sha,
                "synthetic": True,
            }
        ),
        encoding="utf-8",
    )
    return [
        sandbox / "run_manifest.json",
        sandbox / "metrics.json",
        audit,
        sandbox / "reproducibility.json",
        checkpoint,
        launcher_log,
        ready,
        tee_complete,
    ]


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


def _set_group_patch_axis(groups: list[Path], patch_ids: list[str]) -> None:
    for path in groups:
        exported = zarr.open_group(str(path), mode="a")
        exported["patch_id"][:] = np.asarray(patch_ids, dtype="<U9")
        exported.attrs["patch_ids"] = patch_ids


def _add_prepared_evidence_declared_by_audit(sandbox: Path) -> dict[str, Path]:
    """补齐真实 smoke 中必须由 SUCCESS 覆盖的 prepare/CPU/synthetic 证据。"""
    files: dict[str, Path] = {}
    selection = sandbox / "manifests/patch_selection.json"
    selection_sha = sha256(selection.read_bytes()).hexdigest()
    aef_metadata = {
        "synthetic": True,
        "synthetic_kind": "annual_s2_fixed_projection",
        "allowed_use": "smoke_test_only",
        "formal_training_allowed": False,
        "formal_evaluation_allowed": False,
    }
    highres_metadata = {
        "synthetic": True,
        "synthetic_kind": "annual_s2_rgb_5x_deterministic_texture",
        "allowed_use": "smoke_test_only",
        "formal_training_allowed": False,
        "formal_evaluation_allowed": False,
        "claimed_native_gsd_m": None,
        "model_input_gsd_m": 2,
        "contains_real_2m_information": False,
    }
    entries: dict[str, list[dict[str, object]]] = {"aef": [], "highres_2m": []}
    for index, (patch_id, year) in enumerate(
        patch_year for patch_id in PATCH_IDS for patch_year in ((patch_id, 2020), (patch_id, 2021))
    ):
        for kind, metadata, tensor_payload in (
            (
                "aef",
                aef_metadata,
                {
                    "aef": torch.ones((64, 1, 1), dtype=torch.float32),
                    "aef_valid": torch.ones((1, 1, 1), dtype=torch.bool),
                },
            ),
            (
                "highres_2m",
                highres_metadata,
                {
                    "highres": torch.ones((3, 5, 5), dtype=torch.float32),
                    "highres_valid": torch.ones((1, 5, 5), dtype=torch.bool),
                },
            ),
        ):
            relative = f"synthetic/{kind}/patch_{index:02d}_{year}.pt"
            path = sandbox / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                torch.save(
                    {
                        **tensor_payload,
                        "metadata": metadata,
                        "patch_id": patch_id,
                        "year": year,
                    },
                    path,
                )
            files[relative] = path
            entries[kind].append(
                {
                    "patch_id": patch_id,
                    "year": year,
                    "path": relative,
                    "sha256": sha256(path.read_bytes()).hexdigest(),
                }
            )
    projection_relative = "synthetic/aef/fixed_projection_10x64.pt"
    projection_path = sandbox / projection_relative
    if not projection_path.exists():
        torch.save(
            {
                "projection": torch.ones((64, 10), dtype=torch.float32),
                "seed": 20260815,
                "shape": [64, 10],
                "dtype": "float32",
                "synthetic": True,
                "formal_training_allowed": False,
                "formal_evaluation_allowed": False,
            },
            projection_path,
        )
    files[projection_relative] = projection_path
    source = {
        "archives": {f"archive-{index:02d}.zip": {"type": "file"} for index in range(48)},
        "directories": {},
        "hashing_performed": False,
    }
    cpu_groups = {
        group: {
            "shape": [2, 4, 64, 16, 16],
            "dtype": "torch.float32",
            "finite": True,
            "max_vmf_norm_error": 0.0,
        }
        for group in GROUPS
    }
    payloads = {
        "manifests/aef_registry.json": {
            **aef_metadata,
            "entries": entries["aef"],
            "projection": {
                "path": projection_relative,
                "sha256": sha256(projection_path.read_bytes()).hexdigest(),
                "seed": 20260815,
                "shape": [64, 10],
                "dtype": "float32",
            },
        },
        "manifests/highres_2m_registry.json": {
            **highres_metadata,
            "entries": entries["highres_2m"],
        },
        "manifests/prepare_manifest.json": {
            "patch_ids": list(PATCH_IDS),
            "patch_years": 8,
            "years": [2020, 2021],
            "source_archive_count": 48,
            "source_unchanged": True,
            "selection_manifest_sha256": selection_sha,
            "sandbox_bytes_before_path_audit": 1,
            "synthetic": True,
            "allowed_use": "smoke_test_only",
            "formal_training_allowed": False,
            "formal_evaluation_allowed": False,
            "accuracy_conclusion_allowed": False,
        },
        "manifests/cpu_contract.json": {
            "fixture": "small",
            "groups": cpu_groups,
            "full_zero_gate_matches_base": True,
            "synthetic": True,
            "allowed_use": "smoke_test_only",
            "formal_training_allowed": False,
            "formal_evaluation_allowed": False,
            "accuracy_conclusion_allowed": False,
        },
        "manifests/source_snapshot_before.json": source,
        "manifests/source_snapshot_after.json": source,
    }
    for relative, payload in payloads.items():
        path = sandbox / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        files[relative] = path
    audit_path = sandbox / "path_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    declared = audit["created_or_modified"]
    assert isinstance(declared, list)
    additions = [
        "manifests",
        *payloads,
        "synthetic",
        "synthetic/aef",
        projection_relative,
        *(str(entry["path"]) for entry in entries["aef"]),
        "synthetic/highres_2m",
        *(str(entry["path"]) for entry in entries["highres_2m"]),
    ]
    declared.extend(relative for relative in additions if relative not in declared)
    audit_path.write_text(json.dumps(audit, sort_keys=True) + "\n", encoding="utf-8")
    return files


def _valid_group_metrics() -> dict[str, object]:
    return {
        group: {
            "shape": [4, 8, 64, 128, 128],
            "latency_seconds": 0.25,
            "finite": True,
            "pre_export_fp32_vmf_norm": {
                "source_dtype": "float32",
                "computation_dtype": "float32",
                "min": 1.0,
                "median": 1.0,
                "max": 1.0,
            },
            "reopened_fp16_zarr_vmf_norm": {
                "source_dtype": "float16",
                "computation_dtype": "float32",
                "min": 1.0,
                "median": 1.0,
                "max": 1.0,
            },
            "npu_peak_memory": {
                "unit": "bytes",
                "baseline_allocated": 100,
                "baseline_reserved": 200,
                "peak_allocated": 300,
                "peak_reserved": 400,
                "peak_allocated_delta": 200,
                "peak_reserved_delta": 200,
            },
        }
        for group in GROUPS
    }


def _valid_run_manifest(sandbox: Path) -> dict[str, object]:
    cpu_contract = sandbox / "manifests/cpu_contract.json"
    return {
        "git_commit": "a" * 40,
        "config_sha256": "b" * 64,
        "patch_ids": list(PATCH_IDS),
        "periods": list(PERIODS),
        "physical_npu": 2,
        "logical_device": "npu:0",
        "source_unchanged": True,
        "synthetic": True,
        "allowed_use": "smoke_test_only",
        "formal_training_allowed": False,
        "formal_evaluation_allowed": False,
        "accuracy_conclusion_allowed": False,
        "runtime_provenance": {
            "sys_executable": "/sandbox/env/bin/python",
            "python_version": "3.11.15",
            "torch_version": "2.6.0+cpu",
            "torch_npu_version": "2.6.0.post5",
            "cann": {
                "root": "/usr/local/Ascend/cann-9.0.0",
                "version": "9.0.0",
                "install_info": (
                    "/usr/local/Ascend/cann-9.0.0/aarch64-linux/ascend_toolkit_install.info"
                ),
            },
            "driver": {
                "version": "26.0.rc1",
                "version_info": "/usr/local/Ascend/driver/version.info",
            },
            "device_mapping": {
                "physical_device": "/dev/davinci2",
                "physical_npu": 2,
                "visible_devices": "2",
                "logical_device": "npu:0",
                "logical_device_count": 1,
                "device_name": "Ascend 910B4-1",
            },
            "module_paths": {
                "torch": "/site-packages/torch/__init__.py",
                "torch_npu": "/site-packages/torch_npu/__init__.py",
                "runner": "/worktree/experiments/china_v1_fusion_smoke/runner.py",
                "model": "/worktree/experiments/china_v1_fusion_smoke/model.py",
            },
            "occupancy_checks": [
                {"stage": "launcher_preload", "idle": True},
                {"stage": "post_load_pre_set_device", "idle": True},
            ],
        },
        "cpu_contract_fallback": {
            "path": "manifests/cpu_contract.json",
            "sha256": sha256(cpu_contract.read_bytes()).hexdigest(),
        },
    }


def _unsafe_marker(marker: str) -> None:
    Path(marker).write_text("unsafe pickle executed", encoding="utf-8")


class _UnsafeLegacyPayload:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return (_unsafe_marker, (str(self.marker),))


class _TwoParameterModel(torch.nn.Module):
    """让第一参数先被加载，第二参数可触发后续 application 失败。"""

    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Parameter(torch.tensor([1.0]))
        self.second = torch.nn.Parameter(torch.tensor([2.0, 3.0]))


class _MutatesThenFailsModel(torch.nn.Module):
    """模拟同 shape/dtype payload 在应用阶段失败的非标准模块。"""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        del state_dict, strict
        with torch.no_grad():
            self.weight.fill_(99.0)
        raise RuntimeError("simulated application failure")


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


@pytest.mark.parametrize("mutation", ["sparse", "application"], ids=str)
def test_checkpoint_application_failures_restore_model_state(
    tmp_sandbox: Path,
    mutation: str,
) -> None:
    """同形稀疏或应用阶段失败时，任何已写参数都必须被事务性恢复。"""
    model: torch.nn.Module
    if mutation == "sparse":
        model = _TwoParameterModel()
        state_dict = dict(model.state_dict())
        state_dict["first"] = torch.tensor([7.0])
        state_dict["second"] = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 1]]),
            values=torch.tensor([8.0, 9.0]),
            size=(2,),
        )
    else:
        model = _MutatesThenFailsModel()
        state_dict = dict(model.state_dict())
    metadata = _metadata_for(model)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    checkpoint = tmp_sandbox / "smoke_checkpoint.pt"
    torch.save({"state_dict": state_dict, "metadata": metadata}, checkpoint)

    with pytest.raises(ExportError, match="state_dict"):
        load_smoke_checkpoint(model, checkpoint, metadata)

    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_seal_success_requires_complete_sealed_false_formal_evidence(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """SUCCESS 只能在四组已封存且全部审计证据存在时写入。"""
    groups, evidence = _complete_seal_inputs(tmp_sandbox, full_contract_tensors)
    launcher_log = tmp_sandbox / "logs" / "npu_smoke.log"
    log_state_before = launcher_log.stat()
    log_sha_before = sha256(launcher_log.read_bytes()).hexdigest()

    success = seal_success(tmp_sandbox, [*groups, *evidence])

    assert success == tmp_sandbox / "SUCCESS"
    assert launcher_log.stat().st_size == log_state_before.st_size
    assert launcher_log.stat().st_mtime_ns == log_state_before.st_mtime_ns
    assert sha256(launcher_log.read_bytes()).hexdigest() == log_sha_before
    assert success.stat().st_mtime_ns >= launcher_log.stat().st_mtime_ns
    payload = success.read_text(encoding="utf-8")
    assert "combined_sha256" in payload
    assert "utc" in payload
    verify_success(tmp_sandbox)


def test_seal_success_retries_an_authenticated_temporary_after_publish_failure(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完整 SUCCESS.tmp 只因原子发布失败时，下一次调用应安全完成同一 seal。"""
    from experiments.china_v1_fusion_smoke import export as export_module

    groups, evidence = _complete_seal_inputs(tmp_sandbox, full_contract_tensors)
    real_link = export_module.os.link

    def fail_success_publish(source: str, target: str, **kwargs: object) -> None:
        if target == "SUCCESS":
            raise OSError("injected atomic publish failure")
        real_link(source, target, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(export_module.os, "link", fail_success_publish)
        with pytest.raises(ExportError, match="atomically publish"):
            seal_success(tmp_sandbox, [*groups, *evidence])

    temporary = tmp_sandbox / "SUCCESS.tmp"
    payload_before_retry = temporary.read_bytes()
    assert not (tmp_sandbox / "SUCCESS").exists()

    success = seal_success(tmp_sandbox, [*groups, *evidence])

    assert success.read_bytes() == payload_before_retry
    assert not temporary.exists()
    verify_success(tmp_sandbox)


def test_seal_success_does_not_follow_a_symlink_inserted_before_temporary_open(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """temp 创建竞态中插入外部 symlink 时，外部文件一字节也不能被写。"""
    from experiments.china_v1_fusion_smoke import export as export_module

    groups, evidence = _complete_seal_inputs(tmp_sandbox, full_contract_tensors)
    outside = tmp_path / "outside-success.txt"
    outside.write_bytes(b"preserve outside bytes")
    real_open = export_module.os.open
    injected = False

    def racing_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal injected
        if path == "SUCCESS.tmp" and flags & os.O_CREAT and not injected:
            (tmp_sandbox / "SUCCESS.tmp").symlink_to(outside)
            injected = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(export_module.os, "open", racing_open)
    with pytest.raises(ExportError, match="SUCCESS.tmp"):
        seal_success(tmp_sandbox, [*groups, *evidence])

    assert outside.read_bytes() == b"preserve outside bytes"
    assert not (tmp_sandbox / "SUCCESS").exists()


def test_seal_success_rejects_if_temporary_path_is_swapped_before_publish(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """发布前 temp 路径被替换时必须拒绝，且不得留下错误 SUCCESS。"""
    from experiments.china_v1_fusion_smoke import export as export_module

    groups, evidence = _complete_seal_inputs(tmp_sandbox, full_contract_tensors)
    outside = tmp_path / "outside-success.txt"
    outside.write_bytes(b"outside must stay unchanged")
    real_link = export_module.os.link
    swapped = False

    def swap_then_link(source: str, target: str, **kwargs: object) -> None:
        nonlocal swapped
        if target == "SUCCESS" and not swapped:
            temporary = tmp_sandbox / "SUCCESS.tmp"
            temporary.unlink()
            temporary.symlink_to(outside)
            swapped = True
        real_link(source, target, **kwargs)

    monkeypatch.setattr(export_module.os, "link", swap_then_link)
    with pytest.raises(ExportError, match="verified temporary inode"):
        seal_success(tmp_sandbox, [*groups, *evidence])

    assert outside.read_bytes() == b"outside must stay unchanged"
    assert (tmp_sandbox / "SUCCESS.tmp").is_symlink()
    assert (tmp_sandbox / "SUCCESS").is_symlink()


def test_seal_success_never_overwrites_a_concurrent_success_winner(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """另一 sealer 抢先发布 SUCCESS 时，本进程必须保留 winner 并 fail closed。"""
    from experiments.china_v1_fusion_smoke import export as export_module

    groups, evidence = _complete_seal_inputs(tmp_sandbox, full_contract_tensors)
    real_link = export_module.os.link
    winner = b"concurrent winner must remain immutable\n"

    def publish_winner_first(source: str, target: str, **kwargs: object) -> None:
        if target == "SUCCESS":
            (tmp_sandbox / "SUCCESS").write_bytes(winner)
        real_link(source, target, **kwargs)

    monkeypatch.setattr(export_module.os, "link", publish_winner_first)
    with pytest.raises(ExportError, match="concurrently|overwrite"):
        seal_success(tmp_sandbox, [*groups, *evidence])

    assert (tmp_sandbox / "SUCCESS").read_bytes() == winner


@pytest.mark.parametrize(
    "payload",
    [
        b"{incomplete",
        json.dumps({"combined_sha256": "0" * 64}).encode("utf-8"),
        json.dumps(
            {
                "combined_sha256": "0" * 64,
                "sealed_at_utc": "2026-08-15T00:00:00Z",
                "synthetic": True,
                "formal_training_allowed": False,
                "formal_evaluation_allowed": False,
            }
        ).encode("utf-8"),
        json.dumps(
            {
                "combined_sha256": "f" * 64,
                "sealed_at_utc": "2026-08-15T00:00:00Z",
                "synthetic": True,
                "formal_training_allowed": True,
                "formal_evaluation_allowed": False,
            }
        ).encode("utf-8"),
    ],
    ids=("malformed", "incomplete-schema", "forged-digest", "forged-policy"),
)
def test_seal_success_rejects_untrusted_existing_temporary_without_mutating_it(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
    payload: bytes,
) -> None:
    """未知、残缺或语义不符的 SUCCESS.tmp 必须保留原样并 fail closed。"""
    groups, evidence = _complete_seal_inputs(tmp_sandbox, full_contract_tensors)
    temporary = tmp_sandbox / "SUCCESS.tmp"
    temporary.write_bytes(payload)

    with pytest.raises(ExportError, match="SUCCESS.tmp"):
        seal_success(tmp_sandbox, [*groups, *evidence])

    assert temporary.read_bytes() == payload
    assert not (tmp_sandbox / "SUCCESS").exists()


def test_seal_success_rejects_final_audit_with_only_two_stages(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """仅保留最终两阶段不能伪装成完整 prepare→CPU→NPU→finalize 历史。"""
    groups, evidence = _complete_seal_inputs(tmp_sandbox, full_contract_tensors)
    audit_path = tmp_sandbox / "path_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["stages"] = ["npu-smoke", "finalize-seal"]
    audit_path.write_text(json.dumps(audit, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ExportError, match="complete.*stage|stages"):
        seal_success(tmp_sandbox, [*groups, *evidence])
    assert not (tmp_sandbox / "SUCCESS").exists()


def test_seal_success_rejects_preliminary_audit_without_prepare_and_cpu_history(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """READY 绑定的 preliminary audit 也必须包含 prepare→CPU→NPU 三阶段。"""
    groups, evidence = _complete_seal_inputs(tmp_sandbox, full_contract_tensors)
    preliminary = tmp_sandbox / "manifests/preliminary_path_audit.json"
    raw = json.loads(preliminary.read_text(encoding="utf-8"))
    raw["stages"] = ["npu-smoke"]
    preliminary.write_text(json.dumps(raw, sort_keys=True) + "\n", encoding="utf-8")
    ready = tmp_sandbox / "READY_TO_SEAL"
    ready_raw = json.loads(ready.read_text(encoding="utf-8"))
    ready_raw["preliminary_path_audit"]["sha256"] = sha256(preliminary.read_bytes()).hexdigest()
    ready.write_text(json.dumps(ready_raw, sort_keys=True) + "\n", encoding="utf-8")
    tee = tmp_sandbox / "TEE_COMPLETE"
    tee_raw = json.loads(tee.read_text(encoding="utf-8"))
    tee_raw["ready_to_seal_sha256"] = sha256(ready.read_bytes()).hexdigest()
    tee.write_text(json.dumps(tee_raw, sort_keys=True) + "\n", encoding="utf-8")
    audit = tmp_sandbox / "path_audit.json"
    audit_raw = json.loads(audit.read_text(encoding="utf-8"))
    for relative, path in {
        "READY_TO_SEAL": ready,
        "TEE_COMPLETE": tee,
    }.items():
        audit_raw["declared_files"][relative] = {
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": sha256(path.read_bytes()).hexdigest(),
        }
    audit.write_text(json.dumps(audit_raw, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ExportError, match="preliminary audit stage"):
        seal_success(tmp_sandbox, [*groups, *evidence])
    assert not (tmp_sandbox / "SUCCESS").exists()


def test_success_verification_detects_launcher_log_tampering_after_seal(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """最终日志进入 SUCCESS 摘要后，任何 seal 后追加都必须使正式验证失败。"""
    groups, evidence = _complete_seal_inputs(tmp_sandbox, full_contract_tensors)
    launcher_log = tmp_sandbox / "logs" / "npu_smoke.log"
    seal_success(tmp_sandbox, [*groups, *evidence])
    verify_success(tmp_sandbox)

    with launcher_log.open("a", encoding="utf-8") as handle:
        handle.write("tampered after SUCCESS\n")

    with pytest.raises(ExportError, match="launcher log|combined SHA-256|declared"):
        verify_success(tmp_sandbox)


def test_seal_success_rejects_path_audit_without_expected_final_success(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """seal 必须亲自验证 audit 已在 SUCCESS 创建前声明它是最后写入。"""
    groups, evidence = _complete_seal_inputs(tmp_sandbox, full_contract_tensors)
    (tmp_sandbox / "path_audit.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ExportError, match="path audit.*SUCCESS"):
        seal_success(tmp_sandbox, [*groups, *evidence])
    assert not (tmp_sandbox / "SUCCESS").exists()


@pytest.mark.parametrize(
    "tampered_final_seal",
    [
        {
            "path": "SUCCESS",
            "status": "already_written",
            "exists_when_audit_written": False,
        },
        {
            "path": "outputs/SUCCESS",
            "status": "expected_last_write",
            "exists_when_audit_written": False,
        },
    ],
    ids=("wrong-status", "wrong-path"),
)
def test_seal_success_rejects_tampered_final_success_declaration(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
    tampered_final_seal: dict[str, object],
) -> None:
    """planned SUCCESS 的状态或路径被篡改时，seal 不得只相信 created list。"""
    groups, evidence = _complete_seal_inputs(tmp_sandbox, full_contract_tensors)
    audit = tmp_sandbox / "path_audit.json"
    payload = json.loads(audit.read_text(encoding="utf-8"))
    payload["final_seal"] = tampered_final_seal
    audit.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExportError, match="path audit.*SUCCESS"):
        seal_success(tmp_sandbox, [*groups, *evidence])
    assert not (tmp_sandbox / "SUCCESS").exists()


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


def test_seal_success_rejects_coordinated_cross_group_patch_axis_mutation(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """四组共同伪造自洽 patch_id/attrs 时仍须服从可信 patch selection。"""
    groups, evidence = _complete_seal_inputs(tmp_sandbox, full_contract_tensors)
    changed_ids = ["patch-100", "patch-101", "patch-102", "patch-103"]
    _set_group_patch_axis(groups, changed_ids)

    with pytest.raises(ExportError, match="patch"):
        seal_success(tmp_sandbox, [*groups, *evidence])
    assert not (tmp_sandbox / "SUCCESS").exists()


def test_seal_success_rejects_supplied_directory_with_external_symlink_child(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
    tmp_path: Path,
) -> None:
    """required 目录的子 symlink 不得在摘要阶段把沙箱外内容引入 SUCCESS。"""
    groups, evidence = _complete_seal_inputs(tmp_sandbox, full_contract_tensors)
    supplied = tmp_sandbox / "extra-evidence"
    supplied.mkdir()
    outside = tmp_path / "outside-child.txt"
    outside.write_text("must not be hashed", encoding="utf-8")
    os.symlink(outside, supplied / "external-child.txt")

    with pytest.raises(ExportError, match="symlink"):
        seal_success(tmp_sandbox, [*groups, *evidence, supplied])
    assert not (tmp_sandbox / "SUCCESS").exists()


def test_seal_success_requires_patch_selection_even_for_self_consistent_fake_axes(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """四组共同自报 fake IDs 时，缺失可信 selection 也必须 fail closed。"""
    groups, evidence = _complete_seal_inputs(tmp_sandbox, full_contract_tensors)
    _set_group_patch_axis(groups, ["patch-100", "patch-101", "patch-102", "patch-103"])
    (tmp_sandbox / "manifests" / "patch_selection.json").unlink()

    with pytest.raises(ExportError, match="patch selection"):
        seal_success(tmp_sandbox, [*groups, *evidence])
    assert not (tmp_sandbox / "SUCCESS").exists()


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        ({"unknown": PATCH_IDS}, "schema"),
        ({"patch_ids": ["patch-000", "patch-000", "patch-002", "patch-003"]}, "duplicate"),
        ({"patch_ids": list(PATCH_IDS[:3])}, "exactly four"),
        ({"patches": [{"id": patch_id} for patch_id in PATCH_IDS]}, "schema"),
    ],
    ids=["unknown_schema", "duplicate", "wrong_count", "missing_patch_id_field"],
)
def test_seal_success_rejects_invalid_patch_selection_manifest(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
    manifest: dict[str, object],
    message: str,
) -> None:
    """selection manifest 的 schema、唯一性和四 patch 基数均属于封存硬合同。"""
    groups, evidence = _complete_seal_inputs(tmp_sandbox, full_contract_tensors)
    selection = tmp_sandbox / "manifests" / "patch_selection.json"
    selection.write_text(json.dumps(manifest), encoding="utf-8")
    if manifest.get("patch_ids") == ["patch-000", "patch-000", "patch-002", "patch-003"]:
        _set_group_patch_axis(groups, list(manifest["patch_ids"]))

    with pytest.raises(ExportError, match=message):
        seal_success(tmp_sandbox, [*groups, *evidence])
    assert not (tmp_sandbox / "SUCCESS").exists()


def test_success_combined_sha256_covers_mandatory_patch_selection(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """SUCCESS 摘要必须包含 manifests/patch_selection.json，不能只作为读取旁证。"""
    groups, evidence = _complete_seal_inputs(tmp_sandbox, full_contract_tensors)
    selection = tmp_sandbox / "manifests" / "patch_selection.json"

    seal_success(tmp_sandbox, [*groups, *evidence])
    with selection.open("a", encoding="utf-8") as handle:
        handle.write("tampered")

    with pytest.raises(ExportError, match="combined SHA-256|patch selection"):
        verify_success(tmp_sandbox)


@pytest.mark.parametrize(
    "relative",
    [
        "manifests/source_snapshot_before.json",
        "manifests/aef_registry.json",
        "synthetic/aef/patch_00_2020.pt",
        "manifests/prepare_manifest.json",
        "manifests/cpu_contract.json",
    ],
    ids=("source-snapshot", "registry", "synthetic-cache", "prepare", "cpu-contract"),
)
def test_success_digest_covers_each_declared_prepared_evidence_file(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
    relative: str,
) -> None:
    """audit 声明的每份前置证据在 seal 后被改一字节都必须使验证失败。"""
    groups, evidence = _complete_seal_inputs(tmp_sandbox, full_contract_tensors)
    prepared = _add_prepared_evidence_declared_by_audit(tmp_sandbox)
    seal_success(tmp_sandbox, [*groups, *evidence])

    with prepared[relative].open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(
        ExportError,
        match="combined SHA-256|fallback hash|semantic evidence|semantic .* checksum",
    ):
        verify_success(tmp_sandbox)


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "missing", "outside", "unknown", "symlink"],
    ids=str,
)
def test_seal_rejects_invalid_or_unknown_created_evidence_paths(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
    tmp_path: Path,
    mutation: str,
) -> None:
    """审计路径不能重复、缺失、逃逸、未知或通过 symlink 改写证据边界。"""
    groups, evidence = _complete_seal_inputs(tmp_sandbox, full_contract_tensors)
    _add_prepared_evidence_declared_by_audit(tmp_sandbox)
    audit_path = tmp_sandbox / "path_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    created = audit["created_or_modified"]
    assert isinstance(created, list)
    if mutation == "duplicate":
        created.append("manifests/cpu_contract.json")
    elif mutation == "missing":
        created.append("manifests/missing.json")
    elif mutation == "outside":
        created.append("../outside.json")
    elif mutation == "unknown":
        (tmp_sandbox / "unknown.txt").write_text("unknown", encoding="utf-8")
        created.append("unknown.txt")
    else:
        outside = tmp_path / "outside.json"
        outside.write_text("outside", encoding="utf-8")
        (tmp_sandbox / "manifests/external.json").symlink_to(outside)
        created.append("manifests/external.json")
    audit_path.write_text(json.dumps(audit, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ExportError, match="audit|evidence|path|symlink|duplicate|unknown"):
        seal_success(tmp_sandbox, [*groups, *evidence])
    assert not (tmp_sandbox / "SUCCESS").exists()


@pytest.mark.parametrize(
    ("target", "field"),
    [("metrics", "npu_peak_memory"), ("manifest", "runtime_provenance")],
    ids=("missing-peak-hbm", "missing-runtime-provenance"),
)
def test_seal_requires_peak_npu_memory_and_runtime_provenance(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
    target: str,
    field: str,
) -> None:
    """缺 peak HBM 或运行环境 provenance 的 smoke 不能获得 SUCCESS。"""
    groups, evidence = _complete_seal_inputs(tmp_sandbox, full_contract_tensors)
    _add_prepared_evidence_declared_by_audit(tmp_sandbox)
    metrics = {
        "groups": _valid_group_metrics(),
        "zero_gate_max_abs_error": 0.0,
        "gradient_l1": {
            "aef_adapter": 1.0,
            "aef_gate": 1.0,
            "highres_stem": 1.0,
            "highres_adapter": 1.0,
            "highres_gate": 1.0,
            "output_projection": 1.0,
        },
        "checkpoint_reload_max_abs_error": 0.0,
        "accuracy_conclusion_allowed": False,
    }
    manifest = _valid_run_manifest(tmp_sandbox)
    if target == "metrics":
        del metrics["groups"]["full"][field]
    else:
        del manifest[field]
    (tmp_sandbox / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (tmp_sandbox / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ExportError, match="memory|provenance"):
        seal_success(tmp_sandbox, [*groups, *evidence])
    assert not (tmp_sandbox / "SUCCESS").exists()


def test_reopened_fp16_norm_summary_reads_the_exported_zarr(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """FP16 norm 证据必须来自重开后的 Zarr，而不是复用 export 前 FP32 tensor。"""
    import experiments.china_v1_fusion_smoke.export as export_module

    embedding, valid = full_contract_tensors
    embedding[:, :, 0] = 1.0003
    output = export_group_zarr(
        "full",
        embedding,
        valid,
        PATCH_IDS,
        PERIODS,
        tmp_sandbox / "outputs/full/embedding.zarr",
        tmp_sandbox,
    )
    stored = zarr.open_group(str(output), mode="r")["embedding"][:]
    expected = np.sqrt(np.square(stored.astype(np.float32)).sum(axis=2))

    summary = export_module.reopened_fp16_vmf_norm_summary(output)

    assert summary == {
        "source_dtype": "float16",
        "computation_dtype": "float32",
        "min": float(expected.min()),
        "median": float(np.median(expected)),
        "max": float(expected.max()),
    }


@pytest.mark.parametrize("scale", [0.0, 2.0], ids=("all-zero", "norm-two"))
def test_reopened_fp16_norm_summary_rejects_nonunit_persisted_vectors(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
    scale: float,
) -> None:
    """重开实际 FP16 Zarr 后，全零和 norm=2 都不能只因 finite 而通过。"""
    import experiments.china_v1_fusion_smoke.export as export_module

    embedding, valid = full_contract_tensors
    embedding[:, :, 0] = scale
    output = export_group_zarr(
        "full",
        embedding,
        valid,
        PATCH_IDS,
        PERIODS,
        tmp_sandbox / "outputs/full/embedding.zarr",
        tmp_sandbox,
    )

    with pytest.raises(ExportError, match="vMF.*tolerance"):
        export_module.reopened_fp16_vmf_norm_summary(output)


@pytest.mark.parametrize(
    ("source_dtype", "minimum", "maximum"),
    [("float32", 0.99998, 1.0), ("float16", 0.9994, 1.0)],
)
def test_norm_evidence_enforces_dtype_specific_unit_tolerances(
    source_dtype: str,
    minimum: float,
    maximum: float,
) -> None:
    """FP32 使用 1e-5、持久化 FP16 使用 5e-4，越界摘要必须拒绝。"""
    import experiments.china_v1_fusion_smoke.export as export_module

    summary = {
        "source_dtype": source_dtype,
        "computation_dtype": "float32",
        "min": minimum,
        "median": 1.0,
        "max": maximum,
    }

    with pytest.raises(ExportError, match="vMF.*tolerance"):
        export_module._validate_norm_summary(summary, source_dtype=source_dtype)


@pytest.mark.parametrize(
    "mutation",
    [
        "repro-matches",
        "repro-max-error",
        "repro-checkpoint-sha",
        "run-git",
        "run-config",
        "run-patch-axis",
        "empty-aef-registry",
        "prepare-patch-years",
        "source-snapshot-mismatch",
        "cpu-identity",
        "checkpoint-state",
        "persisted-norm-mismatch",
        "missing-gate-gradients",
    ],
)
def test_seal_rejects_semantically_inconsistent_evidence_graph(
    tmp_sandbox: Path,
    full_contract_tensors: tuple[torch.Tensor, torch.Tensor],
    mutation: str,
) -> None:
    """各 JSON 自洽但跨 artifact 矛盾时，最终 seal 必须逐项 fail closed。"""
    groups, evidence = _complete_seal_inputs(tmp_sandbox, full_contract_tensors)
    checkpoint = tmp_sandbox / "smoke_checkpoint.pt"
    checkpoint_sha = sha256(checkpoint.read_bytes()).hexdigest()
    reproducibility = {
        "fixed_seed": 20260815,
        "max_abs_error": 0.0,
        "matches": True,
        "checkpoint_sha256": checkpoint_sha,
        "synthetic": True,
    }
    if mutation == "repro-matches":
        reproducibility["matches"] = False
    elif mutation == "repro-max-error":
        reproducibility["max_abs_error"] = 1.0
    elif mutation == "repro-checkpoint-sha":
        reproducibility["checkpoint_sha256"] = "0" * 64
    (tmp_sandbox / "reproducibility.json").write_text(json.dumps(reproducibility), encoding="utf-8")

    manifest_path = tmp_sandbox / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "run-git":
        manifest["git_commit"] = "d" * 40
    elif mutation == "run-config":
        manifest["config_sha256"] = "e" * 64
    elif mutation == "run-patch-axis":
        manifest["patch_ids"] = list(reversed(PATCH_IDS))
    elif mutation == "cpu-identity":
        cpu_path = tmp_sandbox / "manifests/cpu_contract.json"
        cpu = json.loads(cpu_path.read_text(encoding="utf-8"))
        cpu["full_zero_gate_matches_base"] = False
        cpu_path.write_text(json.dumps(cpu), encoding="utf-8")
        manifest["cpu_contract_fallback"]["sha256"] = sha256(cpu_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    if mutation == "empty-aef-registry":
        registry = tmp_sandbox / "manifests/aef_registry.json"
        registry.write_text(
            json.dumps(
                {
                    "synthetic": True,
                    "synthetic_kind": "annual_s2_fixed_projection",
                    "allowed_use": "smoke_test_only",
                    "formal_training_allowed": False,
                    "formal_evaluation_allowed": False,
                    "entries": [],
                }
            ),
            encoding="utf-8",
        )
    elif mutation == "prepare-patch-years":
        prepare = tmp_sandbox / "manifests/prepare_manifest.json"
        raw = json.loads(prepare.read_text(encoding="utf-8"))
        raw["patch_years"] = 7
        prepare.write_text(json.dumps(raw), encoding="utf-8")
    elif mutation == "source-snapshot-mismatch":
        after = tmp_sandbox / "manifests/source_snapshot_after.json"
        raw = json.loads(after.read_text(encoding="utf-8"))
        raw["unexpected_change"] = True
        after.write_text(json.dumps(raw), encoding="utf-8")
    elif mutation == "checkpoint-state":
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        payload["state_dict"] = {"wrong.weight": torch.ones((1,), dtype=torch.float32)}
        torch.save(payload, checkpoint)
        reproducibility["checkpoint_sha256"] = sha256(checkpoint.read_bytes()).hexdigest()
        (tmp_sandbox / "reproducibility.json").write_text(
            json.dumps(reproducibility), encoding="utf-8"
        )
    elif mutation == "persisted-norm-mismatch":
        metrics = tmp_sandbox / "metrics.json"
        raw = json.loads(metrics.read_text(encoding="utf-8"))
        raw["groups"]["full"]["reopened_fp16_zarr_vmf_norm"]["median"] = 1.0002
        metrics.write_text(json.dumps(raw), encoding="utf-8")
    elif mutation == "missing-gate-gradients":
        metrics = tmp_sandbox / "metrics.json"
        raw = json.loads(metrics.read_text(encoding="utf-8"))
        raw["gradient_l1"].pop("aef_gate")
        metrics.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(
        ExportError,
        match="semantic|reproducibility|registry|source|CPU|checkpoint|gradient|norm|axis",
    ):
        seal_success(tmp_sandbox, [*groups, *evidence])
    assert not (tmp_sandbox / "SUCCESS").exists()
