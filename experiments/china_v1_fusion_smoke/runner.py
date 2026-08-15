"""China V1 隔离三路融合 smoke 的分阶段编排入口。"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
import yaml

from experiments.china_v1_fusion_smoke.archive import (
    MonthRef,
    PatchSelection,
    RasterHeader,
    inspect_selection_headers,
    select_complete_patches,
)
from experiments.china_v1_fusion_smoke.config import SmokeConfig, load_smoke_config
from experiments.china_v1_fusion_smoke.data import PatchYearBatch, load_patch_year_batch
from experiments.china_v1_fusion_smoke.export import (
    export_group_zarr,
    load_smoke_checkpoint,
    reopened_fp16_vmf_norm_summary,
    save_smoke_checkpoint,
    seal_success,
    verify_success,
)
from experiments.china_v1_fusion_smoke.model import IsolatedFusionSmokeModel
from experiments.china_v1_fusion_smoke.registry import (
    RegistryError,
    validate_formal_registry,
    validate_smoke_registry,
)
from experiments.china_v1_fusion_smoke.safety import ensure_sandbox, validate_write_path
from experiments.china_v1_fusion_smoke.synthetic import (
    SyntheticAnnualContext,
    generate_synthetic_context,
)

STAGES = ("inspect", "prepare", "cpu-contract", "npu-smoke")
FINALIZE_STAGE = "finalize-seal"
AUDIT_STAGES = (*STAGES, FINALIZE_STAGE)
GROUPS: dict[str, tuple[bool, bool]] = {
    "base": (False, False),
    "base_aef": (True, False),
    "base_highres": (False, True),
    "full": (True, True),
}
MAX_SANDBOX_BYTES = 5 * 1024**3
EXPECTED_SEED = 20260815
EXPECTED_WORKTREE = Path("/root/workspace/xuannv/.worktrees/codex-china-v1-fusion-smoke")
TASK7_LAUNCHER = EXPECTED_WORKTREE / "scripts/smoke/run_china_v1_isolated_fusion_smoke.sh"
PHYSICAL_NPU2 = Path("/dev/davinci2")
READY_TO_SEAL = "READY_TO_SEAL"
TEE_COMPLETE = "TEE_COMPLETE"


class RunnerError(RuntimeError):
    """阶段编排、provenance 或资源边界不满足时抛出。"""


@dataclass(frozen=True)
class StageResult:
    """供 CLI 与测试共同消费的精简阶段结果。"""

    stage: str
    sandbox_root: Path
    groups: tuple[str, ...] = ()
    full_zero_gate_matches_base: bool = False
    created_paths: tuple[Path, ...] = ()
    patch_ids: tuple[str, ...] = ()
    patch_years: int = 0
    headers: tuple[Mapping[str, object], ...] = ()
    source_unchanged: bool | None = None
    sandbox_bytes: int = 0


def _load_seed(config_path: Path) -> int:
    """Task 1 dataclass 未暴露 experiment.seed，因此在 Runner 边界再次严格读取。"""
    with Path(config_path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    try:
        seed = raw["experiment"]["seed"]
        use_wandb = raw["experiment"]["use_wandb"]
    except (KeyError, TypeError) as exc:
        raise RunnerError("experiment seed/use_wandb is missing") from exc
    if type(seed) is not int or seed != EXPECTED_SEED:
        raise RunnerError(f"smoke seed must be exactly {EXPECTED_SEED}")
    if use_wandb is not False:
        raise RunnerError("wandb must remain disabled for the isolated smoke")
    return seed


def _archive_paths(source_root: Path) -> tuple[Path, ...]:
    return tuple(
        source_root / sensor / str(year) / f"{month:02d}" / f"{sensor}_{year}_{month:02d}.zip"
        for sensor in ("pc-s1", "pc-s2")
        for year in (2020, 2021)
        for month in range(1, 13)
    )


def _source_entry_metadata(path: Path) -> dict[str, object]:
    metadata = path.lstat()
    if stat.S_ISREG(metadata.st_mode):
        kind = "file"
    elif stat.S_ISDIR(metadata.st_mode):
        kind = "directory"
    elif stat.S_ISLNK(metadata.st_mode):
        kind = "symlink"
    else:
        kind = "other"
    return {"type": kind, "size": metadata.st_size, "mtime_ns": metadata.st_mtime_ns}


def _snapshot_source_archives(source_root: Path) -> dict[str, object]:
    """快照 48 个 ZIP 及其有限相关目录的完整直接 entry，不读取大文件内容。"""
    root = Path(os.path.abspath(source_root))
    if root.is_symlink() or not root.is_dir():
        raise RunnerError(f"source root must be a real directory: {root}")
    archives: dict[str, dict[str, object]] = {}
    watched_directories: set[Path] = {root}
    for archive in _archive_paths(root):
        if archive.is_symlink() or not archive.is_file():
            raise RunnerError(f"required source archive is missing or is a symlink: {archive}")
        relative = archive.relative_to(root).as_posix()
        archives[relative] = _source_entry_metadata(archive)
        directory = archive.parent
        while directory.is_relative_to(root):
            watched_directories.add(directory)
            if directory == root:
                break
            directory = directory.parent

    directories: dict[str, object] = {}
    for directory in sorted(watched_directories, key=lambda value: value.as_posix()):
        if directory.is_symlink() or not directory.is_dir():
            raise RunnerError(f"watched source path must be a real directory: {directory}")
        relative = "." if directory == root else directory.relative_to(root).as_posix()
        entries = {
            child.name: _source_entry_metadata(child)
            for child in sorted(directory.iterdir(), key=lambda value: value.name)
        }
        directories[relative] = {
            "metadata": _source_entry_metadata(directory),
            "entries": entries,
        }
    return {"archives": archives, "directories": directories}


def _snapshot_sandbox(sandbox_root: Path) -> dict[str, tuple[int, int, int]]:
    """仅遍历沙箱，用 lstat 捕获 path 类型、size 与 mtime。"""
    root = ensure_sandbox(sandbox_root)
    snapshot: dict[str, tuple[int, int, int]] = {}
    for path in root.rglob("*"):
        stat = path.lstat()
        snapshot[path.relative_to(root).as_posix()] = (
            stat.st_mode,
            stat.st_size,
            stat.st_mtime_ns,
        )
    return snapshot


def _sandbox_size(sandbox_root: Path) -> int:
    """计算沙箱自身目录项大小；symlink 只计 link，不跟随到外部环境。"""
    root = ensure_sandbox(sandbox_root)
    return sum(path.lstat().st_size for path in root.rglob("*") if not path.is_dir())


def _require_disk_budget(sandbox_root: Path) -> int:
    size = _sandbox_size(sandbox_root)
    if size >= MAX_SANDBOX_BYTES:
        raise RunnerError(f"sandbox exceeds the 5 GB stop rule: {size} bytes")
    return size


def _write_json(path: Path, payload: Mapping[str, object], sandbox_root: Path) -> Path:
    target = validate_write_path(path, sandbox_root)
    temporary = validate_write_path(target.with_name(f"{target.name}.partial"), sandbox_root)
    if temporary.exists():
        raise RunnerError(f"diagnostic partial already exists: {temporary}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def _write_tensor_cache(
    path: Path, payload: Mapping[str, object], sandbox_root: Path
) -> tuple[Path, str]:
    target = validate_write_path(path, sandbox_root)
    temporary = validate_write_path(target.with_name(f"{target.name}.partial"), sandbox_root)
    if temporary.exists():
        raise RunnerError(f"diagnostic partial already exists: {temporary}")
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), temporary)
    temporary.replace(target)
    return target, _sha256_file(target)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_file_metadata(path: Path) -> dict[str, object]:
    """在文件未发生并发变化时返回可审计 size/mtime/SHA-256。"""
    if path.is_symlink() or not path.is_file():
        raise RunnerError(f"evidence path must be a real file: {path}")
    before = path.stat()
    checksum = _sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RunnerError(f"evidence file changed while hashing: {path}")
    return {
        "size": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "sha256": checksum,
    }


def _record_path_audit(
    stage: str,
    sandbox_root: Path,
    before: Mapping[str, tuple[int, int, int]],
    *,
    final_seal_path: Path | None = None,
    declared_paths: Sequence[Path] = (),
) -> tuple[Path, ...]:
    """记录本阶段新增或修改路径；audit 文件自身也显式纳入。"""
    if stage not in AUDIT_STAGES:
        raise RunnerError(f"invalid path audit stage: {stage}")
    after = _snapshot_sandbox(sandbox_root)
    changed = {path for path, state in after.items() if before.get(path) != state}
    audit_relative = "path_audit.json"
    audit_path = validate_write_path(sandbox_root / audit_relative, sandbox_root)
    previous_stages: list[str] = []
    if audit_path.exists():
        try:
            previous = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RunnerError(f"existing path audit is malformed: {audit_path}") from exc
        if not isinstance(previous, dict):
            raise RunnerError("existing path audit must be a JSON object")
        previous_paths = previous.get("created_or_modified")
        if not isinstance(previous_paths, list) or not all(
            isinstance(relative, str)
            and relative
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts
            for relative in previous_paths
        ):
            raise RunnerError("existing path audit contains invalid relative paths")
        changed.update(previous_paths)
        raw_stages = previous.get("stages")
        if raw_stages is None and isinstance(previous.get("stage"), str):
            raw_stages = [previous["stage"]]
        if not isinstance(raw_stages, list) or not all(
            isinstance(value, str) and value in AUDIT_STAGES for value in raw_stages
        ):
            raise RunnerError("existing path audit contains invalid stages")
        previous_stages = raw_stages
    changed.add(audit_relative)
    declared_files: dict[str, dict[str, object]] = {}
    for declared_path in declared_paths:
        declared = validate_write_path(declared_path, sandbox_root)
        if declared.is_symlink() or not declared.is_file():
            raise RunnerError(f"declared audit path must be a real file: {declared}")
        relative = declared.relative_to(sandbox_root).as_posix()
        before_hash = declared.stat()
        checksum = _sha256_file(declared)
        after_hash = declared.stat()
        if (before_hash.st_size, before_hash.st_mtime_ns) != (
            after_hash.st_size,
            after_hash.st_mtime_ns,
        ):
            raise RunnerError(f"declared audit file changed while hashing: {declared}")
        changed.add(relative)
        declared_files[relative] = {
            "size": after_hash.st_size,
            "mtime_ns": after_hash.st_mtime_ns,
            "sha256": checksum,
        }
    stages = [*previous_stages, stage]
    final_seal: dict[str, object] | None = None
    if final_seal_path is not None:
        success = validate_write_path(final_seal_path, sandbox_root)
        if success != sandbox_root / "SUCCESS":
            raise RunnerError("final seal path must be the sandbox SUCCESS file")
        if success.exists() or success.is_symlink():
            raise RunnerError("SUCCESS must not exist when the authoritative audit is written")
        changed.add("SUCCESS")
        final_seal = {
            "path": "SUCCESS",
            "status": "expected_last_write",
            "exists_when_audit_written": False,
        }
    payload: dict[str, object] = {
        "stage": stage,
        "stages": stages,
        "sandbox_root": str(sandbox_root),
        "created_or_modified": sorted(changed),
        "formal_training_allowed": False,
        "formal_evaluation_allowed": False,
    }
    if final_seal is not None:
        payload["final_seal"] = final_seal
    if declared_files:
        payload["declared_files"] = dict(sorted(declared_files.items()))
    _write_json(
        audit_path,
        payload,
        sandbox_root,
    )
    return tuple(sandbox_root / relative for relative in sorted(changed))


def _checked_selections(config: SmokeConfig) -> tuple[PatchSelection, ...]:
    selections = select_complete_patches(config.source_root, count=config.max_patches)
    if len(selections) != 4:
        raise RunnerError(f"selector must return exactly four patches; got {len(selections)}")
    patch_ids = tuple(selection.patch_id for selection in selections)
    if len(set(patch_ids)) != 4 or any(not patch_id for patch_id in patch_ids):
        raise RunnerError("selector must return exactly four unique non-empty patch IDs")
    if any(len(selection.months) != 48 for selection in selections):
        raise RunnerError("every selected patch must contain exactly 48 month references")
    return selections


def _representative_selection(selection: PatchSelection) -> PatchSelection:
    """每个 sensor/year 仅检查一月头，共四个代表性 raster。"""
    months = tuple(month for month in selection.months if month.month == 1)
    if len(months) != 4:
        raise RunnerError(f"selection {selection.patch_id} has invalid representative headers")
    return PatchSelection(selection.patch_id, months)


def _header_payload(header: RasterHeader, patch_id: str) -> dict[str, object]:
    return {"patch_id": patch_id, **asdict(header)}


def _inspect(
    config: SmokeConfig,
) -> tuple[tuple[PatchSelection, ...], tuple[dict[str, object], ...]]:
    selections = _checked_selections(config)
    headers: list[dict[str, object]] = []
    for selection in selections:
        inspected = inspect_selection_headers(_representative_selection(selection))
        if len(inspected) != 4:
            raise RunnerError("header inspection must return four representative rasters")
        headers.extend(_header_payload(header, selection.patch_id) for header in inspected)
    return selections, tuple(headers)


def _month_payload(month: MonthRef, source_root: Path) -> dict[str, object]:
    archive = month.archive.resolve(strict=True)
    root = source_root.resolve(strict=True)
    if not archive.is_relative_to(root):
        raise RunnerError(f"selected source archive escapes fixed source root: {archive}")
    return {
        "sensor": month.sensor,
        "year": month.year,
        "month": month.month,
        "archive": archive.relative_to(root).as_posix(),
        "member": month.member,
        "crc": month.crc,
    }


def _selection_manifest(
    selections: Sequence[PatchSelection],
    headers: Sequence[Mapping[str, object]],
    source_root: Path,
) -> dict[str, object]:
    headers_by_patch: dict[str, list[Mapping[str, object]]] = {}
    for header in headers:
        headers_by_patch.setdefault(str(header["patch_id"]), []).append(header)
    return {
        "patch_ids": [selection.patch_id for selection in selections],
        "patches": [
            {
                "patch_id": selection.patch_id,
                "months": [_month_payload(month, source_root) for month in selection.months],
                "representative_headers": headers_by_patch[selection.patch_id],
            }
            for selection in selections
        ],
        "source_root": str(source_root),
        "synthetic": False,
    }


def _validate_batch(batch: PatchYearBatch, selections: Sequence[PatchSelection]) -> None:
    expected_patch_ids = tuple(
        selection.patch_id for selection in selections for _year in (2020, 2021)
    )
    expected_years = (2020, 2021) * 4
    expected_shapes = {
        "s2": (8, 4, 3, 10, 128, 128),
        "s1": (8, 4, 3, 2, 128, 128),
        "valid_s2": (8, 4, 3, 1, 128, 128),
        "valid_s1": (8, 4, 3, 1, 128, 128),
    }
    for name, shape in expected_shapes.items():
        if tuple(getattr(batch, name).shape) != shape:
            raise RunnerError(f"prepared batch {name} must have shape {shape}")
    if batch.patch_ids != expected_patch_ids or batch.years != expected_years:
        raise RunnerError("prepared batch patch/year ordering does not match the fixed selection")


def _validate_context(context: SyntheticAnnualContext) -> None:
    expected = {
        "aef": ((64, 128, 128), torch.float32),
        "aef_valid": ((1, 128, 128), torch.bool),
        "highres": ((3, 640, 640), torch.float32),
        "highres_valid": ((1, 640, 640), torch.bool),
    }
    for name, (shape, dtype) in expected.items():
        tensor = getattr(context, name)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise RunnerError(f"synthetic context {name} must have shape {shape}")
        if tensor.dtype != dtype or tensor.device.type != "cpu":
            raise RunnerError(f"synthetic context {name} has an invalid dtype or device")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise RunnerError(f"synthetic context {name} contains NaN or Inf")
    if set(context.metadata) != {"aef", "highres"}:
        raise RunnerError("synthetic context metadata must contain exactly AEF and highres")
    for metadata in context.metadata.values():
        validate_smoke_registry(metadata)
        try:
            validate_formal_registry(metadata)
        except RegistryError:
            pass
        else:
            raise RunnerError("formal validator accepted synthetic smoke metadata")


def _registry_payload(
    metadata: Mapping[str, object], entries: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    return {**dict(metadata), "entries": list(entries)}


def _run_prepare(
    config: SmokeConfig,
    seed: int,
    before_sandbox: Mapping[str, tuple[int, int, int]],
) -> StageResult:
    _require_disk_budget(config.sandbox_root)
    source_before = _snapshot_source_archives(config.source_root)
    selections, headers = _inspect(config)
    batch = load_patch_year_batch(selections, years=config.years)
    _validate_batch(batch, selections)
    source_after = _snapshot_source_archives(config.source_root)
    if source_after != source_before:
        raise RunnerError("source archive snapshot changed during prepare")

    manifests = config.sandbox_root / "manifests"
    _write_json(
        manifests / "source_snapshot_before.json",
        {**source_before, "hashing_performed": False},
        config.sandbox_root,
    )
    _write_json(
        manifests / "source_snapshot_after.json",
        {**source_after, "hashing_performed": False},
        config.sandbox_root,
    )
    selection_path = _write_json(
        manifests / "patch_selection.json",
        _selection_manifest(selections, headers, config.source_root),
        config.sandbox_root,
    )

    aef_entries: list[dict[str, object]] = []
    highres_entries: list[dict[str, object]] = []
    registry_metadata: Mapping[str, Mapping[str, object]] | None = None
    for sample_index, (patch_id, year) in enumerate(zip(batch.patch_ids, batch.years)):
        context = generate_synthetic_context(
            batch.s2[sample_index],
            batch.valid_s2[sample_index],
            patch_id,
            year,
            seed,
        )
        _validate_context(context)
        if registry_metadata is None:
            registry_metadata = context.metadata
        elif context.metadata != registry_metadata:
            raise RunnerError("synthetic metadata changed between patch-years")
        basename = f"patch_{sample_index:02d}_{year}.pt"
        aef_path, aef_sha = _write_tensor_cache(
            config.sandbox_root / "synthetic" / "aef" / basename,
            {
                "aef": context.aef,
                "aef_valid": context.aef_valid,
                "metadata": dict(context.metadata["aef"]),
                "patch_id": patch_id,
                "year": year,
            },
            config.sandbox_root,
        )
        highres_path, highres_sha = _write_tensor_cache(
            config.sandbox_root / "synthetic" / "highres_2m" / basename,
            {
                "highres": context.highres,
                "highres_valid": context.highres_valid,
                "metadata": dict(context.metadata["highres"]),
                "patch_id": patch_id,
                "year": year,
            },
            config.sandbox_root,
        )
        aef_entries.append(
            {
                "patch_id": patch_id,
                "year": year,
                "path": aef_path.relative_to(config.sandbox_root).as_posix(),
                "sha256": aef_sha,
            }
        )
        highres_entries.append(
            {
                "patch_id": patch_id,
                "year": year,
                "path": highres_path.relative_to(config.sandbox_root).as_posix(),
                "sha256": highres_sha,
            }
        )
        _require_disk_budget(config.sandbox_root)
    assert registry_metadata is not None
    _write_json(
        manifests / "aef_registry.json",
        _registry_payload(registry_metadata["aef"], aef_entries),
        config.sandbox_root,
    )
    _write_json(
        manifests / "highres_2m_registry.json",
        _registry_payload(registry_metadata["highres"], highres_entries),
        config.sandbox_root,
    )
    sandbox_bytes = _require_disk_budget(config.sandbox_root)
    _write_json(
        manifests / "prepare_manifest.json",
        {
            "patch_ids": [selection.patch_id for selection in selections],
            "patch_years": 8,
            "years": list(config.years),
            "source_archive_count": len(source_before["archives"]),
            "source_unchanged": True,
            "selection_manifest_sha256": _sha256_file(selection_path),
            "sandbox_bytes_before_path_audit": sandbox_bytes,
            "synthetic": True,
            "allowed_use": "smoke_test_only",
            "formal_training_allowed": False,
            "formal_evaluation_allowed": False,
            "accuracy_conclusion_allowed": False,
        },
        config.sandbox_root,
    )
    created = _record_path_audit("prepare", config.sandbox_root, before_sandbox)
    sandbox_bytes = _require_disk_budget(config.sandbox_root)
    return StageResult(
        stage="prepare",
        sandbox_root=config.sandbox_root,
        created_paths=created,
        patch_ids=tuple(selection.patch_id for selection in selections),
        patch_years=8,
        headers=headers,
        source_unchanged=True,
        sandbox_bytes=sandbox_bytes,
    )


def _fixture_inputs(full_shape: bool) -> dict[str, torch.Tensor]:
    batch = 8 if full_shape else 2
    height = 128 if full_shape else 16
    generator = torch.Generator(device="cpu").manual_seed(EXPECTED_SEED)
    return {
        "s2": torch.randn((batch, 4, 3, 10, height, height), generator=generator),
        "s1": torch.randn((batch, 4, 3, 2, height, height), generator=generator),
        "valid_s2": torch.ones((batch, 4, 3, 1, height, height), dtype=torch.bool),
        "valid_s1": torch.ones((batch, 4, 3, 1, height, height), dtype=torch.bool),
        "aef": torch.randn((batch, 64, height, height), generator=generator),
        "aef_valid": torch.ones((batch, 1, height, height), dtype=torch.bool),
        "highres": torch.randn((batch, 3, height * 5, height * 5), generator=generator),
        "highres_valid": torch.ones((batch, 1, height * 5, height * 5), dtype=torch.bool),
    }


def _run_cpu_contract(
    config: SmokeConfig,
    seed: int,
    full_shape: bool,
    before_sandbox: Mapping[str, tuple[int, int, int]],
) -> StageResult:
    _require_disk_budget(config.sandbox_root)
    inputs = _fixture_inputs(full_shape)
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(seed)
        model = IsolatedFusionSmokeModel(embed_dim=config.embed_dim).eval()
        outputs = {
            group: model(
                **inputs,
                use_aef=use_aef,
                use_highres=use_highres,
            )
            for group, (use_aef, use_highres) in GROUPS.items()
        }
    base = outputs["base"]
    full = outputs["full"]
    identity = torch.equal(full.pre_vmf, base.pre_vmf) and torch.equal(
        full.embedding, base.embedding
    )
    if not identity:
        raise RunnerError("zero-gate Full/Base identity failed on the CPU contract")
    group_payload: dict[str, object] = {}
    for group, output in outputs.items():
        if not bool(torch.isfinite(output.embedding).all()):
            raise RunnerError(f"CPU contract group {group} contains NaN or Inf")
        norms = torch.linalg.vector_norm(output.embedding, dim=2)
        max_norm_error = float((norms - 1.0).abs().max())
        if max_norm_error > 1.0e-5:
            raise RunnerError(f"CPU contract group {group} violates the vMF norm contract")
        group_payload[group] = {
            "shape": list(output.embedding.shape),
            "dtype": str(output.embedding.dtype),
            "finite": True,
            "max_vmf_norm_error": max_norm_error,
        }
    _write_json(
        config.sandbox_root / "manifests" / "cpu_contract.json",
        {
            "fixture": "full-shape" if full_shape else "small",
            "groups": group_payload,
            "full_zero_gate_matches_base": True,
            "synthetic": True,
            "allowed_use": "smoke_test_only",
            "formal_training_allowed": False,
            "formal_evaluation_allowed": False,
            "accuracy_conclusion_allowed": False,
        },
        config.sandbox_root,
    )
    created = _record_path_audit("cpu-contract", config.sandbox_root, before_sandbox)
    return StageResult(
        stage="cpu-contract",
        sandbox_root=config.sandbox_root,
        groups=tuple(GROUPS),
        full_zero_gate_matches_base=True,
        created_paths=created,
        sandbox_bytes=_require_disk_budget(config.sandbox_root),
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunnerError(f"prepared manifest is missing or malformed: {path}") from exc
    if not isinstance(raw, dict):
        raise RunnerError(f"prepared manifest must be a JSON object: {path}")
    return raw


def _load_prepared_contexts(
    config: SmokeConfig, batch: PatchYearBatch
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    manifests = config.sandbox_root / "manifests"
    aef_registry = _load_json_object(manifests / "aef_registry.json")
    highres_registry = _load_json_object(manifests / "highres_2m_registry.json")
    validate_smoke_registry(aef_registry)
    validate_smoke_registry(highres_registry)
    tensors: dict[str, list[torch.Tensor]] = {
        "aef": [],
        "aef_valid": [],
        "highres": [],
        "highres_valid": [],
    }
    for registry, kind, tensor_name, valid_name in (
        (aef_registry, "aef", "aef", "aef_valid"),
        (highres_registry, "highres", "highres", "highres_valid"),
    ):
        entries = registry.get("entries")
        if not isinstance(entries, list) or len(entries) != 8:
            raise RunnerError(f"prepared {kind} registry must contain eight entries")
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise RunnerError(f"prepared {kind} registry entry is malformed")
            if (entry.get("patch_id"), entry.get("year")) != (
                batch.patch_ids[index],
                batch.years[index],
            ):
                raise RunnerError(f"prepared {kind} registry ordering is invalid")
            relative = entry.get("path")
            if not isinstance(relative, str):
                raise RunnerError(f"prepared {kind} cache path is invalid")
            path = validate_write_path(config.sandbox_root / relative, config.sandbox_root)
            if not path.is_file() or _sha256_file(path) != entry.get("sha256"):
                raise RunnerError(f"prepared {kind} cache checksum mismatch: {path}")
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            except Exception as exc:
                raise RunnerError(f"cannot safely load prepared cache: {path}") from exc
            if not isinstance(payload, dict) or payload.get("metadata") != {
                key: value for key, value in registry.items() if key != "entries"
            }:
                raise RunnerError(f"prepared {kind} cache metadata mismatch: {path}")
            validate_smoke_registry(payload["metadata"])
            tensors[tensor_name].append(payload[tensor_name])
            tensors[valid_name].append(payload[valid_name])
    context = SyntheticAnnualContext(
        aef=torch.stack(tensors["aef"]),
        aef_valid=torch.stack(tensors["aef_valid"]),
        highres=torch.stack(tensors["highres"]),
        highres_valid=torch.stack(tensors["highres_valid"]),
        metadata={"aef": aef_registry, "highres": highres_registry},
    )
    _validate_context(
        SyntheticAnnualContext(
            aef=context.aef[0],
            aef_valid=context.aef_valid[0],
            highres=context.highres[0],
            highres_valid=context.highres_valid[0],
            metadata={
                "aef": {key: value for key, value in aef_registry.items() if key != "entries"},
                "highres": {
                    key: value for key, value in highres_registry.items() if key != "entries"
                },
            },
        )
    )
    return context.aef, context.aef_valid, context.highres, context.highres_valid


def _is_expected_smoke_interpreter(config: SmokeConfig) -> bool:
    expected = Path(os.path.abspath(config.sandbox_root / "env/bin/python"))
    return Path(os.path.abspath(sys.executable)) == expected


def _physical_npu2_exists() -> bool:
    return PHYSICAL_NPU2.exists() and PHYSICAL_NPU2.is_char_device()


def _proc_device_user_pids(device: Path, *, proc_root: Path) -> tuple[int, ...]:
    """在没有 fuser 时按字符设备 st_rdev 扫描完整可读的进程 fd。"""
    try:
        device_metadata = device.stat()
    except OSError as exc:
        raise RunnerError(f"cannot stat physical NPU device: {device}") from exc
    if not stat.S_ISCHR(device_metadata.st_mode):
        raise RunnerError(f"physical NPU path is not a character device: {device}")
    try:
        processes = tuple(proc_root.iterdir())
    except OSError as exc:
        raise RunnerError("cannot independently scan the process table") from exc

    users: set[int] = set()
    for process in processes:
        if not process.name.isdigit():
            continue
        fd_root = process / "fd"
        try:
            descriptors = tuple(fd_root.iterdir())
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise RunnerError(
                f"cannot independently scan process file descriptors: {fd_root}"
            ) from exc
        for descriptor in descriptors:
            try:
                metadata = descriptor.stat()
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise RunnerError(
                    f"cannot independently inspect process file descriptor: {descriptor}"
                ) from exc
            if stat.S_ISCHR(metadata.st_mode) and metadata.st_rdev == device_metadata.st_rdev:
                users.add(int(process.name))
    return tuple(sorted(users))


def _physical_device_is_idle(device: Path, *, proc_root: Path = Path("/proc")) -> bool:
    fuser = shutil.which("fuser")
    if fuser is None:
        return not _proc_device_user_pids(device, proc_root=proc_root)
    try:
        result = subprocess.run(
            [fuser, str(device)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RunnerError("cannot independently query NPU 2 occupancy with fuser") from exc
    if result.returncode == 1:
        return True
    if result.returncode == 0:
        return False
    raise RunnerError(
        f"cannot independently query NPU 2 occupancy: fuser exited {result.returncode}"
    )


def _physical_npu2_is_idle() -> bool:
    return _physical_device_is_idle(PHYSICAL_NPU2)


def _proc_parent_pid(process: Path) -> int | None:
    try:
        lines = process.joinpath("status").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    for line in lines:
        if line.startswith("PPid:"):
            try:
                return int(line.split(":", maxsplit=1)[1].strip())
            except ValueError:
                return None
    return None


def _has_task7_launcher_ancestor(
    *,
    start_pid: int | None = None,
    proc_root: Path = Path("/proc"),
) -> bool:
    """沿真实父进程链查找由固定 worktree 执行固定 Task 7 脚本的 shell。"""
    pid = os.getppid() if start_pid is None else start_pid
    expected_launcher = TASK7_LAUNCHER.resolve(strict=False)
    expected_worktree = EXPECTED_WORKTREE.resolve(strict=True)
    visited: set[int] = set()
    for _depth in range(16):
        if pid <= 1 or pid in visited:
            return False
        visited.add(pid)
        process = proc_root / str(pid)
        try:
            command = process.joinpath("cmdline").read_bytes().split(b"\0")
            command = [part.decode("utf-8") for part in command if part]
            process_cwd = process.joinpath("cwd").resolve(strict=True)
        except (OSError, UnicodeDecodeError):
            return False
        if len(command) == 2 and Path(command[0]).name in {"bash", "sh"}:
            script_operand = Path(command[1])
            if not script_operand.is_absolute():
                script_operand = process_cwd / script_operand
            if (
                script_operand.resolve(strict=False) == expected_launcher
                and process_cwd == expected_worktree
            ):
                return True
        parent = _proc_parent_pid(process)
        if parent is None:
            return False
        pid = parent
    return False


def _validate_npu_launcher(config: SmokeConfig) -> None:
    expected = {
        "PYTHONNOUSERSITE": "1",
        "ASCEND_RT_VISIBLE_DEVICES": str(config.physical_npu),
        "XUANNV_SMOKE_DEVICE": config.logical_device,
        "WANDB_MODE": "disabled",
    }
    mismatches = [f"{key}={value!r}" for key, value in expected.items() if os.getenv(key) != value]
    if mismatches:
        raise RunnerError(
            "npu-smoke is supported only through the Task 7 launcher; missing exact environment: "
            + ", ".join(mismatches)
        )
    if Path.cwd().resolve(strict=True) != EXPECTED_WORKTREE.resolve(strict=True):
        raise RunnerError("Task 7 launcher must run from the fixed isolated worktree")
    if not _is_expected_smoke_interpreter(config):
        raise RunnerError("Task 7 launcher must use the fixed sandbox Python interpreter")
    if not _physical_npu2_exists():
        raise RunnerError("physical /dev/davinci2 is absent")
    if not _physical_npu2_is_idle():
        raise RunnerError("NPU 2 is busy; refusing smoke run")
    if not _has_task7_launcher_ancestor():
        raise RunnerError("npu-smoke requires a verified Task 7 launcher ancestor")
    if TASK7_LAUNCHER.is_symlink() or not TASK7_LAUNCHER.is_file():
        raise RunnerError("fixed Task 7 launcher is missing or is a symlink")


def _patch_period_layout(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[0] != 8 or tensor.shape[1] != 4:
        raise RunnerError("NPU output must contain eight patch-years and four quarters")
    return tensor.reshape(4, 2, 4, *tensor.shape[2:]).reshape(4, 8, *tensor.shape[2:])


def _git_commit() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(commit) != 40:
        raise RunnerError("cannot record the current Git commit")
    return commit


def _key_value_version(path: Path, key: str) -> str:
    """读取 CANN/driver 的固定 key=value 版本证据。"""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise RunnerError(f"cannot read runtime version evidence: {path}") from exc
    prefix = f"{key}="
    for line in lines:
        if line.startswith(prefix):
            value = line.removeprefix(prefix).strip().strip('"')
            if value:
                return value
    raise RunnerError(f"runtime version evidence is missing {key}: {path}")


def _runtime_provenance(torch_npu_module: object, device: torch.device) -> dict[str, object]:
    """记录本次 NPU 进程实际解释器、模块、CANN/driver 与设备映射。"""
    cann_root = Path("/usr/local/Ascend/cann-9.0.0")
    cann_info = cann_root / "aarch64-linux/ascend_toolkit_install.info"
    driver_info = Path("/usr/local/Ascend/driver/version.info")
    model_module = sys.modules[IsolatedFusionSmokeModel.__module__]
    module_paths = {
        "torch": str(Path(torch.__file__).resolve(strict=True)),
        "torch_npu": str(Path(torch_npu_module.__file__).resolve(strict=True)),
        "runner": str(Path(__file__).resolve(strict=True)),
        "model": str(Path(model_module.__file__).resolve(strict=True)),
    }
    return {
        "sys_executable": os.path.abspath(sys.executable),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_npu_version": str(torch_npu_module.__version__),
        "cann": {
            "root": str(cann_root),
            "version": _key_value_version(cann_info, "version"),
            "install_info": str(cann_info),
        },
        "driver": {
            "version": _key_value_version(driver_info, "Version"),
            "version_info": str(driver_info),
        },
        "device_mapping": {
            "physical_device": str(PHYSICAL_NPU2),
            "physical_npu": 2,
            "visible_devices": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
            "logical_device": str(device),
            "logical_device_count": torch.npu.device_count(),
            "device_name": torch.npu.get_device_name(device),
        },
        "module_paths": module_paths,
    }


def _run_npu_smoke(
    config: SmokeConfig,
    config_path: Path,
    seed: int,
    before_sandbox: Mapping[str, tuple[int, int, int]],
) -> StageResult:
    """在 Task 7 启动守卫之后执行全尺寸四组、反向、恢复与封存。"""
    _validate_npu_launcher(config)
    prepare = _load_json_object(config.sandbox_root / "manifests" / "prepare_manifest.json")
    if prepare.get("patch_years") != 8 or prepare.get("source_unchanged") is not True:
        raise RunnerError("npu-smoke requires a successful fixed prepare stage")
    cpu_contract_path = config.sandbox_root / "manifests" / "cpu_contract.json"
    cpu_contract = _load_json_object(cpu_contract_path)
    if (
        cpu_contract.get("full_zero_gate_matches_base") is not True
        or not isinstance(cpu_contract.get("groups"), Mapping)
        or set(cpu_contract["groups"]) != set(GROUPS)
    ):
        raise RunnerError("npu-smoke requires the successful CPU contract as fallback evidence")
    _require_disk_budget(config.sandbox_root)
    source_before = _snapshot_source_archives(config.source_root)
    selections = _checked_selections(config)
    selection_manifest = _load_json_object(
        config.sandbox_root / "manifests" / "patch_selection.json"
    )
    patch_ids = tuple(selection.patch_id for selection in selections)
    if selection_manifest.get("patch_ids") != list(patch_ids):
        raise RunnerError("prepared patch selection differs from the current stable selection")
    batch = load_patch_year_batch(selections, years=config.years)
    _validate_batch(batch, selections)
    aef, aef_valid, highres, highres_valid = _load_prepared_contexts(config, batch)

    try:
        import torch_npu
    except ImportError as exc:
        raise RunnerError("torch_npu is unavailable after the Task 7 launcher setup") from exc
    if torch.npu.device_count() != 1:
        raise RunnerError("Task 7 launcher must expose exactly one logical NPU")
    device = torch.device(config.logical_device)
    torch.npu.set_device(device)
    runtime_provenance = _runtime_provenance(torch_npu, device)
    inputs = {
        "s2": batch.s2.to(device),
        "s1": batch.s1.to(device),
        "valid_s2": batch.valid_s2.to(device),
        "valid_s1": batch.valid_s1.to(device),
        "aef": aef.to(device),
        "aef_valid": aef_valid.to(device),
        "highres": highres.to(device),
        "highres_valid": highres_valid.to(device),
    }
    torch.manual_seed(seed)
    model = IsolatedFusionSmokeModel(embed_dim=config.embed_dim).to(device).eval()
    output_cpu: dict[str, torch.Tensor] = {}
    group_metrics: dict[str, object] = {}
    with torch.no_grad():
        for group, (use_aef, use_highres) in GROUPS.items():
            torch.npu.synchronize()
            torch.npu.empty_cache()
            torch.npu.synchronize()
            torch.npu.reset_peak_memory_stats(device)
            baseline_allocated = int(torch.npu.memory_allocated(device))
            baseline_reserved = int(torch.npu.memory_reserved(device))
            started = time.perf_counter()
            output = model(
                **inputs,
                use_aef=use_aef,
                use_highres=use_highres,
            )
            torch.npu.synchronize()
            latency = time.perf_counter() - started
            peak_allocated = int(torch.npu.max_memory_allocated(device))
            peak_reserved = int(torch.npu.max_memory_reserved(device))
            embedding = output.embedding.detach().cpu()
            output_cpu[group] = embedding
            norms = torch.linalg.vector_norm(embedding, dim=2)
            group_metrics[group] = {
                "shape": list(_patch_period_layout(embedding).shape),
                "latency_seconds": latency,
                "finite": bool(torch.isfinite(embedding).all()),
                "pre_export_fp32_vmf_norm": {
                    "source_dtype": str(embedding.dtype).removeprefix("torch."),
                    "computation_dtype": "float32",
                    "min": float(norms.min()),
                    "median": float(norms.median()),
                    "max": float(norms.max()),
                },
                "npu_peak_memory": {
                    "unit": "bytes",
                    "baseline_allocated": baseline_allocated,
                    "baseline_reserved": baseline_reserved,
                    "peak_allocated": peak_allocated,
                    "peak_reserved": peak_reserved,
                    "peak_allocated_delta": peak_allocated - baseline_allocated,
                    "peak_reserved_delta": peak_reserved - baseline_reserved,
                },
            }
    zero_gate_error = float((output_cpu["full"] - output_cpu["base"]).abs().max())
    if zero_gate_error != 0.0:
        raise RunnerError("zero-gate Full/Base identity failed on NPU")
    if not all(bool(torch.isfinite(value).all()) for value in output_cpu.values()):
        raise RunnerError("NPU forward produced NaN or Inf")

    selection_path = config.sandbox_root / "manifests" / "patch_selection.json"
    metadata = {
        "git_commit": _git_commit(),
        "config_sha256": _sha256_file(config_path),
        "selected_patch_manifest_sha256": _sha256_file(selection_path),
        "seed": seed,
        "model_class": ("experiments.china_v1_fusion_smoke.model.IsolatedFusionSmokeModel"),
        "synthetic": True,
        "formal_training_allowed": False,
        "formal_evaluation_allowed": False,
    }
    checkpoint = config.sandbox_root / "smoke_checkpoint.pt"
    checkpoint_sha = save_smoke_checkpoint(model, checkpoint, metadata)
    reloaded = IsolatedFusionSmokeModel(embed_dim=config.embed_dim)
    load_smoke_checkpoint(reloaded, checkpoint, metadata)
    reloaded = reloaded.to(device).eval()
    with torch.no_grad():
        restored = reloaded(**inputs, use_aef=True, use_highres=True).embedding.detach().cpu()
    reload_error = float((restored - output_cpu["full"]).abs().max())
    if reload_error != 0.0:
        raise RunnerError("checkpoint reload output differs from the original Full output")

    model.train()
    model.s2_stem.requires_grad_(False)
    model.s1_stem.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=1.0e-4)
    gradient_norms: dict[str, float] = {}
    for _step in range(config.max_optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        gradient_output = model(
            **inputs,
            use_aef=True,
            use_highres=True,
            gate_override=config.gradient_smoke_gate,
        )
        gradient_output.pre_vmf.square().mean().backward()
        gradient_norms = {
            "aef_adapter": _gradient_l1(model.aef_adapter),
            "highres_stem": _gradient_l1(model.highres_stem),
            "highres_adapter": _gradient_l1(model.highres_adapter),
            "output_projection": _gradient_l1(model.output_projection),
        }
        if not all(value > 0.0 for value in gradient_norms.values()):
            raise RunnerError("gradient smoke produced a missing, nonfinite or zero gradient")
        optimizer.step()

    torch.manual_seed(seed)
    repeated_model = IsolatedFusionSmokeModel(embed_dim=config.embed_dim).to(device).eval()
    with torch.no_grad():
        repeated = repeated_model(**inputs, use_aef=True, use_highres=True).embedding.detach().cpu()
    reproducibility_error = float((repeated - output_cpu["full"]).abs().max())
    if reproducibility_error != 0.0:
        raise RunnerError("fixed-seed NPU rerun is not reproducible")

    valid = batch.valid_s2.any(dim=2) & batch.valid_s1.any(dim=2)
    exported_valid = _patch_period_layout(valid)
    for group, embedding in output_cpu.items():
        exported = export_group_zarr(
            group,
            _patch_period_layout(embedding),
            exported_valid,
            patch_ids,
            config.periods,
            config.sandbox_root / "outputs" / group / "embedding.zarr",
            config.sandbox_root,
        )
        metric = group_metrics[group]
        assert isinstance(metric, dict)
        metric["reopened_fp16_zarr_vmf_norm"] = reopened_fp16_vmf_norm_summary(exported)
    source_after = _snapshot_source_archives(config.source_root)
    if source_after != source_before:
        raise RunnerError("source archive snapshot changed during npu-smoke")
    _write_json(
        config.sandbox_root / "run_manifest.json",
        {
            "git_commit": metadata["git_commit"],
            "config_sha256": metadata["config_sha256"],
            "patch_ids": list(patch_ids),
            "periods": list(config.periods),
            "physical_npu": config.physical_npu,
            "logical_device": config.logical_device,
            "source_unchanged": True,
            "synthetic": True,
            "allowed_use": "smoke_test_only",
            "formal_training_allowed": False,
            "formal_evaluation_allowed": False,
            "accuracy_conclusion_allowed": False,
            "runtime_provenance": runtime_provenance,
            "cpu_contract_fallback": {
                "path": "manifests/cpu_contract.json",
                "sha256": _sha256_file(cpu_contract_path),
            },
        },
        config.sandbox_root,
    )
    _write_json(
        config.sandbox_root / "metrics.json",
        {
            "groups": group_metrics,
            "zero_gate_max_abs_error": zero_gate_error,
            "gradient_l1": gradient_norms,
            "checkpoint_reload_max_abs_error": reload_error,
            "accuracy_conclusion_allowed": False,
        },
        config.sandbox_root,
    )
    _write_json(
        config.sandbox_root / "reproducibility.json",
        {
            "fixed_seed": seed,
            "max_abs_error": reproducibility_error,
            "matches": True,
            "checkpoint_sha256": checkpoint_sha,
            "synthetic": True,
        },
        config.sandbox_root,
    )
    created = _record_path_audit(
        "npu-smoke",
        config.sandbox_root,
        before_sandbox,
    )
    path_audit = config.sandbox_root / "path_audit.json"
    ready = _write_json(
        config.sandbox_root / READY_TO_SEAL,
        {
            "status": "npu_compute_complete",
            "git_commit": metadata["git_commit"],
            "config_sha256": metadata["config_sha256"],
            "preliminary_path_audit_sha256": _sha256_file(path_audit),
            "source_unchanged": True,
            "synthetic": True,
            "formal_training_allowed": False,
            "formal_evaluation_allowed": False,
        },
        config.sandbox_root,
    )
    created = (*created, ready)
    return StageResult(
        stage="npu-smoke",
        sandbox_root=config.sandbox_root,
        groups=tuple(GROUPS),
        full_zero_gate_matches_base=True,
        created_paths=created,
        patch_ids=patch_ids,
        patch_years=8,
        source_unchanged=True,
        sandbox_bytes=_require_disk_budget(config.sandbox_root),
    )


def _gradient_l1(module: torch.nn.Module) -> float:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    if not gradients or any(gradient is None for gradient in gradients):
        return 0.0
    if any(
        not bool(torch.isfinite(gradient).all()) for gradient in gradients if gradient is not None
    ):
        return 0.0
    return float(sum(gradient.abs().sum().item() for gradient in gradients if gradient is not None))


def _validated_ready_to_seal(
    config: SmokeConfig,
    config_path: Path,
) -> tuple[Path, Path]:
    """校验 compute marker 并返回 READY 与其绑定的 preliminary audit。"""
    sandbox_root = config.sandbox_root
    ready_path = validate_write_path(sandbox_root / READY_TO_SEAL, sandbox_root)
    if ready_path.is_symlink() or not ready_path.is_file():
        raise RunnerError("operation requires a real READY_TO_SEAL file")
    ready = _load_json_object(ready_path)
    expected_ready_keys = {
        "status",
        "git_commit",
        "config_sha256",
        "preliminary_path_audit_sha256",
        "source_unchanged",
        "synthetic",
        "formal_training_allowed",
        "formal_evaluation_allowed",
    }
    if set(ready) != expected_ready_keys:
        raise RunnerError("READY_TO_SEAL has an invalid schema")
    expected_ready = {
        "status": "npu_compute_complete",
        "git_commit": _git_commit(),
        "config_sha256": _sha256_file(config_path),
        "source_unchanged": True,
        "synthetic": True,
        "formal_training_allowed": False,
        "formal_evaluation_allowed": False,
    }
    if any(ready.get(key) != value for key, value in expected_ready.items()):
        raise RunnerError("READY_TO_SEAL provenance does not match the current smoke")

    audit_path = validate_write_path(sandbox_root / "path_audit.json", sandbox_root)
    preliminary_audit = _load_json_object(audit_path)
    if preliminary_audit.get("stage") != "npu-smoke" or ready.get(
        "preliminary_path_audit_sha256"
    ) != _sha256_file(audit_path):
        raise RunnerError("READY_TO_SEAL does not match the preliminary path audit")
    return ready_path, audit_path


def _mark_tee_complete(config_path: Path) -> Path:
    """仅在 launcher 的 pipefail pipeline 成功返回后原子绑定最终 tee 日志。"""
    config_path = Path(config_path)
    config = load_smoke_config(config_path)
    _load_seed(config_path)
    sandbox_root = ensure_sandbox(config.sandbox_root)
    _validate_npu_launcher(config)
    if (sandbox_root / "SUCCESS").exists():
        raise RunnerError("SUCCESS already exists; sealed smoke output is immutable")
    marker = validate_write_path(sandbox_root / TEE_COMPLETE, sandbox_root)
    if marker.exists() or marker.is_symlink():
        raise RunnerError("TEE_COMPLETE already exists; refusing to overwrite completion evidence")
    ready_path, _audit_path = _validated_ready_to_seal(config, config_path)
    launcher_log = validate_write_path(sandbox_root / "logs" / "npu_smoke.log", sandbox_root)
    log_metadata = _stable_file_metadata(launcher_log)
    return _write_json(
        marker,
        {
            "status": "tee_pipeline_complete",
            "git_commit": _git_commit(),
            "config_sha256": _sha256_file(config_path),
            "ready_to_seal_sha256": _sha256_file(ready_path),
            "launcher_log": {
                "path": "logs/npu_smoke.log",
                **log_metadata,
            },
            "synthetic": True,
            "formal_training_allowed": False,
            "formal_evaluation_allowed": False,
        },
        sandbox_root,
    )


def _validated_tee_complete(
    config: SmokeConfig,
    config_path: Path,
    ready_path: Path,
) -> tuple[Path, Path]:
    """验证 tee marker 的 provenance 与当前最终日志逐字段一致。"""
    sandbox_root = config.sandbox_root
    marker = validate_write_path(sandbox_root / TEE_COMPLETE, sandbox_root)
    if marker.is_symlink() or not marker.is_file():
        raise RunnerError("finalize-seal requires a real TEE_COMPLETE file")
    raw = _load_json_object(marker)
    expected_keys = {
        "status",
        "git_commit",
        "config_sha256",
        "ready_to_seal_sha256",
        "launcher_log",
        "synthetic",
        "formal_training_allowed",
        "formal_evaluation_allowed",
    }
    if set(raw) != expected_keys:
        raise RunnerError("TEE_COMPLETE has an invalid schema")
    expected = {
        "status": "tee_pipeline_complete",
        "git_commit": _git_commit(),
        "config_sha256": _sha256_file(config_path),
        "ready_to_seal_sha256": _sha256_file(ready_path),
        "synthetic": True,
        "formal_training_allowed": False,
        "formal_evaluation_allowed": False,
    }
    if any(raw.get(key) != value for key, value in expected.items()):
        raise RunnerError("TEE_COMPLETE provenance does not match the current smoke")
    launcher_log = validate_write_path(sandbox_root / "logs" / "npu_smoke.log", sandbox_root)
    expected_log = {"path": "logs/npu_smoke.log", **_stable_file_metadata(launcher_log)}
    if raw.get("launcher_log") != expected_log:
        raise RunnerError("TEE_COMPLETE launcher log metadata does not match the final log")
    return marker, launcher_log


def _finalize_npu_smoke(config_path: Path) -> Path:
    """tee 完全结束后重建最终审计，并把 SUCCESS 作为最后一次沙箱写入。"""
    config_path = Path(config_path)
    config = load_smoke_config(config_path)
    _load_seed(config_path)
    sandbox_root = ensure_sandbox(config.sandbox_root)
    _validate_npu_launcher(config)
    if (sandbox_root / "SUCCESS").exists():
        raise RunnerError("SUCCESS already exists; sealed smoke output is immutable")

    ready_path, _audit_path = _validated_ready_to_seal(config, config_path)
    tee_complete, launcher_log = _validated_tee_complete(config, config_path, ready_path)

    expected_source = _load_json_object(sandbox_root / "manifests" / "source_snapshot_after.json")
    if expected_source.pop("hashing_performed", None) is not False:
        raise RunnerError("prepared source snapshot has an invalid hashing policy")
    if _snapshot_source_archives(config.source_root) != expected_source:
        raise RunnerError("source archive snapshot changed before finalize-seal")

    before_finalize = _snapshot_sandbox(sandbox_root)
    _record_path_audit(
        FINALIZE_STAGE,
        sandbox_root,
        before_finalize,
        final_seal_path=sandbox_root / "SUCCESS",
        declared_paths=(launcher_log, ready_path, tee_complete),
    )
    _require_disk_budget(sandbox_root)
    success = seal_success(sandbox_root, [])
    verify_success(sandbox_root)
    _require_disk_budget(sandbox_root)
    return success


def run_stage(config_path: Path, stage: str, *, full_shape: bool = False) -> StageResult:
    """校验配置与 sentinel 后，执行一个且仅一个隔离 smoke 阶段。"""
    if stage not in STAGES:
        raise ValueError(f"unknown smoke stage: {stage!r}")
    config_path = Path(config_path)
    config = load_smoke_config(config_path)
    seed = _load_seed(config_path)
    sandbox_root = ensure_sandbox(config.sandbox_root)
    if sandbox_root != config.sandbox_root.resolve(strict=True):
        raise RunnerError("resolved sandbox root differs from the configured root")
    before_sandbox = _snapshot_sandbox(sandbox_root)
    if stage == "inspect":
        selections, headers = _inspect(config)
        return StageResult(
            stage=stage,
            sandbox_root=sandbox_root,
            patch_ids=tuple(selection.patch_id for selection in selections),
            patch_years=8,
            headers=headers,
            sandbox_bytes=_require_disk_budget(sandbox_root),
        )
    if stage == "prepare":
        return _run_prepare(config, seed, before_sandbox)
    if stage == "cpu-contract":
        return _run_cpu_contract(config, seed, full_shape, before_sandbox)
    return _run_npu_smoke(config, config_path, seed, before_sandbox)


def _result_payload(result: StageResult) -> dict[str, object]:
    return {
        "stage": result.stage,
        "sandbox_root": str(result.sandbox_root),
        "groups": list(result.groups),
        "full_zero_gate_matches_base": result.full_zero_gate_matches_base,
        "created_paths": [str(path) for path in result.created_paths],
        "patch_ids": list(result.patch_ids),
        "patch_years": result.patch_years,
        "headers": list(result.headers),
        "source_unchanged": result.source_unchanged,
        "sandbox_bytes": result.sandbox_bytes,
        "accuracy_conclusion_allowed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--stage", choices=STAGES)
    action.add_argument("--mark-tee-complete", action="store_true")
    action.add_argument("--finalize-seal", action="store_true")
    parser.add_argument(
        "--full-shape",
        action="store_true",
        help="cpu-contract only: use B=8 and 128x128 instead of the small fixture",
    )
    args = parser.parse_args(argv)
    if args.mark_tee_complete:
        if args.full_shape:
            parser.error("--full-shape is not supported by --mark-tee-complete")
        _mark_tee_complete(args.config)
        return 0
    if args.finalize_seal:
        if args.full_shape:
            parser.error("--full-shape is not supported by --finalize-seal")
        _finalize_npu_smoke(args.config)
        return 0
    if args.full_shape and args.stage != "cpu-contract":
        parser.error("--full-shape is supported only by --stage cpu-contract")
    assert args.stage is not None
    result = run_stage(args.config, args.stage, full_shape=args.full_shape)
    print(json.dumps(_result_payload(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
