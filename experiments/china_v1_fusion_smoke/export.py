"""隔离 China V1 fusion smoke 的可审计导出与最终封存。"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from numcodecs import Blosc

from experiments.china_v1_fusion_smoke.safety import SENTINEL, validate_write_path


class ExportError(RuntimeError):
    """导出、恢复或 SUCCESS 封存合同不满足时抛出。"""


_GROUPS = ("base", "base_aef", "base_highres", "full")
_EMBEDDING_SHAPE = (4, 8, 64, 128, 128)
_VALID_SHAPE = (4, 8, 1, 128, 128)
_EMBEDDING_CHUNKS = (1, 1, 64, 128, 128)
_VALID_CHUNKS = (1, 1, 1, 128, 128)
_REQUIRED_EVIDENCE = (
    "run_manifest.json",
    "metrics.json",
    "path_audit.json",
    "reproducibility.json",
    "smoke_checkpoint.pt",
)
_PATCH_SELECTION_RELATIVE_PATH = Path("manifests") / "patch_selection.json"
_LAUNCHER_LOG_RELATIVE_PATH = Path("logs") / "npu_smoke.log"
_READY_TO_SEAL_RELATIVE_PATH = Path("READY_TO_SEAL")
_TEE_COMPLETE_RELATIVE_PATH = Path("TEE_COMPLETE")
_MANDATORY_EVIDENCE_ROOTS = (Path("manifests"), Path("synthetic"), Path("outputs"))
_MANDATORY_MANIFEST_FILES = (
    Path("manifests/aef_registry.json"),
    Path("manifests/cpu_contract.json"),
    Path("manifests/highres_2m_registry.json"),
    Path("manifests/patch_selection.json"),
    Path("manifests/prepare_manifest.json"),
    Path("manifests/source_snapshot_after.json"),
    Path("manifests/source_snapshot_before.json"),
)
_ALLOWED_AUDIT_EXACT_PATHS = {
    "logs",
    _LAUNCHER_LOG_RELATIVE_PATH.as_posix(),
    _READY_TO_SEAL_RELATIVE_PATH.as_posix(),
    _TEE_COMPLETE_RELATIVE_PATH.as_posix(),
    "metrics.json",
    "path_audit.json",
    "reproducibility.json",
    "run_manifest.json",
    "smoke_checkpoint.pt",
    "SUCCESS",
}
_REQUIRED_METADATA = (
    "git_commit",
    "config_sha256",
    "selected_patch_manifest_sha256",
    "seed",
    "model_class",
    "synthetic",
)
_BLOSC = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
_EXPECTED_PERIODS = tuple(f"{year}Q{quarter}" for year in (2020, 2021) for quarter in range(1, 5))
_VMF_NORM_TOLERANCE = {"float32": 1.0e-5, "float16": 5.0e-4}


def _model_class_name(model: torch.nn.Module) -> str:
    cls = model.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _is_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _validate_checkpoint_metadata_schema(metadata: Mapping[str, object]) -> dict[str, object]:
    """校验 checkpoint provenance 自身的不可放宽字段。"""
    raw = dict(metadata)
    missing = [key for key in _REQUIRED_METADATA if key not in raw]
    if missing:
        raise ExportError(f"checkpoint metadata is missing required fields: {missing}")
    if not _is_hex(raw["git_commit"], 40):
        raise ExportError("checkpoint metadata git_commit must be a 40-character hexadecimal SHA")
    for key in ("config_sha256", "selected_patch_manifest_sha256"):
        if not _is_hex(raw[key], 64):
            raise ExportError(f"checkpoint metadata {key} must be a SHA-256")
    if type(raw["seed"]) is not int:
        raise ExportError("checkpoint metadata seed must be an integer")
    if not isinstance(raw["model_class"], str) or not raw["model_class"]:
        raise ExportError("checkpoint metadata model_class must be a non-empty string")
    if raw["synthetic"] is not True:
        raise ExportError("checkpoint metadata must declare synthetic=true")
    for key in ("formal_training_allowed", "formal_evaluation_allowed"):
        if key in raw and raw[key] is not False:
            raise ExportError(f"checkpoint metadata {key} must be false when present")
    return raw


def _validate_checkpoint_metadata(
    metadata: Mapping[str, object], model: torch.nn.Module
) -> dict[str, object]:
    """严格校验足以绑定权重和本次 smoke 输入的 provenance。"""
    raw = _validate_checkpoint_metadata_schema(metadata)
    if raw["model_class"] != _model_class_name(model):
        raise ExportError("checkpoint metadata model_class does not match the model")
    return raw


def _exactly_equal(left: object, right: object) -> bool:
    """比较 provenance 时不允许 bool/int 等 Python 宽松相等绕过合同。"""
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        return left.keys() == right.keys() and all(
            _exactly_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _exactly_equal(item_left, item_right) for item_left, item_right in zip(left, right)
        )
    return left == right


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_without_resolving(path: Path) -> Path:
    """绝对化路径但不跟随 symlink，令后续检查能看到损坏 link。"""
    return Path(os.path.abspath(path))


def _guard_non_symlink_sandbox_path(path: Path, sandbox_root: Path) -> Path:
    """仅接受沙箱内、各既有路径节点均非 symlink 的路径。"""
    root = _absolute_without_resolving(sandbox_root)
    candidate = _absolute_without_resolving(path)
    if root.is_symlink():
        raise ExportError(f"sandbox root must not be a symlink: {root}")
    for ancestor in root.parents:
        if ancestor.is_symlink():
            raise ExportError(f"sandbox ancestor must not be a symlink: {ancestor}")
    sentinel = root / SENTINEL
    if sentinel.is_symlink() or not sentinel.is_file():
        raise ExportError(f"sandbox sentinel must be a regular file: {sentinel}")
    if not candidate.is_relative_to(root):
        # Keep Task 1's canonical boundary validation load-bearing.
        validate_write_path(candidate, root)
        raise ExportError(f"path escapes sandbox: {candidate}")
    current = root
    for component in candidate.relative_to(root).parts:
        current = current / component
        if current.is_symlink():
            raise ExportError(f"symlink paths are forbidden in the smoke sandbox: {current}")
    validate_write_path(candidate, root)
    return candidate


def _checkpoint_path_and_root(path: Path) -> tuple[Path, Path]:
    """从未 resolve 的 checkpoint 路径向上查找带常规 sentinel 的沙箱。"""
    candidate = _absolute_without_resolving(path)
    for root in candidate.parents:
        sentinel = root / SENTINEL
        if sentinel.is_symlink():
            raise ExportError(f"sandbox sentinel must not be a symlink: {sentinel}")
        if sentinel.is_file():
            return _guard_non_symlink_sandbox_path(candidate, root), root
    raise ExportError("smoke checkpoint path is not inside a sentinel sandbox")


def _validate_group_name(group: str) -> None:
    if group not in _GROUPS:
        raise ExportError(f"unknown smoke output group: {group!r}")


def _validate_export_inputs(
    embedding: torch.Tensor,
    valid: torch.Tensor,
    patch_ids: Sequence[str],
    periods: Sequence[str],
) -> None:
    if not isinstance(embedding, torch.Tensor) or embedding.shape != _EMBEDDING_SHAPE:
        raise ExportError(f"embedding must have exact shape {_EMBEDDING_SHAPE}")
    if not embedding.is_floating_point():
        raise ExportError("embedding must be a floating-point tensor")
    if not bool(torch.isfinite(embedding).all()):
        raise ExportError("embedding contains NaN or Inf")
    if not isinstance(valid, torch.Tensor) or valid.shape != _VALID_SHAPE:
        raise ExportError(f"valid must have exact shape {_VALID_SHAPE}")
    if valid.dtype is not torch.bool:
        raise ExportError("valid must be a bool tensor")
    if len(patch_ids) != _EMBEDDING_SHAPE[0] or not all(
        isinstance(value, str) and value for value in patch_ids
    ):
        raise ExportError("patch_ids must contain four non-empty strings")
    if len(periods) != _EMBEDDING_SHAPE[1] or not all(
        isinstance(value, str) and value for value in periods
    ):
        raise ExportError("periods must contain eight non-empty strings")


def _string_array(values: Sequence[str]) -> np.ndarray:
    width = max(len(value) for value in values)
    return np.asarray(values, dtype=f"<U{width}")


def _required_group_paths(sandbox_root: Path) -> tuple[Path, ...]:
    return tuple(sandbox_root / "outputs" / group / "embedding.zarr" for group in _GROUPS)


def _assert_tree_has_no_symlinks(path: Path) -> None:
    """Zarr 目录内部也不得放置能把读取或哈希引出沙箱的 symlink。"""
    if path.is_symlink():
        raise ExportError(f"symlink Zarr path is forbidden: {path}")
    if not path.is_dir():
        raise ExportError(f"Zarr group must be a directory: {path}")
    _assert_directory_has_no_symlink_children(path)


def _assert_directory_has_no_symlink_children(path: Path) -> None:
    """递归检查任意将被读取或哈希的目录，拒绝其所有 symlink 子项。"""
    if path.is_symlink():
        raise ExportError(f"symlink directory is forbidden: {path}")
    if not path.is_dir():
        raise ExportError(f"required directory is missing: {path}")
    for child in path.rglob("*"):
        if child.is_symlink():
            raise ExportError(f"symlink child is forbidden in a hashed directory: {child}")


def _validate_identifier_array(
    exported: zarr.Group,
    name: str,
    expected: Sequence[str],
) -> None:
    array = exported[name]
    expected_values = list(expected)
    if array.shape != (len(expected_values),):
        raise ExportError(f"Zarr {name} shape is invalid")
    if array.dtype != _string_array(expected_values).dtype:
        raise ExportError(f"Zarr {name} dtype is invalid")
    if array[:].tolist() != expected_values:
        raise ExportError(f"Zarr {name} values do not match the export contract")


def _is_zstd_blosc(compressor: object) -> bool:
    return isinstance(compressor, Blosc) and compressor.cname == "zstd"


def _verify_exported_group(
    path: Path,
    expected_group: str | None = None,
    expected_patch_ids: Sequence[str] | None = None,
    expected_periods: Sequence[str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """在 rename 和 SUCCESS 前重新打开 Zarr，避免把坏目录当作成功产物。"""
    _assert_tree_has_no_symlinks(path)
    try:
        exported = zarr.open_group(str(path), mode="r")
    except Exception as exc:  # zarr reports several concrete error classes across releases.
        raise ExportError(f"cannot reopen exported Zarr group: {path}") from exc
    if set(exported.array_keys()) != {"embedding", "valid", "patch_id", "period"}:
        raise ExportError(f"Zarr array set is invalid: {path}")
    if exported["embedding"].shape != _EMBEDDING_SHAPE:
        raise ExportError(f"Zarr embedding shape is not {_EMBEDDING_SHAPE}: {path}")
    if exported["embedding"].dtype != np.dtype("float16"):
        raise ExportError(f"Zarr embedding dtype is not float16: {path}")
    if exported["embedding"].chunks != _EMBEDDING_CHUNKS:
        raise ExportError(f"Zarr embedding chunks are not {_EMBEDDING_CHUNKS}: {path}")
    if not _is_zstd_blosc(exported["embedding"].compressor):
        raise ExportError(f"Zarr embedding compressor must be Blosc Zstd: {path}")
    if exported["valid"].shape != _VALID_SHAPE or exported["valid"].dtype != np.dtype("bool"):
        raise ExportError(f"Zarr valid array contract is invalid: {path}")
    if exported["valid"].chunks != _VALID_CHUNKS:
        raise ExportError(f"Zarr valid chunks are not {_VALID_CHUNKS}: {path}")
    if not _is_zstd_blosc(exported["valid"].compressor):
        raise ExportError(f"Zarr valid compressor must be Blosc Zstd: {path}")
    attribute_patch_ids = exported.attrs.get("patch_ids")
    attribute_periods = exported.attrs.get("periods")
    if not isinstance(attribute_patch_ids, list) or not all(
        isinstance(value, str) for value in attribute_patch_ids
    ):
        raise ExportError(f"Zarr patch_ids attribute is invalid: {path}")
    if not isinstance(attribute_periods, list) or not all(
        isinstance(value, str) for value in attribute_periods
    ):
        raise ExportError(f"Zarr periods attribute is invalid: {path}")
    patch_ids = attribute_patch_ids if expected_patch_ids is None else list(expected_patch_ids)
    periods = attribute_periods if expected_periods is None else list(expected_periods)
    _validate_identifier_array(exported, "patch_id", patch_ids)
    _validate_identifier_array(exported, "period", periods)
    if expected_group is not None and exported.attrs.get("group") != expected_group:
        raise ExportError(f"Zarr group attribute does not match {expected_group!r}: {path}")
    if expected_periods is None and attribute_periods != list(_EXPECTED_PERIODS):
        raise ExportError(f"Zarr periods must be the fixed smoke quarters: {path}")
    if expected_patch_ids is not None and attribute_patch_ids != list(expected_patch_ids):
        raise ExportError(f"Zarr patch_ids attribute does not match the export contract: {path}")
    if expected_periods is not None and attribute_periods != list(expected_periods):
        raise ExportError(f"Zarr periods attribute does not match the export contract: {path}")
    if exported.attrs.get("synthetic") is not True:
        raise ExportError(f"Zarr must declare synthetic=true: {path}")
    if exported.attrs.get("allowed_use") != "smoke_test_only":
        raise ExportError(f"Zarr must be limited to smoke_test_only: {path}")
    for key in ("formal_training_allowed", "formal_evaluation_allowed"):
        if exported.attrs.get(key) is not False:
            raise ExportError(f"Zarr {key} must be false: {path}")
    return tuple(patch_ids), tuple(periods)


def reopened_fp16_vmf_norm_summary(path: Path) -> dict[str, object]:
    """重开已导出的 FP16 Zarr，并以 FP32 累加计算实际持久化向量范数。"""
    target = Path(path)
    _verify_exported_group(target)
    exported = zarr.open_group(str(target), mode="r")
    stored = exported["embedding"][:]
    if stored.dtype != np.dtype("float16"):
        raise ExportError(f"reopened Zarr embedding is not float16: {target}")
    values = stored.astype(np.float32)
    norms = np.sqrt(np.square(values).sum(axis=2))
    if not np.isfinite(norms).all():
        raise ExportError(f"reopened Zarr vMF norms contain NaN or Inf: {target}")
    summary = {
        "source_dtype": "float16",
        "computation_dtype": "float32",
        "min": float(norms.min()),
        "median": float(np.median(norms)),
        "max": float(norms.max()),
    }
    _validate_norm_summary(summary, source_dtype="float16")
    return summary


def export_group_zarr(
    group: str,
    embedding: torch.Tensor,
    valid: torch.Tensor,
    patch_ids: Sequence[str],
    periods: Sequence[str],
    output: Path,
    sandbox_root: Path,
) -> Path:
    """原子写出一个完整季度组；失败时只保留可诊断的 ``.partial`` 目录。"""
    _validate_group_name(group)
    _validate_export_inputs(embedding, valid, patch_ids, periods)
    target = _guard_non_symlink_sandbox_path(Path(output), sandbox_root)
    if target.name != "embedding.zarr":
        raise ExportError("smoke Zarr outputs must be named embedding.zarr")
    partial = _guard_non_symlink_sandbox_path(
        target.with_name(f"{target.name}.partial"), sandbox_root
    )
    if target.exists():
        raise ExportError(f"refusing to overwrite sealed Zarr output: {target}")
    if partial.exists():
        raise ExportError(f"refusing to overwrite diagnostic partial Zarr output: {partial}")
    target.parent.mkdir(parents=True, exist_ok=True)

    embedding_array = embedding.detach().to(device="cpu", dtype=torch.float16).contiguous().numpy()
    valid_array = valid.detach().to(device="cpu", dtype=torch.bool).contiguous().numpy()
    exported = zarr.open_group(str(partial), mode="w", zarr_version=2)
    exported.create_dataset(
        "embedding",
        data=embedding_array,
        shape=_EMBEDDING_SHAPE,
        chunks=_EMBEDDING_CHUNKS,
        compressor=_BLOSC,
        overwrite=False,
    )
    exported.create_dataset(
        "valid",
        data=valid_array,
        shape=_VALID_SHAPE,
        chunks=_VALID_CHUNKS,
        compressor=_BLOSC,
        overwrite=False,
    )
    exported.create_dataset("patch_id", data=_string_array(patch_ids), shape=(4,), overwrite=False)
    exported.create_dataset("period", data=_string_array(periods), shape=(8,), overwrite=False)
    exported.attrs.update(
        {
            "group": group,
            "synthetic": True,
            "allowed_use": "smoke_test_only",
            "formal_training_allowed": False,
            "formal_evaluation_allowed": False,
            "patch_ids": list(patch_ids),
            "periods": list(periods),
        }
    )
    _verify_exported_group(
        partial,
        expected_group=group,
        expected_patch_ids=patch_ids,
        expected_periods=periods,
    )
    _guard_non_symlink_sandbox_path(partial, sandbox_root)
    _guard_non_symlink_sandbox_path(target, sandbox_root)
    partial.replace(target)
    return target


def save_smoke_checkpoint(
    model: torch.nn.Module,
    path: Path,
    metadata: Mapping[str, object],
) -> str:
    """以精确 provenance 和原子 replace 保存仅用于 reload 检查的 checkpoint。"""
    checked_metadata = _validate_checkpoint_metadata(metadata, model)
    target, sandbox_root = _checkpoint_path_and_root(Path(path))
    partial = _guard_non_symlink_sandbox_path(
        target.with_name(f"{target.name}.partial"), sandbox_root
    )
    if target.exists():
        raise ExportError(f"refusing to overwrite smoke checkpoint: {target}")
    if target.suffix != ".pt":
        raise ExportError("smoke checkpoint must use a .pt suffix")
    target.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists():
        raise ExportError(f"refusing to overwrite diagnostic partial checkpoint: {partial}")
    torch.save({"state_dict": model.state_dict(), "metadata": checked_metadata}, partial)
    _guard_non_symlink_sandbox_path(partial, sandbox_root)
    _guard_non_symlink_sandbox_path(target, sandbox_root)
    partial.replace(target)
    return _sha256_file(target)


def _validate_checkpoint_payload(
    raw: object,
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, object]]:
    """仅接受 weights-only 可解析的最小 checkpoint payload 结构。"""
    if not isinstance(raw, Mapping) or set(raw) != {"state_dict", "metadata"}:
        raise ExportError("smoke checkpoint payload has an invalid schema")
    state_dict = raw["state_dict"]
    metadata = raw["metadata"]
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ExportError("smoke checkpoint state_dict has an invalid schema")
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in state_dict.items()
    ):
        raise ExportError("smoke checkpoint state_dict must map strings to tensors")
    if not isinstance(metadata, Mapping):
        raise ExportError("smoke checkpoint metadata has an invalid schema")
    _validate_checkpoint_metadata_schema(metadata)
    return state_dict, metadata


def _load_checkpoint_payload(
    checkpoint: Path,
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, object]]:
    """禁用 legacy pickle 后读取 checkpoint，随后进行显式结构校验。"""
    try:
        raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ExportError(f"cannot safely load smoke checkpoint: {checkpoint}") from exc
    return _validate_checkpoint_payload(raw)


def _validate_state_dict_for_model(
    state_dict: Mapping[str, torch.Tensor], model: torch.nn.Module
) -> None:
    """在任何参数写入前验证 keys、shape、dtype 和 tensor 类型。"""
    expected = model.state_dict()
    if set(state_dict) != set(expected):
        raise ExportError("smoke checkpoint state_dict keys do not exactly match the model")
    for key, target in expected.items():
        value = state_dict[key]
        if not isinstance(value, torch.Tensor):
            raise ExportError(f"smoke checkpoint state_dict value is not a tensor: {key}")
        if value.layout != torch.strided or target.layout != torch.strided:
            raise ExportError(f"smoke checkpoint state_dict tensor must be dense strided: {key}")
        if value.device.type != "cpu":
            raise ExportError(f"smoke checkpoint state_dict tensor must be loaded onto CPU: {key}")
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ExportError(f"smoke checkpoint state_dict tensor contract is invalid: {key}")


def _snapshot_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """在应用 checkpoint 前保存值副本，供异常路径直接恢复而不调用模块 hook。"""
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _restore_model_state(model: torch.nn.Module, snapshot: Mapping[str, torch.Tensor]) -> None:
    """直接 copy 回 state tensors，避免失败模型的 ``load_state_dict`` 再次执行副作用。"""
    current = model.state_dict()
    if set(current) != set(snapshot):
        raise ExportError("model state changed while applying smoke checkpoint")
    with torch.no_grad():
        for key, value in current.items():
            value.copy_(snapshot[key])


def load_smoke_checkpoint(
    model: torch.nn.Module,
    path: Path,
    expected_metadata: Mapping[str, object],
) -> None:
    """只在文件 metadata 与调用方 provenance 完全相同时恢复权重。"""
    expected = _validate_checkpoint_metadata(expected_metadata, model)
    checkpoint, _sandbox_root = _checkpoint_path_and_root(Path(path))
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise ExportError(f"smoke checkpoint is missing: {checkpoint}")
    state_dict, saved_metadata = _load_checkpoint_payload(checkpoint)
    _validate_checkpoint_metadata(saved_metadata, model)
    if not _exactly_equal(dict(saved_metadata), expected):
        raise ExportError("smoke checkpoint metadata does not exactly match expected metadata")
    _validate_state_dict_for_model(state_dict, model)
    snapshot = _snapshot_model_state(model)
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        _restore_model_state(model, snapshot)
        raise ExportError("smoke checkpoint state_dict could not be applied") from exc


def _resolve_required_path(path: Path | str, sandbox_root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = sandbox_root / candidate
    guarded = _guard_non_symlink_sandbox_path(candidate, sandbox_root)
    if guarded.is_dir():
        _assert_directory_has_no_symlink_children(guarded)
    return guarded


def _require_regular_evidence_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ExportError(f"required smoke evidence must be a regular non-symlink file: {path}")


def _finite_number(value: object) -> bool:
    return type(value) in {int, float} and bool(np.isfinite(value))


def _validate_norm_summary(raw: object, *, source_dtype: str) -> None:
    expected = {"source_dtype", "computation_dtype", "min", "median", "max"}
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise ExportError(f"{source_dtype} vMF norm evidence has an invalid schema")
    if raw["source_dtype"] != source_dtype or raw["computation_dtype"] != "float32":
        raise ExportError(f"{source_dtype} vMF norm dtype provenance is invalid")
    values = [raw[key] for key in ("min", "median", "max")]
    if not all(_finite_number(value) for value in values) or not (
        values[0] <= values[1] <= values[2]
    ):
        raise ExportError(f"{source_dtype} vMF norm values are invalid")
    tolerance = _VMF_NORM_TOLERANCE.get(source_dtype)
    if tolerance is None or any(abs(float(value) - 1.0) > tolerance for value in values):
        raise ExportError(f"{source_dtype} vMF norm exceeds the unit-vector tolerance")


def _validate_metrics_evidence(raw: Mapping[str, object]) -> None:
    expected_keys = {
        "groups",
        "zero_gate_max_abs_error",
        "gradient_l1",
        "checkpoint_reload_max_abs_error",
        "accuracy_conclusion_allowed",
    }
    if set(raw) != expected_keys or raw["accuracy_conclusion_allowed"] is not False:
        raise ExportError("metrics evidence has an invalid schema")
    groups = raw["groups"]
    if not isinstance(groups, Mapping) or set(groups) != set(_GROUPS):
        raise ExportError("metrics groups have an invalid schema")
    expected_group_keys = {
        "shape",
        "latency_seconds",
        "finite",
        "pre_export_fp32_vmf_norm",
        "reopened_fp16_zarr_vmf_norm",
        "npu_peak_memory",
    }
    memory_keys = {
        "unit",
        "baseline_allocated",
        "baseline_reserved",
        "peak_allocated",
        "peak_reserved",
        "peak_allocated_delta",
        "peak_reserved_delta",
    }
    for group in _GROUPS:
        evidence = groups[group]
        if not isinstance(evidence, Mapping):
            raise ExportError(f"metrics group {group} has an invalid schema")
        if "npu_peak_memory" not in evidence:
            raise ExportError(f"metrics group {group} is missing NPU peak memory evidence")
        if set(evidence) != expected_group_keys:
            raise ExportError(f"metrics group {group} has an invalid schema")
        if evidence["shape"] != list(_EMBEDDING_SHAPE) or evidence["finite"] is not True:
            raise ExportError(f"metrics group {group} shape/finite evidence is invalid")
        if not _finite_number(evidence["latency_seconds"]) or evidence["latency_seconds"] <= 0:
            raise ExportError(f"metrics group {group} latency is invalid")
        _validate_norm_summary(evidence["pre_export_fp32_vmf_norm"], source_dtype="float32")
        _validate_norm_summary(evidence["reopened_fp16_zarr_vmf_norm"], source_dtype="float16")
        memory = evidence["npu_peak_memory"]
        if not isinstance(memory, Mapping) or set(memory) != memory_keys:
            raise ExportError(f"metrics group {group} NPU peak memory evidence is invalid")
        if memory["unit"] != "bytes":
            raise ExportError(f"metrics group {group} NPU peak memory unit must be bytes")
        numbers = {key: memory[key] for key in memory_keys - {"unit"}}
        if not all(type(value) is int and value >= 0 for value in numbers.values()):
            raise ExportError(f"metrics group {group} NPU peak memory values are invalid")
        if (
            numbers["peak_allocated"] < numbers["baseline_allocated"]
            or numbers["peak_reserved"] < numbers["baseline_reserved"]
            or numbers["peak_allocated_delta"]
            != numbers["peak_allocated"] - numbers["baseline_allocated"]
            or numbers["peak_reserved_delta"]
            != numbers["peak_reserved"] - numbers["baseline_reserved"]
        ):
            raise ExportError(f"metrics group {group} NPU peak memory deltas are invalid")
    if raw["zero_gate_max_abs_error"] != 0.0 or raw["checkpoint_reload_max_abs_error"] != 0.0:
        raise ExportError("metrics exact-identity evidence is invalid")
    gradients = raw["gradient_l1"]
    expected_gradients = {
        "aef_adapter",
        "aef_gate",
        "highres_stem",
        "highres_adapter",
        "highres_gate",
        "output_projection",
    }
    if (
        not isinstance(gradients, Mapping)
        or set(gradients) != expected_gradients
        or not all(_finite_number(value) and value > 0 for value in gradients.values())
    ):
        raise ExportError("metrics gradient evidence is invalid")


def _validate_run_manifest_evidence(raw: Mapping[str, object], sandbox_root: Path) -> None:
    expected_keys = {
        "git_commit",
        "config_sha256",
        "patch_ids",
        "periods",
        "physical_npu",
        "logical_device",
        "source_unchanged",
        "synthetic",
        "allowed_use",
        "formal_training_allowed",
        "formal_evaluation_allowed",
        "accuracy_conclusion_allowed",
        "runtime_provenance",
        "cpu_contract_fallback",
    }
    if set(raw) != expected_keys:
        raise ExportError("run manifest provenance has an invalid schema")
    if not _is_hex(raw["git_commit"], 40) or not _is_hex(raw["config_sha256"], 64):
        raise ExportError("run manifest provenance hashes are invalid")
    if (
        not isinstance(raw["patch_ids"], list)
        or len(raw["patch_ids"]) != 4
        or len(set(raw["patch_ids"])) != 4
        or raw["periods"] != list(_EXPECTED_PERIODS)
    ):
        raise ExportError("run manifest patch/period provenance is invalid")
    if raw["physical_npu"] != 2 or raw["logical_device"] != "npu:0":
        raise ExportError("run manifest NPU mapping provenance is invalid")
    if (
        raw["source_unchanged"] is not True
        or raw["synthetic"] is not True
        or raw["allowed_use"] != "smoke_test_only"
        or raw["formal_training_allowed"] is not False
        or raw["formal_evaluation_allowed"] is not False
        or raw["accuracy_conclusion_allowed"] is not False
    ):
        raise ExportError("run manifest use-policy provenance is invalid")

    runtime = raw["runtime_provenance"]
    runtime_keys = {
        "sys_executable",
        "python_version",
        "torch_version",
        "torch_npu_version",
        "cann",
        "driver",
        "device_mapping",
        "module_paths",
        "occupancy_checks",
    }
    if not isinstance(runtime, Mapping) or set(runtime) != runtime_keys:
        raise ExportError("run manifest runtime provenance is invalid")
    for key in ("sys_executable", "python_version", "torch_version", "torch_npu_version"):
        if not isinstance(runtime[key], str) or not runtime[key]:
            raise ExportError(f"run manifest runtime provenance {key} is invalid")
    if not Path(runtime["sys_executable"]).is_absolute():
        raise ExportError("run manifest sys.executable provenance must be absolute")
    cann = runtime["cann"]
    if not isinstance(cann, Mapping) or set(cann) != {"root", "version", "install_info"}:
        raise ExportError("run manifest CANN provenance is invalid")
    if cann["root"] != "/usr/local/Ascend/cann-9.0.0" or cann["version"] != "9.0.0":
        raise ExportError("run manifest CANN version provenance is invalid")
    if not isinstance(cann["install_info"], str) or not Path(cann["install_info"]).is_absolute():
        raise ExportError("run manifest CANN install-info provenance is invalid")
    driver = runtime["driver"]
    if not isinstance(driver, Mapping) or set(driver) != {"version", "version_info"}:
        raise ExportError("run manifest driver provenance is invalid")
    if not all(isinstance(driver[key], str) and driver[key] for key in driver):
        raise ExportError("run manifest driver values are invalid")
    if not Path(driver["version_info"]).is_absolute():
        raise ExportError("run manifest driver path provenance must be absolute")
    mapping = runtime["device_mapping"]
    expected_mapping = {
        "physical_device": "/dev/davinci2",
        "physical_npu": 2,
        "visible_devices": "2",
        "logical_device": "npu:0",
        "logical_device_count": 1,
    }
    if not isinstance(mapping, Mapping) or set(mapping) != {*expected_mapping, "device_name"}:
        raise ExportError("run manifest device mapping provenance is invalid")
    if (
        any(mapping[key] != value for key, value in expected_mapping.items())
        or not isinstance(mapping["device_name"], str)
        or not mapping["device_name"]
    ):
        raise ExportError("run manifest device mapping provenance is invalid")
    module_paths = runtime["module_paths"]
    if not isinstance(module_paths, Mapping) or set(module_paths) != {
        "torch",
        "torch_npu",
        "runner",
        "model",
    }:
        raise ExportError("run manifest module-path provenance is invalid")
    if not all(
        isinstance(value, str) and Path(value).is_absolute() for value in module_paths.values()
    ):
        raise ExportError("run manifest module paths must be absolute")
    occupancy = runtime["occupancy_checks"]
    expected_occupancy = [
        {"stage": "launcher_preload", "idle": True},
        {"stage": "post_load_pre_set_device", "idle": True},
    ]
    if not _exactly_equal(occupancy, expected_occupancy):
        raise ExportError("run manifest NPU occupancy evidence is invalid")

    fallback = raw["cpu_contract_fallback"]
    if not isinstance(fallback, Mapping) or set(fallback) != {"path", "sha256"}:
        raise ExportError("run manifest CPU contract fallback provenance is invalid")
    if fallback["path"] != "manifests/cpu_contract.json" or not _is_hex(fallback["sha256"], 64):
        raise ExportError("run manifest CPU contract fallback hash is invalid")
    cpu_contract = _guard_non_symlink_sandbox_path(sandbox_root / fallback["path"], sandbox_root)
    _require_regular_evidence_file(cpu_contract)
    if _sha256_file(cpu_contract) != fallback["sha256"]:
        raise ExportError("run manifest CPU contract fallback hash does not match evidence")


def _read_json_evidence(path: Path, sandbox_root: Path) -> dict[str, object]:
    guarded = _guard_non_symlink_sandbox_path(path, sandbox_root)
    _require_regular_evidence_file(guarded)
    try:
        raw = json.loads(guarded.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExportError(f"semantic evidence is malformed: {guarded}") from exc
    if not isinstance(raw, dict):
        raise ExportError(f"semantic evidence must be a JSON object: {guarded}")
    return raw


def _strict_smoke_policy(raw: Mapping[str, object], *, highres: bool = False) -> None:
    expected = {
        "synthetic": True,
        "allowed_use": "smoke_test_only",
        "formal_training_allowed": False,
        "formal_evaluation_allowed": False,
    }
    if any(not _exactly_equal(raw.get(key), value) for key, value in expected.items()):
        raise ExportError("semantic registry use-policy flags are invalid")
    expected_kind = (
        "annual_s2_rgb_5x_deterministic_texture" if highres else "annual_s2_fixed_projection"
    )
    if raw.get("synthetic_kind") != expected_kind:
        raise ExportError("semantic registry synthetic kind is invalid")
    if highres and (
        raw.get("claimed_native_gsd_m") is not None
        or raw.get("model_input_gsd_m") != 2
        or raw.get("contains_real_2m_information") is not False
    ):
        raise ExportError("semantic highres registry claims unsupported real 2 m information")


def _load_safe_tensor_payload(path: Path, sandbox_root: Path) -> dict[str, object]:
    guarded = _guard_non_symlink_sandbox_path(path, sandbox_root)
    _require_regular_evidence_file(guarded)
    try:
        payload = torch.load(guarded, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ExportError(f"semantic tensor evidence cannot be loaded safely: {guarded}") from exc
    if not isinstance(payload, dict) or not payload:
        raise ExportError(f"semantic tensor evidence must be a non-empty mapping: {guarded}")
    return payload


def _validate_registry_graph(
    sandbox_root: Path,
    patch_ids: tuple[str, ...],
) -> None:
    expected_pairs = tuple((patch_id, year) for patch_id in patch_ids for year in (2020, 2021))
    manifest_root = sandbox_root / "manifests"
    registries = (
        ("aef", _read_json_evidence(manifest_root / "aef_registry.json", sandbox_root), False),
        (
            "highres_2m",
            _read_json_evidence(manifest_root / "highres_2m_registry.json", sandbox_root),
            True,
        ),
    )
    for kind, registry, highres in registries:
        _strict_smoke_policy(registry, highres=highres)
        entries = registry.get("entries")
        if not isinstance(entries, list) or len(entries) != 8:
            raise ExportError(f"semantic {kind} registry must contain exactly eight entries")
        observed_pairs: list[tuple[object, object]] = []
        metadata = {
            key: value for key, value in registry.items() if key not in {"entries", "projection"}
        }
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping) or set(entry) != {
                "patch_id",
                "year",
                "path",
                "sha256",
            }:
                raise ExportError(f"semantic {kind} registry entry schema is invalid")
            pair = (entry["patch_id"], entry["year"])
            observed_pairs.append(pair)
            expected_path = f"synthetic/{kind}/patch_{index:02d}_{expected_pairs[index][1]}.pt"
            if pair != expected_pairs[index] or entry["path"] != expected_path:
                raise ExportError(f"semantic {kind} registry patch-year axis is invalid")
            if not _is_hex(entry["sha256"], 64):
                raise ExportError(f"semantic {kind} registry checksum is invalid")
            cache = _guard_non_symlink_sandbox_path(sandbox_root / str(entry["path"]), sandbox_root)
            if _sha256_file(cache) != entry["sha256"]:
                raise ExportError(f"semantic {kind} registry checksum does not match cache")
            payload = _load_safe_tensor_payload(cache, sandbox_root)
            if (
                payload.get("patch_id") != pair[0]
                or payload.get("year") != pair[1]
                or not _exactly_equal(payload.get("metadata"), metadata)
            ):
                raise ExportError(f"semantic {kind} cache provenance is invalid")
        if tuple(observed_pairs) != expected_pairs or len(set(observed_pairs)) != 8:
            raise ExportError(f"semantic {kind} registry does not cover eight patch-years")

    aef_registry = registries[0][1]
    projection = aef_registry.get("projection")
    expected_projection_keys = {"path", "sha256", "seed", "shape", "dtype"}
    if not isinstance(projection, Mapping) or set(projection) != expected_projection_keys:
        raise ExportError("semantic AEF projection provenance is invalid")
    if (
        projection.get("path") != "synthetic/aef/fixed_projection_10x64.pt"
        or projection.get("seed") != 20260815
        or projection.get("shape") != [64, 10]
        or projection.get("dtype") != "float32"
        or not _is_hex(projection.get("sha256"), 64)
    ):
        raise ExportError("semantic AEF projection provenance is invalid")
    projection_path = sandbox_root / str(projection["path"])
    if _sha256_file(projection_path) != projection["sha256"]:
        raise ExportError("semantic AEF projection checksum does not match")
    projection_payload = _load_safe_tensor_payload(projection_path, sandbox_root)
    tensor = projection_payload.get("projection")
    if (
        not isinstance(tensor, torch.Tensor)
        or tuple(tensor.shape) != (64, 10)
        or tensor.dtype != torch.float32
        or not bool(torch.isfinite(tensor).all())
        or projection_payload.get("seed") != 20260815
    ):
        raise ExportError("semantic AEF projection tensor is invalid")


def _validate_prepare_and_cpu_graph(
    sandbox_root: Path,
    patch_ids: tuple[str, ...],
    selection_sha256: str,
) -> None:
    manifests = sandbox_root / "manifests"
    prepare = _read_json_evidence(manifests / "prepare_manifest.json", sandbox_root)
    expected_prepare_keys = {
        "patch_ids",
        "patch_years",
        "years",
        "source_archive_count",
        "source_unchanged",
        "selection_manifest_sha256",
        "sandbox_bytes_before_path_audit",
        "synthetic",
        "allowed_use",
        "formal_training_allowed",
        "formal_evaluation_allowed",
        "accuracy_conclusion_allowed",
    }
    if set(prepare) != expected_prepare_keys:
        raise ExportError("semantic prepare manifest schema is invalid")
    if (
        prepare["patch_ids"] != list(patch_ids)
        or prepare["patch_years"] != 8
        or prepare["years"] != [2020, 2021]
        or prepare["source_archive_count"] != 48
        or prepare["source_unchanged"] is not True
        or prepare["selection_manifest_sha256"] != selection_sha256
        or type(prepare["sandbox_bytes_before_path_audit"]) is not int
        or prepare["sandbox_bytes_before_path_audit"] < 0
        or prepare["synthetic"] is not True
        or prepare["allowed_use"] != "smoke_test_only"
        or prepare["formal_training_allowed"] is not False
        or prepare["formal_evaluation_allowed"] is not False
        or prepare["accuracy_conclusion_allowed"] is not False
    ):
        raise ExportError("semantic prepare manifest provenance is invalid")

    before = _read_json_evidence(manifests / "source_snapshot_before.json", sandbox_root)
    after = _read_json_evidence(manifests / "source_snapshot_after.json", sandbox_root)
    if not _exactly_equal(before, after) or set(before) != {
        "archives",
        "directories",
        "hashing_performed",
    }:
        raise ExportError("semantic source snapshots differ")
    if (
        before["hashing_performed"] is not False
        or not isinstance(before["archives"], Mapping)
        or len(before["archives"]) != 48
        or not isinstance(before["directories"], Mapping)
    ):
        raise ExportError("semantic source snapshot coverage is invalid")

    cpu = _read_json_evidence(manifests / "cpu_contract.json", sandbox_root)
    expected_cpu_keys = {
        "fixture",
        "groups",
        "full_zero_gate_matches_base",
        "synthetic",
        "allowed_use",
        "formal_training_allowed",
        "formal_evaluation_allowed",
        "accuracy_conclusion_allowed",
    }
    if set(cpu) != expected_cpu_keys or cpu["fixture"] not in {"small", "full-shape"}:
        raise ExportError("semantic CPU contract schema is invalid")
    groups = cpu.get("groups")
    if not isinstance(groups, Mapping) or set(groups) != set(_GROUPS):
        raise ExportError("semantic CPU contract must contain the exact four groups")
    for group in _GROUPS:
        value = groups[group]
        if not isinstance(value, Mapping) or set(value) != {
            "shape",
            "dtype",
            "finite",
            "max_vmf_norm_error",
        }:
            raise ExportError("semantic CPU contract group schema is invalid")
        shape = value["shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != 5
            or shape[1] != 4
            or shape[2] != 64
            or shape[3] != shape[4]
            or value["dtype"] != "torch.float32"
            or value["finite"] is not True
            or not _finite_number(value["max_vmf_norm_error"])
            or not 0.0 <= value["max_vmf_norm_error"] <= 1.0e-5
        ):
            raise ExportError("semantic CPU contract group value is invalid")
    if (
        cpu["full_zero_gate_matches_base"] is not True
        or cpu["synthetic"] is not True
        or cpu["allowed_use"] != "smoke_test_only"
        or cpu["formal_training_allowed"] is not False
        or cpu["formal_evaluation_allowed"] is not False
        or cpu["accuracy_conclusion_allowed"] is not False
    ):
        raise ExportError("semantic CPU identity/use-policy evidence is invalid")


def _validate_markers(sandbox_root: Path, git_commit: str, config_sha256: str) -> None:
    preliminary_path = sandbox_root / "manifests/preliminary_path_audit.json"
    preliminary = _read_json_evidence(preliminary_path, sandbox_root)
    if preliminary.get("stage") != "npu-smoke" or preliminary.get("stages") != [
        "prepare",
        "cpu-contract",
        "npu-smoke",
    ]:
        raise ExportError("semantic preliminary audit stage is invalid")
    ready_path = sandbox_root / _READY_TO_SEAL_RELATIVE_PATH
    ready = _read_json_evidence(ready_path, sandbox_root)
    expected_ready_keys = {
        "status",
        "git_commit",
        "config_sha256",
        "preliminary_path_audit",
        "source_unchanged",
        "synthetic",
        "formal_training_allowed",
        "formal_evaluation_allowed",
    }
    binding = {
        "path": "manifests/preliminary_path_audit.json",
        "sha256": _sha256_file(preliminary_path),
    }
    if set(ready) != expected_ready_keys or not _exactly_equal(
        ready,
        {
            "status": "npu_compute_complete",
            "git_commit": git_commit,
            "config_sha256": config_sha256,
            "preliminary_path_audit": binding,
            "source_unchanged": True,
            "synthetic": True,
            "formal_training_allowed": False,
            "formal_evaluation_allowed": False,
        },
    ):
        raise ExportError("semantic READY_TO_SEAL provenance is invalid")
    tee = _read_json_evidence(sandbox_root / _TEE_COMPLETE_RELATIVE_PATH, sandbox_root)
    log = sandbox_root / _LAUNCHER_LOG_RELATIVE_PATH
    stat_result = log.stat()
    log_metadata = {
        "path": _LAUNCHER_LOG_RELATIVE_PATH.as_posix(),
        "size": stat_result.st_size,
        "mtime_ns": stat_result.st_mtime_ns,
        "sha256": _sha256_file(log),
    }
    expected_tee = {
        "status": "tee_pipeline_complete",
        "git_commit": git_commit,
        "config_sha256": config_sha256,
        "ready_to_seal_sha256": _sha256_file(ready_path),
        "launcher_log": log_metadata,
        "synthetic": True,
        "formal_training_allowed": False,
        "formal_evaluation_allowed": False,
    }
    if not _exactly_equal(tee, expected_tee):
        raise ExportError("semantic TEE_COMPLETE/log provenance is invalid")


def _validate_semantic_evidence_graph(sandbox_root: Path) -> None:
    selection_path = sandbox_root / _PATCH_SELECTION_RELATIVE_PATH
    patch_ids = _required_patch_selection_ids(sandbox_root)
    selection_sha = _sha256_file(selection_path)
    run = _read_json_evidence(sandbox_root / "run_manifest.json", sandbox_root)
    metrics = _read_json_evidence(sandbox_root / "metrics.json", sandbox_root)
    reproducibility = _read_json_evidence(sandbox_root / "reproducibility.json", sandbox_root)
    _validate_run_manifest_evidence(run, sandbox_root)
    _validate_metrics_evidence(metrics)
    if run["patch_ids"] != list(patch_ids):
        raise ExportError("semantic run/selection patch axis differs")

    checkpoint_path = sandbox_root / "smoke_checkpoint.pt"
    state_dict, metadata = _load_checkpoint_payload(checkpoint_path)
    expected_metadata_keys = {
        "git_commit",
        "config_sha256",
        "selected_patch_manifest_sha256",
        "seed",
        "model_class",
        "synthetic",
        "formal_training_allowed",
        "formal_evaluation_allowed",
    }
    if set(metadata) != expected_metadata_keys:
        raise ExportError("semantic checkpoint metadata schema is invalid")
    if (
        metadata["git_commit"] != run["git_commit"]
        or metadata["config_sha256"] != run["config_sha256"]
        or metadata["selected_patch_manifest_sha256"] != selection_sha
        or metadata["seed"] != 20260815
        or metadata["model_class"]
        != "experiments.china_v1_fusion_smoke.model.IsolatedFusionSmokeModel"
        or metadata["synthetic"] is not True
        or metadata["formal_training_allowed"] is not False
        or metadata["formal_evaluation_allowed"] is not False
    ):
        raise ExportError("semantic checkpoint provenance differs from run/selection evidence")
    from experiments.china_v1_fusion_smoke.model import IsolatedFusionSmokeModel

    expected_model = IsolatedFusionSmokeModel(embed_dim=64)
    _validate_state_dict_for_model(state_dict, expected_model)

    expected_repro = {
        "fixed_seed": 20260815,
        "max_abs_error": 0.0,
        "matches": True,
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "synthetic": True,
    }
    if not _exactly_equal(reproducibility, expected_repro):
        raise ExportError("semantic reproducibility evidence is invalid")

    _validate_registry_graph(sandbox_root, patch_ids)
    _validate_prepare_and_cpu_graph(sandbox_root, patch_ids, selection_sha)
    _validate_markers(
        sandbox_root,
        str(run["git_commit"]),
        str(run["config_sha256"]),
    )
    for group in _GROUPS:
        actual = reopened_fp16_vmf_norm_summary(sandbox_root / "outputs" / group / "embedding.zarr")
        reported = metrics["groups"][group]["reopened_fp16_zarr_vmf_norm"]
        if not _exactly_equal(actual, reported):
            raise ExportError(f"semantic persisted norm differs from metrics for {group}")


def _validate_required_evidence(sandbox_root: Path) -> None:
    """封存前要求四份 JSON 与 checkpoint 都是可安全解析的常规文件。"""
    for name in _REQUIRED_EVIDENCE[:-1]:
        path = _guard_non_symlink_sandbox_path(sandbox_root / name, sandbox_root)
        _require_regular_evidence_file(path)
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExportError(f"required JSON evidence is malformed: {path}") from exc
        if not isinstance(parsed, Mapping):
            raise ExportError(f"required JSON evidence must be an object: {path}")
        if name == "path_audit.json":
            _validate_final_path_audit(parsed, sandbox_root)
        elif name == "metrics.json":
            _validate_metrics_evidence(parsed)
        elif name == "run_manifest.json":
            _validate_run_manifest_evidence(parsed, sandbox_root)
    checkpoint = _guard_non_symlink_sandbox_path(sandbox_root / "smoke_checkpoint.pt", sandbox_root)
    _require_regular_evidence_file(checkpoint)
    _load_checkpoint_payload(checkpoint)
    _validate_semantic_evidence_graph(sandbox_root)


def _created_evidence_paths(raw: Mapping[str, object], sandbox_root: Path) -> tuple[Path, ...]:
    """把 audit 声明解析为严格、存在且属于已知证据命名空间的路径。"""
    created = raw.get("created_or_modified")
    if not isinstance(created, list) or not created:
        raise ExportError("path audit created_or_modified must be a non-empty list")
    if not all(isinstance(relative, str) and relative for relative in created):
        raise ExportError("path audit contains a non-string or empty evidence path")
    if len(set(created)) != len(created):
        raise ExportError("path audit contains a duplicate evidence path")

    allowed_roots = tuple(path.as_posix() for path in _MANDATORY_EVIDENCE_ROOTS)
    paths: list[Path] = []
    for relative in created:
        parsed = Path(relative)
        if parsed.is_absolute() or ".." in parsed.parts or parsed.as_posix() != relative:
            raise ExportError(f"path audit contains an outside or noncanonical path: {relative}")
        allowed = relative in _ALLOWED_AUDIT_EXACT_PATHS or any(
            relative == root or relative.startswith(f"{root}/") for root in allowed_roots
        )
        if not allowed:
            raise ExportError(f"path audit contains an unknown evidence path: {relative}")
        if relative == "SUCCESS":
            continue
        path = _guard_non_symlink_sandbox_path(sandbox_root / parsed, sandbox_root)
        if not path.exists():
            raise ExportError(f"path audit declares missing evidence: {relative}")
        if path.is_dir():
            _assert_directory_has_no_symlink_children(path)
        elif not path.is_file():
            raise ExportError(f"path audit evidence is not a regular file or directory: {relative}")
        paths.append(path)
    return tuple(paths)


def _validate_final_path_audit(raw: Mapping[str, object], sandbox_root: Path) -> None:
    """确认 SUCCESS 已在不可再修改的 audit 中声明为最后封存写入。"""
    created = raw.get("created_or_modified")
    required_created = {
        _LAUNCHER_LOG_RELATIVE_PATH.as_posix(),
        _READY_TO_SEAL_RELATIVE_PATH.as_posix(),
        _TEE_COMPLETE_RELATIVE_PATH.as_posix(),
        "SUCCESS",
    }
    if not isinstance(created, list) or not required_created.issubset(created):
        raise ExportError(
            "path audit must include the final log, READY_TO_SEAL, TEE_COMPLETE and SUCCESS"
        )
    _created_evidence_paths(raw, sandbox_root)
    if raw.get("stage") != "finalize-seal":
        raise ExportError("path audit final SUCCESS declaration requires stage finalize-seal")
    stages = raw.get("stages")
    expected_stages = ["prepare", "cpu-contract", "npu-smoke", "finalize-seal"]
    if stages != expected_stages:
        raise ExportError("path audit must contain the complete fixed stage history")
    expected = {
        "path": "SUCCESS",
        "status": "expected_last_write",
        "exists_when_audit_written": False,
    }
    if not _exactly_equal(raw.get("final_seal"), expected):
        raise ExportError("path audit must declare SUCCESS as the expected last write")
    declared = raw.get("declared_files")
    expected_declared = {
        _LAUNCHER_LOG_RELATIVE_PATH.as_posix(),
        _READY_TO_SEAL_RELATIVE_PATH.as_posix(),
        _TEE_COMPLETE_RELATIVE_PATH.as_posix(),
    }
    if not isinstance(declared, Mapping) or set(declared) != expected_declared:
        raise ExportError(
            "path audit declared files must bind the launcher log, READY_TO_SEAL and TEE_COMPLETE"
        )
    for relative in sorted(expected_declared):
        metadata = declared[relative]
        if not isinstance(metadata, Mapping) or set(metadata) != {"size", "mtime_ns", "sha256"}:
            raise ExportError(f"path audit declared metadata is invalid: {relative}")
        path = _guard_non_symlink_sandbox_path(sandbox_root / relative, sandbox_root)
        _require_regular_evidence_file(path)
        current = path.stat()
        expected_metadata = {
            "size": current.st_size,
            "mtime_ns": current.st_mtime_ns,
            "sha256": _sha256_file(path),
        }
        if not _exactly_equal(metadata, expected_metadata):
            raise ExportError(f"path audit declared file changed after final audit: {relative}")


def _required_patch_selection_ids(sandbox_root: Path) -> tuple[str, ...]:
    """读取并严格验证封存必需的四 patch 选择证据。"""
    selection = sandbox_root / _PATCH_SELECTION_RELATIVE_PATH
    if selection.is_symlink():
        _guard_non_symlink_sandbox_path(selection, sandbox_root)
    if not selection.exists():
        raise ExportError(f"required patch selection evidence is missing: {selection}")
    guarded = _guard_non_symlink_sandbox_path(selection, sandbox_root)
    _require_regular_evidence_file(guarded)
    try:
        raw = json.loads(guarded.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExportError(f"patch selection evidence is malformed: {guarded}") from exc
    if not isinstance(raw, Mapping):
        raise ExportError("patch selection evidence schema must be a JSON object")
    values = raw.get("patch_ids")
    if not isinstance(values, list) and isinstance(raw.get("patches"), list):
        values = [
            item.get("patch_id") if isinstance(item, Mapping) else None for item in raw["patches"]
        ]
    if not isinstance(values, list) or not all(
        isinstance(value, str) and value for value in values
    ):
        raise ExportError("patch selection evidence schema must provide patch_ids")
    if len(values) != _EMBEDDING_SHAPE[0]:
        raise ExportError("patch selection evidence must contain exactly four patch IDs")
    if len(set(values)) != len(values):
        raise ExportError("patch selection evidence must not contain duplicate patch IDs")
    return tuple(values)


def _digest_path(digest: Any, path: Path, sandbox_root: Path) -> None:
    """把文件树路径和内容一起纳入封口摘要，避免只哈希目录名。"""
    if path.is_symlink():
        raise ExportError(f"symlink path must not enter SUCCESS digest: {path}")
    if path.is_file():
        relative = path.relative_to(sandbox_root).as_posix().encode("utf-8")
        digest.update(relative + b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
        return
    _assert_directory_has_no_symlink_children(path)
    children = sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.as_posix(),
    )
    for child in children:
        _digest_path(digest, child, sandbox_root)


def _validated_seal_paths(root: Path, required_files: Iterable[Path | str]) -> list[Path]:
    """验证最终证据合同并返回参与 SUCCESS 摘要的稳定路径集合。"""
    supplied_values = list(required_files)
    supplied_paths = [_resolve_required_path(path, root) for path in supplied_values]
    supplied_relatives = [path.relative_to(root).as_posix() for path in supplied_paths]
    if len(set(supplied_relatives)) != len(supplied_relatives):
        raise ExportError("required smoke evidence contains a duplicate path")
    mandatory_paths = [
        *(
            _guard_non_symlink_sandbox_path(root / relative, root)
            for relative in _MANDATORY_EVIDENCE_ROOTS
        ),
        *(_guard_non_symlink_sandbox_path(root / name, root) for name in _REQUIRED_EVIDENCE),
        _guard_non_symlink_sandbox_path(root / _LAUNCHER_LOG_RELATIVE_PATH, root),
        _guard_non_symlink_sandbox_path(root / _READY_TO_SEAL_RELATIVE_PATH, root),
        _guard_non_symlink_sandbox_path(root / _TEE_COMPLETE_RELATIVE_PATH, root),
    ]
    trusted_patch_ids = _required_patch_selection_ids(root)
    for relative in _MANDATORY_MANIFEST_FILES:
        path = _guard_non_symlink_sandbox_path(root / relative, root)
        _require_regular_evidence_file(path)
    for relative in (Path("synthetic/aef"), Path("synthetic/highres_2m")):
        path = _guard_non_symlink_sandbox_path(root / relative, root)
        _assert_directory_has_no_symlink_children(path)
    for path in [*mandatory_paths, *supplied_paths]:
        if not path.exists():
            raise ExportError(f"required smoke evidence is missing: {path}")
    partials = sorted(root.rglob("*.partial"))
    if partials:
        raise ExportError(f"partial smoke output prevents SUCCESS sealing: {partials[0]}")

    group_axes: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for group, path in zip(_GROUPS, _required_group_paths(root)):
        guarded = _guard_non_symlink_sandbox_path(path, root)
        group_axes.append(_verify_exported_group(guarded, expected_group=group))
    if any(axes != group_axes[0] for axes in group_axes[1:]):
        raise ExportError("all smoke Zarr groups must share exactly the same patch and period axes")
    if group_axes[0][0] != trusted_patch_ids:
        raise ExportError("Zarr patch axis does not match trusted patch selection evidence")
    _validate_required_evidence(root)
    audit = json.loads((root / "path_audit.json").read_text(encoding="utf-8"))
    assert isinstance(audit, Mapping)  # already established by _validate_required_evidence.
    audit_paths = _created_evidence_paths(audit, root)
    allowed_roots = tuple(path.as_posix() for path in _MANDATORY_EVIDENCE_ROOTS)
    for path, relative in zip(supplied_paths, supplied_relatives):
        allowed = relative in _ALLOWED_AUDIT_EXACT_PATHS or any(
            relative == allowed_root or relative.startswith(f"{allowed_root}/")
            for allowed_root in allowed_roots
        )
        if not allowed:
            raise ExportError(f"required smoke evidence contains an unknown path: {path}")
    return list(dict.fromkeys([*mandatory_paths, *audit_paths, *supplied_paths]))


def _combined_paths_sha256(root: Path, paths: Iterable[Path]) -> str:
    files: set[Path] = set()
    for path in paths:
        if path.is_file():
            files.add(path)
            continue
        _assert_directory_has_no_symlink_children(path)
        files.update(item for item in path.rglob("*") if item.is_file())
    digest = sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        _digest_path(digest, path, root)
    return digest.hexdigest()


def _validate_success_summary(raw: object, *, label: str) -> Mapping[str, object]:
    """校验 SUCCESS/SUCCESS.tmp 的精确 fail-closed schema。"""
    expected_keys = {
        "combined_sha256",
        "sealed_at_utc",
        "synthetic",
        "formal_training_allowed",
        "formal_evaluation_allowed",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_keys:
        raise ExportError(f"{label} summary has an invalid schema")
    if not _is_hex(raw["combined_sha256"], 64):
        raise ExportError(f"{label} combined SHA-256 is invalid")
    if not isinstance(raw["sealed_at_utc"], str) or not raw["sealed_at_utc"].endswith("Z"):
        raise ExportError(f"{label} sealed_at_utc is invalid")
    if (
        raw["synthetic"] is not True
        or raw["formal_training_allowed"] is not False
        or raw["formal_evaluation_allowed"] is not False
    ):
        raise ExportError(f"{label} formal-use policy is invalid")
    return raw


def _read_success_summary(path: Path, *, label: str) -> Mapping[str, object]:
    """从常规文件路径读取并校验 SUCCESS 摘要。"""
    _require_regular_evidence_file(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExportError(f"{label} summary is malformed") from exc
    return _validate_success_summary(raw, label=label)


def _read_success_summary_fd(descriptor: int, *, label: str) -> Mapping[str, object]:
    """从已打开的稳定 inode 读取摘要，避免再次解析可替换路径。"""
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 64 * 1024):
        chunks.append(chunk)
    try:
        raw = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExportError(f"{label} summary is malformed") from exc
    return _validate_success_summary(raw, label=label)


def _write_all_fd(descriptor: int, payload: bytes) -> None:
    """把 seal payload 完整写入已独占打开的 inode。"""
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        written = os.write(descriptor, view[offset:])
        if written <= 0 or written > len(view) - offset:
            raise ExportError("SUCCESS.tmp write made no valid progress")
        offset += written


def _open_success_temporary(root_descriptor: int, combined_sha256: str) -> int:
    """独占创建或只读恢复绑定当前 evidence 的稳定临时 inode。"""
    relative = "SUCCESS.tmp"
    create_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    read_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        create_flags |= os.O_NOFOLLOW
        read_flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(relative, create_flags, 0o600, dir_fd=root_descriptor)
    except FileExistsError:
        try:
            descriptor = os.open(relative, read_flags, dir_fd=root_descriptor)
        except OSError as exc:
            raise ExportError("SUCCESS.tmp cannot be safely reopened") from exc
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            os.close(descriptor)
            raise ExportError("SUCCESS.tmp must be one private regular inode")
        try:
            summary = _read_success_summary_fd(descriptor, label="SUCCESS.tmp")
        except Exception:
            os.close(descriptor)
            raise
        if summary["combined_sha256"] != combined_sha256:
            os.close(descriptor)
            raise ExportError("SUCCESS.tmp combined SHA-256 does not match current evidence")
        return descriptor
    except OSError as exc:
        raise ExportError("SUCCESS.tmp cannot be safely created") from exc

    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise ExportError("new SUCCESS.tmp must be one private regular inode")
    try:
        sealed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        payload = (
            "{\n"
            f'  "combined_sha256": "{combined_sha256}",\n'
            f'  "sealed_at_utc": "{sealed_at}",\n'
            '  "synthetic": true,\n'
            '  "formal_training_allowed": false,\n'
            '  "formal_evaluation_allowed": false\n'
            "}\n"
        ).encode("utf-8")
        _write_all_fd(descriptor, payload)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _publish_success_no_replace(root_descriptor: int, temporary_descriptor: int) -> None:
    """原子 no-replace 发布，并确认目标仍是已验证的临时 inode。"""
    temporary_metadata = os.fstat(temporary_descriptor)
    try:
        os.link(
            "SUCCESS.tmp",
            "SUCCESS",
            src_dir_fd=root_descriptor,
            dst_dir_fd=root_descriptor,
            follow_symlinks=False,
        )
    except FileExistsError as exc:
        raise ExportError("SUCCESS appeared concurrently; refusing to overwrite it") from exc
    except OSError as exc:
        raise ExportError("cannot atomically publish SUCCESS without replacement") from exc
    published_metadata = os.stat("SUCCESS", dir_fd=root_descriptor, follow_symlinks=False)
    if not stat.S_ISREG(published_metadata.st_mode) or (
        published_metadata.st_dev,
        published_metadata.st_ino,
    ) != (temporary_metadata.st_dev, temporary_metadata.st_ino):
        raise ExportError("published SUCCESS does not match the verified temporary inode")
    os.fsync(root_descriptor)
    try:
        os.unlink("SUCCESS.tmp", dir_fd=root_descriptor)
    except FileNotFoundError:
        pass
    os.fsync(root_descriptor)


def seal_success(sandbox_root: Path, required_files: Iterable[Path | str]) -> Path:
    """确认固定 evidence 与四个 sealed Zarr 均完整后，最后写入 ``SUCCESS``。"""
    root = _guard_non_symlink_sandbox_path(Path(sandbox_root), Path(sandbox_root))
    success = _guard_non_symlink_sandbox_path(root / "SUCCESS", root)
    if success.exists():
        raise ExportError("SUCCESS already exists; a sealed smoke run is immutable")

    all_paths = _validated_seal_paths(root, required_files)
    combined_sha256 = _combined_paths_sha256(root, all_paths)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    try:
        root_descriptor = os.open(root, directory_flags)
    except OSError as exc:
        raise ExportError("sandbox root cannot be safely opened for sealing") from exc
    temporary_descriptor: int | None = None
    try:
        temporary_descriptor = _open_success_temporary(root_descriptor, combined_sha256)
        _publish_success_no_replace(root_descriptor, temporary_descriptor)
    finally:
        if temporary_descriptor is not None:
            os.close(temporary_descriptor)
        os.close(root_descriptor)
    verify_success(root)
    return success


def verify_success(sandbox_root: Path) -> None:
    """只读复核 SUCCESS 摘要与最终日志、证据和四组 Zarr 的当前内容。"""
    root = _guard_non_symlink_sandbox_path(Path(sandbox_root), Path(sandbox_root))
    success = _guard_non_symlink_sandbox_path(root / "SUCCESS", root)
    raw = _read_success_summary(success, label="SUCCESS")
    all_paths = _validated_seal_paths(root, ())
    if _combined_paths_sha256(root, all_paths) != raw["combined_sha256"]:
        raise ExportError("SUCCESS combined SHA-256 does not match current smoke evidence")
