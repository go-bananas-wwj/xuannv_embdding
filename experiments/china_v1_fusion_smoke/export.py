"""隔离 China V1 fusion smoke 的可审计导出与最终封存。"""

from __future__ import annotations

import json
import os
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
    checkpoint = _guard_non_symlink_sandbox_path(sandbox_root / "smoke_checkpoint.pt", sandbox_root)
    _require_regular_evidence_file(checkpoint)
    _load_checkpoint_payload(checkpoint)


def _validate_final_path_audit(raw: Mapping[str, object], sandbox_root: Path) -> None:
    """确认 SUCCESS 已在不可再修改的 audit 中声明为最后封存写入。"""
    created = raw.get("created_or_modified")
    required_created = {
        _LAUNCHER_LOG_RELATIVE_PATH.as_posix(),
        _READY_TO_SEAL_RELATIVE_PATH.as_posix(),
        "SUCCESS",
    }
    if not isinstance(created, list) or not required_created.issubset(created):
        raise ExportError("path audit must include the final log, READY_TO_SEAL and SUCCESS")
    if raw.get("stage") != "finalize-seal":
        raise ExportError("path audit final SUCCESS declaration requires stage finalize-seal")
    stages = raw.get("stages")
    if not isinstance(stages, list) or stages[-2:] != ["npu-smoke", "finalize-seal"]:
        raise ExportError("path audit must finish with npu-smoke then finalize-seal")
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
    }
    if not isinstance(declared, Mapping) or set(declared) != expected_declared:
        raise ExportError("path audit declared files must bind the launcher log and READY_TO_SEAL")
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
    supplied_paths = [_resolve_required_path(path, root) for path in required_files]
    mandatory_paths = [
        *(_guard_non_symlink_sandbox_path(path, root) for path in _required_group_paths(root)),
        *(_guard_non_symlink_sandbox_path(root / name, root) for name in _REQUIRED_EVIDENCE),
        _guard_non_symlink_sandbox_path(root / _PATCH_SELECTION_RELATIVE_PATH, root),
        _guard_non_symlink_sandbox_path(root / _LAUNCHER_LOG_RELATIVE_PATH, root),
        _guard_non_symlink_sandbox_path(root / _READY_TO_SEAL_RELATIVE_PATH, root),
    ]
    trusted_patch_ids = _required_patch_selection_ids(root)
    all_paths = list(dict.fromkeys([*mandatory_paths, *supplied_paths]))
    for path in all_paths:
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
    return all_paths


def _combined_paths_sha256(root: Path, paths: Iterable[Path]) -> str:
    digest = sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        _digest_path(digest, path, root)
    return digest.hexdigest()


def seal_success(sandbox_root: Path, required_files: Iterable[Path | str]) -> Path:
    """确认固定 evidence 与四个 sealed Zarr 均完整后，最后写入 ``SUCCESS``。"""
    root = _guard_non_symlink_sandbox_path(Path(sandbox_root), Path(sandbox_root))
    success = _guard_non_symlink_sandbox_path(root / "SUCCESS", root)
    if success.exists():
        raise ExportError("SUCCESS already exists; a sealed smoke run is immutable")

    all_paths = _validated_seal_paths(root, required_files)
    combined_sha256 = _combined_paths_sha256(root, all_paths)
    sealed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    payload = (
        "{\n"
        f'  "combined_sha256": "{combined_sha256}",\n'
        f'  "sealed_at_utc": "{sealed_at}",\n'
        '  "synthetic": true,\n'
        '  "formal_training_allowed": false,\n'
        '  "formal_evaluation_allowed": false\n'
        "}\n"
    )
    temporary = _guard_non_symlink_sandbox_path(root / "SUCCESS.tmp", root)
    if temporary.exists():
        raise ExportError("SUCCESS.tmp already exists; refusing to overwrite seal evidence")
    temporary.write_text(payload, encoding="utf-8")
    _guard_non_symlink_sandbox_path(temporary, root)
    _guard_non_symlink_sandbox_path(success, root)
    temporary.replace(success)
    verify_success(root)
    return success


def verify_success(sandbox_root: Path) -> None:
    """只读复核 SUCCESS 摘要与最终日志、证据和四组 Zarr 的当前内容。"""
    root = _guard_non_symlink_sandbox_path(Path(sandbox_root), Path(sandbox_root))
    success = _guard_non_symlink_sandbox_path(root / "SUCCESS", root)
    _require_regular_evidence_file(success)
    try:
        raw = json.loads(success.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExportError("SUCCESS summary is malformed") from exc
    expected_keys = {
        "combined_sha256",
        "sealed_at_utc",
        "synthetic",
        "formal_training_allowed",
        "formal_evaluation_allowed",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_keys:
        raise ExportError("SUCCESS summary has an invalid schema")
    if not _is_hex(raw["combined_sha256"], 64):
        raise ExportError("SUCCESS combined SHA-256 is invalid")
    if not isinstance(raw["sealed_at_utc"], str) or not raw["sealed_at_utc"].endswith("Z"):
        raise ExportError("SUCCESS sealed_at_utc is invalid")
    if (
        raw["synthetic"] is not True
        or raw["formal_training_allowed"] is not False
        or raw["formal_evaluation_allowed"] is not False
    ):
        raise ExportError("SUCCESS formal-use policy is invalid")
    all_paths = _validated_seal_paths(root, ())
    if _combined_paths_sha256(root, all_paths) != raw["combined_sha256"]:
        raise ExportError("SUCCESS combined SHA-256 does not match current smoke evidence")
