"""隔离 China V1 fusion smoke 的可审计导出与最终封存。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zarr
from numcodecs import Blosc

from experiments.china_v1_fusion_smoke.safety import validate_write_path


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
_REQUIRED_METADATA = (
    "git_commit",
    "config_sha256",
    "selected_patch_manifest_sha256",
    "seed",
    "model_class",
    "synthetic",
)
_BLOSC = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)


def _model_class_name(model: torch.nn.Module) -> str:
    cls = model.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _is_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _validate_checkpoint_metadata(
    metadata: Mapping[str, object], model: torch.nn.Module
) -> dict[str, object]:
    """严格校验足以绑定权重和本次 smoke 输入的 provenance。"""
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
    if raw["model_class"] != _model_class_name(model):
        raise ExportError("checkpoint metadata model_class does not match the model")
    if raw["synthetic"] is not True:
        raise ExportError("checkpoint metadata must declare synthetic=true")
    for key in ("formal_training_allowed", "formal_evaluation_allowed"):
        if key in raw and raw[key] is not False:
            raise ExportError(f"checkpoint metadata {key} must be false when present")
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


def _validated_checkpoint_path(path: Path) -> Path:
    """从 checkpoint 目标向上寻找唯一的隔离哨兵并复用路径守卫。"""
    resolved = path.resolve(strict=False)
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / ".xuannv_isolated_smoke").is_file():
            return validate_write_path(resolved, candidate)
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


def _verify_exported_group(path: Path, expected_group: str | None = None) -> None:
    """在 rename 和 SUCCESS 前重新打开 Zarr，避免把坏目录当作成功产物。"""
    try:
        exported = zarr.open_group(str(path), mode="r")
    except Exception as exc:  # zarr reports several concrete error classes across releases.
        raise ExportError(f"cannot reopen exported Zarr group: {path}") from exc
    if exported["embedding"].shape != _EMBEDDING_SHAPE:
        raise ExportError(f"Zarr embedding shape is not {_EMBEDDING_SHAPE}: {path}")
    if exported["embedding"].dtype != np.dtype("float16"):
        raise ExportError(f"Zarr embedding dtype is not float16: {path}")
    if exported["embedding"].chunks != _EMBEDDING_CHUNKS:
        raise ExportError(f"Zarr embedding chunks are not {_EMBEDDING_CHUNKS}: {path}")
    if exported["valid"].shape != _VALID_SHAPE or exported["valid"].dtype != np.dtype("bool"):
        raise ExportError(f"Zarr valid array contract is invalid: {path}")
    if exported["valid"].chunks != _VALID_CHUNKS:
        raise ExportError(f"Zarr valid chunks are not {_VALID_CHUNKS}: {path}")
    if exported["patch_id"].shape != (4,) or exported["period"].shape != (8,):
        raise ExportError(f"Zarr patch_id or period contract is invalid: {path}")
    if expected_group is not None and exported.attrs.get("group") != expected_group:
        raise ExportError(f"Zarr group attribute does not match {expected_group!r}: {path}")
    if exported.attrs.get("synthetic") is not True:
        raise ExportError(f"Zarr must declare synthetic=true: {path}")
    if exported.attrs.get("allowed_use") != "smoke_test_only":
        raise ExportError(f"Zarr must be limited to smoke_test_only: {path}")
    for key in ("formal_training_allowed", "formal_evaluation_allowed"):
        if exported.attrs.get(key) is not False:
            raise ExportError(f"Zarr {key} must be false: {path}")


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
    target = validate_write_path(Path(output), sandbox_root)
    if target.name != "embedding.zarr":
        raise ExportError("smoke Zarr outputs must be named embedding.zarr")
    partial = validate_write_path(target.with_name(f"{target.name}.partial"), sandbox_root)
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
        }
    )
    _verify_exported_group(partial, expected_group=group)
    partial.replace(target)
    return target


def save_smoke_checkpoint(
    model: torch.nn.Module,
    path: Path,
    metadata: Mapping[str, object],
) -> str:
    """以精确 provenance 和原子 replace 保存仅用于 reload 检查的 checkpoint。"""
    checked_metadata = _validate_checkpoint_metadata(metadata, model)
    target = _validated_checkpoint_path(Path(path))
    if target.exists():
        raise ExportError(f"refusing to overwrite smoke checkpoint: {target}")
    if target.suffix != ".pt":
        raise ExportError("smoke checkpoint must use a .pt suffix")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.partial")
    if partial.exists():
        raise ExportError(f"refusing to overwrite diagnostic partial checkpoint: {partial}")
    torch.save({"state_dict": model.state_dict(), "metadata": checked_metadata}, partial)
    partial.replace(target)
    return _sha256_file(target)


def load_smoke_checkpoint(
    model: torch.nn.Module,
    path: Path,
    expected_metadata: Mapping[str, object],
) -> None:
    """只在文件 metadata 与调用方 provenance 完全相同时恢复权重。"""
    expected = _validate_checkpoint_metadata(expected_metadata, model)
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise ExportError(f"smoke checkpoint is missing: {checkpoint}")
    try:
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ExportError(f"cannot load smoke checkpoint: {checkpoint}") from exc
    if not isinstance(raw, Mapping) or set(raw) != {"state_dict", "metadata"}:
        raise ExportError("smoke checkpoint payload has an invalid schema")
    saved_metadata = raw["metadata"]
    if not isinstance(saved_metadata, Mapping):
        raise ExportError("smoke checkpoint metadata has an invalid schema")
    _validate_checkpoint_metadata(saved_metadata, model)
    if not _exactly_equal(dict(saved_metadata), expected):
        raise ExportError("smoke checkpoint metadata does not exactly match expected metadata")
    if not isinstance(raw["state_dict"], Mapping):
        raise ExportError("smoke checkpoint state_dict has an invalid schema")
    model.load_state_dict(raw["state_dict"], strict=True)


def _resolve_required_path(path: Path | str, sandbox_root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = sandbox_root / candidate
    return validate_write_path(candidate, sandbox_root)


def _digest_path(digest: Any, path: Path, sandbox_root: Path) -> None:
    """把文件树路径和内容一起纳入封口摘要，避免只哈希目录名。"""
    if path.is_file():
        relative = path.relative_to(sandbox_root).as_posix().encode("utf-8")
        digest.update(relative + b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
        return
    children = sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.as_posix(),
    )
    for child in children:
        _digest_path(digest, child, sandbox_root)


def seal_success(sandbox_root: Path, required_files: Iterable[Path | str]) -> Path:
    """确认固定 evidence 与四个 sealed Zarr 均完整后，最后写入 ``SUCCESS``。"""
    root = validate_write_path(Path(sandbox_root), sandbox_root)
    success = validate_write_path(root / "SUCCESS", root)
    if success.exists():
        raise ExportError("SUCCESS already exists; a sealed smoke run is immutable")

    supplied_paths = [_resolve_required_path(path, root) for path in required_files]
    mandatory_paths = [
        *_required_group_paths(root),
        *(root / name for name in _REQUIRED_EVIDENCE),
    ]
    all_paths = list(dict.fromkeys([*mandatory_paths, *supplied_paths]))
    for path in all_paths:
        if not path.exists():
            raise ExportError(f"required smoke evidence is missing: {path}")
    partials = sorted(root.rglob("*.partial"))
    if partials:
        raise ExportError(f"partial smoke output prevents SUCCESS sealing: {partials[0]}")

    for group, path in zip(_GROUPS, _required_group_paths(root)):
        _verify_exported_group(path, expected_group=group)

    digest = sha256()
    for path in sorted(all_paths, key=lambda item: item.relative_to(root).as_posix()):
        _digest_path(digest, path, root)
    sealed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    payload = (
        "{\n"
        f'  "combined_sha256": "{digest.hexdigest()}",\n'
        f'  "sealed_at_utc": "{sealed_at}",\n'
        '  "synthetic": true,\n'
        '  "formal_training_allowed": false,\n'
        '  "formal_evaluation_allowed": false\n'
        "}\n"
    )
    temporary = validate_write_path(root / "SUCCESS.tmp", root)
    if temporary.exists():
        raise ExportError("SUCCESS.tmp already exists; refusing to overwrite seal evidence")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(success)
    return success
