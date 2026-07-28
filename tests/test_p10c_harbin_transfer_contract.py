from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parent.parent
CONFIG_PATH = REPO_ROOT / "configs/paper_p10c_haidian_frozen_to_harbin_20260728.yaml"
SCRIPT_PATH = REPO_ROOT / "scripts/eval/validate_p10c_harbin_transfer.py"


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("p10c_harbin_transfer", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载脚本: {SCRIPT_PATH}")
    module = types.ModuleType(spec.name)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_transfer_contract_requires_explicit_missing_sar_and_source_statistics() -> None:
    """冻结迁移不得把缺失 SAR 静默当作真实观测，也不得拟合目标域统计量。"""
    module = _load_module()

    contract = module.load_and_validate_contract(CONFIG_PATH)

    assert contract["transfer"]["freeze_encoder"] is True
    assert contract["transfer"]["normalization"] == "haidian_p10c_training_statistics"
    assert contract["transfer"]["modality_contract"]["s2"] == {
        "channels": 12,
        "availability": "provided",
    }
    assert contract["transfer"]["modality_contract"]["highres_optical"] == {
        "channels": 3,
        "availability": "provided_with_physical_source_provenance",
    }
    assert contract["transfer"]["modality_contract"]["highres_sar"] == {
        "channels": 1,
        "availability": "explicit_missing",
    }


def test_transfer_contract_rejects_target_statistics_and_silent_missing_sar(tmp_path: Path) -> None:
    """风险配置必须被拒绝，避免迁移评测在不知情下发生目标域适配。"""
    module = _load_module()
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        """
transfer:
  freeze_encoder: true
  normalization: harbin_statistics
  modality_contract:
    s2: {channels: 12, availability: provided}
    s1: {channels: 2, availability: provided}
    landsat: {channels: 7, availability: provided}
    highres_optical: {channels: 3, availability: provided_with_physical_source_provenance}
    highres_sar: {channels: 1, availability: provided}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="normalization|highres_sar"):
        module.load_and_validate_contract(invalid)


def test_transfer_manifest_maps_only_optical_schema_slot_and_keeps_sar_missing(
    tmp_path: Path,
) -> None:
    """哈尔滨高分光学可进入兼容输入槽位，但 SAR 缺失必须保持为空。"""
    module = _load_module()
    source_manifest = tmp_path / "harbin.json"
    target_manifest = tmp_path / "transfer.json"
    source_manifest.write_text(
        json.dumps(
            [
                {
                    "patch_id": "harbin_patch_000001",
                    "highres_optical_harbin": ["patches/highres_optical/a.tif"],
                    "highres_sar_haidian": [],
                }
            ]
        ),
        encoding="utf-8",
    )

    module.materialize_transfer_manifest(source_manifest, target_manifest)

    mapped = json.loads(target_manifest.read_text(encoding="utf-8"))
    assert mapped[0]["highres_optical_haidian"] == ["patches/highres_optical/a.tif"]
    assert mapped[0]["highres_sar_haidian"] == []
    assert mapped[0]["transfer_highres_optical_source"] == "highres_optical_harbin"


def test_transfer_manifest_rejects_a_new_parent_that_breaks_relative_paths(tmp_path: Path) -> None:
    """原始 manifest 使用相对影像路径，迁移清单必须保留相同父目录。"""
    module = _load_module()
    source_manifest = tmp_path / "harbin.json"
    source_manifest.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="父目录"):
        module.materialize_transfer_manifest(source_manifest, tmp_path / "other/transfer.json")
