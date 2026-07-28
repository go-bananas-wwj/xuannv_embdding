"""校验海淀 P10C 冻结迁移到哈尔滨时不可放宽的输入合同。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


EXPECTED_MODALITIES = {
    "s2": {"channels": 12, "availability": "provided"},
    "s1": {"channels": 2, "availability": "provided"},
    "landsat": {"channels": 7, "availability": "provided"},
    "highres_optical": {
        "channels": 3,
        "availability": "provided_with_physical_source_provenance",
    },
    "highres_sar": {"channels": 1, "availability": "explicit_missing"},
}


def load_and_validate_contract(path: str | Path) -> dict[str, Any]:
    """读取迁移合同，并拒绝会把迁移变成目标域适配的配置。"""
    contract_path = Path(path)
    raw = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("transfer"), dict):
        raise ValueError("迁移合同必须包含 transfer 映射")

    transfer = raw["transfer"]
    if transfer.get("freeze_encoder") is not True:
        raise ValueError("freeze_encoder 必须为 true")
    if transfer.get("normalization") != "haidian_p10c_training_statistics":
        raise ValueError("normalization 必须使用 haidian_p10c_training_statistics")
    if transfer.get("target_statistics_fitting") != "forbidden":
        raise ValueError("target_statistics_fitting 必须为 forbidden")

    modalities = transfer.get("modality_contract")
    if not isinstance(modalities, dict):
        raise ValueError("transfer.modality_contract 必须为映射")
    for name, expected in EXPECTED_MODALITIES.items():
        actual = modalities.get(name)
        if actual != expected:
            raise ValueError(
                f"{name} 合同不符合冻结迁移要求: expected={expected!r}, actual={actual!r}"
            )

    forbidden = set(transfer.get("forbidden", []))
    required_forbidden = {
        "encoder_finetuning",
        "learned_adapter",
        "target_statistics_fitting",
        "silent_modality_rename",
        "missing_sensor_as_observation",
    }
    missing_forbidden = sorted(required_forbidden - forbidden)
    if missing_forbidden:
        raise ValueError(f"迁移合同缺少禁止项: {', '.join(missing_forbidden)}")
    return raw


def materialize_transfer_manifest(source_path: str | Path, target_path: str | Path) -> None:
    """把哈尔滨高分光学映射到冻结 checkpoint 的输入槽位。

    映射仅服务于 checkpoint 的 schema key；原始物理来源通过
    ``transfer_highres_optical_source`` 保留。高分 SAR 在哈尔滨不存在，因而必须
    保留为空列表，使 dataset 产生零 availability，而不是伪造观测。
    """
    source_path = Path(source_path)
    target_path = Path(target_path)
    if source_path.parent.resolve() != target_path.parent.resolve():
        raise ValueError("迁移 manifest 的父目录必须与源 manifest 一致，以保持相对路径有效")
    records = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("source manifest 必须是记录列表")

    mapped_records: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("manifest 中每条记录必须是对象")
        optical_paths = record.get("highres_optical_harbin", [])
        if not isinstance(optical_paths, list):
            raise ValueError("highres_optical_harbin 必须是列表")
        sar_paths = record.get("highres_sar_haidian", [])
        if sar_paths not in (None, []):
            raise ValueError("哈尔滨冻结迁移不允许提供 highres_sar_haidian")

        mapped = dict(record)
        mapped["highres_optical_haidian"] = optical_paths
        mapped["highres_sar_haidian"] = []
        mapped["transfer_highres_optical_source"] = "highres_optical_harbin"
        mapped_records.append(mapped)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(mapped_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    contract = load_and_validate_contract(args.config)
    print(
        "P10C Haidian-to-Harbin transfer contract accepted: "
        f"target={contract['transfer']['target_region']} "
        f"month={contract['transfer']['target_month']}"
    )


if __name__ == "__main__":
    main()
