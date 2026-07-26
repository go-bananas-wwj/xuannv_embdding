#!/usr/bin/env python3
"""Generate self-contained clean paper experiment YAML files."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "configs/v2_p10c_haidian_202512_202605_osm_semantic_hardneg_20260704.yaml"
DEFAULT_CONFIG_DIR = ROOT / "configs/paper_registered_v5_20260726"
DEFAULT_DATA_DIR = Path("/data/xuannv_embedding/processed/haidian")
DEFAULT_OUTPUT_ROOT = Path("/data/xuannv_embedding/outputs/paper_registered_v5_20260726")
MANIFEST_PREFIX = "paper_registered_v5_20260726"
SPATIAL_SPLIT = ROOT / "configs/eval/haidian_spatial_5fold_complete2x2_v5_seed42.json"
SUBSET_REGISTRY = ROOT / "configs/eval/haidian_paper_subsets_40_80_150_complete2x2_v5_seed42.json"
NORMALIZATION_STATISTICS = ROOT / "configs/eval/haidian_paper_v5_normalization_statistics.json"
PROTOCOL_NAME = "rse_v5_registered_20260726"


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", type=int, default=(0, 1, 2, 3, 4))
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output", dest="config_dir", type=Path, help="生成 YAML 的目录。")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split", type=Path, default=SPATIAL_SPLIT)
    parser.add_argument("--subset-registry", type=Path, default=SUBSET_REGISTRY)
    parser.add_argument("--statistics-registry", type=Path, default=NORMALIZATION_STATISTICS)
    parser.add_argument("--manifest-prefix", default=MANIFEST_PREFIX)
    parser.add_argument("--protocol-name", default=PROTOCOL_NAME)
    return parser.parse_args()


def base_config(
    name: str,
    train_size: int,
    fold: int,
    data_dir: Path,
    output_root: Path,
    split_path: Path = SPATIAL_SPLIT,
    subset_registry_path: Path = SUBSET_REGISTRY,
    statistics_registry_path: Path = NORMALIZATION_STATISTICS,
    manifest_prefix: str = MANIFEST_PREFIX,
    protocol_name: str = PROTOCOL_NAME,
) -> dict[str, Any]:
    statistics_registry = json.loads(statistics_registry_path.read_text(encoding="utf-8"))
    if statistics_registry.get("source_split_sha256") != sha256(split_path):
        raise ValueError("Normalization statistics registry does not match the requested split")
    fold_statistics = statistics_registry.get("folds", {}).get(str(fold))
    if fold_statistics is None:
        raise ValueError(f"Normalization statistics registry has no fold {fold}")
    manifest_audit_path = Path(fold_statistics["manifest_audit"])
    manifest_audit = json.loads(manifest_audit_path.read_text(encoding="utf-8"))
    expected_train = manifest_audit["manifests"][f"train_{train_size}"]
    train_path = data_dir / f"{manifest_prefix}_train_{train_size}_fold{fold}_seed42.json"
    if str(train_path) != expected_train["path"] or sha256(train_path) != expected_train["sha256"]:
        raise ValueError("Generated config train manifest does not match the fold audit sidecar")
    cfg = copy.deepcopy(yaml.safe_load(BASE.read_text(encoding="utf-8")))
    cfg["experiment"].update(
        name=name,
        output_dir=str(output_root / name),
        wandb_run_name=name,
    )
    cfg["training"]["gradient_accumulation_steps"] = 6
    cfg["data"].update(
        manifest_path=str(train_path),
        train_manifest_path=str(train_path),
        val_manifest_path=str(data_dir / f"{manifest_prefix}_val_fold{fold}.json"),
        statistics_dir=fold_statistics["statistics_dir"],
        num_samples=train_size,
    )
    cfg["data"].update(
        paper_protocol_name=protocol_name,
        paper_spatial_split=str(split_path),
        paper_spatial_split_sha256=sha256(split_path),
        paper_subset_registry=str(subset_registry_path),
        paper_subset_registry_sha256=sha256(subset_registry_path),
        paper_manifest_audit=str(manifest_audit_path),
        paper_manifest_audit_sha256=sha256(manifest_audit_path),
        paper_normalization_statistics=str(statistics_registry_path),
        paper_normalization_statistics_sha256=sha256(statistics_registry_path),
        paper_normalization_statistics_audit=fold_statistics.get("statistics_audit"),
        paper_normalization_statistics_audit_sha256=fold_statistics.get("statistics_audit_sha256"),
        paper_fold=fold,
    )
    return cfg


def bind_normalization_statistics_audit(
    cfg: dict[str, Any], statistics_registry_path: Path
) -> None:
    """Bind a config to the sealed audit variant for its exact source set."""
    data = cfg["data"]
    registry = json.loads(statistics_registry_path.read_text(encoding="utf-8"))
    fold_statistics = registry["folds"][str(data["paper_fold"])]
    source_key = "__".join(sorted(data["sources"]))
    audit = fold_statistics.get("statistics_audits", {}).get(source_key)
    if audit is None:
        raise ValueError(
            "Normalization statistics registry has no sealed audit for sources: "
            f"{', '.join(sorted(data['sources']))}"
        )
    data["paper_normalization_statistics_audit"] = audit["path"]
    data["paper_normalization_statistics_audit_sha256"] = audit["sha256"]


def remove_fine_osm_probe(cfg: dict[str, Any]) -> None:
    training = cfg["training"]
    training["semantic_probe_weight"] = 0.0
    training["semantic_probe_tasks"] = []
    training["semantic_probe_task_weights"] = {}
    training["semantic_probe_pos_weights"] = {}
    training["semantic_probe_hard_negative_ratio"] = 0.0
    training["semantic_probe_hard_negative_weight"] = 0.0
    cfg["data"]["supervised_label_roots"] = {}
    cfg["data"]["supervised_sampling"] = {"enabled": False}


def remove_all_osm(cfg: dict[str, Any]) -> None:
    remove_fine_osm_probe(cfg)
    cfg["model"]["target_heads"].pop("worldcover", None)
    cfg["data"]["sources"] = [source for source in cfg["data"]["sources"] if source != "worldcover"]


def remove_highres(cfg: dict[str, Any]) -> None:
    cfg["model"]["stp"]["highres_fusion_to_embedding"] = False
    for source in ("highres_optical_haidian", "highres_sar_haidian"):
        cfg["model"]["sensor_channels"].pop(source, None)
        cfg["data"]["sources"] = [value for value in cfg["data"]["sources"] if value != source]
    cfg["model"]["target_heads"].pop("highres_optical_haidian_recon", None)
    cfg["model"]["target_heads"].pop("highres_sar_haidian_recon", None)


def variants(
    folds: list[int] | tuple[int, ...],
    data_dir: Path,
    output_root: Path,
    split_path: Path,
    subset_registry_path: Path,
    statistics_registry_path: Path,
    manifest_prefix: str,
    protocol_name: str,
) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for fold in folds:
        for size in (40, 80, 150):
            name = f"paper_registered_v5_full_{size}_fold{fold}_20260726"
            configs[name] = base_config(
                name,
                size,
                fold,
                data_dir,
                output_root,
                split_path,
                subset_registry_path,
                statistics_registry_path,
                manifest_prefix,
                protocol_name,
            )

        name = f"paper_registered_v5_coarse_osm_only_150_fold{fold}_20260726"
        configs[name] = base_config(
            name,
            150,
            fold,
            data_dir,
            output_root,
            split_path,
            subset_registry_path,
            statistics_registry_path,
            manifest_prefix,
            protocol_name,
        )
        remove_fine_osm_probe(configs[name])

        name = f"paper_registered_v5_no_osm_150_fold{fold}_20260726"
        configs[name] = base_config(
            name,
            150,
            fold,
            data_dir,
            output_root,
            split_path,
            subset_registry_path,
            statistics_registry_path,
            manifest_prefix,
            protocol_name,
        )
        remove_all_osm(configs[name])

        name = f"paper_registered_v5_probe_nohardneg_150_fold{fold}_20260726"
        configs[name] = base_config(
            name,
            150,
            fold,
            data_dir,
            output_root,
            split_path,
            subset_registry_path,
            statistics_registry_path,
            manifest_prefix,
            protocol_name,
        )
        configs[name]["training"]["semantic_probe_hard_negative_ratio"] = 0.0
        configs[name]["training"]["semantic_probe_hard_negative_weight"] = 0.0

        name = f"paper_registered_v5_no_highres_path_150_fold{fold}_20260726"
        configs[name] = base_config(
            name,
            150,
            fold,
            data_dir,
            output_root,
            split_path,
            subset_registry_path,
            statistics_registry_path,
            manifest_prefix,
            protocol_name,
        )
        remove_highres(configs[name])

        name = f"paper_registered_v5_no_masking_150_fold{fold}_20260726"
        configs[name] = base_config(
            name,
            150,
            fold,
            data_dir,
            output_root,
            split_path,
            subset_registry_path,
            statistics_registry_path,
            manifest_prefix,
            protocol_name,
        )
        configs[name]["training"]["input_masking"]["enabled"] = False
    for cfg in configs.values():
        bind_normalization_statistics_audit(cfg, statistics_registry_path)
    return configs


def main() -> None:
    args = parse_args()
    args.config_dir.mkdir(parents=True, exist_ok=True)
    for name, cfg in variants(
        args.folds,
        args.data_dir,
        args.output_root,
        args.split,
        args.subset_registry,
        args.statistics_registry,
        args.manifest_prefix,
        args.protocol_name,
    ).items():
        path = args.config_dir / f"{name}.yaml"
        path.write_text(
            yaml.dump(
                cfg,
                Dumper=NoAliasDumper,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            ),
            encoding="utf-8",
        )
        print(path)


if __name__ == "__main__":
    main()
