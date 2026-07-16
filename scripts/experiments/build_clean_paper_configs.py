#!/usr/bin/env python3
"""Generate self-contained clean paper experiment YAML files."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "configs/v2_p10c_haidian_202512_202605_osm_semantic_hardneg_20260704.yaml"
DEFAULT_CONFIG_DIR = ROOT / "configs/paper_registered_20260716"
DEFAULT_DATA_DIR = Path("/data/xuannv_embedding/processed/haidian")
DEFAULT_OUTPUT_ROOT = Path("/data/xuannv_embedding/outputs/paper_registered_20260716")
MANIFEST_PREFIX = "paper_registered_20260716"
SPATIAL_SPLIT = ROOT / "configs/eval/haidian_spatial_5fold_buffer1_seed42.json"
SUBSET_REGISTRY = ROOT / "configs/eval/haidian_paper_subsets_40_80_150_seed42.json"


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", type=int, default=(0, 1, 2, 3, 4))
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def base_config(
    name: str,
    train_size: int,
    fold: int,
    data_dir: Path,
    output_root: Path,
) -> dict[str, Any]:
    cfg = copy.deepcopy(yaml.safe_load(BASE.read_text(encoding="utf-8")))
    cfg["experiment"].update(
        name=name,
        output_dir=str(output_root / name),
        wandb_run_name=name,
    )
    cfg["training"]["gradient_accumulation_steps"] = 6
    cfg["data"].update(
        manifest_path=str(data_dir / f"{MANIFEST_PREFIX}_train_{train_size}_fold{fold}_seed42.json"),
        train_manifest_path=str(data_dir / f"{MANIFEST_PREFIX}_train_{train_size}_fold{fold}_seed42.json"),
        val_manifest_path=str(data_dir / f"{MANIFEST_PREFIX}_val_fold{fold}.json"),
        num_samples=train_size,
    )
    cfg["data"]["paper_spatial_split"] = str(SPATIAL_SPLIT)
    cfg["data"]["paper_subset_registry"] = str(SUBSET_REGISTRY)
    cfg["data"]["paper_subset_registry_sha256"] = sha256(SUBSET_REGISTRY)
    cfg["data"]["paper_fold"] = fold
    return cfg


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
) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for fold in folds:
        for size in (40, 80, 150):
            name = f"paper_registered_full_{size}_fold{fold}_20260716"
            configs[name] = base_config(name, size, fold, data_dir, output_root)

        name = f"paper_registered_coarse_osm_only_150_fold{fold}_20260716"
        configs[name] = base_config(name, 150, fold, data_dir, output_root)
        remove_fine_osm_probe(configs[name])

        name = f"paper_registered_no_osm_150_fold{fold}_20260716"
        configs[name] = base_config(name, 150, fold, data_dir, output_root)
        remove_all_osm(configs[name])

        name = f"paper_registered_probe_nohardneg_150_fold{fold}_20260716"
        configs[name] = base_config(name, 150, fold, data_dir, output_root)
        configs[name]["training"]["semantic_probe_hard_negative_ratio"] = 0.0
        configs[name]["training"]["semantic_probe_hard_negative_weight"] = 0.0

        name = f"paper_registered_no_highres_path_150_fold{fold}_20260716"
        configs[name] = base_config(name, 150, fold, data_dir, output_root)
        remove_highres(configs[name])

        name = f"paper_registered_no_masking_150_fold{fold}_20260716"
        configs[name] = base_config(name, 150, fold, data_dir, output_root)
        configs[name]["training"]["input_masking"]["enabled"] = False
    return configs


def main() -> None:
    args = parse_args()
    args.config_dir.mkdir(parents=True, exist_ok=True)
    for name, cfg in variants(args.folds, args.data_dir, args.output_root).items():
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
