#!/usr/bin/env python3
"""Generate self-contained clean paper experiment YAML files."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "configs/v2_p10c_haidian_202512_202605_osm_semantic_hardneg_20260704.yaml"
CONFIG_DIR = ROOT / "configs/paper_20260715"
DATA_DIR = Path("/data/xuannv_embedding/processed/haidian")
OUTPUT_ROOT = Path("/data/xuannv_embedding/outputs/paper_20260715")


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def base_config(name: str, train_size: int) -> dict[str, Any]:
    cfg = copy.deepcopy(yaml.safe_load(BASE.read_text(encoding="utf-8")))
    cfg["experiment"].update(
        name=name,
        output_dir=str(OUTPUT_ROOT / name),
        wandb_run_name=name,
    )
    cfg["training"]["gradient_accumulation_steps"] = 6
    cfg["data"].update(
        manifest_path=str(DATA_DIR / f"paper_20260715_train_{train_size}_fold0_seed42.json"),
        train_manifest_path=str(DATA_DIR / f"paper_20260715_train_{train_size}_fold0_seed42.json"),
        val_manifest_path=str(DATA_DIR / "paper_20260715_val_fold0.json"),
        num_samples=train_size,
    )
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


def variants() -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for size in (40, 80, 160):
        name = f"paper_clean_full_{size}_fold0_20260715"
        configs[name] = base_config(name, size)

    name = "paper_clean_coarse_osm_only_160_fold0_20260715"
    configs[name] = base_config(name, 160)
    remove_fine_osm_probe(configs[name])

    name = "paper_clean_no_osm_160_fold0_20260715"
    configs[name] = base_config(name, 160)
    remove_all_osm(configs[name])

    name = "paper_clean_probe_nohardneg_160_fold0_20260715"
    configs[name] = base_config(name, 160)
    configs[name]["training"]["semantic_probe_hard_negative_ratio"] = 0.0
    configs[name]["training"]["semantic_probe_hard_negative_weight"] = 0.0

    name = "paper_clean_no_highres_160_fold0_20260715"
    configs[name] = base_config(name, 160)
    remove_highres(configs[name])

    name = "paper_clean_no_masking_160_fold0_20260715"
    configs[name] = base_config(name, 160)
    configs[name]["training"]["input_masking"]["enabled"] = False
    return configs


def main() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, cfg in variants().items():
        path = CONFIG_DIR / f"{name}.yaml"
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
