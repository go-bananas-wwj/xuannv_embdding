from __future__ import annotations

from pathlib import Path

from downstreams.utils.config import load_config


CONFIG_DIR = Path(__file__).parent / "configs"

TASKS = {
    "construction_upernet_bitemporal": {
        "head_type": "upernet",
        "pos_weight": 500,
        "pos_prior": 0.012,
        "loss": "bce_dice_tversky",
    },
    "building_change_linear_bitemporal": {
        "head_type": "linear",
        "pos_weight": 2000,
        "pos_prior": 0.0015,
        "loss": "focal_tversky",
    },
    "farm_change_unet_bitemporal": {
        "head_type": "unet",
        "pos_weight": 2000,
        "pos_prior": 0.0008,
        "loss": "focal_tversky",
    },
    "rubbish_fcn_bitemporal": {
        "head_type": "fcn",
        "pos_weight": 2000,
        "pos_prior": 0.0008,
        "loss": "focal_tversky",
    },
    "construction_upernet_joint_bitemporal": {
        "head_type": "upernet",
        "pos_weight": 500,
        "pos_prior": 0.012,
        "loss": "bce_dice_tversky",
    },
}


def validate() -> None:
    for name, expected in TASKS.items():
        path = CONFIG_DIR / f"{name}.yaml"
        assert path.exists(), f"missing config: {path}"
        cfg = load_config(path)

        exp = cfg["experiment"]
        assert "name" in exp and exp["name"], f"{name}: missing experiment.name"

        tr = cfg["training"]
        assert tr["head_type"] == expected["head_type"], f"{name}: head_type mismatch"
        assert tr["months"] == [202512, 202605], f"{name}: months mismatch"
        assert tr["loss"] == expected["loss"], f"{name}: loss mismatch"
        assert tr["pos_weight"] == expected["pos_weight"], f"{name}: pos_weight mismatch"
        assert tr["tversky_beta"] == 0.7, f"{name}: tversky_beta mismatch"
        assert tr["tversky_weight"] == 2.0, f"{name}: tversky_weight mismatch"
        assert tr["early_stop_metric"] == "f1_best", f"{name}: early_stop_metric mismatch"
        assert tr["early_stop_patience"] == 30, f"{name}: early_stop_patience mismatch"
        assert tr["batch_size"] == 8, f"{name}: batch_size mismatch"
        assert tr["lr"] == 3.0e-5, f"{name}: lr mismatch"
        assert tr["epochs"] == 200, f"{name}: epochs mismatch"
        assert tr["use_weighted_sampler"] is True, f"{name}: use_weighted_sampler mismatch"
        assert tr["use_threshold_tuning"] is True, f"{name}: use_threshold_tuning mismatch"
        assert tr["pos_prior"] == expected["pos_prior"], f"{name}: pos_prior mismatch"

        if expected["loss"] == "focal_tversky":
            assert tr["focal_gamma"] == 2.0, f"{name}: focal_gamma mismatch"

        data = cfg["data"]
        assert data["embed_dim"] == 64, f"{name}: embed_dim mismatch"
        assert data["num_classes"] == 2, f"{name}: num_classes mismatch"

        print(f"✓ {name}")

    print("All bitemporal configs validated successfully.")


if __name__ == "__main__":
    validate()
