from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    base = raw.pop("_base_", None)
    if base:
        base_path = path.parent / base
        with open(base_path, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f)
        base_cfg.update(raw)
        return base_cfg
    return raw
