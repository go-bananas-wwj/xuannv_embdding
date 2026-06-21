from __future__ import annotations

import torch

from xuannv_embedding.utils.device import get_device


def get_downstream_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        return get_device()
    return torch.device(preference)
