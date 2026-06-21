from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader


class BaseTask(ABC):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def build_head(self) -> nn.Module: ...

    @abstractmethod
    def build_loss(self) -> nn.Module: ...

    @abstractmethod
    def train_one_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> float: ...

    @abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        threshold: float | None = None,
    ) -> dict[str, float]: ...
