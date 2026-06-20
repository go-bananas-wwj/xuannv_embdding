from __future__ import annotations

from pathlib import Path

import pytest
import torch
from downstreams.tasks.construction_segmentation import ConstructionSegmentationTask, FocalDiceLoss
from downstreams.utils.config import load_config
from torch import nn
from torch.utils.data import DataLoader


class DummyDataset:
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "embedding_map": torch.randn(2, 16, 16),
            "mask": torch.randint(0, 2, (16, 16)),
        }


class DummyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 输出 (B, 2, H, W) logits
        return torch.zeros(x.size(0), 2, x.size(2), x.size(3)) + self.bias


def test_task_build() -> None:
    cfg = load_config(Path("downstreams/configs/construction_segmentation.yaml"))
    task_cfg = {**cfg["training"], **cfg["data"]}
    task = ConstructionSegmentationTask(task_cfg)
    head = task.build_head()
    assert head is not None


def test_focal_dice_loss() -> None:
    loss_fn = FocalDiceLoss()
    logits = torch.randn(2, 16, 16, requires_grad=True)
    target = torch.randint(0, 2, (2, 16, 16))
    loss = loss_fn(logits, target)
    assert loss.ndim == 0
    loss.backward()
    assert logits.grad is not None


def test_build_loss_bce() -> None:
    cfg = {
        "head_type": "linear",
        "embed_dim": 64,
        "num_classes": 2,
        "loss": "bce",
        "pos_weight": 10.0,
    }
    task = ConstructionSegmentationTask(cfg)
    loss_fn = task.build_loss()
    assert isinstance(loss_fn, nn.BCEWithLogitsLoss)


def test_build_loss_unknown() -> None:
    cfg = {"head_type": "linear", "embed_dim": 64, "num_classes": 2, "loss": "foo"}
    task = ConstructionSegmentationTask(cfg)
    with pytest.raises(ValueError):
        task.build_loss()


def test_train_one_epoch() -> None:
    cfg = {"head_type": "linear", "embed_dim": 64, "num_classes": 2, "loss": "focal_dice"}
    task = ConstructionSegmentationTask(cfg)
    model = DummyHead()
    loader = DataLoader(DummyDataset(4), batch_size=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = task.train_one_epoch(model, loader, optimizer, torch.device("cpu"))
    assert isinstance(loss, float)


def test_evaluate() -> None:
    cfg = {"head_type": "linear", "embed_dim": 64, "num_classes": 2, "loss": "focal_dice"}
    task = ConstructionSegmentationTask(cfg)
    model = DummyHead()
    loader = DataLoader(DummyDataset(4), batch_size=2)
    metrics = task.evaluate(model, loader, torch.device("cpu"))
    assert "miou" in metrics
    assert "f1_0.5" in metrics
