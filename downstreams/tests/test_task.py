from __future__ import annotations

from pathlib import Path

import pytest
import torch
from downstreams.heads.linear_probe import LinearProbeHead
from downstreams.tasks.construction_segmentation import (
    BCEDiceTverskyLoss,
    ConstructionSegmentationTask,
    FocalDiceLoss,
)
from downstreams.utils.config import load_config
from torch import nn
from torch.utils.data import DataLoader

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "construction_segmentation.yaml"


class DummyDataset:
    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
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
    cfg = load_config(CONFIG_PATH)
    assert cfg["experiment"]["seed"] == 42
    assert cfg["training"]["epochs"] == 100
    assert cfg["data"]["embed_dim"] == 64
    task = ConstructionSegmentationTask(cfg)
    head = task.build_head()
    assert head is not None


def test_build_head_initializes_foreground_bias_from_positive_prior() -> None:
    cfg = {
        "training": {"head_type": "linear", "loss": "focal_dice", "pos_prior": 0.04},
        "data": {"embed_dim": 64, "num_classes": 2},
    }
    task = ConstructionSegmentationTask(cfg)
    head = task.build_head()
    assert isinstance(head, LinearProbeHead)
    expected = torch.logit(torch.tensor(0.04))
    assert head.conv.bias[1].item() == pytest.approx(expected.item())
    assert head.conv.bias[0].item() == pytest.approx(0.0)


def test_focal_dice_loss() -> None:
    loss_fn = FocalDiceLoss()
    logits = torch.randn(2, 16, 16, requires_grad=True)
    target = torch.randint(0, 2, (2, 16, 16))
    loss = loss_fn(logits, target)
    assert loss.ndim == 0
    loss.backward()
    assert logits.grad is not None


def test_focal_dice_loss_uses_positive_weight() -> None:
    logits = torch.tensor([[[0.0, 0.0]]])
    target = torch.tensor([[[1.0, 0.0]]])
    unweighted = FocalDiceLoss(pos_weight=1.0)(logits, target)
    weighted = FocalDiceLoss(pos_weight=20.0)(logits, target)
    assert weighted > unweighted


def test_bce_dice_tversky_loss_penalizes_false_negatives_more() -> None:
    target = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 0.0]],
        ]
    )
    false_negative_logits = torch.tensor(
        [
            [[-4.0, -4.0], [-4.0, -4.0]],
            [[-4.0, -4.0], [-4.0, -4.0]],
        ]
    )
    false_positive_logits = torch.tensor(
        [
            [[4.0, 4.0], [-4.0, -4.0]],
            [[4.0, 4.0], [-4.0, -4.0]],
        ]
    )

    loss_fn = BCEDiceTverskyLoss(pos_weight=1.0, tversky_beta=0.8)

    assert loss_fn(false_negative_logits, target) > loss_fn(false_positive_logits, target)


def test_build_loss_bce_dice_tversky() -> None:
    cfg = {
        "training": {
            "head_type": "linear",
            "loss": "bce_dice_tversky",
            "pos_weight": 150.0,
            "tversky_beta": 0.7,
        },
        "data": {"embed_dim": 64, "num_classes": 2},
    }
    task = ConstructionSegmentationTask(cfg)
    loss_fn = task.build_loss()
    assert isinstance(loss_fn, BCEDiceTverskyLoss)
    assert loss_fn.pos_weight.item() == pytest.approx(150.0)
    assert loss_fn.tversky_beta == pytest.approx(0.7)


def test_build_loss_bce() -> None:
    cfg = {
        "training": {"head_type": "linear", "loss": "bce", "pos_weight": 10.0},
        "data": {"embed_dim": 64, "num_classes": 2},
    }
    task = ConstructionSegmentationTask(cfg)
    loss_fn = task.build_loss()
    assert isinstance(loss_fn, nn.BCEWithLogitsLoss)


def test_build_loss_unknown() -> None:
    cfg = {
        "training": {"head_type": "linear", "loss": "foo"},
        "data": {"embed_dim": 64, "num_classes": 2},
    }
    task = ConstructionSegmentationTask(cfg)
    with pytest.raises(ValueError):
        task.build_loss()


def test_train_one_epoch() -> None:
    cfg = {
        "training": {"head_type": "linear", "loss": "focal_dice"},
        "data": {"embed_dim": 64, "num_classes": 2},
    }
    task = ConstructionSegmentationTask(cfg)
    model = DummyHead()
    loader = DataLoader(DummyDataset(4), batch_size=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = task.train_one_epoch(model, loader, optimizer, torch.device("cpu"))
    assert isinstance(loss, float)


def test_evaluate() -> None:
    cfg = {
        "training": {"head_type": "linear", "loss": "focal_dice"},
        "data": {"embed_dim": 64, "num_classes": 2},
    }
    task = ConstructionSegmentationTask(cfg)
    model = DummyHead()
    loader = DataLoader(DummyDataset(4), batch_size=2)
    metrics = task.evaluate(model, loader, torch.device("cpu"))
    assert "miou" in metrics
    assert "f1_0.5" in metrics
