"""v3/P14 新增损失的单元测试：原型对比与边界对比。"""

import torch

from xuannv_embedding.training.losses import (
    SemanticPrototypeContrastLoss,
    semantic_boundary_contrast_loss,
)

TASKS = ["osm_building", "osm_water"]


def _fake_inputs(batch=2, months=2, dim=8, size=16):
    torch.manual_seed(0)
    emb_map = torch.randn(batch, months, dim, size, size, requires_grad=True)
    labels = {
        "osm_building": (torch.rand(batch, size, size) > 0.7).float(),
        "osm_water": (torch.rand(batch, size, size) > 0.8).float(),
    }
    masks = {t: torch.ones(batch) for t in TASKS}
    return emb_map, labels, masks


def test_prototype_contrast_basic():
    emb_map, labels, masks = _fake_inputs()
    loss_fn = SemanticPrototypeContrastLoss(embed_dim=8, tasks=TASKS)
    loss, stats = loss_fn(emb_map, labels, masks)
    assert loss.requires_grad
    assert torch.isfinite(loss)
    loss.backward()
    assert emb_map.grad is not None
    assert loss_fn.prototypes.grad is not None
    assert stats["prototype_contrast_positive_pixels"].item() > 0


def test_prototype_contrast_empty_labels():
    emb_map, _, _ = _fake_inputs()
    loss_fn = SemanticPrototypeContrastLoss(embed_dim=8, tasks=TASKS)
    loss, _ = loss_fn(emb_map, None, None)
    assert loss.item() == 0.0


def test_boundary_contrast_basic():
    emb_map, labels, masks = _fake_inputs()
    loss, stats = semantic_boundary_contrast_loss(
        emb_map, labels, masks, tasks=TASKS
    )
    assert loss.requires_grad
    assert torch.isfinite(loss)
    loss.backward()
    assert emb_map.grad is not None
    assert stats["boundary_contrast_pairs"].item() > 0


def test_boundary_contrast_no_tasks():
    emb_map, labels, masks = _fake_inputs()
    loss, _ = semantic_boundary_contrast_loss(emb_map, labels, masks, tasks=[])
    assert loss.item() == 0.0
