"""V4 DINOv3 蒸馏损失测试。"""
import torch

from xuannv_embedding.training.losses import DinoDistillLoss


def test_distill_loss_basic():
    loss_fn = DinoDistillLoss(embed_dim=64, teacher_dim=1024, max_tokens=64)
    emb = torch.randn(2, 3, 64, 128, 128, requires_grad=True)  # [B,T,C,H,W]
    teacher = torch.randn(2, 32, 32, 1024)
    valid = torch.tensor([True, True])
    cos_loss, gram_loss, stats = loss_fn(emb, teacher, valid)
    assert cos_loss.requires_grad and gram_loss.requires_grad
    assert cos_loss.item() > 0 and gram_loss.item() > 0
    assert stats["distill_valid_samples"].item() == 2.0
    (cos_loss + gram_loss).backward()
    assert loss_fn.proj.weight.grad is not None


def test_distill_loss_partial_valid():
    loss_fn = DinoDistillLoss(embed_dim=64, teacher_dim=1024, max_tokens=64)
    emb = torch.randn(2, 64, 128, 128)  # 4 维兼容
    teacher = torch.randn(2, 32, 32, 1024)
    valid = torch.tensor([True, False])
    cos_loss, gram_loss, stats = loss_fn(emb, teacher, valid)
    assert stats["distill_valid_samples"].item() == 1.0
    (cos_loss + gram_loss).backward()


def test_distill_loss_no_valid():
    loss_fn = DinoDistillLoss(embed_dim=64, teacher_dim=1024, max_tokens=64)
    emb = torch.randn(1, 64, 128, 128)
    teacher = torch.zeros(1, 32, 32, 1024)
    valid = torch.tensor([False])
    cos_loss, gram_loss, stats = loss_fn(emb, teacher, valid)
    # 无有效样本时损失为 0 但梯度图完整（DDP 安全）
    total = cos_loss + gram_loss
    total.backward()
    assert total.item() == 0.0


def test_distill_perfect_alignment_low_loss():
    """学生与教师结构一致时 Gram 损失应显著低于随机。"""
    torch.manual_seed(0)
    loss_fn = DinoDistillLoss(embed_dim=64, teacher_dim=64, max_tokens=64)
    emb = torch.randn(1, 64, 128, 128)
    pooled = torch.nn.functional.adaptive_avg_pool2d(emb, (32, 32))
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    teacher_same = pooled.permute(0, 2, 3, 1)  # 与学生完全同构
    _, gram_same, _ = loss_fn(emb, teacher_same, torch.tensor([True]))
    teacher_rand = torch.randn(1, 32, 32, 64)
    _, gram_rand, _ = loss_fn(emb, teacher_rand, torch.tensor([True]))
    assert gram_same.item() < gram_rand.item() * 0.1
