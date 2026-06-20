from __future__ import annotations

# DataLoader 工厂
from torch.utils.data import DataLoader

from xuannv_embedding.config import DataConfig
from xuannv_embedding.data.collate import collate_fn
from xuannv_embedding.data.dataset import MonthlyEmbeddingDataset


def build_dataloader(
    cfg: DataConfig,
    split: str = "train",
) -> DataLoader:
    """根据 ``DataConfig`` 构造 ``DataLoader``。

    统计量目录优先使用 ``cfg.statistics_dir``；若未提供，
    则按约定从 ``processed/{region}`` 推导为 ``{data_root}/statistics/{region}``。

    参数:
        cfg: 数据配置。
        split: 数据集划分，``train`` 会启用 shuffle 与 drop_last。

    返回:
        配置好的 ``DataLoader``。
    """
    if cfg.statistics_dir is not None:
        statistics_dir = cfg.statistics_dir
    else:
        statistics_dir = cfg.root.parent / "statistics" / cfg.region

    dataset = MonthlyEmbeddingDataset(
        manifest_path=cfg.manifest_path,
        statistics_dir=statistics_dir,
        sources=cfg.sources,
        patch_size=cfg.patch_size,
        max_patches=cfg.max_patches,
        num_months=cfg.num_months,
    )

    shuffle = split == "train"
    return DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=shuffle,
    )
