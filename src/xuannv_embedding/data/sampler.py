from __future__ import annotations

import math
from collections.abc import Iterator

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class DistributedWeightedSampler(Sampler[int]):
    """Distributed weighted sampler with deterministic epoch shuffling.

    The sampler draws a single global weighted index list, pads it to be evenly
    divisible by the distributed world size, then returns this rank's slice.
    """

    def __init__(
        self,
        weights: torch.Tensor,
        num_replicas: int | None = None,
        rank: int | None = None,
        replacement: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"Invalid rank {rank} for num_replicas {num_replicas}")

        weights = torch.as_tensor(weights, dtype=torch.double)
        if weights.dim() != 1:
            raise ValueError("weights must be a 1D tensor")
        if len(weights) == 0:
            raise ValueError("weights must not be empty")
        if torch.any(weights < 0):
            raise ValueError("weights must be non-negative")
        if float(weights.sum().item()) <= 0:
            raise ValueError("weights must contain at least one positive value")

        self.weights = weights
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        if self.drop_last and len(self.weights) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.weights) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(self.weights) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=self.replacement,
            generator=generator,
        ).tolist()
        if not self.drop_last and len(indices) < self.total_size:
            indices += indices[: self.total_size - len(indices)]
        else:
            indices = indices[: self.total_size]
        rank_indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(rank_indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
