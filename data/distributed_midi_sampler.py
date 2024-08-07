from typing import Iterator

import torch
from torch.utils.data import DistributedSampler


class DistributedMidiSampler(DistributedSampler):
    def __init__(self, dataset_length: int, shuffle: bool = False, seed: int = 4):
        self.dataset_length = dataset_length
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            generator = torch.Generator()
            if self.seed is not None:
                generator.manual_seed(self.seed)
            return iter(torch.randperm(self.dataset_length, generator=generator).tolist())
        else:
            return iter(range(self.dataset_length))

    def __len__(self) -> int:
        return self.dataset_length
