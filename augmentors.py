
## Augmentation
from __future__ import annotations

import torch
from datasets import Dataset
from typing import Optional, Tuple, NamedTuple, List

class Augmentor():
    """Base class for graph augmentors."""
    def __init__(self):
        pass

    def augment(self, x: Dataset) -> Dataset:
        raise NotImplementedError(f"Dataset.augment should be implemented.")

    def __call__(
            self, x: Dataset,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.augment(x)

class Identity(Augmentor):
    def __init__(self):
        super(Identity, self).__init__()

    def augment(self, x: Dataset) -> Dataset:
        return x