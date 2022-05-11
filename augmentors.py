
## Augmentation
from __future__ import annotations

import torch
from datasets import Dataset
from typing import Tuple

from torch._C import device

def get_model_inputs(x, device):
    ids = torch.tensor(x['input_ids']).to(device)
    mask = torch.tensor(x['attention_mask']).to(device)
    return ids, mask

class Augmentor():
    """Base class for text augmentors."""
    def __init__(self):
        pass

    def augment(self, x: Dataset) -> Dataset:
        raise NotImplementedError(f"Dataset.augment should be implemented.")

    def __call__(
            self, x: Dataset, device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.augment(x, device=device)

class Identity(Augmentor):
    def __init__(self):
        super(Identity, self).__init__()

    def augment(self, x: Dataset, device='cpu') -> Dataset:
        device = torch.device(device)
        return get_model_inputs(x, device)

class RandomSampling(Augmentor):
    def __init__(self):
        super(RandomSampling, self).__init__()

    def augment(self, x: Dataset, device='cpu') -> Dataset:
        ids, mask = [], []
        for data in x:
          l = len(data['text'])
          upper = l - 510
          random = random.randint(1, upper)
          ids.append([data['input_ids'][0]] + data['input_ids'][random:511] + [data['input_ids'][l-1]])
          mask.append([data['attention_mask'][0]] + data['attention_mask'][random:511] + data['attention_mask'][l-1])
        device = torch.device(device)
        ids_tensor = torch.tensor(ids).to(device)
        mask_tensor = torch.tensor(mask).to(device)
        return ids_tensor, mask_tensor