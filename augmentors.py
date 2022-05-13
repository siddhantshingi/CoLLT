
## Augmentation
from __future__ import annotations

import torch
from datasets import Dataset
from typing import Tuple
import random

from torch._C import device

def get_model_inputs(x, device):
    ids = torch.tensor(x['input_ids']).to(device)
    mask = torch.tensor(x['attention_mask']).to(device)
    return ids, mask

class Augmentor():
    """Base class for text augmentors."""
    def __init__(self):
        pass

    def augment(self, x: Dataset, idx: int, device: str):
        raise NotImplementedError(f"Dataset.augment should be implemented.")

    def __call__(
            self, x: Dataset, idx: int, device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.augment(x, idx=idx, device=device)

class Identity(Augmentor):
    def __init__(self):
        super(Identity, self).__init__()

    def augment(self, x: Dataset, idx=None, device='cpu'):
        device = torch.device(device)
        return get_model_inputs(x, device)

class RandomSampling(Augmentor):
    def __init__(self):
        super(RandomSampling, self).__init__()

    def augment(self, x: Dataset, idx=None, device='cpu'):
        ids, mask = [], []
        l = len(x['text'])
        for i in range(l):
            if len(x["input_ids"][i]) <= 512:
                ids.append(x["input_ids"][i] + [0]*(512 - len(x["input_ids"][i])))
                mask.append(x["attention_mask"][i] + [0]*(512 - len(x["attention_mask"][i])))
                continue
            upper = len(x['input_ids'][i]) - 510
            r = random.randint(1, upper)

            ids.append([x['input_ids'][i][0]] + x['input_ids'][i][r:r+510] + [x['input_ids'][i][-1]])
            mask.append([x['attention_mask'][i][0]] + x['attention_mask'][i][r:r+510] + [x['attention_mask'][i][-1]])
        device = torch.device(device)
        ids_tensor = torch.tensor(ids).to(device)
        mask_tensor = torch.tensor(mask).to(device)
        return ids_tensor, mask_tensor

class NonOverlappingChunks(Augmentor):
    def __init__(self):
        super(NonOverlappingChunks, self).__init__()

    def augment(self, x: Dataset, idx=None, device='cpu'):
        ids, mask = [], []
        l = len(x['text'])
        for i in range(l):
            input_x = x["input_ids"][i]
            input_len = len(x["input_ids"][i])
            cls_id = [x['input_ids'][i][0]]
            cls_mask = [x['attention_mask'][i][0]]
            sep_id = [x['input_ids'][i][-1]]
            sep_mask = [x['attention_mask'][i][-1]]
            start = 1
            while(input_len>512):
              ids.append(cls_id + x['input_ids'][i][start:start + 511] + sep_id)
              mask.append(cls_mask+ x['attention_mask'][i][start:start + 511] + sep_mask)
              start += 511
              input_len -= 510
            if input_len <= 512:
                ids.append(cls_id + x["input_ids"][i] + [0]*(511 - input_len))
                mask.append(cls_mask + x["attention_mask"][i] + [0]*(511 - input_len))
                
        device = torch.device(device)
        ids_tensor = torch.tensor(ids).to(device)
        mask_tensor = torch.tensor(mask).to(device)
        return ids_tensor, mask_tensor

class OverlappingChunks(Augmentor):
    def __init__(self):
        super(OverlappingChunks, self).__init__()

    def augment(self, x: Dataset, idx=None, device='cpu'):
        ids, mask = [], []
        l = len(x['text'])
        for i in range(l):
            input_x = x["input_ids"][i]
            input_len = len(x["input_ids"][i])
            cls_id = [x['input_ids'][i][0]]
            cls_mask = [x['attention_mask'][i][0]]
            sep_id = [x['input_ids'][i][-1]]
            sep_mask = [x['attention_mask'][i][-1]]
            start = 1
            while(input_len>512):
              ids.append(cls_id + x['input_ids'][i][start:start + 511] + sep_id)
              mask.append(cls_mask+ x['attention_mask'][i][start:start + 511] + sep_mask)
              start += 311
              input_len -= 510
            if input_len <= 512:
                ids.append(cls_id + x["input_ids"][i] + [0]*(511 - input_len))
                mask.append(cls_mask + x["attention_mask"][i] + [0]*(511 - input_len))
                
        device = torch.device(device)
        ids_tensor = torch.tensor(ids).to(device)
        mask_tensor = torch.tensor(mask).to(device)
        return ids_tensor, mask_tensor