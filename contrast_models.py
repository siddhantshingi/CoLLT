import torch
from losses import Loss

class WithinEmbedContrast(torch.nn.Module):
    def __init__(self, loss: Loss, **kwargs):
        super(WithinEmbedContrast, self).__init__()
        self.loss = loss
        self.kwargs = kwargs

    def forward(self, h1, h2):
        l1 = self.loss(anchor=h1, sample=h2, **self.kwargs)
        l2 = self.loss(anchor=h2, sample=h1, **self.kwargs)
        return (l1 + l2) * 0.5

class WithinEmbedContrastMultiple(torch.nn.Module):
    def __init__(self, loss: Loss, **kwargs):
        super(WithinEmbedContrast, self).__init__()
        self.loss = loss
        self.kwargs = kwargs

    def forward(self, hs):
        n = len(hs)
        l = 0
        for i in range(n):
            for j in range(i+1, n):
                l1 = self.loss(anchor=hs[i], sample=hs[j], **self.kwargs)
                l2 = self.loss(anchor=hs[j], sample=hs[i], **self.kwargs)
                l += (l1 + l2) * 0.5
        return l
