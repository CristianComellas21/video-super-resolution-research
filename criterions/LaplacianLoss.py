import torch
from torch.nn import functional as F
from torch import nn

class Criterion(nn.Module):
    def __init__(self, base_loss: nn.Module):
        super().__init__()

        self.base_loss = base_loss

        k = torch.tensor([[0, -1, 0],
                          [-1,  4, -1],
                          [0, -1, 0]], dtype=torch.float32)
        self.register_buffer("kernel", k.view(1, 1, 3, 3))

    def _to_nchw(self, x):
        if x.ndim == 5:
            x = x.flatten(0, 1)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return x

    def _lap(self, x):
        x = self._to_nchw(x)
        c = x.size(1)
        k = self.kernel.repeat(c, 1, 1, 1)
        return F.conv2d(x, k, padding=1, groups=c)

    def forward(self, pred, target):
        return self.base_loss(self._lap(pred), self._lap(target))
