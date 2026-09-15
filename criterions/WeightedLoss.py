import torch
from torch import nn

class Criterion(nn.Module):
    def __init__(self, loss_dict: dict[str, tuple[nn.Module, float]], normalize=False):
        super().__init__()

        self.losses = nn.ModuleDict()
        weights = []

        for name, (loss_fn, weight) in loss_dict.items():
            self.losses[name] = loss_fn
            weights.append(weight)

        weights = torch.tensor(weights, dtype=torch.float32)
        if normalize:
            weights = weights / weights.sum()

        self.register_buffer("weights", weights)
        self.names = list(loss_dict.keys())
        

    def forward(self, pred, target):
        total = 0.0
        log_dict = {}

        for name, weight in zip(self.names, self.weights):
            value = self.losses[name](pred, target)
            total = total + weight * value
            log_dict[name] = value.detach()

        return total, log_dict
