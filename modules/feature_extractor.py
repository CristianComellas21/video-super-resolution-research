import torch
import torch.nn as nn
from .blocks import ResBlock

class FeatureExtractor(nn.Module):
    def __init__(self, in_ch: int = 3, feat_ch: int = 64, num_blocks: int = 5):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, feat_ch, 3, 1, 1)
        self.blocks = nn.Sequential(*[ResBlock(feat_ch) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.conv_in(x))
