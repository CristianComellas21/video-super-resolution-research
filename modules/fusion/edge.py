import torch
import torch.nn as nn

from .base import BaseFusion
from ..blocks import ResBlock


class WarpConcatEdgeFusion(BaseFusion):
    """
    Edge fusion for boundary frames.

    API:
      forward(side, cur, flow_cur_to_side) -> updated_cur
    """

    def __init__(self, feat_ch: int = 64, num_blocks: int = 2):
        super().__init__(feat_ch=feat_ch)
        self.conv_in = nn.Conv2d(feat_ch * 2, feat_ch, 3, 1, 1)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.blocks = nn.Sequential(*[ResBlock(feat_ch) for _ in range(num_blocks)])

    def forward(self, side: torch.Tensor, cur: torch.Tensor, flow_cur_to_side: torch.Tensor) -> torch.Tensor:
        side_w = self.warp(side, flow_cur_to_side)
        x = torch.cat([side_w, cur], dim=1)
        x = self.act(self.conv_in(x))
        return self.blocks(x)