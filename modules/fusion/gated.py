import torch
import torch.nn as nn

from .base import BaseFusion
from ..blocks import ResBlock


class WarpGatedFusion(BaseFusion):
    """
    Warp prev/next to cur, then learn per-pixel weights for [prev_w, cur, next_w].
    """
    def __init__(self, feat_ch: int = 64, num_blocks: int = 1):
        super().__init__(feat_ch=feat_ch)
        self.gate_net = nn.Sequential(
            nn.Conv2d(feat_ch * 3 + 4, feat_ch, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feat_ch, 3, 1, 1, 0),  # logits for 3 weights
        )
        self.refine = nn.Sequential(*[ResBlock(feat_ch) for _ in range(num_blocks)])

    def forward(self, prev, cur, nxt, flow_cur_to_prev, flow_cur_to_next):
        prev_w = self.warp(prev, flow_cur_to_prev)
        nxt_w  = self.warp(nxt,  flow_cur_to_next)

        gate_in = torch.cat([prev_w, cur, nxt_w, flow_cur_to_prev, flow_cur_to_next], dim=1)
        w = torch.softmax(self.gate_net(gate_in), dim=1)  # (B,3,H,W)

        out = w[:, 0:1] * prev_w + w[:, 1:2] * cur + w[:, 2:3] * nxt_w
        return self.refine(out)
