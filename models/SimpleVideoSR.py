from enum import Enum
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.arch_util import flow_warp
from basicsr.archs.spynet_arch import SpyNet
from basicsr.utils.registry import ARCH_REGISTRY


# -------------------------
# Minimal blocks
# -------------------------
class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act(self.conv1(x)))


class FeatureExtractor(nn.Module):
    def __init__(self, in_ch: int = 3, feat_ch: int = 64, num_blocks: int = 5):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, feat_ch, 3, 1, 1)
        self.blocks = nn.Sequential(*[ResBlock(feat_ch) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.conv_in(x))


class Fusion(nn.Module):
    """Fusión mínima de 3 features alineadas: concat + conv + resblocks."""
    def __init__(self, feat_ch: int = 64, num_blocks: int = 2):
        super().__init__()
        self.conv_in = nn.Conv2d(feat_ch * 3, feat_ch, 3, 1, 1)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.blocks = nn.Sequential(*[ResBlock(feat_ch) for _ in range(num_blocks)])

    def forward(self, f_prev_w: torch.Tensor, f_cur: torch.Tensor, f_next_w: torch.Tensor) -> torch.Tensor:
        x = torch.cat([f_prev_w, f_cur, f_next_w], dim=1)
        x = self.act(self.conv_in(x))
        x = self.blocks(x)
        return x


class Upsampler(nn.Module):
    def __init__(self, feat_ch: int = 64, scale: int = 4, out_ch: int = 3):
        super().__init__()
        assert scale in (2, 4)
        self.scale = scale

        self.conv_before = nn.Conv2d(feat_ch, feat_ch, 3, 1, 1)
        self.up1 = nn.Conv2d(feat_ch, feat_ch * 4, 3, 1, 1)
        self.ps = nn.PixelShuffle(2)
        self.up2 = nn.Conv2d(feat_ch, feat_ch * 4, 3, 1, 1) if scale == 4 else None

        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.conv_hr = nn.Conv2d(feat_ch, feat_ch, 3, 1, 1)
        self.conv_out = nn.Conv2d(feat_ch, out_ch, 3, 1, 1)

    def forward(self, feat: torch.Tensor, lq: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv_before(feat))
        x = self.act(self.ps(self.up1(x)))
        if self.scale == 4:
            x = self.act(self.ps(self.up2(x)))
        x = self.act(self.conv_hr(x))
        x = self.conv_out(x)

        base = F.interpolate(lq, scale_factor=self.scale, mode="bilinear", align_corners=False)
        return x + base


# -------------------------
# Main arch
# -------------------------
class OutputMode(str, Enum):
    CENTER = "center"
    ALL = "all"


@ARCH_REGISTRY.register()
class Model(nn.Module):
    """
    IART-like (but non-recurrent) iterative refinement over the whole sequence:
      - Compute features for ALL frames at level 0
      - For each stage k: update ALL frame features from stage k-1 using (prev,cur,next) + flow warp
      - At the end: decode either ALL frames or only the center frame
      - ALWAYS returns 5D output: (N, T_out, C, sH, sW)
    """

    def __init__(
        self,
        in_channels: int = 3,
        feat_ch: int = 64,
        num_feat_blocks: int = 5,
        fusion_blocks: int = 2,
        num_stages: int = 3,
        share_fusion: bool = False,
        scale: int = 4,
        spynet_path=None,
        freeze_spynet: bool = True,
        output_mode: Union[OutputMode, str] = OutputMode.CENTER,
    ):
        super().__init__()
        self.scale = scale
        self.output_mode = OutputMode(output_mode)
        self.freeze_spynet = freeze_spynet
        self.num_stages = int(num_stages)
        assert self.num_stages >= 1

        self.spynet = SpyNet(spynet_path)
        self.feat_extractor = FeatureExtractor(in_ch=in_channels, feat_ch=feat_ch, num_blocks=num_feat_blocks)

        if share_fusion:
            self.fusions = nn.ModuleList([Fusion(feat_ch=feat_ch, num_blocks=fusion_blocks)])
        else:
            self.fusions = nn.ModuleList(
                [Fusion(feat_ch=feat_ch, num_blocks=fusion_blocks) for _ in range(self.num_stages)]
            )

        self.upsampler = Upsampler(feat_ch=feat_ch, scale=scale, out_ch=in_channels)

        if freeze_spynet:
            for p in self.spynet.parameters():
                p.requires_grad = False

    def compute_flows_adjacent(self, lqs: torch.Tensor):
        """
        lqs: (N, T, 3, H, W)

        Returns:
          flow_fwd: (N, T-1, 2, H, W)  flow i -> i+1
          flow_bwd: (N, T-1, 2, H, W)  flow i+1 -> i
        """
        N, T, C, H, W = lqs.shape
        lqs_0 = lqs[:, :-1].reshape(-1, C, H, W)
        lqs_1 = lqs[:, 1:].reshape(-1, C, H, W)

        ctx = torch.no_grad() if self.freeze_spynet else torch.enable_grad()
        with ctx:
            flow_fwd = self.spynet(lqs_0, lqs_1).view(N, T - 1, 2, H, W)  # i->i+1
            flow_bwd = self.spynet(lqs_1, lqs_0).view(N, T - 1, 2, H, W)  # i+1->i

        return flow_fwd, flow_bwd

    @staticmethod
    def _pad_time_replicate(x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, T, F, H, W) -> (N, T+2, F, H, W)
        Replicate border frames: [x0, x0, x1, ..., x_{T-1}, x_{T-1}]
        """
        left = x[:, 0:1]
        right = x[:, -1:]
        return torch.cat([left, x, right], dim=1)

    @staticmethod
    def _warp_feat(feat: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        # feat: (N,F,H,W), flow: (N,2,H,W) -> flow_warp expects (N,H,W,2)
        return flow_warp(feat, flow.permute(0, 2, 3, 1))

    def forward(self, lqs: torch.Tensor) -> torch.Tensor:
        """
        lqs: (N, T, 3, H, W)

        Returns ALWAYS:
          (N, T_out, 3, sH, sW)
            * CENTER -> T_out = 1
            * ALL    -> T_out = T
        """
        N, T, C, H, W = lqs.shape
        assert T >= 1

        # 0) Level-0 features for all frames
        feats = self.feat_extractor(lqs.reshape(-1, C, H, W)).reshape(N, T, -1, H, W)  # (N,T,F,H,W)

        # 1) Precompute adjacent flows once
        if T == 1:
            flow_fwd = lqs.new_zeros(N, 0, 2, H, W)
            flow_bwd = lqs.new_zeros(N, 0, 2, H, W)
        else:
            flow_fwd, flow_bwd = self.compute_flows_adjacent(lqs)

        # 2) Iterative refinement over the whole sequence
        for s in range(self.num_stages):
            fusion_mod = self.fusions[0] if len(self.fusions) == 1 else self.fusions[s]

            # replicate-pad in time so we can take prev/next without conditionals
            feats_pad = self._pad_time_replicate(feats)  # (N, T+2, F, H, W)

            prev = feats_pad[:, 0:T]       # (N,T,F,H,W) corresponds to t-1 (replicated at borders)
            cur  = feats_pad[:, 1:T+1]     # (N,T,F,H,W) corresponds to t
            nxt  = feats_pad[:, 2:T+2]     # (N,T,F,H,W) corresponds to t+1 (replicated at borders)

            if T == 1:
                feats = fusion_mod(prev[:, 0], cur[:, 0], nxt[:, 0]).unsqueeze(1)
                continue

            # flow_warp samples source at (target + flow), so to warp prev/nxt into cur
            # coordinates we need flows cur->prev and cur->next respectively.
            # We'll do it vectorized with two warps + border fix.

            # prev warp for t=1..T-1
            prev_w = prev.clone()
            prev_w[:, 1:] = self._warp_feat(
                prev[:, 1:].reshape(-1, prev.size(2), H, W),
                flow_bwd.reshape(-1, 2, H, W)
            ).reshape(N, T - 1, -1, H, W)

            # next warp for t=0..T-2
            nxt_w = nxt.clone()
            nxt_w[:, :-1] = self._warp_feat(
                nxt[:, :-1].reshape(-1, nxt.size(2), H, W),
                flow_fwd.reshape(-1, 2, H, W)
            ).reshape(N, T - 1, -1, H, W)

            # Now update feats for all t (vectorized via flatten in batch)
            # fusion_mod expects (N,F,H,W) so we flatten time into batch
            prev_w_flat = prev_w.reshape(-1, prev_w.size(2), H, W)
            cur_flat    = cur.reshape(-1, cur.size(2), H, W)
            nxt_w_flat  = nxt_w.reshape(-1, nxt_w.size(2), H, W)

            feats_new = fusion_mod(prev_w_flat, cur_flat, nxt_w_flat)  # (N*T,F,H,W)
            feats = feats_new.reshape(N, T, -1, H, W)

        # 3) Decode outputs
        if self.output_mode == OutputMode.CENTER:
            t = T // 2
            sr = self.upsampler(feats[:, t], lqs[:, t]).unsqueeze(1)  # (N,1,3,sH,sW)
            return sr

        # ALL frames
        outs = []
        for t in range(T):
            outs.append(self.upsampler(feats[:, t], lqs[:, t]))
        return torch.stack(outs, dim=1)  # (N,T,3,sH,sW)
