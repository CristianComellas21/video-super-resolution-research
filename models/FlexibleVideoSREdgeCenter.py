from enum import Enum
from typing import Sequence, Optional, Union
import copy

import torch
import torch.nn as nn

from basicsr.utils.registry import ARCH_REGISTRY

from modules.feature_extractor import FeatureExtractor
from modules.upsampler import UpsamplerBest as Upsampler


class OutputMode(str, Enum):
    CENTER = "center"
    ALL = "all"


class FillMode(str, Enum):
    CYCLE = "cycle"
    HOLD_LAST = "last"


@ARCH_REGISTRY.register()
class Model(nn.Module):
    """
    Flexible non-recurrent iterative refinement with separate fusion modules:

    - Center fusion (for t in [1, ..., T-2]):
      forward(prev, cur, nxt, flow_cur_to_prev, flow_cur_to_next) -> updated_cur

    - Edge fusion (for t = 0 and t = T-1):
      forward(side, cur, flow_cur_to_side) -> updated_cur

    For each stage, two modules are used: one center module and one edge module.
    """

    def __init__(
        self,
        in_channels: int = 3,
        feat_ch: int = 64,
        num_feat_blocks: int = 5,
        num_stages: int = 3,
        scale: int = 4,
        freeze_flownet: bool = True,
        flownet: nn.Module | None = None,
        output_mode: Union[OutputMode, str] = OutputMode.CENTER,
        center_fusions: Optional[Sequence[nn.Module]] = None,
        edge_fusions: Optional[Sequence[nn.Module]] = None,
        share_center_fusion: bool = True,
        share_edge_fusion: bool = True,
        fill_mode: Union[FillMode, str] = FillMode.HOLD_LAST,
    ):
        super().__init__()
        self.scale = scale
        self.output_mode = OutputMode(output_mode)
        self.freeze_flownet = freeze_flownet
        self.num_stages = int(num_stages)
        assert self.num_stages >= 1

        self.flownet = flownet if flownet is not None else nn.Module()
        self.feat_extractor = FeatureExtractor(in_ch=in_channels, feat_ch=feat_ch, num_blocks=num_feat_blocks)
        self.upsampler = Upsampler(feat_ch=feat_ch, scale=scale, out_ch=in_channels)

        if freeze_flownet:
            for p in self.flownet.parameters():
                p.requires_grad = False

        if center_fusions is None or len(center_fusions) == 0:
            raise ValueError("ModelEdgeCenter requires `center_fusions` as a non-empty list.")
        if edge_fusions is None or len(edge_fusions) == 0:
            raise ValueError("ModelEdgeCenter requires `edge_fusions` as a non-empty list.")

        fill_mode = FillMode(fill_mode)
        self.center_fusions = self._normalize_fusions(
            fusions=list(center_fusions),
            num_stages=self.num_stages,
            share_fusion=bool(share_center_fusion),
            fill_mode=fill_mode,
        )
        self.edge_fusions = self._normalize_fusions(
            fusions=list(edge_fusions),
            num_stages=self.num_stages,
            share_fusion=bool(share_edge_fusion),
            fill_mode=fill_mode,
        )

    @staticmethod
    def _normalize_fusions(
        fusions: list[nn.Module],
        num_stages: int,
        share_fusion: bool,
        fill_mode: FillMode,
    ) -> nn.ModuleList:
        assert num_stages >= 1

        if share_fusion:
            return nn.ModuleList([fusions[0]])

        n = len(fusions)
        if n == num_stages:
            return nn.ModuleList(fusions)
        if n > num_stages:
            return nn.ModuleList(fusions[:num_stages])
        if n == 1:
            base = fusions[0]
            return nn.ModuleList([copy.deepcopy(base) for _ in range(num_stages)])

        out: list[nn.Module] = []
        if fill_mode == FillMode.CYCLE:
            for i in range(num_stages):
                out.append(copy.deepcopy(fusions[i % n]))
        else:
            out.extend(fusions)
            last = fusions[-1]
            for _ in range(num_stages - n):
                out.append(copy.deepcopy(last))

        return nn.ModuleList(out)

    def compute_flows_adjacent(self, lqs: torch.Tensor):
        n, t, c, h, w = lqs.shape
        lqs_0 = lqs[:, :-1].reshape(-1, c, h, w)
        lqs_1 = lqs[:, 1:].reshape(-1, c, h, w)

        if self.freeze_flownet:
            self.flownet.eval()

        ctx = torch.no_grad() if self.freeze_flownet else torch.enable_grad()
        with ctx:
            flow_fwd = self.flownet(lqs_0, lqs_1).view(n, t - 1, 2, h, w)
            flow_bwd = self.flownet(lqs_1, lqs_0).view(n, t - 1, 2, h, w)

        return flow_fwd, flow_bwd

    @staticmethod
    def _pad_flow_time_zeros(flow: torch.Tensor, left_zeros: int, right_zeros: int) -> torch.Tensor:
        if flow.numel() == 0:
            return flow
        n, _, c, h, w = flow.shape
        z_left = flow.new_zeros(n, left_zeros, c, h, w)
        z_right = flow.new_zeros(n, right_zeros, c, h, w)
        return torch.cat([z_left, flow, z_right], dim=1)

    def forward(self, lqs: torch.Tensor) -> torch.Tensor:
        n, t, c, h, w = lqs.shape
        assert t >= 1

        feats = self.feat_extractor(lqs.reshape(-1, c, h, w)).reshape(n, t, -1, h, w)

        if t == 1:
            flow_cur_to_prev = lqs.new_zeros(n, 1, 2, h, w)
            flow_cur_to_next = lqs.new_zeros(n, 1, 2, h, w)
        else:
            flow_fwd, flow_bwd = self.compute_flows_adjacent(lqs)
            flow_cur_to_prev = self._pad_flow_time_zeros(flow_bwd, left_zeros=1, right_zeros=0)
            flow_cur_to_next = self._pad_flow_time_zeros(flow_fwd, left_zeros=0, right_zeros=1)

        for s in range(self.num_stages):
            center_fusion = self.center_fusions[0] if len(self.center_fusions) == 1 else self.center_fusions[s]
            edge_fusion = self.edge_fusions[0] if len(self.edge_fusions) == 1 else self.edge_fusions[s]

            feats_new = feats.clone()

            if t >= 3:
                prev_mid = feats[:, 0 : t - 2]
                cur_mid = feats[:, 1 : t - 1]
                nxt_mid = feats[:, 2:t]

                fp2c_mid = flow_cur_to_prev[:, 1 : t - 1]
                fn2c_mid = flow_cur_to_next[:, 1 : t - 1]

                mid_out = center_fusion(
                    prev_mid.reshape(-1, prev_mid.size(2), h, w),
                    cur_mid.reshape(-1, cur_mid.size(2), h, w),
                    nxt_mid.reshape(-1, nxt_mid.size(2), h, w),
                    fp2c_mid.reshape(-1, 2, h, w),
                    fn2c_mid.reshape(-1, 2, h, w),
                )
                feats_new[:, 1 : t - 1] = mid_out.reshape(n, t - 2, -1, h, w)

            if t >= 2:
                left_out = edge_fusion(
                    feats[:, 1],
                    feats[:, 0],
                    flow_cur_to_next[:, 0],
                )
                right_out = edge_fusion(
                    feats[:, t - 2],
                    feats[:, t - 1],
                    flow_cur_to_prev[:, t - 1],
                )
                feats_new[:, 0] = left_out
                feats_new[:, t - 1] = right_out

            feats = feats_new

        if self.output_mode == OutputMode.CENTER:
            idx = t // 2
            sr = self.upsampler(feats[:, idx], lqs[:, idx]).unsqueeze(1)
            return sr

        outs = [self.upsampler(feats[:, i], lqs[:, i]) for i in range(t)]
        return torch.stack(outs, dim=1)