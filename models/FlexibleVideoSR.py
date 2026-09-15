from enum import Enum
from typing import Union, Sequence, Optional
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY

from modules.feature_extractor import FeatureExtractor
from modules.upsampler import UpsamplerBest as Upsampler

# -------------------------
# Main arch
# -------------------------
class OutputMode(str, Enum):
    CENTER = "center"
    ALL = "all"


class FillMode(str, Enum):
    CYCLE = "cycle"      # repeat modules in a loop: [A,B] -> A,B,A,B,...
    HOLD_LAST = "last"   # keep last: [A,B] -> A,B,B,B,...


@ARCH_REGISTRY.register()
class Model(nn.Module):
    """
    Flexible IART-like (non-recurrent) iterative refinement.
    Fusion modules are provided as a list and normalized to length `num_stages`.

    Fusion API (must match):
      forward(prev, cur, nxt, flow_prev_to_cur, flow_next_to_cur) -> updated_cur
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
        fusions: Optional[Sequence[nn.Module]] = None,
        share_fusion: bool = True,
        fill_mode: Union[FillMode, str] = FillMode.HOLD_LAST,
    ):
        super().__init__()
        self.scale = scale
        self.output_mode = OutputMode(output_mode)
        self.freeze_flownet = freeze_flownet
        self.num_stages = int(num_stages)
        assert self.num_stages >= 1

        self.flownet = flownet if flownet is not None else nn.Module()  # dummy if not provided
        self.feat_extractor = FeatureExtractor(in_ch=in_channels, feat_ch=feat_ch, num_blocks=num_feat_blocks)
        self.upsampler = Upsampler(feat_ch=feat_ch, scale=scale, out_ch=in_channels)

        if freeze_flownet:
            for p in self.flownet.parameters():
                p.requires_grad = False

        if fusions is None or len(fusions) == 0:
            raise ValueError("FlexibleVideoSR requires `fusions` as a non-empty list of Fusion modules.")

        self.fusions = self._normalize_fusions(
            fusions=list(fusions),
            num_stages=self.num_stages,
            share_fusion=bool(share_fusion),
            fill_mode=FillMode(fill_mode),
        )

    @staticmethod
    def _normalize_fusions(
        fusions: list[nn.Module],
        num_stages: int,
        share_fusion: bool,
        fill_mode: FillMode,
    ) -> nn.ModuleList:
        """
        Normalize user-provided fusion modules to a ModuleList.

        Rules:
          - If share_fusion=True:
              * If len(fusions)==1: ModuleList([that])
              * If len(fusions)>1:  ModuleList([fusions[0]])  (explicitly "shared" means one instance)
                (If you prefer to error instead, change here.)
          - If share_fusion=False:
              * If len(fusions)==num_stages: use as-is
              * If len(fusions)==1: deepcopy to length num_stages
              * If 1 < len(fusions) < num_stages:
                  - CYCLE: repeat in a loop
                  - HOLD_LAST: extend with deepcopies of last
              * If len(fusions) > num_stages: truncate
        """
        assert num_stages >= 1

        if share_fusion:
            # "Shared" means a single module instance used for all stages.
            # If user passed many, we take the first (simple + predictable).
            return nn.ModuleList([fusions[0]])

        # Non-shared: need exactly num_stages modules (instances).
        n = len(fusions)

        if n == num_stages:
            return nn.ModuleList(fusions)

        if n > num_stages:
            return nn.ModuleList(fusions[:num_stages])

        if n == 1:
            base = fusions[0]
            return nn.ModuleList([copy.deepcopy(base) for _ in range(num_stages)])

        # 1 < n < num_stages
        out: list[nn.Module] = []
        if fill_mode == FillMode.CYCLE:
            for i in range(num_stages):
                out.append(copy.deepcopy(fusions[i % n]))
        else:  # HOLD_LAST
            out.extend(fusions)  # keep original instances for first n
            last = fusions[-1]
            for _ in range(num_stages - n):
                out.append(copy.deepcopy(last))

        return nn.ModuleList(out)

    def compute_flows_adjacent(self, lqs: torch.Tensor):
        N, T, C, H, W = lqs.shape
        lqs_0 = lqs[:, :-1].reshape(-1, C, H, W)
        lqs_1 = lqs[:, 1:].reshape(-1, C, H, W)

        if self.freeze_flownet:
            # Keep frozen flow networks deterministic and avoid BN stats drift.
            self.flownet.eval()

        ctx = torch.no_grad() if self.freeze_flownet else torch.enable_grad()
        with ctx:
            flow_fwd = self.flownet(lqs_0, lqs_1).view(N, T - 1, 2, H, W)  # i->i+1
            flow_bwd = self.flownet(lqs_1, lqs_0).view(N, T - 1, 2, H, W)  # i+1->i

        return flow_fwd, flow_bwd

    @staticmethod
    def _pad_time_replicate(x: torch.Tensor) -> torch.Tensor:
        left = x[:, 0:1]
        right = x[:, -1:]
        return torch.cat([left, x, right], dim=1)

    @staticmethod
    def _pad_flow_time_zeros(flow: torch.Tensor, left_zeros: int, right_zeros: int) -> torch.Tensor:
        if flow.numel() == 0:
            return flow
        N, TT, C, H, W = flow.shape
        zL = flow.new_zeros(N, left_zeros, C, H, W)
        zR = flow.new_zeros(N, right_zeros, C, H, W)
        return torch.cat([zL, flow, zR], dim=1)


    def forward(self, lqs: torch.Tensor) -> torch.Tensor:
        N, T, C, H, W = lqs.shape
        assert T >= 1

        # Extract features for all frames at once for efficiency.
        feats = self.feat_extractor(lqs.reshape(-1, C, H, W)).reshape(N, T, -1, H, W)

        # Compute flows and pad to length T with zeros (no flow) if needed.
        if T == 1:
            flow_cur_to_prev = lqs.new_zeros(N, 1, 2, H, W)
            flow_cur_to_next = lqs.new_zeros(N, 1, 2, H, W)
        else:
            flow_fwd, flow_bwd = self.compute_flows_adjacent(lqs)
            # flow_warp samples source at (target + flow), so to warp prev/nxt into cur
            # coordinates we need flows cur->prev and cur->next respectively.
            flow_cur_to_prev = self._pad_flow_time_zeros(flow_bwd, left_zeros=1, right_zeros=0)  # (N,T,2,H,W)
            flow_cur_to_next = self._pad_flow_time_zeros(flow_fwd, left_zeros=0, right_zeros=1)  # (N,T,2,H,W)

        for s in range(self.num_stages):
            fusion_mod = self.fusions[0] if len(self.fusions) == 1 else self.fusions[s]

            feats_pad = self._pad_time_replicate(feats)
            prev = feats_pad[:, 0:T]
            cur = feats_pad[:, 1:T + 1]
            nxt = feats_pad[:, 2:T + 2]

            prev_flat = prev.reshape(-1, prev.size(2), H, W)
            cur_flat = cur.reshape(-1, cur.size(2), H, W)
            nxt_flat = nxt.reshape(-1, nxt.size(2), H, W)

            fp2c_flat = flow_cur_to_prev.reshape(-1, 2, H, W)
            fn2c_flat = flow_cur_to_next.reshape(-1, 2, H, W)

            feats_new = fusion_mod(prev_flat, cur_flat, nxt_flat, fp2c_flat, fn2c_flat)
            feats = feats_new.reshape(N, T, -1, H, W)

        if self.output_mode == OutputMode.CENTER:
            t = T // 2
            sr = self.upsampler(feats[:, t], lqs[:, t]).unsqueeze(1)
            return sr

        outs = [self.upsampler(feats[:, t], lqs[:, t]) for t in range(T)]
        return torch.stack(outs, dim=1)
