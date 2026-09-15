from enum import Enum
from typing import Union, Sequence, Optional
import copy

import torch
import torch.nn as nn

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import flow_warp

from modules.feature_extractor import FeatureExtractor
from modules.upsampler import UpsamplerBest as Upsampler


class OutputMode(str, Enum):
    CENTER = "center"
    ALL = "all"


class FillMode(str, Enum):
    CYCLE = "cycle"
    HOLD_LAST = "last"


class HistoryFillMode(str, Enum):
    REPLICATE = "replicate"
    ZEROS = "zeros"


class StageDirection(str, Enum):
    FORWARD = "forward"    # left -> right
    BACKWARD = "backward"  # right -> left


@ARCH_REGISTRY.register()
class Model(nn.Module):
    """
    Alternating-direction recurrent refinement, IART-style.

    Each stage runs sequentially across time:
      - even stage:  left -> right
      - odd stage:   right -> left

    At each time t:
      out[t] = fusion(base[t], mem1, mem2, flow_t_to_mem1, flow_t_to_mem2)

    where:
      - base[t] is feats from previous stage (or extracted features at stage 0)
      - mem1/mem2 are previous states along the traversal direction
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
        history_fill: Union[HistoryFillMode, str] = HistoryFillMode.REPLICATE,
        reconstruction_fusion: nn.Module | None = None,
    ):
        super().__init__()
        self.scale = scale
        self.output_mode = OutputMode(output_mode)
        self.freeze_flownet = freeze_flownet
        self.num_stages = int(num_stages)
        self.history_fill = HistoryFillMode(history_fill)
        assert self.num_stages >= 1

        self.flownet = flownet if flownet is not None else nn.Module()
        self.feat_extractor = FeatureExtractor(
            in_ch=in_channels,
            feat_ch=feat_ch,
            num_blocks=num_feat_blocks
        )
        self.upsampler = Upsampler(feat_ch=feat_ch, scale=scale, out_ch=in_channels)
        
        self.reconstruction_fusion = reconstruction_fusion

        if freeze_flownet:
            for p in self.flownet.parameters():
                p.requires_grad = False

        if fusions is None or len(fusions) == 0:
            raise ValueError("Model requires `fusions` as a non-empty list of fusion modules.")

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
        N, T, C, H, W = lqs.shape

        if T == 1:
            empty = lqs.new_zeros(N, 0, 2, H, W)
            return empty, empty

        lqs_0 = lqs[:, :-1].reshape(-1, C, H, W)
        lqs_1 = lqs[:, 1:].reshape(-1, C, H, W)

        if self.freeze_flownet:
            self.flownet.eval()

        ctx = torch.no_grad() if self.freeze_flownet else torch.enable_grad()
        with ctx:
            flow_fwd = self.flownet(lqs_0, lqs_1).view(N, T - 1, 2, H, W)  # i -> i+1
            flow_bwd = self.flownet(lqs_1, lqs_0).view(N, T - 1, 2, H, W)  # i+1 -> i

        return flow_fwd, flow_bwd

    @staticmethod
    def _pad_flow_time_zeros(flow: torch.Tensor, left_zeros: int, right_zeros: int) -> torch.Tensor:
        N = flow.shape[0]
        C = flow.shape[2] if flow.ndim == 5 and flow.shape[1] > 0 else 2
        H = flow.shape[-2]
        W = flow.shape[-1]

        zL = flow.new_zeros(N, left_zeros, C, H, W)
        zR = flow.new_zeros(N, right_zeros, C, H, W)
        return torch.cat([zL, flow, zR], dim=1)

    @staticmethod
    def _warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Warp x with flow using the internal flow convention (N, 2, H, W).
        basicsr.flow_warp expects flow as (N, H, W, 2).
        """
        return flow_warp(x, flow.permute(0, 2, 3, 1).contiguous())

    @staticmethod
    def _compose_flow(flow_a_to_b: torch.Tensor, flow_b_to_c: torch.Tensor) -> torch.Tensor:
        """
        Compose optical flows:
            a -> c = (a -> b) + warp(b -> c, a -> b)

        Args:
            flow_a_to_b: (N, 2, H, W)
            flow_b_to_c: (N, 2, H, W)

        Returns:
            flow_a_to_c: (N, 2, H, W)
        """
        return flow_a_to_b + Model._warp(flow_b_to_c, flow_a_to_b)

    def _stage_direction(self, s: int) -> StageDirection:
        return StageDirection.FORWARD if (s % 2 == 0) else StageDirection.BACKWARD

    def _default_history(self, base_t: torch.Tensor) -> torch.Tensor:
        if self.history_fill == HistoryFillMode.ZEROS:
            return torch.zeros_like(base_t)
        return base_t

    def _get_forward_flows(
        self,
        flow_cur_to_prev: torch.Tensor,
        t: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          flow_t_to_t1 = t -> t-1
          flow_t_to_t2 = t -> t-2
        """
        flow1 = flow_cur_to_prev[:, t]

        if t >= 2:
            flow_prev_to_prev2 = flow_cur_to_prev[:, t - 1]   # (t-1) -> (t-2)
            flow2 = self._compose_flow(flow1, flow_prev_to_prev2)
        else:
            flow2 = torch.zeros_like(flow1)

        return flow1, flow2

    def _get_backward_flows(
        self,
        flow_cur_to_next: torch.Tensor,
        t: int,
        T: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          flow_t_to_t1 = t -> t+1
          flow_t_to_t2 = t -> t+2
        """
        flow1 = flow_cur_to_next[:, t]

        if t <= T - 3:
            flow_next_to_next2 = flow_cur_to_next[:, t + 1]   # (t+1) -> (t+2)
            flow2 = self._compose_flow(flow1, flow_next_to_next2)
        else:
            flow2 = torch.zeros_like(flow1)

        return flow1, flow2
    
    def _get_reconstruction_feature(
        self,
        feats: torch.Tensor,
        t: int,
        flow_cur_to_prev: torch.Tensor,
        flow_cur_to_next: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build the feature passed to the spatial upsampler.

        The optional reconstruction fusion follows the same interface as the
        recurrent stage fusion modules:

            fusion(prev, cur, nxt, flow_cur_to_prev, flow_cur_to_next)

        Args:
            feats:
                Final-stage features, shape (N, T, C, H, W).
            t:
                Target frame index.
            flow_cur_to_prev:
                Shape (N, T, 2, H, W). At index t, stores t -> t-1.
            flow_cur_to_next:
                Shape (N, T, 2, H, W). At index t, stores t -> t+1.

        Returns:
            Reconstruction feature with shape (N, C, H, W).
        """
        cur = feats[:, t]

        # # TODO: remove this
        # return cur

        if self.reconstruction_fusion is None:
            return cur

        T = feats.shape[1]

        prev = feats[:, t - 1] if t > 0 else self._default_history(cur)
        nxt = feats[:, t + 1] if t + 1 < T else self._default_history(cur)

        flow_prev = (
            flow_cur_to_prev[:, t]
            if t > 0
            else torch.zeros_like(flow_cur_to_prev[:, t])
        )

        flow_next = (
            flow_cur_to_next[:, t]
            if t + 1 < T
            else torch.zeros_like(flow_cur_to_next[:, t])
        )

        return self.reconstruction_fusion(
            prev,
            cur,
            nxt,
            flow_prev,
            flow_next,
        )

    def _run_stage(
        self,
        feats_in: torch.Tensor,          # (N,T,C,H,W)
        fusion_mod: nn.Module,
        direction: StageDirection,
        flow_cur_to_prev: torch.Tensor,  # (N,T,2,H,W), index t stores t -> t-1
        flow_cur_to_next: torch.Tensor,  # (N,T,2,H,W), index t stores t -> t+1
    ) -> torch.Tensor:
        N, T, C, H, W = feats_in.shape
        out = [None] * T

        if direction == StageDirection.FORWARD:
            time_indices = range(T)
        else:
            time_indices = range(T - 1, -1, -1)

        for t in time_indices:
            base_t = feats_in[:, t]

            if direction == StageDirection.FORWARD:
                mem1 = out[t - 1] if t - 1 >= 0 else self._default_history(base_t)
                mem2 = out[t - 2] if t - 2 >= 0 else self._default_history(base_t)
                flow1, flow2 = self._get_forward_flows(flow_cur_to_prev, t)
            else:
                mem1 = out[t + 1] if t + 1 < T else self._default_history(base_t)
                mem2 = out[t + 2] if t + 2 < T else self._default_history(base_t)
                flow1, flow2 = self._get_backward_flows(flow_cur_to_next, t, T)

            out_t = fusion_mod(mem1, base_t, mem2, flow1, flow2)
            out[t] = out_t

        return torch.stack(out, dim=1)

    def forward(self, lqs: torch.Tensor) -> torch.Tensor:
        N, T, C, H, W = lqs.shape
        assert T >= 1

        feats = self.feat_extractor(lqs.reshape(-1, C, H, W)).reshape(N, T, -1, H, W)

        if T == 1:
            flow_cur_to_prev = lqs.new_zeros(N, 1, 2, H, W)
            flow_cur_to_next = lqs.new_zeros(N, 1, 2, H, W)
        else:
            flow_fwd, flow_bwd = self.compute_flows_adjacent(lqs)

            # index t contains t -> t-1
            flow_cur_to_prev = self._pad_flow_time_zeros(
                flow_bwd, left_zeros=1, right_zeros=0
            )

            # index t contains t -> t+1
            flow_cur_to_next = self._pad_flow_time_zeros(
                flow_fwd, left_zeros=0, right_zeros=1
            )

        for s in range(self.num_stages):
            fusion_mod = self.fusions[0] if len(self.fusions) == 1 else self.fusions[s]
            direction = self._stage_direction(s)

            feats = self._run_stage(
                feats_in=feats,
                fusion_mod=fusion_mod,
                direction=direction,
                flow_cur_to_prev=flow_cur_to_prev,
                flow_cur_to_next=flow_cur_to_next,
            )

        if self.output_mode == OutputMode.CENTER:
            t = T // 2
            
            reconstruction_feat = self._get_reconstruction_feature(
                feats=feats,
                t=t,
                flow_cur_to_prev=flow_cur_to_prev,
                flow_cur_to_next=flow_cur_to_next,
            )
            
            return self.upsampler(
                reconstruction_feat,
                lqs[:, t],
            ).unsqueeze(1)

        outs = []

        for t in range(T):
            reconstruction_feat = self._get_reconstruction_feature(
                feats=feats,
                t=t,
                flow_cur_to_prev=flow_cur_to_prev,
                flow_cur_to_next=flow_cur_to_next,
            )

            sr_t = self.upsampler(
                reconstruction_feat,
                lqs[:, t],
            )

            outs.append(sr_t)

        return torch.stack(outs, dim=1)