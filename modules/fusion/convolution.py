import torch
import torch.nn as nn
from torch.nn import functional as F

from torchvision.ops import DeformConv2d
from .base import BaseFusion
from ..blocks import ResBlock
from ..blocks import NAFBlock


##################################################################
######### WARP + CONCAT FUSION (with optional DCN) ###############
##################################################################


class WarpConcatFusion(BaseFusion):
    def __init__(self, feat_ch: int = 64, num_blocks: int = 2):
        super().__init__(feat_ch=feat_ch)
        self.conv_in = nn.Conv2d(feat_ch * 3, feat_ch, 3, 1, 1)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.blocks = nn.Sequential(*[ResBlock(feat_ch) for _ in range(num_blocks)])

    def forward(self, prev, cur, nxt, flow_cur_to_prev, flow_cur_to_next):
        # self._check(prev, cur, nxt, flow_cur_to_prev, flow_cur_to_next)  # opcional
        prev_w = self.warp(prev, flow_cur_to_prev)
        nxt_w  = self.warp(nxt,  flow_cur_to_next)

        x = torch.cat([prev_w, cur, nxt_w], dim=1)
        x = self.act(self.conv_in(x))
        return self.blocks(x)



class WarpConcatNAFFusion(BaseFusion):
    def __init__(self, feat_ch: int = 64, num_blocks: int = 2):
        super().__init__(feat_ch=feat_ch)
        self.conv_in = nn.Conv2d(feat_ch * 3, feat_ch, 3, 1, 1)
        self.blocks = nn.Sequential(*[NAFBlock(feat_ch) for _ in range(num_blocks)])

    def forward(self, prev, cur, nxt, flow_cur_to_prev, flow_cur_to_next):
        prev_w = self.warp(prev, flow_cur_to_prev)
        nxt_w  = self.warp(nxt,  flow_cur_to_next)

        x = torch.cat([prev_w, cur, nxt_w], dim=1)
        x = self.conv_in(x)
        x = self.blocks(x)

        return cur + x


##############################################################################
######### FLOW-GUIDED DEFORMABLE FUSION (with optional learned offsets) ######
##############################################################################

class FlowGuidedDeformableConcatFusion(BaseFusion):
    """
    Simple flow-guided deformable fusion.

    Key idea:
      - Do NOT warp prev/nxt beforehand.
      - Use flow as the base offset for deformable convolutions.
      - Optionally learn a small residual offset on top of the flow.

    Branches:
      prev -> DCN guided by flow_cur_to_prev
      cur  -> regular conv
      nxt  -> DCN guided by flow_cur_to_next

    Then concatenate and fuse.
    """

    def __init__(
        self,
        feat_ch: int = 64,
        num_blocks: int = 2,
        kernel_size: int = 3,
        use_residual_offsets: bool = True,
        offset_hidden_ch: int = 64,
        use_final_residual: bool = True,
    ):
        super().__init__(feat_ch=feat_ch)

        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.k = int(kernel_size)
        self.pad = self.k // 2
        self.use_residual_offsets = bool(use_residual_offsets)
        self.use_final_residual = bool(use_final_residual)

        # Current-frame branch: regular conv
        self.cur_conv = nn.Conv2d(feat_ch, feat_ch, self.k, 1, self.pad)

        # Neighbor branches: deformable convs
        self.prev_dcn = DeformConv2d(feat_ch, feat_ch, self.k, 1, self.pad)
        self.next_dcn = DeformConv2d(feat_ch, feat_ch, self.k, 1, self.pad)

        # Optional learned residual offsets
        # We condition on [neighbor, cur, flow]
        if self.use_residual_offsets:
            in_ch = feat_ch * 2 + 2
            out_ch = 2 * self.k * self.k

            self.prev_offset_net = nn.Sequential(
                nn.Conv2d(in_ch, offset_hidden_ch, self.k, 1, self.pad),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(offset_hidden_ch, out_ch, self.k, 1, self.pad),
            )
            self.next_offset_net = nn.Sequential(
                nn.Conv2d(in_ch, offset_hidden_ch, self.k, 1, self.pad),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(offset_hidden_ch, out_ch, self.k, 1, self.pad),
            )

            # Important: start from pure flow guidance
            nn.init.zeros_(self.prev_offset_net[-1].weight)
            nn.init.zeros_(self.prev_offset_net[-1].bias)
            nn.init.zeros_(self.next_offset_net[-1].weight)
            nn.init.zeros_(self.next_offset_net[-1].bias)

        self.fuse = nn.Conv2d(feat_ch * 3, feat_ch, self.k, 1, 1)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.blocks = nn.Sequential(*[ResBlock(feat_ch) for _ in range(num_blocks)])

    def _flow_to_dcn_offset(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Convert flow (dx, dy) into DCN offsets of shape (B, 2*k*k, H, W).

        IMPORTANT:
        For torchvision/mmcv-style deformable convs, offsets are typically ordered
        per kernel location as (dy, dx), not (dx, dy).

        So if flow is:
            flow[:, 0] = dx
            flow[:, 1] = dy

        then DCN offset channels become:
            [dy, dx, dy, dx, ..., dy, dx]
        repeated k*k times.
        """
        dx = flow[:, 0:1, :, :]
        dy = flow[:, 1:2, :, :]

        # Repeat the same flow for every kernel sampling location
        offset = torch.cat([dy, dx] * (self.k * self.k), dim=1)
        return offset

    def forward(self, prev, cur, nxt, flow_cur_to_prev, flow_cur_to_next):
        # self._check(prev, cur, nxt, flow_cur_to_prev, flow_cur_to_next)  # opcional

        # Base offsets from optical flow
        prev_offset = self._flow_to_dcn_offset(flow_cur_to_prev)
        next_offset = self._flow_to_dcn_offset(flow_cur_to_next)

        # Optional learned residual offsets around the flow-guided position
        if self.use_residual_offsets:
            prev_res = self.prev_offset_net(torch.cat([prev, cur, flow_cur_to_prev], dim=1))
            next_res = self.next_offset_net(torch.cat([nxt, cur, flow_cur_to_next], dim=1))
            prev_offset = prev_offset + prev_res
            next_offset = next_offset + next_res

        prev_f = self.prev_dcn(prev, prev_offset)
        cur_f = self.cur_conv(cur)
        next_f = self.next_dcn(nxt, next_offset)

        x = torch.cat([prev_f, cur_f, next_f], dim=1)
        x = self.act(self.fuse(x))
        x = self.blocks(x)

        if self.use_final_residual:
            return cur + x
        return x
    
    
    
class FlowGuidedDeformableDeepConcatFusion(BaseFusion):
    """
    Simple flow-guided deformable fusion.

    Key idea:
      - Do NOT warp prev/nxt beforehand.
      - Use flow as the base offset for deformable convolutions.
      - Optionally learn a small residual offset on top of the flow.

    Branches:
      prev -> DCN guided by flow_cur_to_prev
      cur  -> regular conv
      nxt  -> DCN guided by flow_cur_to_next

    Then concatenate and fuse.
    """

    def __init__(
        self,
        feat_ch: int = 64,
        alignment_kernel_size: int = 3,
        fusion_depth: int = 1,
        fusion_kernel_size: int = 3,
        fusion_hidden_ch: int | None = None,
        num_refinement_blocks: int = 2,
        use_residual_offsets: bool = True,
        offset_hidden_ch: int = 64,
        use_final_residual: bool = True,
    ):
        super().__init__(feat_ch=feat_ch)
        
        assert alignment_kernel_size % 2 == 1, "alignment_kernel_size must be odd"
        assert fusion_depth >= 1, "fusion_depth must be >= 1"
        assert fusion_kernel_size % 2 == 1, "fusion_kernal_size must be odd"
        
        self.align_k = int(alignment_kernel_size)
        self.align_pad = self.align_k // 2
        
        self.fusion_k = int(fusion_kernel_size)
        self.fusion_depth = int(fusion_depth)
        self.fusion_pad = self.fusion_k // 2
        
        self.use_residual_offsets = bool(use_residual_offsets)
        self.use_final_residual = bool(use_final_residual)

        # Current-frame branch: regular conv
        self.cur_conv = nn.Conv2d(feat_ch, feat_ch, self.align_k, 1, self.align_pad)

        # Neighbor branches: deformable convs
        self.prev_dcn = DeformConv2d(feat_ch, feat_ch, self.align_k, 1, self.align_pad)
        self.next_dcn = DeformConv2d(feat_ch, feat_ch, self.align_k, 1, self.align_pad)

        # Optional learned residual offsets
        # We condition on [neighbor, cur, flow]
        if self.use_residual_offsets:
            in_ch = feat_ch * 2 + 2
            out_ch = 2 * self.align_k * self.align_k

            self.prev_offset_net = nn.Sequential(
                nn.Conv2d(in_ch, offset_hidden_ch, self.align_k, 1, self.align_pad),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(offset_hidden_ch, out_ch, self.align_k, 1, self.align_pad),
            )
            self.next_offset_net = nn.Sequential(
                nn.Conv2d(in_ch, offset_hidden_ch, self.align_k, 1, self.align_pad),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(offset_hidden_ch, out_ch, self.align_k, 1, self.align_pad),
            )

            # Important: start from pure flow guidance
            nn.init.zeros_(self.prev_offset_net[-1].weight)
            nn.init.zeros_(self.prev_offset_net[-1].bias)
            nn.init.zeros_(self.next_offset_net[-1].weight)
            nn.init.zeros_(self.next_offset_net[-1].bias)


        self.fusion_hidden_ch = fusion_hidden_ch or feat_ch * 3
        self.fuse = nn.Sequential(
                *[nn.Sequential(
                    nn.Conv2d(
                        feat_ch * 3 if i == 0 else self.fusion_hidden_ch,
                        self.fusion_hidden_ch if i < self.fusion_depth - 1 else feat_ch,
                        self.fusion_k, 
                        1, 
                        self.fusion_pad), 
                    nn.LeakyReLU(0.1, inplace=True)
                    ) 
                  for i in range(self.fusion_depth)],
            )
        self.blocks = nn.Sequential(*[ResBlock(feat_ch) for _ in range(num_refinement_blocks)])

    def _flow_to_dcn_offset(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Convert flow (dx, dy) into DCN offsets of shape (B, 2*k*k, H, W).

        IMPORTANT:
        For torchvision/mmcv-style deformable convs, offsets are typically ordered
        per kernel location as (dy, dx), not (dx, dy).

        So if flow is:
            flow[:, 0] = dx
            flow[:, 1] = dy

        then DCN offset channels become:
            [dy, dx, dy, dx, ..., dy, dx]
        repeated k*k times.
        """
        dx = flow[:, 0:1, :, :]
        dy = flow[:, 1:2, :, :]

        # Repeat the same flow for every kernel sampling location
        offset = torch.cat([dy, dx] * (self.align_k * self.align_k), dim=1)
        return offset

    def forward(self, prev, cur, nxt, flow_cur_to_prev, flow_cur_to_next):
        # self._check(prev, cur, nxt, flow_cur_to_prev, flow_cur_to_next)  # opcional

        # Base offsets from optical flow
        prev_offset = self._flow_to_dcn_offset(flow_cur_to_prev)
        next_offset = self._flow_to_dcn_offset(flow_cur_to_next)

        # Optional learned residual offsets around the flow-guided position
        if self.use_residual_offsets:
            prev_res = self.prev_offset_net(torch.cat([prev, cur, flow_cur_to_prev], dim=1))
            next_res = self.next_offset_net(torch.cat([nxt, cur, flow_cur_to_next], dim=1))
            prev_offset = prev_offset + prev_res
            next_offset = next_offset + next_res

        prev_f = self.prev_dcn(prev, prev_offset)
        cur_f = self.cur_conv(cur)
        next_f = self.next_dcn(nxt, next_offset)

        x = torch.cat([prev_f, cur_f, next_f], dim=1)
        x = self.fuse(x)
        x = self.blocks(x)

        if self.use_final_residual:
            return cur + x
        return x
    
    
class FlowGuidedDeformableAttentionFusion(BaseFusion):
    """
    Simple flow-guided deformable fusion.

    Key idea:
      - Do NOT warp prev/nxt beforehand.
      - Use flow as the base offset for deformable convolutions.
      - Optionally learn a small residual offset on top of the flow.

    Branches:
      prev -> DCN guided by flow_cur_to_prev
      cur  -> regular conv
      nxt  -> DCN guided by flow_cur_to_next

    Then concatenate and fuse with attention.
    """

    def __init__(
        self,
        feat_ch: int = 64,
        num_blocks: int = 2,
        kernel_size: int = 3,
        use_residual_offsets: bool = True,
        offset_hidden_ch: int = 64,
        use_final_residual: bool = True,
    ):
        super().__init__(feat_ch=feat_ch)

        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.k = int(kernel_size)
        self.pad = self.k // 2
        self.use_residual_offsets = bool(use_residual_offsets)
        self.use_final_residual = bool(use_final_residual)

        # Current-frame branch: regular conv
        self.cur_conv = nn.Conv2d(feat_ch, feat_ch, self.k, 1, self.pad)

        # Neighbor branches: deformable convs
        self.prev_dcn = DeformConv2d(feat_ch, feat_ch, self.k, 1, self.pad)
        self.next_dcn = DeformConv2d(feat_ch, feat_ch, self.k, 1, self.pad)

        # Optional learned residual offsets
        # We condition on [neighbor, cur, flow]
        if self.use_residual_offsets:
            in_ch = feat_ch * 2 + 2
            out_ch = 2 * self.k * self.k

            self.prev_offset_net = nn.Sequential(
                nn.Conv2d(in_ch, offset_hidden_ch, 3, 1, 1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(offset_hidden_ch, out_ch, 3, 1, 1),
            )
            self.next_offset_net = nn.Sequential(
                nn.Conv2d(in_ch, offset_hidden_ch, 3, 1, 1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(offset_hidden_ch, out_ch, 3, 1, 1),
            )

            # Important: start from pure flow guidance
            nn.init.zeros_(self.prev_offset_net[-1].weight)
            nn.init.zeros_(self.prev_offset_net[-1].bias)
            nn.init.zeros_(self.next_offset_net[-1].weight)
            nn.init.zeros_(self.next_offset_net[-1].bias)

        self.fuse = TemporalSourceAttentionFuse(
            feat_ch=feat_ch,
            attn_ch=feat_ch // 2,
            use_residual_value=True,
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.blocks = nn.Sequential(*[ResBlock(feat_ch) for _ in range(num_blocks)])

    def _flow_to_dcn_offset(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Convert flow (dx, dy) into DCN offsets of shape (B, 2*k*k, H, W).

        IMPORTANT:
        For torchvision/mmcv-style deformable convs, offsets are typically ordered
        per kernel location as (dy, dx), not (dx, dy).

        So if flow is:
            flow[:, 0] = dx
            flow[:, 1] = dy

        then DCN offset channels become:
            [dy, dx, dy, dx, ..., dy, dx]
        repeated k*k times.
        """
        dx = flow[:, 0:1, :, :]
        dy = flow[:, 1:2, :, :]

        # Repeat the same flow for every kernel sampling location
        offset = torch.cat([dy, dx] * (self.k * self.k), dim=1)
        return offset

    def forward(self, prev, cur, nxt, flow_cur_to_prev, flow_cur_to_next):
        # self._check(prev, cur, nxt, flow_cur_to_prev, flow_cur_to_next)  # opcional

        # Base offsets from optical flow
        prev_offset = self._flow_to_dcn_offset(flow_cur_to_prev)
        next_offset = self._flow_to_dcn_offset(flow_cur_to_next)

        # Optional learned residual offsets around the flow-guided position
        if self.use_residual_offsets:
            prev_res = self.prev_offset_net(torch.cat([prev, cur, flow_cur_to_prev], dim=1))
            next_res = self.next_offset_net(torch.cat([nxt, cur, flow_cur_to_next], dim=1))
            prev_offset = prev_offset + prev_res
            next_offset = next_offset + next_res

        prev_f = self.prev_dcn(prev, prev_offset)
        cur_f = self.cur_conv(cur)
        next_f = self.next_dcn(nxt, next_offset)

        x = self.fuse(prev_f, cur_f, next_f)
        x = self.act(x)
        x = self.blocks(x)

        if self.use_final_residual:
            return cur + x
        return x
    
### FUSION TMP ###
    
class TemporalSourceAttentionFuse(nn.Module):
    """
    Attention-based fusion over already aligned temporal sources:
        [prev_f, cur_f, next_f]

    Input:
        prev_f, cur_f, next_f: (B, C, H, W)

    Output:
        fused: (B, C, H, W)
        attn:  (B, 3, H, W)
    """

    def __init__(
        self,
        feat_ch: int,
        attn_ch: int | None = None,
        use_residual_value: bool = True,
    ):
        super().__init__()

        attn_ch = attn_ch or feat_ch
        self.feat_ch = feat_ch
        self.attn_ch = attn_ch
        self.use_residual_value = use_residual_value

        # Query only from current frame
        self.q = nn.Conv2d(feat_ch, attn_ch, 1, 1, 0)

        # Keys and values from each temporal source
        self.k = nn.Conv2d(feat_ch, attn_ch, 1, 1, 0)
        self.v = nn.Conv2d(feat_ch, attn_ch, 1, 1, 0)

        # Back to feat_ch
        self.proj = nn.Conv2d(attn_ch, feat_ch, 1, 1, 0)

    def forward(self, prev_f, cur_f, next_f, return_attn: bool = False):
        B, C, H, W = cur_f.shape

        # Stack temporal sources
        sources = torch.stack([prev_f, cur_f, next_f], dim=1)
        # (B, 3, C, H, W)

        sources_flat = sources.reshape(B * 3, C, H, W)

        q = self.q(cur_f)
        # (B, A, H, W)

        k = self.k(sources_flat).view(B, 3, self.attn_ch, H, W)
        v = self.v(sources_flat).view(B, 3, self.attn_ch, H, W)

        # Dot product between current query and each source key
        scores = (q.unsqueeze(1) * k).sum(dim=2)
        scores = scores / (self.attn_ch ** 0.5)
        # (B, 3, H, W)

        attn = torch.softmax(scores, dim=1)
        # (B, 3, H, W)

        fused = (attn.unsqueeze(2) * v).sum(dim=1)
        # (B, A, H, W)

        fused = self.proj(fused)
        # (B, C, H, W)

        if self.use_residual_value:
            fused = fused + cur_f

        if return_attn:
            return fused, attn

        return fused
        
###########################################################################
###### NEW TRANSFORMER-BASED FUSION (with optional learned offsets) ######
###########################################################################

class FlowGuidedDeformableTransformerFusion(BaseFusion):
    """
    Simple flow-guided deformable fusion.

    Key idea:
      - Do NOT warp prev/nxt beforehand.
      - Use flow as the base offset for deformable convolutions.
      - Optionally learn a small residual offset on top of the flow.

    Branches:
      prev -> DCN guided by flow_cur_to_prev
      cur  -> regular conv
      nxt  -> DCN guided by flow_cur_to_next

    Then concatenate and fuse with attention.
    """

    def __init__(
        self,
        feat_ch: int = 64,
        num_blocks: int = 2,
        kernel_size: int = 3,
        use_residual_offsets: bool = True,
        offset_hidden_ch: int = 64,
        use_final_residual: bool = True,
        transformer_depth: int = 2,
        transformer_num_heads: int = 4,
        transformer_win: int = 3,
        transformer_mlp_ratio: float = 2.0,
        transformer_chunk_size: int = 1024,
        transformer_return_delta: bool = True,
    ):
        super().__init__(feat_ch=feat_ch)

        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.k = int(kernel_size)
        self.pad = self.k // 2
        self.use_residual_offsets = bool(use_residual_offsets)
        self.use_final_residual = bool(use_final_residual)

        # Current-frame branch: regular conv
        self.cur_conv = nn.Conv2d(feat_ch, feat_ch, self.k, 1, self.pad)

        # Neighbor branches: deformable convs
        self.prev_dcn = DeformConv2d(feat_ch, feat_ch, self.k, 1, self.pad)
        self.next_dcn = DeformConv2d(feat_ch, feat_ch, self.k, 1, self.pad)

        # Optional learned residual offsets
        # We condition on [neighbor, cur, flow]
        if self.use_residual_offsets:
            in_ch = feat_ch * 2 + 2
            out_ch = 2 * self.k * self.k

            self.prev_offset_net = nn.Sequential(
                nn.Conv2d(in_ch, offset_hidden_ch, 3, 1, 1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(offset_hidden_ch, out_ch, 3, 1, 1),
            )
            self.next_offset_net = nn.Sequential(
                nn.Conv2d(in_ch, offset_hidden_ch, 3, 1, 1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(offset_hidden_ch, out_ch, 3, 1, 1),
            )

            # Important: start from pure flow guidance
            nn.init.zeros_(self.prev_offset_net[-1].weight)
            nn.init.zeros_(self.prev_offset_net[-1].bias)
            nn.init.zeros_(self.next_offset_net[-1].weight)
            nn.init.zeros_(self.next_offset_net[-1].bias)

        self.fuse = AlignedLocalTransformerFuse(
            feat_ch=feat_ch,
            depth=transformer_depth,
            num_heads=transformer_num_heads,
            win=transformer_win,
            mlp_ratio=transformer_mlp_ratio,
            chunk_size=transformer_chunk_size,
            return_delta=True,
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.blocks = nn.Sequential(*[ResBlock(feat_ch) for _ in range(num_blocks)])

    def _flow_to_dcn_offset(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Convert flow (dx, dy) into DCN offsets of shape (B, 2*k*k, H, W).

        IMPORTANT:
        For torchvision/mmcv-style deformable convs, offsets are typically ordered
        per kernel location as (dy, dx), not (dx, dy).

        So if flow is:
            flow[:, 0] = dx
            flow[:, 1] = dy

        then DCN offset channels become:
            [dy, dx, dy, dx, ..., dy, dx]
        repeated k*k times.
        """
        dx = flow[:, 0:1, :, :]
        dy = flow[:, 1:2, :, :]

        # Repeat the same flow for every kernel sampling location
        offset = torch.cat([dy, dx] * (self.k * self.k), dim=1)
        return offset

    def forward(self, prev, cur, nxt, flow_cur_to_prev, flow_cur_to_next):
        # self._check(prev, cur, nxt, flow_cur_to_prev, flow_cur_to_next)  # opcional

        # Base offsets from optical flow
        prev_offset = self._flow_to_dcn_offset(flow_cur_to_prev)
        next_offset = self._flow_to_dcn_offset(flow_cur_to_next)

        # Optional learned residual offsets around the flow-guided position
        if self.use_residual_offsets:
            prev_res = self.prev_offset_net(torch.cat([prev, cur, flow_cur_to_prev], dim=1))
            next_res = self.next_offset_net(torch.cat([nxt, cur, flow_cur_to_next], dim=1))
            prev_offset = prev_offset + prev_res
            next_offset = next_offset + next_res

        prev_f = self.prev_dcn(prev, prev_offset)
        cur_f = self.cur_conv(cur)
        next_f = self.next_dcn(nxt, next_offset)

        x = self.fuse(prev_f, cur_f, next_f)
        x = self.act(x)
        x = self.blocks(x)

        if self.use_final_residual:
            return cur + x
        return x

class LayerNorm2d(nn.Module):
    """
    LayerNorm sobre el eje de canales para tensores (B, C, H, W).

    Equivalente conceptualmente a:
        x = x.permute(0, 2, 3, 1)
        x = LayerNorm(C)(x)
        x = x.permute(0, 3, 1, 2)

    Pero evitando permutaciones externas en el resto del código.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)

        x = (x - mean) / torch.sqrt(var + self.eps)

        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)

        return x * weight + bias


class ConvTransformerFFN(nn.Module):
    """
    FFN convolucional para restauración de imagen.

    Estructura:
        1x1: C -> hidden
        GELU
        depthwise 3x3
        GELU
        1x1: hidden -> C

    La depthwise conv introduce mezcla espacial local, cosa que una FFN
    transformer clásica aplicada píxel a píxel no haría.
    """

    def __init__(
        self,
        dim: int,
        expansion_ratio: float = 2.0,
        bias: bool = True,
    ):
        super().__init__()

        hidden_dim = int(dim * expansion_ratio)

        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=bias),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=hidden_dim,
                bias=bias,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AlignedLocalSpatioTemporalAttention(nn.Module):
    """
    Multi-head local attention sobre tres fuentes temporales ya alineadas.

    Para cada posición (y, x) del frame actual:

        Query:
            cur[:, :, y, x]

        Keys/Values:
            ventana local de prev
            ventana local de cur
            ventana local de next

    Número de candidatos por query:
        3 * win * win

    Inputs:
        prev: (B, C, H, W)
        cur:  (B, C, H, W)
        nxt:  (B, C, H, W)

    Output:
        (B, C, H, W)

    Importante:
        prev y nxt deberían estar previamente alineados con cur,
        por ejemplo mediante deformable convolution guiada por flujo.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        win: int = 3,
        chunk_size: int = 1024,
        qkv_bias: bool = True,
        proj_bias: bool = True,
    ):
        super().__init__()

        if win % 2 != 1:
            raise ValueError(f"win debe ser impar, recibido: {win}")

        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} debe ser divisible por num_heads={num_heads}"
            )

        if chunk_size <= 0:
            raise ValueError(
                f"chunk_size debe ser positivo, recibido: {chunk_size}"
            )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.win = win
        self.win2 = win * win
        self.pad = win // 2
        self.chunk_size = int(chunk_size)

        self.scale = self.head_dim**-0.5

        # Query únicamente desde el estado central.
        self.q = nn.Conv2d(dim, dim, 1, bias=qkv_bias)

        # K y V compartidas entre las tres fuentes.
        self.kv = nn.Conv2d(dim, 2 * dim, 1, bias=qkv_bias)

        self.proj = nn.Conv2d(dim, dim, 1, bias=proj_bias)

    def _reshape_unfolded(
        self,
        x: torch.Tensor,
        length: int,
    ) -> torch.Tensor:
        """
        Convierte:

            (B, C * win², L)

        en:

            (B, heads, L, win², head_dim)
        """

        batch = x.shape[0]

        x = x.view(
            batch,
            self.num_heads,
            self.head_dim,
            self.win2,
            length,
        )

        return x.permute(0, 1, 4, 3, 2).contiguous()

    def forward(
        self,
        prev: torch.Tensor,
        cur: torch.Tensor,
        nxt: torch.Tensor,
        return_attention: bool = False,
    ):
        if prev.shape != cur.shape or nxt.shape != cur.shape:
            raise ValueError(
                "prev, cur y nxt deben tener exactamente la misma shape. "
                f"Recibidas: prev={prev.shape}, cur={cur.shape}, nxt={nxt.shape}"
            )

        batch, channels, height, width = cur.shape

        if channels != self.dim:
            raise ValueError(
                f"Se esperaban {self.dim} canales, recibidos {channels}"
            )

        q = self.q(cur)

        k_prev, v_prev = self.kv(prev).chunk(2, dim=1)
        k_cur, v_cur = self.kv(cur).chunk(2, dim=1)
        k_next, v_next = self.kv(nxt).chunk(2, dim=1)

        # Pad una única vez. Después hacemos unfold únicamente por tiras.
        padding = (self.pad, self.pad, self.pad, self.pad)

        k_prev = F.pad(k_prev, padding)
        v_prev = F.pad(v_prev, padding)

        k_cur = F.pad(k_cur, padding)
        v_cur = F.pad(v_cur, padding)

        k_next = F.pad(k_next, padding)
        v_next = F.pad(v_next, padding)

        output = q.new_empty(batch, channels, height, width)

        # Cada chunk contiene aproximadamente chunk_size queries.
        chunk_rows = max(1, self.chunk_size // width)

        attention_chunks = [] if return_attention else None

        for y0 in range(0, height, chunk_rows):
            y1 = min(height, y0 + chunk_rows)

            rows = y1 - y0
            length = rows * width

            # Query del chunk:
            # (B, C, rows, W)
            q_chunk = q[:, :, y0:y1, :]

            # (B, heads, L, head_dim)
            q_chunk = q_chunk.view(
                batch,
                self.num_heads,
                self.head_dim,
                length,
            ).permute(0, 1, 3, 2)

            # Como los mapas están paddeados, una tira correspondiente a
            # y0:y1 requiere rows + 2*pad filas.
            strip_start = y0
            strip_end = y1 + 2 * self.pad

            kp_strip = k_prev[:, :, strip_start:strip_end, :]
            vp_strip = v_prev[:, :, strip_start:strip_end, :]

            kc_strip = k_cur[:, :, strip_start:strip_end, :]
            vc_strip = v_cur[:, :, strip_start:strip_end, :]

            kn_strip = k_next[:, :, strip_start:strip_end, :]
            vn_strip = v_next[:, :, strip_start:strip_end, :]

            # Cada unfold produce:
            # (B, C * win², L)
            kp = F.unfold(kp_strip, kernel_size=self.win)
            vp = F.unfold(vp_strip, kernel_size=self.win)

            kc = F.unfold(kc_strip, kernel_size=self.win)
            vc = F.unfold(vc_strip, kernel_size=self.win)

            kn = F.unfold(kn_strip, kernel_size=self.win)
            vn = F.unfold(vn_strip, kernel_size=self.win)

            # Cada tensor:
            # (B, heads, L, win², head_dim)
            kp = self._reshape_unfolded(kp, length)
            vp = self._reshape_unfolded(vp, length)

            kc = self._reshape_unfolded(kc, length)
            vc = self._reshape_unfolded(vc, length)

            kn = self._reshape_unfolded(kn, length)
            vn = self._reshape_unfolded(vn, length)

            # Pool espacio-temporal:
            # (B, heads, L, 3*win², head_dim)
            keys = torch.cat([kp, kc, kn], dim=3)
            values = torch.cat([vp, vc, vn], dim=3)

            # Producto escalar por cabeza:
            # q_chunk: (B, heads, L, head_dim)
            # keys:    (B, heads, L, candidates, head_dim)
            #
            # scores:  (B, heads, L, candidates)
            scores = (
                q_chunk.unsqueeze(3) * keys
            ).sum(dim=-1) * self.scale

            attention = torch.softmax(scores, dim=-1)

            # (B, heads, L, head_dim)
            out_chunk = (
                attention.unsqueeze(-1) * values
            ).sum(dim=3)

            # Volver a:
            # (B, C, rows, W)
            out_chunk = out_chunk.permute(0, 1, 3, 2).contiguous()
            out_chunk = out_chunk.view(
                batch,
                channels,
                rows,
                width,
            )

            output[:, :, y0:y1, :] = out_chunk

            if return_attention:
                # Guardamos en CPU para no retener toda la atención en GPU.
                attention_chunks.append(attention.detach().cpu())

        output = self.proj(output)

        if return_attention:
            return output, attention_chunks

        return output


class AlignedLocalTransformerBlock(nn.Module):
    """
    Bloque transformer local espacio-temporal.

    Pre-norm:

        x = x + LocalAttention(
            norm(prev),
            norm(x),
            norm(next)
        )

        x = x + FFN(norm(x))

    prev y next permanecen como fuentes temporales externas.
    x representa el estado central que se va refinando.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        win: int = 3,
        mlp_ratio: float = 2.0,
        chunk_size: int = 1024,
        qkv_bias: bool = True,
    ):
        super().__init__()

        self.norm_sources = LayerNorm2d(dim)
        self.norm_ffn = LayerNorm2d(dim)

        self.attention = AlignedLocalSpatioTemporalAttention(
            dim=dim,
            num_heads=num_heads,
            win=win,
            chunk_size=chunk_size,
            qkv_bias=qkv_bias,
        )

        self.ffn = ConvTransformerFFN(
            dim=dim,
            expansion_ratio=mlp_ratio,
        )

    def forward(
        self,
        prev: torch.Tensor,
        cur: torch.Tensor,
        nxt: torch.Tensor,
    ) -> torch.Tensor:
        prev_norm = self.norm_sources(prev)
        cur_norm = self.norm_sources(cur)
        next_norm = self.norm_sources(nxt)

        cur = cur + self.attention(
            prev=prev_norm,
            cur=cur_norm,
            nxt=next_norm,
        )

        cur = cur + self.ffn(self.norm_ffn(cur))

        return cur


class AlignedLocalTransformerFuse(nn.Module):
    """
    Fusión completa mediante varios bloques transformer locales.

    Está diseñada para reemplazar:

        TemporalSourceAttentionFuse

    dentro de FlowGuidedDeformableAttentionFusion.

    Por defecto devuelve un residual/delta respecto al frame central:

        delta = transformed_cur - original_cur

    Esto permite conservar la lógica exterior:

        x = self.fuse(prev_f, cur_f, next_f)
        x = self.blocks(x)
        return cur + x

    Si return_delta=False, devuelve directamente la feature fusionada.
    """

    def __init__(
        self,
        feat_ch: int,
        depth: int = 2,
        num_heads: int = 4,
        win: int = 3,
        mlp_ratio: float = 2.0,
        chunk_size: int = 1024,
        return_delta: bool = True,
    ):
        super().__init__()

        if depth < 1:
            raise ValueError(f"depth debe ser >= 1, recibido: {depth}")

        self.return_delta = return_delta

        self.blocks = nn.ModuleList(
            [
                AlignedLocalTransformerBlock(
                    dim=feat_ch,
                    num_heads=num_heads,
                    win=win,
                    mlp_ratio=mlp_ratio,
                    chunk_size=chunk_size,
                )
                for _ in range(depth)
            ]
        )

        # Normalización final habitual en stacks transformer.
        self.final_norm = LayerNorm2d(feat_ch)

    def forward(
        self,
        prev_f: torch.Tensor,
        cur_f: torch.Tensor,
        next_f: torch.Tensor,
    ) -> torch.Tensor:
        initial_cur = cur_f
        x = cur_f

        for block in self.blocks:
            x = block(
                prev=prev_f,
                cur=x,
                nxt=next_f,
            )

        x = self.final_norm(x)

        if self.return_delta:
            return x - initial_cur

        return x        


#####################################################################
############### FLOW-GUIDED DEFORMABLE FUSION (with NAF blocks)######
#####################################################################
    
class FlowGuidedDeformableConcatNAFFusion(BaseFusion):
    """
    Simple flow-guided deformable fusion.

    Key idea:
      - Do NOT warp prev/nxt beforehand.
      - Use flow as the base offset for deformable convolutions.
      - Optionally learn a small residual offset on top of the flow.

    Branches:
      prev -> DCN guided by flow_cur_to_prev
      cur  -> regular conv
      nxt  -> DCN guided by flow_cur_to_next

    Then concatenate and fuse.
    """

    def __init__(
        self,
        feat_ch: int = 64,
        num_blocks: int = 2,
        kernel_size: int = 3,
        use_residual_offsets: bool = True,
        offset_hidden_ch: int = 64,
        use_final_residual: bool = True,
    ):
        super().__init__(feat_ch=feat_ch)

        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.k = int(kernel_size)
        self.pad = self.k // 2
        self.use_residual_offsets = bool(use_residual_offsets)
        self.use_final_residual = bool(use_final_residual)

        # Current-frame branch: regular conv
        self.cur_conv = nn.Conv2d(feat_ch, feat_ch, self.k, 1, self.pad)

        # Neighbor branches: deformable convs
        self.prev_dcn = DeformConv2d(feat_ch, feat_ch, self.k, 1, self.pad)
        self.next_dcn = DeformConv2d(feat_ch, feat_ch, self.k, 1, self.pad)

        # Optional learned residual offsets
        # We condition on [neighbor, cur, flow]
        if self.use_residual_offsets:
            in_ch = feat_ch * 2 + 2
            out_ch = 2 * self.k * self.k

            self.prev_offset_net = nn.Sequential(
                nn.Conv2d(in_ch, offset_hidden_ch, 3, 1, 1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(offset_hidden_ch, out_ch, 3, 1, 1),
            )
            self.next_offset_net = nn.Sequential(
                nn.Conv2d(in_ch, offset_hidden_ch, 3, 1, 1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(offset_hidden_ch, out_ch, 3, 1, 1),
            )

            # Important: start from pure flow guidance
            nn.init.zeros_(self.prev_offset_net[-1].weight)
            nn.init.zeros_(self.prev_offset_net[-1].bias)
            nn.init.zeros_(self.next_offset_net[-1].weight)
            nn.init.zeros_(self.next_offset_net[-1].bias)

        self.fuse = nn.Conv2d(feat_ch * 3, feat_ch, 3, 1, 1)
        self.blocks = nn.Sequential(*[NAFBlock(feat_ch) for _ in range(num_blocks)])
        
        if self.use_final_residual:
            self.alpha = nn.Parameter(torch.zeros(1, feat_ch, 1, 1))

    def _flow_to_dcn_offset(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Convert flow (dx, dy) into DCN offsets of shape (B, 2*k*k, H, W).

        IMPORTANT:
        For torchvision/mmcv-style deformable convs, offsets are typically ordered
        per kernel location as (dy, dx), not (dx, dy).

        So if flow is:
            flow[:, 0] = dx
            flow[:, 1] = dy

        then DCN offset channels become:
            [dy, dx, dy, dx, ..., dy, dx]
        repeated k*k times.
        """
        dx = flow[:, 0:1, :, :]
        dy = flow[:, 1:2, :, :]

        # Repeat the same flow for every kernel sampling location
        offset = torch.cat([dy, dx] * (self.k * self.k), dim=1)
        return offset

    def forward(self, prev, cur, nxt, flow_cur_to_prev, flow_cur_to_next):

        # Base offsets from optical flow
        prev_offset = self._flow_to_dcn_offset(flow_cur_to_prev)
        next_offset = self._flow_to_dcn_offset(flow_cur_to_next)

        # Optional learned residual offsets around the flow-guided position
        if self.use_residual_offsets:
            prev_res = self.prev_offset_net(torch.cat([prev, cur, flow_cur_to_prev], dim=1))
            next_res = self.next_offset_net(torch.cat([nxt, cur, flow_cur_to_next], dim=1))
            prev_offset = prev_offset + prev_res
            next_offset = next_offset + next_res

        prev_f = self.prev_dcn(prev, prev_offset)
        cur_f = self.cur_conv(cur)
        next_f = self.next_dcn(nxt, next_offset)

        x = torch.cat([prev_f, cur_f, next_f], dim=1)
        x = self.fuse(x)
        x = self.blocks(x)

        if self.use_final_residual:
            return cur + self.alpha * x
        return x
    
    
    
#####################################################################
######################## FLOW-GUIDED UNET FUSION #####################
#####################################################################
    
    
class FlowGuidedDeformableUnetFusion(BaseFusion):
    """
    Simple flow-guided deformable fusion.

    Key idea:
      - Do NOT warp prev/nxt beforehand.
      - Use flow as the base offset for deformable convolutions.
      - Optionally learn a small residual offset on top of the flow.

    Branches:
      prev -> DCN guided by flow_cur_to_prev
      cur  -> regular conv
      nxt  -> DCN guided by flow_cur_to_next

    Then concatenate and fuse.
    """

    def __init__(
        self,
        feat_ch: int = 64,
        low_ch: int | None = None,
        num_low_blocks: int = 2,
        num_high_blocks: int = 1,
        kernel_size: int = 3,
        use_residual_offsets: bool = True,
        offset_hidden_ch: int = 64,
        use_final_residual: bool = True,
    ):
        super().__init__(feat_ch=feat_ch)

        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.k = int(kernel_size)
        self.pad = self.k // 2
        self.use_residual_offsets = bool(use_residual_offsets)
        self.use_final_residual = bool(use_final_residual)

        # Current-frame branch: regular conv
        self.cur_conv = nn.Conv2d(feat_ch, feat_ch, self.k, 1, self.pad)

        # Neighbor branches: deformable convs
        self.prev_dcn = DeformConv2d(feat_ch, feat_ch, self.k, 1, self.pad)
        self.next_dcn = DeformConv2d(feat_ch, feat_ch, self.k, 1, self.pad)

        # Optional learned residual offsets
        # We condition on [neighbor, cur, flow]
        if self.use_residual_offsets:
            in_ch = feat_ch * 2 + 2
            out_ch = 2 * self.k * self.k

            self.prev_offset_net = nn.Sequential(
                nn.Conv2d(in_ch, offset_hidden_ch, 3, 1, 1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(offset_hidden_ch, out_ch, 3, 1, 1),
            )
            self.next_offset_net = nn.Sequential(
                nn.Conv2d(in_ch, offset_hidden_ch, 3, 1, 1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(offset_hidden_ch, out_ch, 3, 1, 1),
            )

            # Important: start from pure flow guidance
            nn.init.zeros_(self.prev_offset_net[-1].weight)
            nn.init.zeros_(self.prev_offset_net[-1].bias)
            nn.init.zeros_(self.next_offset_net[-1].weight)
            nn.init.zeros_(self.next_offset_net[-1].bias)

        self.fuse = OneLevelUNetFuse(
            feat_ch=feat_ch,
            low_ch=low_ch,
            num_low_blocks=num_low_blocks,
            num_high_blocks=num_high_blocks,
        )

    def _flow_to_dcn_offset(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Convert flow (dx, dy) into DCN offsets of shape (B, 2*k*k, H, W).

        IMPORTANT:
        For torchvision/mmcv-style deformable convs, offsets are typically ordered
        per kernel location as (dy, dx), not (dx, dy).

        So if flow is:
            flow[:, 0] = dx
            flow[:, 1] = dy

        then DCN offset channels become:
            [dy, dx, dy, dx, ..., dy, dx]
        repeated k*k times.
        """
        dx = flow[:, 0:1, :, :]
        dy = flow[:, 1:2, :, :]

        # Repeat the same flow for every kernel sampling location
        offset = torch.cat([dy, dx] * (self.k * self.k), dim=1)
        return offset

    def forward(self, prev, cur, nxt, flow_cur_to_prev, flow_cur_to_next):
        # self._check(prev, cur, nxt, flow_cur_to_prev, flow_cur_to_next)  # opcional

        # Base offsets from optical flow
        prev_offset = self._flow_to_dcn_offset(flow_cur_to_prev)
        next_offset = self._flow_to_dcn_offset(flow_cur_to_next)

        # Optional learned residual offsets around the flow-guided position
        if self.use_residual_offsets:
            prev_res = self.prev_offset_net(torch.cat([prev, cur, flow_cur_to_prev], dim=1))
            next_res = self.next_offset_net(torch.cat([nxt, cur, flow_cur_to_next], dim=1))
            prev_offset = prev_offset + prev_res
            next_offset = next_offset + next_res

        prev_f = self.prev_dcn(prev, prev_offset)
        cur_f = self.cur_conv(cur)
        next_f = self.next_dcn(nxt, next_offset)

        # U-Net fusion
        x = self.fuse(
            prev_f,
            cur_f,
            next_f,
        )

        if self.use_final_residual:
            return cur + x
        return x
    
class OneLevelUNetFuse(nn.Module):
    """
    One-level U-Net-like fusion for three already aligned temporal features.

    Design:
        - prev_f, cur_f, next_f are already aligned in cur coordinates.
        - All three sources are propagated independently to half resolution.
        - Temporal fusion happens at half resolution.
        - Only cur_f is used as the full-resolution skip.
        - The decoder combines:
              low-resolution temporal context
              +
              high-resolution current-frame detail

    Inputs:
        prev_f: (B, C, H, W)
        cur_f:  (B, C, H, W)
        next_f: (B, C, H, W)

    Output:
        residual: (B, C, H, W)

    The outer fusion module can then do:
        return cur + residual
    """

    def __init__(
        self,
        feat_ch: int,
        low_ch: int | None = None,
        num_low_blocks: int = 2,
        num_high_blocks: int = 1,
    ):
        super().__init__()

        low_ch = low_ch or feat_ch

        self.feat_ch = feat_ch
        self.low_ch = low_ch

        self.act = nn.LeakyReLU(0.1, inplace=True)

        # ---------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------
        # Shared downsampler for prev / cur / next.
        #
        # Using the same weights keeps the three temporal sources
        # in the same feature space.
        self.down = nn.Conv2d(
            feat_ch,
            low_ch,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        # ---------------------------------------------------------
        # Low-resolution temporal fusion
        # ---------------------------------------------------------
        #
        # Three independent streams:
        #     prev_low
        #     cur_low
        #     next_low
        #
        # are concatenated only here.
        self.low_fuse = nn.Conv2d(
            low_ch * 3,
            low_ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.low_blocks = nn.Sequential(
            *[
                ResBlock(low_ch)
                for _ in range(num_low_blocks)
            ]
        )

        # ---------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------
        #
        # After bilinear upsampling, project low-resolution
        # representation back to feat_ch.
        # self.up_proj = nn.Conv2d(
        #     low_ch,
        #     feat_ch,
        #     kernel_size=3,
        #     stride=1,
        #     padding=1,
        # )

        self.up = nn.Sequential(
            nn.Conv2d(
                low_ch,
                feat_ch * 4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # ---------------------------------------------------------
        # High-resolution fusion
        # ---------------------------------------------------------
        #
        # Only cur_f is used as high-resolution skip.
        #
        # Input:
        #     cur_f : C
        #     up    : C
        #
        # Total = 2C
        self.high_fuse = nn.Conv2d(
            feat_ch * 2,
            feat_ch,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.high_blocks = nn.Sequential(
            *[
                ResBlock(feat_ch)
                for _ in range(num_high_blocks)
            ]
        )

    def forward(
        self,
        prev_f: torch.Tensor,
        cur_f: torch.Tensor,
        next_f: torch.Tensor,
    ) -> torch.Tensor:

        # =========================================================
        # 1. Encoder
        # =========================================================
        #
        # Keep all temporal sources separate while going down.
        #
        # (B, C, H, W)
        #       ↓
        # (B, Clow, H/2, W/2)

        prev_low = self.act(self.down(prev_f))
        cur_low = self.act(self.down(cur_f))
        next_low = self.act(self.down(next_f))

        # =========================================================
        # 2. Temporal fusion at low resolution
        # =========================================================

        low = torch.cat(
            [
                prev_low,
                cur_low,
                next_low,
            ],
            dim=1,
        )

        # (B, 3*Clow, H/2, W/2)
        #       ↓
        # (B, Clow, H/2, W/2)

        low = self.act(self.low_fuse(low))
        low = self.low_blocks(low)

        # =========================================================
        # 3. Decoder / upsample
        # =========================================================

        up = self.up(low)
        
        # Match exactly the full-resolution skip size.
        if up.shape[-2:] != cur_f.shape[-2:]:
            up = up[
                ...,
                :cur_f.shape[-2],
                :cur_f.shape[-1],
            ]

        # =========================================================
        # 4. High-resolution skip fusion
        # =========================================================
        #
        # The temporal information is already summarized in `up`.
        # At full resolution we only reintroduce cur_f to preserve
        # spatial detail and fine texture.

        x = torch.cat(
            [
                cur_f,
                up,
            ],
            dim=1,
        )

        x = self.act(self.high_fuse(x))
        x = self.high_blocks(x)

        # Return a correction/residual.
        return x
    
    
# #################################################################################
# ######################## FLOW-GUIDED Multiscale Post FUSION #####################
# #################################################################################

# class FlowGuidedDeformableMultiScaleFusion(BaseFusion):
#     def __init__(
#         self,
#         feat_ch: int = 64,
#         low_ch: int | None = None,
#         num_high_blocks: int = 1,
#         num_low_blocks: int = 2,
#         num_final_blocks: int = 1,
#         kernel_size: int = 3,
#         use_residual_offsets: bool = True,
#         offset_hidden_ch: int = 64,
#         use_final_residual: bool = True,
#     ):
#         super().__init__(feat_ch=feat_ch)

#         assert kernel_size % 2 == 1

#         self.k = kernel_size
#         self.pad = kernel_size // 2
#         self.use_residual_offsets = use_residual_offsets
#         self.use_final_residual = use_final_residual

#         # Align
#         self.cur_conv = nn.Conv2d(
#             feat_ch, feat_ch, self.k, 1, self.pad
#         )

#         self.prev_dcn = DeformConv2d(
#             feat_ch, feat_ch, self.k, 1, self.pad
#         )

#         self.next_dcn = DeformConv2d(
#             feat_ch, feat_ch, self.k, 1, self.pad
#         )

#         if self.use_residual_offsets:
#             in_ch = feat_ch * 2 + 2
#             out_ch = 2 * self.k * self.k

#             self.prev_offset_net = nn.Sequential(
#                 nn.Conv2d(in_ch, offset_hidden_ch, 3, 1, 1),
#                 nn.LeakyReLU(0.1, inplace=True),
#                 nn.Conv2d(offset_hidden_ch, out_ch, 3, 1, 1),
#             )

#             self.next_offset_net = nn.Sequential(
#                 nn.Conv2d(in_ch, offset_hidden_ch, 3, 1, 1),
#                 nn.LeakyReLU(0.1, inplace=True),
#                 nn.Conv2d(offset_hidden_ch, out_ch, 3, 1, 1),
#             )

#             nn.init.zeros_(self.prev_offset_net[-1].weight)
#             nn.init.zeros_(self.prev_offset_net[-1].bias)
#             nn.init.zeros_(self.next_offset_net[-1].weight)
#             nn.init.zeros_(self.next_offset_net[-1].bias)

#         # Fuse: identical to current best
#         self.fuse = nn.Conv2d(
#             feat_ch * 3,
#             feat_ch,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#         )

#         self.act = nn.LeakyReLU(0.1, inplace=True)

#         # Refine: new multiscale module
#         self.refine = OneLevelPostFusionUNetRefine(
#             feat_ch=feat_ch,
#             low_ch=low_ch,
#             num_high_blocks=num_high_blocks,
#             num_low_blocks=num_low_blocks,
#             num_final_blocks=num_final_blocks,
#         )
        
#     def _flow_to_dcn_offset(self, flow: torch.Tensor) -> torch.Tensor:
#         """
#         Convert flow (dx, dy) into DCN offsets of shape (B, 2*k*k, H, W).

#         IMPORTANT:
#         For torchvision/mmcv-style deformable convs, offsets are typically ordered
#         per kernel location as (dy, dx), not (dx, dy).

#         So if flow is:
#             flow[:, 0] = dx
#             flow[:, 1] = dy

#         then DCN offset channels become:
#             [dy, dx, dy, dx, ..., dy, dx]
#         repeated k*k times.
#         """
#         dx = flow[:, 0:1, :, :]
#         dy = flow[:, 1:2, :, :]

#         # Repeat the same flow for every kernel sampling location
#         offset = torch.cat([dy, dx] * (self.k * self.k), dim=1)
#         return offset
        
#     def forward(
#         self,
#         prev,
#         cur,
#         nxt,
#         flow_cur_to_prev,
#         flow_cur_to_next,
#     ):
#         prev_offset = self._flow_to_dcn_offset(flow_cur_to_prev)
#         next_offset = self._flow_to_dcn_offset(flow_cur_to_next)

#         if self.use_residual_offsets:
#             prev_res = self.prev_offset_net(
#                 torch.cat(
#                     [prev, cur, flow_cur_to_prev],
#                     dim=1,
#                 )
#             )

#             next_res = self.next_offset_net(
#                 torch.cat(
#                     [nxt, cur, flow_cur_to_next],
#                     dim=1,
#                 )
#             )

#             prev_offset = prev_offset + prev_res
#             next_offset = next_offset + next_res

#         # ---------------------------------------------------------
#         # ALIGN
#         # ---------------------------------------------------------

#         prev_f = self.prev_dcn(prev, prev_offset)
#         cur_f = self.cur_conv(cur)
#         next_f = self.next_dcn(nxt, next_offset)

#         # ---------------------------------------------------------
#         # FUSE
#         # Exactly the same mechanism as current best
#         # ---------------------------------------------------------

#         x = torch.cat(
#             [prev_f, cur_f, next_f],
#             dim=1,
#         )

#         x = self.act(self.fuse(x))

#         # ---------------------------------------------------------
#         # REFINE
#         # New multiscale spatial context
#         # ---------------------------------------------------------

#         x = self.refine(x)

#         if self.use_final_residual:
#             return cur + x

#         return x
    
    
# class OneLevelPostFusionUNetRefine(nn.Module):
#     def __init__(
#         self,
#         feat_ch: int,
#         low_ch: int | None = None,
#         num_high_blocks: int = 1,
#         num_low_blocks: int = 2,
#         num_final_blocks: int = 1,
#     ):
#         super().__init__()

#         low_ch = low_ch or feat_ch // 2

#         self.act = nn.LeakyReLU(0.1, inplace=True)

#         # ---------------------------------------------------------
#         # High-resolution branch
#         # ---------------------------------------------------------
#         self.high_blocks = nn.Sequential(
#             *[ResBlock(feat_ch) for _ in range(num_high_blocks)]
#         )

#         # ---------------------------------------------------------
#         # Low-resolution context branch
#         # ---------------------------------------------------------
#         self.down = nn.Conv2d(
#             feat_ch,
#             low_ch,
#             kernel_size=3,
#             stride=2,
#             padding=1,
#         )

#         self.low_blocks = nn.Sequential(
#             *[ResBlock(low_ch) for _ in range(num_low_blocks)]
#         )

#         self.up = nn.Sequential(
#             nn.Conv2d(
#                 low_ch,
#                 feat_ch * 4,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#             ),
#             nn.PixelShuffle(2),
#             nn.LeakyReLU(0.1, inplace=True),
#         )

#         # ---------------------------------------------------------
#         # Merge high + low
#         # ---------------------------------------------------------
#         self.merge = nn.Conv2d(
#             feat_ch * 2,
#             feat_ch,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#         )

#         self.final_blocks = nn.Sequential(
#             *[ResBlock(feat_ch) for _ in range(num_final_blocks)]
#         )

#     def forward(self, x):
#         # Full-resolution refinement
#         high = self.high_blocks(x)

#         # Low-resolution contextual refinement
#         low = self.act(self.down(x))
#         low = self.low_blocks(low)
#         low = self.up(low)

#         if low.shape[-2:] != high.shape[-2:]:
#             low = low[..., :high.shape[-2], :high.shape[-1]]

#         # Combine both scales
#         out = torch.cat([high, low], dim=1)
#         out = self.act(self.merge(out))
#         out = self.final_blocks(out)

#         return out