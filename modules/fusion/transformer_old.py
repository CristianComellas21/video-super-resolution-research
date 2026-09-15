from torch import nn
import torch
import torch.nn.functional as F

from .base import BaseFusion
from ..blocks import ConvMLP


class FlowGuidedGatedLocalTransformerFusionBlock(BaseFusion):
    """
    Pre-norm transformer-style fusion block using separate flow-guided
    local attentions + gated branch fusion.

    Structure:
        prev_att = Attn(Norm(cur), Norm(prev), flow_cur_to_prev)
        cur_att  = Attn(Norm(cur), Norm(cur),  None)
        next_att = Attn(Norm(cur), Norm(nxt), flow_cur_to_next)

        attn_out = Gate(prev_att, cur_att, next_att, cur) weighted fusion

        x = cur + gamma * Proj(attn_out)
        x = x   + MLP(Norm(x))
    """

    def __init__(
        self,
        feat_ch=64,
        attn_ch=None,
        win=7,
        chunk_size=1024,
        share_qk=False,
        mlp_ratio=4.0,
        attn_use_pos_bias=True,
        attn_out_proj=False,
    ):
        super().__init__(feat_ch=feat_ch)

        hidden_dim = int(feat_ch * mlp_ratio)

        self.norm1 = nn.GroupNorm(1, feat_ch)

        self.attn_prev = _FlowGuidedLocalAttentionChunked(
            feat_ch=feat_ch,
            attn_ch=attn_ch,
            win=win,
            chunk_size=chunk_size,
            share_qk=share_qk,
            use_pos_bias=attn_use_pos_bias,
            out_proj=attn_out_proj,
        )

        self.attn_cur = _FlowGuidedLocalAttentionChunked(
            feat_ch=feat_ch,
            attn_ch=attn_ch,
            win=win,
            chunk_size=chunk_size,
            share_qk=share_qk,
            use_pos_bias=attn_use_pos_bias,
            out_proj=attn_out_proj,
        )

        self.attn_next = _FlowGuidedLocalAttentionChunked(
            feat_ch=feat_ch,
            attn_ch=attn_ch,
            win=win,
            chunk_size=chunk_size,
            share_qk=share_qk,
            use_pos_bias=attn_use_pos_bias,
            out_proj=attn_out_proj,
        )

        attn_ch_eff = feat_ch if attn_out_proj else (attn_ch or feat_ch)

        self.gate = nn.Sequential(
            nn.Conv2d(attn_ch_eff * 3 + feat_ch, feat_ch, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feat_ch, 3, 1, 1, 0),
        )

        self.proj = nn.Conv2d(attn_ch_eff, feat_ch, 1, 1, 0) if not attn_out_proj else nn.Identity()

        self.gamma = nn.Parameter(torch.zeros(1))

        self.norm2 = nn.GroupNorm(1, feat_ch)
        self.mlp = ConvMLP(feat_ch, hidden_dim=hidden_dim)

    def forward(self, prev, cur, nxt, flow_cur_to_prev, flow_cur_to_next):
        prev_n = self.norm1(prev)
        cur_n = self.norm1(cur)
        nxt_n = self.norm1(nxt)

        prev_att = self.attn_prev(cur_n, prev_n, flow_cur_to_prev)
        cur_att = self.attn_cur(cur_n, cur_n, None)
        next_att = self.attn_next(cur_n, nxt_n, flow_cur_to_next)

        gate_in = torch.cat([prev_att, cur_att, next_att, cur_n], dim=1)
        gate_logits = self.gate(gate_in)
        gates = torch.softmax(gate_logits, dim=1)

        attn_out = (
            gates[:, 0:1] * prev_att +
            gates[:, 1:2] * cur_att +
            gates[:, 2:3] * next_att
        )

        x = cur + self.gamma * self.proj(attn_out)

        x = x + self.mlp(self.norm2(x))
        return x


class _FlowGuidedLocalAttentionChunked(nn.Module):
    """
    Local attention for one source.

    Usage:
        out_prev = attn(cur, prev, flow_cur_to_prev)
        out_cur  = attn(cur, cur,  None)
        out_next = attn(cur, nxt,  flow_cur_to_next)

    query:  (B, C, H, W)
    source: (B, C, H, W)
    flow:   (B, 2, H, W) or None

    If flow is None:
        windows are centered at the same spatial position.
    If flow is given:
        windows are centered at x + round(flow_x), y + round(flow_y).
    """

    def __init__(
        self,
        feat_ch: int = 64,
        attn_ch: int | None = None,
        win: int = 7,
        chunk_size: int = 1024,
        share_qk: bool = False,
        use_pos_bias: bool = True,
        out_proj: bool = False,
    ):
        super().__init__()
        assert win % 2 == 1, "win must be odd"

        attn_ch = attn_ch or feat_ch

        self.feat_ch = feat_ch
        self.attn_ch = attn_ch
        self.win = int(win)
        self.chunk_size = int(chunk_size)
        self.share_qk = bool(share_qk)
        self.use_pos_bias = bool(use_pos_bias)

        self.q = nn.Conv2d(feat_ch, attn_ch, 1, 1, 0)

        if share_qk:
            self.k = None
        else:
            self.k = nn.Conv2d(feat_ch, attn_ch, 1, 1, 0)

        self.v = nn.Conv2d(feat_ch, attn_ch, 1, 1, 0)

        self.proj = nn.Conv2d(attn_ch, feat_ch, 1, 1, 0) if out_proj else nn.Identity()

        if self.use_pos_bias:
            self.rel_pos_bias = nn.Parameter(torch.zeros(win * win))
        else:
            self.register_parameter("rel_pos_bias", None)

        pad = self.win // 2
        oy, ox = torch.meshgrid(
            torch.arange(-pad, pad + 1),
            torch.arange(-pad, pad + 1),
            indexing="ij",
        )
        self.register_buffer("offset_y", oy.reshape(-1).long(), persistent=False)
        self.register_buffer("offset_x", ox.reshape(-1).long(), persistent=False)

    def _get_k(self, x: torch.Tensor) -> torch.Tensor:
        return self.q(x) if self.k is None else self.k(x)

    def _make_regular_indices(
        self,
        B: int,
        y0: int,
        y1: int,
        H: int,
        W: int,
        device,
    ) -> torch.Tensor:
        pad = self.win // 2
        win2 = self.win * self.win
        rows = y1 - y0
        L = rows * W
        Wpad = W + 2 * pad

        yy = torch.arange(y0, y1, device=device)
        xx = torch.arange(W, device=device)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")

        gx = gx.reshape(1, L) + pad
        gy = gy.reshape(1, L) + pad

        xs = gx.unsqueeze(-1) + self.offset_x.view(1, 1, win2)
        ys = gy.unsqueeze(-1) + self.offset_y.view(1, 1, win2)

        idx = ys * Wpad + xs
        idx = idx.long().expand(B, -1, -1)
        return idx

    def _make_flow_guided_indices(
        self,
        flow: torch.Tensor,
        y0: int,
        y1: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        device = flow.device
        B = flow.shape[0]

        pad = self.win // 2
        win2 = self.win * self.win
        rows = y1 - y0
        L = rows * W
        Wpad = W + 2 * pad

        yy = torch.arange(y0, y1, device=device)
        xx = torch.arange(W, device=device)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")

        gx = gx.reshape(1, L)
        gy = gy.reshape(1, L)

        dx = torch.round(flow[:, 0, y0:y1, :]).to(torch.long).reshape(B, L)
        dy = torch.round(flow[:, 1, y0:y1, :]).to(torch.long).reshape(B, L)

        cx = (gx + dx).clamp(0, W - 1) + pad
        cy = (gy + dy).clamp(0, H - 1) + pad

        xs = cx.unsqueeze(-1) + self.offset_x.view(1, 1, win2)
        ys = cy.unsqueeze(-1) + self.offset_y.view(1, 1, win2)

        idx = ys * Wpad + xs
        return idx.long()

    @staticmethod
    def _gather_windows_from_indices(
        x_pad: torch.Tensor,
        idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        x_pad: (B, C, Hpad, Wpad)
        idx:   (B, L, win2)

        return:
            windows: (B, L, win2, C)
        """
        B, C, Hpad, Wpad = x_pad.shape
        _, L, win2 = idx.shape

        flat = x_pad.view(B, C, Hpad * Wpad)
        idx_flat = idx.view(B, 1, L * win2).expand(-1, C, -1)

        gathered = flat.gather(2, idx_flat)
        gathered = gathered.view(B, C, L, win2)
        gathered = gathered.permute(0, 2, 3, 1).contiguous()

        return gathered

    def forward(
        self,
        query: torch.Tensor,
        source: torch.Tensor,
        flow: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self.q(query)
        k = self._get_k(source)
        v = self.v(source)

        B, C, H, W = q.shape
        pad = self.win // 2
        win2 = self.win * self.win
        HW = H * W

        k_pad = F.pad(k, (pad, pad, pad, pad))
        v_pad = F.pad(v, (pad, pad, pad, pad))

        q_seq = q.view(B, C, HW).permute(0, 2, 1).contiguous()
        out_seq = q_seq.new_empty(B, HW, C)

        cs = self.chunk_size if self.chunk_size > 0 else HW
        chunk_rows = max(1, cs // W)

        base_index = 0

        for y0 in range(0, H, chunk_rows):
            y1 = min(H, y0 + chunk_rows)
            rows = y1 - y0
            L = rows * W

            if flow is None:
                idx = self._make_regular_indices(B, y0, y1, H, W, query.device)
            else:
                idx = self._make_flow_guided_indices(flow, y0, y1, H, W)

            k_win = self._gather_windows_from_indices(k_pad, idx)
            v_win = self._gather_windows_from_indices(v_pad, idx)

            q_chunk = q_seq[:, base_index:base_index + L, :]

            scores = (k_win * q_chunk.unsqueeze(2)).sum(dim=-1)
            scores = scores / (C ** 0.5)

            if self.rel_pos_bias is not None:
                scores = scores + self.rel_pos_bias.view(1, 1, win2)

            att = torch.softmax(scores, dim=-1)

            out_chunk = (att.unsqueeze(-1) * v_win).sum(dim=2)

            out_seq[:, base_index:base_index + L, :] = out_chunk
            base_index += L

        out = out_seq.permute(0, 2, 1).view(B, C, H, W)
        out = self.proj(out)

        return out


### ---------------------------------------------------------------------------------- ###
### ---------------------------------------------------------------------------------- ###
### -------- TRANSFORMER STYLE FUSION BLOCKS WITH FLOW-GUIDED LOCAL ATTENTION -------- ###
### ---------------------------------------------------------------------------------- ###
### ---------------------------------------------------------------------------------- ###



class FlowGuidedLocalTransformerFusionBlock(BaseFusion):
    """
    Pre-norm transformer-style fusion block using flow-guided local dense attention.

    Structure:
        x = cur + Attention(Norm(prev), Norm(cur), Norm(nxt), flow)
        x = x   + MLP(Norm(x))
    """

    def __init__(
        self,
        feat_ch=64,
        attn_ch=None,
        win=7,
        chunk_size=1024,
        share_qk=False,
        mlp_ratio=4.0,
    ):
        super().__init__(feat_ch=feat_ch)

        hidden_dim = int(feat_ch * mlp_ratio)

        # Shared norm for the three inputs before attention
        self.norm1 = nn.GroupNorm(1, feat_ch)

        self.attn = _FlowGuidedLocalDenseAttentionChunked(
            feat_ch=feat_ch,
            attn_ch=attn_ch,
            win=win,
            chunk_size=chunk_size,
            share_qk=share_qk,
        )

        self.norm2 = nn.GroupNorm(1, feat_ch)
        self.mlp = ConvMLP(feat_ch, hidden_dim=hidden_dim)

    def forward(self, prev, cur, nxt, flow_cur_to_prev, flow_cur_to_next):
        prev_n = self.norm1(prev)
        cur_n = self.norm1(cur)
        nxt_n = self.norm1(nxt)

        x = cur + self.attn(
            prev_n,
            cur_n,
            nxt_n,
            flow_cur_to_prev,
            flow_cur_to_next,
        )

        x = x + self.mlp(self.norm2(x))
        return x

class _FlowGuidedLocalDenseAttentionChunked(nn.Module):
    """
    Dense local attention with integer flow-guided search windows:
      - No global warp
      - cur: regular local window centered at (x, y)
      - prev/nxt: local window centered at (x + round(flow_x), y + round(flow_y))
      - Chunked over rows to reduce memory
    """

    def __init__(
        self,
        feat_ch=64,
        attn_ch=None,
        win=7,
        chunk_size=1024,
        share_qk=False,
    ):
        super().__init__()
        assert win % 2 == 1
        attn_ch = attn_ch or feat_ch

        self.win = win
        self.chunk_size = int(chunk_size)

        self.q = nn.Conv2d(feat_ch, attn_ch, 1, 1, 0)
        self.k = nn.Conv2d(feat_ch, attn_ch, 1, 1, 0) if not share_qk else None
        self.v = nn.Conv2d(feat_ch, attn_ch, 1, 1, 0)
        self.proj = nn.Conv2d(attn_ch, feat_ch, 1, 1, 0)

        pad = self.win // 2
        oy, ox = torch.meshgrid(
            torch.arange(-pad, pad + 1),
            torch.arange(-pad, pad + 1),
            indexing="ij",
        )
        self.register_buffer("offset_y", oy.reshape(-1).long(), persistent=False)  # (win2,)
        self.register_buffer("offset_x", ox.reshape(-1).long(), persistent=False)  # (win2,)

    def _gather_windows_from_indices(self, x_pad: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        x_pad: (B, C, Hpad, Wpad)
        idx:   (B, L, win2) linear indices over padded spatial dimension
        return: (B, L, win2, C)
        """
        B, C, Hpad, Wpad = x_pad.shape
        _, L, win2 = idx.shape

        flat = x_pad.view(B, C, Hpad * Wpad)  # (B, C, Hpad*Wpad)
        idx_flat = idx.view(B, 1, L * win2).expand(-1, C, -1)  # (B, C, L*win2)

        gathered = flat.gather(2, idx_flat)  # (B, C, L*win2)
        gathered = gathered.view(B, C, L, win2).permute(0, 2, 3, 1).contiguous()  # (B, L, win2, C)
        return gathered

    def _make_flow_guided_indices(
        self,
        flow: torch.Tensor,
        y0: int,
        y1: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Build linear padded-image indices for flow-guided windows.

        flow: (B, 2, H, W), interpreted as:
              flow[:, 0] = dx, flow[:, 1] = dy
        returns:
              idx: (B, L, win2), where L = (y1-y0)*W
        """
        device = flow.device
        B = flow.shape[0]
        pad = self.win // 2
        win2 = self.win * self.win
        rows = y1 - y0
        L = rows * W
        Wpad = W + 2 * pad

        # Base coordinates for this strip in original image coordinates
        yy = torch.arange(y0, y1, device=device)
        xx = torch.arange(W, device=device)
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")  # (rows, W)

        gx = gx.reshape(1, L)  # (1, L)
        gy = gy.reshape(1, L)  # (1, L)

        dx = torch.round(flow[:, 0, y0:y1, :]).to(torch.long).reshape(B, L)
        dy = torch.round(flow[:, 1, y0:y1, :]).to(torch.long).reshape(B, L)

        # Shift center according to integer flow, clamp to valid ORIGINAL coords
        cx = (gx + dx).clamp(0, W - 1) + pad  # padded coords
        cy = (gy + dy).clamp(0, H - 1) + pad  # padded coords

        # Add local window offsets
        xs = cx.unsqueeze(-1) + self.offset_x.view(1, 1, win2)  # type: ignore # (B, L, win2)
        ys = cy.unsqueeze(-1) + self.offset_y.view(1, 1, win2)  # type: ignore # (B, L, win2) 

        # Convert to linear indices over padded spatial dimension
        idx = ys * Wpad + xs  # (B, L, win2)
        return idx.long()

    def forward(self, prev, cur, nxt, flow_cur_to_prev, flow_cur_to_next):
        q_c = self.q(cur)
        k_c = self.k(cur) if self.k is not None else self.q(cur)
        v_c = self.v(cur)
        k_p = self.k(prev) if self.k is not None else self.q(prev)
        v_p = self.v(prev)
        k_n = self.k(nxt) if self.k is not None else self.q(nxt)
        v_n = self.v(nxt)

        B, C, H, W = q_c.shape
        d = C
        pad = self.win // 2
        win2 = self.win * self.win

        # Pad once
        k_c_pad = F.pad(k_c, (pad, pad, pad, pad))
        v_c_pad = F.pad(v_c, (pad, pad, pad, pad))
        k_p_pad = F.pad(k_p, (pad, pad, pad, pad))
        v_p_pad = F.pad(v_p, (pad, pad, pad, pad))
        k_n_pad = F.pad(k_n, (pad, pad, pad, pad))
        v_n_pad = F.pad(v_n, (pad, pad, pad, pad))

        HW = H * W
        q_seq = q_c.view(B, C, HW).permute(0, 2, 1).contiguous()  # (B, HW, C)
        out_seq = q_seq.new_empty(B, HW, C)

        cs = self.chunk_size if self.chunk_size > 0 else HW
        chunk_rows = max(1, cs // W)

        base_index = 0
        for y0 in range(0, H, chunk_rows):
            y1 = min(H, y0 + chunk_rows)
            rows = y1 - y0
            L = rows * W

            # -------------------------
            # cur windows: regular local windows via unfold
            # -------------------------
            ys = y0
            ye = y1 + 2 * pad

            kc_strip = k_c_pad[:, :, ys:ye, :]  # (B, C, rows+2*pad, W+2*pad)
            vc_strip = v_c_pad[:, :, ys:ye, :]

            kc_u = F.unfold(kc_strip, kernel_size=self.win, padding=0)  # (B, C*win2, L)
            vc_u = F.unfold(vc_strip, kernel_size=self.win, padding=0)

            kc = kc_u.view(B, C, win2, L).permute(0, 3, 2, 1).contiguous()  # (B, L, win2, C)
            vc = vc_u.view(B, C, win2, L).permute(0, 3, 2, 1).contiguous()

            # -------------------------
            # prev / nxt windows: flow-guided centers
            # -------------------------
            idx_prev = self._make_flow_guided_indices(flow_cur_to_prev, y0, y1, H, W)  # (B, L, win2)
            idx_next = self._make_flow_guided_indices(flow_cur_to_next, y0, y1, H, W)

            kp = self._gather_windows_from_indices(k_p_pad, idx_prev)  # (B, L, win2, C)
            vp = self._gather_windows_from_indices(v_p_pad, idx_prev)
            kn = self._gather_windows_from_indices(k_n_pad, idx_next)
            vn = self._gather_windows_from_indices(v_n_pad, idx_next)

            q_chunk = q_seq[:, base_index:base_index + L, :]  # (B, L, C)

            # Scores per source
            sp = (kp * q_chunk.unsqueeze(2)).sum(dim=-1)
            sc = (kc * q_chunk.unsqueeze(2)).sum(dim=-1)
            sn = (kn * q_chunk.unsqueeze(2)).sum(dim=-1)

            scores = torch.cat([sp, sc, sn], dim=2) / (d ** 0.5)  # (B, L, 3*win2)
            att = torch.softmax(scores, dim=-1)

            ap, ac, an = att.split([win2, win2, win2], dim=2)

            out_chunk = (
                (ap.unsqueeze(-1) * vp).sum(dim=2) +
                (ac.unsqueeze(-1) * vc).sum(dim=2) +
                (an.unsqueeze(-1) * vn).sum(dim=2)
            )  # (B, L, C)

            out_seq[:, base_index:base_index + L, :] = out_chunk
            base_index += L

        out = out_seq.permute(0, 2, 1).view(B, C, H, W)
        return self.proj(out)