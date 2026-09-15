import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseFusion
from ..blocks import ResBlock


class WarpLocalDenseAttentionChunkedFusion(BaseFusion):
    def __init__(self, feat_ch=64, attn_ch=None, win=7, num_blocks=1, chunk_size=1024):
        super().__init__(feat_ch=feat_ch)
        assert win % 2 == 1
        attn_ch = attn_ch or feat_ch

        self.win = win
        self.chunk_size = int(chunk_size)

        self.q = nn.Conv2d(feat_ch, attn_ch, 1, 1, 0)
        self.kv = nn.Conv2d(feat_ch, attn_ch * 2, 1, 1, 0)
        self.proj = nn.Conv2d(attn_ch, feat_ch, 1, 1, 0)
        self.refine = nn.Sequential(*[ResBlock(feat_ch) for _ in range(num_blocks)])

    def forward(self, prev, cur, nxt, flow_cur_to_prev, flow_cur_to_next):
        prev = self.warp(prev, flow_cur_to_prev)
        nxt  = self.warp(nxt,  flow_cur_to_next)

        q_c = self.q(cur)
        k_c, v_c = self.kv(cur).chunk(2, dim=1)
        k_p, v_p = self.kv(prev).chunk(2, dim=1)
        k_n, v_n = self.kv(nxt).chunk(2, dim=1)

        B, C, H, W = q_c.shape
        d = C
        pad = self.win // 2
        win2 = self.win * self.win
        P = 3 * win2

        # Pad 1 vez (muy barato comparado con unfold gigante)
        k_p = F.pad(k_p, (pad, pad, pad, pad))
        v_p = F.pad(v_p, (pad, pad, pad, pad))
        k_c = F.pad(k_c, (pad, pad, pad, pad))
        v_c = F.pad(v_c, (pad, pad, pad, pad))
        k_n = F.pad(k_n, (pad, pad, pad, pad))
        v_n = F.pad(v_n, (pad, pad, pad, pad))

        # q en secuencia
        HW = H * W
        q_seq = q_c.view(B, C, HW).permute(0, 2, 1)  # (B, HW, C)
        out_seq = q_seq.new_empty(B, HW, C)

        # chunking por tiras: L = chunk_rows * W (aprox <= chunk_size)
        cs = self.chunk_size if self.chunk_size > 0 else HW
        chunk_rows = max(1, cs // W)

        base_index = 0
        for y0 in range(0, H, chunk_rows):
            y1 = min(H, y0 + chunk_rows)
            rows = y1 - y0
            L = rows * W

            # slice vertical en tensores PADDEADOS: (B, C, rows+2*pad, W+2*pad)
            ys = y0
            ye = y1 + 2 * pad
            kp_strip = k_p[:, :, ys:ye, :]
            vp_strip = v_p[:, :, ys:ye, :]
            kc_strip = k_c[:, :, ys:ye, :]
            vc_strip = v_c[:, :, ys:ye, :]
            kn_strip = k_n[:, :, ys:ye, :]
            vn_strip = v_n[:, :, ys:ye, :]

            # unfold SOLO de la tira (padding=0 porque ya paddeamos)
            # -> (B, C*win2, L)
            kp_u = F.unfold(kp_strip, kernel_size=self.win, padding=0)
            vp_u = F.unfold(vp_strip, kernel_size=self.win, padding=0)
            kc_u = F.unfold(kc_strip, kernel_size=self.win, padding=0)
            vc_u = F.unfold(vc_strip, kernel_size=self.win, padding=0)
            kn_u = F.unfold(kn_strip, kernel_size=self.win, padding=0)
            vn_u = F.unfold(vn_strip, kernel_size=self.win, padding=0)

            # a (B, L, win2, C)
            kp = kp_u.view(B, C, win2, L).permute(0, 3, 2, 1)
            vp = vp_u.view(B, C, win2, L).permute(0, 3, 2, 1)
            kc = kc_u.view(B, C, win2, L).permute(0, 3, 2, 1)
            vc = vc_u.view(B, C, win2, L).permute(0, 3, 2, 1)
            kn = kn_u.view(B, C, win2, L).permute(0, 3, 2, 1)
            vn = vn_u.view(B, C, win2, L).permute(0, 3, 2, 1)

            q_chunk = q_seq[:, base_index:base_index + L, :]  # (B, L, C)

            # scores por fuente (evitas cat de Kpool gigante)
            sp = (kp * q_chunk.unsqueeze(2)).sum(dim=-1)
            sc = (kc * q_chunk.unsqueeze(2)).sum(dim=-1)
            sn = (kn * q_chunk.unsqueeze(2)).sum(dim=-1)

            scores = torch.cat([sp, sc, sn], dim=2) / (d ** 0.5)  # (B, L, P)
            att = torch.softmax(scores, dim=-1)  # (B, L, P)

            # split atención y suma ponderada sin construir Vpool gigante
            ap, ac, an = att.split([win2, win2, win2], dim=2)  # cada una (B, L, win2)
            out_chunk = (
                (ap.unsqueeze(-1) * vp).sum(dim=2) +
                (ac.unsqueeze(-1) * vc).sum(dim=2) +
                (an.unsqueeze(-1) * vn).sum(dim=2)
            )  # (B, L, C)

            out_seq[:, base_index:base_index + L, :] = out_chunk
            base_index += L

        out = out_seq.permute(0, 2, 1).view(B, C, H, W)
        out = self.proj(out)
        return cur + self.refine(out)

class WarpPatchDenseAttentionChunkedFusion(BaseFusion):
    """
    Patch-token dense local attention, chunked over token HW:
      1) Warp prev/nxt -> cur coords (full-res)
      2) Downsample to tokens with strided conv (token_stride)
      3) Dense local attention on token grid over all P=3*win^2 candidates
      4) Project + upsample back to full-res, refine, residual
    """

    def __init__(
        self,
        feat_ch=64,
        attn_ch: int | None = None,
        win=7,
        num_blocks=1,
        chunk_size: int = 1024,
        token_stride: int = 4,
        token_ch: int | None = None,
    ):
        super().__init__(feat_ch=feat_ch)
        assert win % 2 == 1
        attn_ch = attn_ch or feat_ch
        token_ch = token_ch or feat_ch
        assert token_stride >= 1

        self.win = win
        self.chunk_size = int(chunk_size)
        self.token_stride = int(token_stride)

        # Tokenization (downsample)
        self.tokenize = nn.Conv2d(
            feat_ch, token_ch,
            kernel_size=token_stride,
            stride=token_stride,
            padding=0
        )

        # Projections on token grid
        self.q = nn.Conv2d(token_ch, attn_ch, 1, 1, 0)
        self.kv = nn.Conv2d(token_ch, attn_ch * 2, 1, 1, 0)

        # Back to feature channels
        self.proj_token = nn.Conv2d(attn_ch, token_ch, 1, 1, 0)
        self.detokenize = nn.Conv2d(token_ch, feat_ch, 1, 1, 0)

        self.refine = nn.Sequential(*[ResBlock(feat_ch) for _ in range(num_blocks)])

    def _unfold(self, x: torch.Tensor) -> torch.Tensor:
        return F.unfold(x, kernel_size=self.win, padding=self.win // 2)

    def forward(self, prev, cur, nxt, flow_cur_to_prev, flow_cur_to_next):
        prev = self.warp(prev, flow_cur_to_prev)
        nxt = self.warp(nxt, flow_cur_to_next)

        # Token grid
        cur_t = self.tokenize(cur)
        prev_t = self.tokenize(prev)
        nxt_t = self.tokenize(nxt)

        q_c = self.q(cur_t)
        k_c, v_c = self.kv(cur_t).chunk(2, dim=1)
        k_p, v_p = self.kv(prev_t).chunk(2, dim=1)
        k_n, v_n = self.kv(nxt_t).chunk(2, dim=1)

        B, C, Ht, Wt = q_c.shape
        HWt = Ht * Wt
        d = C
        win2 = self.win * self.win

        kp_u = self._unfold(k_p)  # (B, C*win2, HWt)
        vp_u = self._unfold(v_p)
        kc_u = self._unfold(k_c)
        vc_u = self._unfold(v_c)
        kn_u = self._unfold(k_n)
        vn_u = self._unfold(v_n)

        q_seq = q_c.view(B, C, HWt).permute(0, 2, 1)  # (B, HWt, C)
        out_seq = q_seq.new_empty(B, HWt, C)

        cs = self.chunk_size if self.chunk_size > 0 else HWt
        for start in range(0, HWt, cs):
            end = min(start + cs, HWt)
            L = end - start

            kp = kp_u[:, :, start:end].view(B, C, win2, L).permute(0, 3, 2, 1)  # (B,L,win2,C)
            vp = vp_u[:, :, start:end].view(B, C, win2, L).permute(0, 3, 2, 1)
            kc = kc_u[:, :, start:end].view(B, C, win2, L).permute(0, 3, 2, 1)
            vc = vc_u[:, :, start:end].view(B, C, win2, L).permute(0, 3, 2, 1)
            kn = kn_u[:, :, start:end].view(B, C, win2, L).permute(0, 3, 2, 1)
            vn = vn_u[:, :, start:end].view(B, C, win2, L).permute(0, 3, 2, 1)

            Kpool = torch.cat([kp, kc, kn], dim=2)  # (B,L,3*win2,C)
            Vpool = torch.cat([vp, vc, vn], dim=2)

            q_chunk = q_seq[:, start:end, :]  # (B,L,C)
            scores = (Kpool * q_chunk.unsqueeze(2)).sum(dim=-1) / (d ** 0.5)  # (B,L,P)

            att = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B,L,P,1)
            out_chunk = (att * Vpool).sum(dim=2)               # (B,L,C)

            out_seq[:, start:end, :] = out_chunk

        out_t = out_seq.permute(0, 2, 1).view(B, C, Ht, Wt)  # (B,C,Ht,Wt)
        out_t = self.proj_token(out_t)                       # (B,token_ch,Ht,Wt)

        out_up = F.interpolate(out_t, scale_factor=self.token_stride, mode="bilinear", align_corners=False)
        out_up = self.detokenize(out_up)

        if out_up.shape[-2:] != cur.shape[-2:]:
            out_up = F.interpolate(out_up, size=cur.shape[-2:], mode="bilinear", align_corners=False)

        return cur + self.refine(out_up)


class FlowGuidedLocalDenseAttentionChunkedFusion(BaseFusion):
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
        num_blocks=1,
        chunk_size=1024,
        share_qk=False,
    ):
        super().__init__(feat_ch=feat_ch)
        assert win % 2 == 1
        attn_ch = attn_ch or feat_ch

        self.win = win
        self.chunk_size = int(chunk_size)

        self.q = nn.Conv2d(feat_ch, attn_ch, 1, 1, 0)
        self.k = nn.Conv2d(feat_ch, attn_ch, 1, 1, 0) if share_qk else None
        self.v = nn.Conv2d(feat_ch, attn_ch, 1, 1, 0)
        self.proj = nn.Conv2d(attn_ch, feat_ch, 1, 1, 0)
        self.refine = nn.Sequential(*[ResBlock(feat_ch) for _ in range(num_blocks)])

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
        out = self.proj(out)
        return cur + self.refine(out)


class WarpLocalTopKThenDenseAttentionChunkedFusion(BaseFusion):
    """
    Two-step attention (retrieval + refinement), chunked over HW:

      Step 0: warp prev/nxt -> cur coords
      Step 1 (retrieval): compute scores over the full local pool P=3*win^2 and select Top-K
      Step 2 (refinement): run dense attention ONLY over the selected K candidates
                           (softmax over K), producing the fused output

    Notes:
      - This is "hard selection" + "soft mixing".
      - Gradients flow through the refinement softmax and the selected values; selection itself is non-smooth.
    """

    def __init__(
        self,
        feat_ch=64,
        attn_ch: int | None = None,
        win=7,
        k=16,                    # per-frame k
        k_total: int | None = None,  # optional override for total topk
        num_blocks=1,
        chunk_size: int = 1024,
    ):
        super().__init__(feat_ch=feat_ch)
        assert win % 2 == 1
        attn_ch = attn_ch or feat_ch

        self.win = win
        self.chunk_size = int(chunk_size)

        # TopK size
        self.k_total = int(k_total) if k_total is not None else int(3 * k)

        # Projections
        self.q = nn.Conv2d(feat_ch, attn_ch, 1, 1, 0)          # only for cur
        self.kv = nn.Conv2d(feat_ch, attn_ch * 2, 1, 1, 0)     # for prev/cur/nxt
        
        self.v_refine = nn.Sequential(
            nn.Linear(attn_ch, attn_ch),
            nn.GELU(),
            nn.Linear(attn_ch, attn_ch)
        )


        self.proj = nn.Conv2d(attn_ch, feat_ch, 1, 1, 0)
        self.refine = nn.Sequential(*[ResBlock(feat_ch) for _ in range(num_blocks)])

    def _unfold(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W) -> (B, C*win*win, H*W)
        return F.unfold(x, kernel_size=self.win, padding=self.win // 2)

    def forward(self, prev, cur, nxt, flow_cur_to_prev, flow_cur_to_next):
        # 0) warp neighbors -> cur coordinates
        prev = self.warp(prev, flow_cur_to_prev)
        nxt = self.warp(nxt, flow_cur_to_next)

        # 1) Q, K, V
        q_c = self.q(cur)
        k_c, v_c = self.kv(cur).chunk(2, dim=1)
        k_p, v_p = self.kv(prev).chunk(2, dim=1)
        k_n, v_n = self.kv(nxt).chunk(2, dim=1)

        B, C, H, W = q_c.shape
        HW = H * W
        d = C
        win2 = self.win * self.win
        P = 3 * win2
        topk = min(self.k_total, P)

        # 2) compact unfolds: (B, C*win2, HW)
        kp_u = self._unfold(k_p)
        vp_u = self._unfold(v_p)
        kc_u = self._unfold(k_c)
        vc_u = self._unfold(v_c)
        kn_u = self._unfold(k_n)
        vn_u = self._unfold(v_n)

        q_seq = q_c.view(B, C, HW).permute(0, 2, 1)  # (B, HW, C)
        out_seq = q_seq.new_empty(B, HW, C)

        # 3) chunked over query positions
        cs = self.chunk_size if self.chunk_size > 0 else HW
        for start in range(0, HW, cs):
            end = min(start + cs, HW)
            L = end - start

            # (B, L, win2, C)
            kp = kp_u[:, :, start:end].view(B, C, win2, L).permute(0, 3, 2, 1)
            vp = vp_u[:, :, start:end].view(B, C, win2, L).permute(0, 3, 2, 1)
            kc = kc_u[:, :, start:end].view(B, C, win2, L).permute(0, 3, 2, 1)
            vc = vc_u[:, :, start:end].view(B, C, win2, L).permute(0, 3, 2, 1)
            kn = kn_u[:, :, start:end].view(B, C, win2, L).permute(0, 3, 2, 1)
            vn = vn_u[:, :, start:end].view(B, C, win2, L).permute(0, 3, 2, 1)

            # Full pools: (B, L, P, C)
            Kpool = torch.cat([kp, kc, kn], dim=2)
            Vpool = torch.cat([vp, vc, vn], dim=2)

            # Queries: (B, L, C)
            q_chunk = q_seq[:, start:end, :]

            # Step 1: retrieval
            scores = (Kpool * q_chunk.unsqueeze(2)).sum(dim=-1) / (d ** 0.5)
            vals, idx = torch.topk(scores, k=topk, dim=-1)

            idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, C)
            Vsel = torch.gather(Vpool, dim=2, index=idx_exp)  # (B,L,topk,C)

            # Step 2: reuse logits + refine values
            att = torch.softmax(vals, dim=-1).unsqueeze(-1)  # (B,L,topk,1)

            Vsel_ref = self.v_refine(Vsel)  # apply per token (broadcast over batch/L/topk)

            out_chunk = (att * Vsel_ref).sum(dim=2)                          # (B,L,C)

            out_seq[:, start:end, :] = out_chunk

        out = out_seq.permute(0, 2, 1).view(B, C, H, W)
        out = self.proj(out)
        return cur + self.refine(out)



class WarpLocalTopKThenFullAttentionChunkedFusion(BaseFusion):
    """
    Two-step attention (retrieval + full re-attention), chunked over HW:

      Step 0: warp prev/nxt -> cur coords
      Step 1 (retrieval): compute scores over full local pool P=3*win^2 and select Top-K indices
      Step 2 (full attention): compute a NEW attention on the selected set with NEW projections
            q2 = Wq2(cur)     (per-query)
            k2,v2 = Wk2/Wv2(sel) (per-selected token)
          then softmax(q2·k2) and mix v2

    This is the "option 1" you described: a truly new attention pass on the retrieved candidates.
    """

    def __init__(
        self,
        feat_ch=64,
        attn_ch: int | None = None,
        win=7,
        k=16,                      # default: topk_total = 3*k
        k_total: int | None = None,
        num_blocks=1,
        chunk_size: int = 1024,
        # optional: separate dims for retrieval vs refine
        attn_ch_retr: int | None = None,
        attn_ch_ref: int | None = None,
    ):
        super().__init__(feat_ch=feat_ch)
        assert win % 2 == 1

        attn_ch = attn_ch or feat_ch
        self.attn_ch_retr = int(attn_ch_retr or attn_ch)
        self.attn_ch_ref = int(attn_ch_ref or attn_ch)

        self.win = win
        self.chunk_size = int(chunk_size)
        self.k_total = int(k_total) if k_total is not None else int(3 * k)

        # --- Retrieval projections (cheap) ---
        self.q1 = nn.Conv2d(feat_ch, self.attn_ch_retr, 1, 1, 0)                 # cur -> q1
        self.kv1 = nn.Conv2d(feat_ch, self.attn_ch_retr * 2, 1, 1, 0)            # prev/cur/nxt -> k1,v1

        # --- Refinement projections (new full attention) ---
        self.q2 = nn.Conv2d(feat_ch, self.attn_ch_ref, 1, 1, 0)                  # cur -> q2
        # apply on selected tokens (B,L,K,C): easiest as Linear over last dim
        self.k2 = nn.Linear(self.attn_ch_retr, self.attn_ch_ref)                 # k1_sel -> k2_sel
        self.v2 = nn.Linear(self.attn_ch_retr, self.attn_ch_ref)                 # v1_sel -> v2_sel

        self.proj = nn.Conv2d(self.attn_ch_ref, feat_ch, 1, 1, 0)
        self.refine = nn.Sequential(*[ResBlock(feat_ch) for _ in range(num_blocks)])

    def _unfold(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W) -> (B, C*win*win, H*W)
        return F.unfold(x, kernel_size=self.win, padding=self.win // 2)

    def forward(self, prev, cur, nxt, flow_cur_to_prev, flow_cur_to_next):
        # 0) warp neighbors -> cur coordinates
        prev = self.warp(prev, flow_cur_to_prev)
        nxt = self.warp(nxt, flow_cur_to_next)

        B, _, H, W = cur.shape
        HW = H * W
        win2 = self.win * self.win
        P = 3 * win2

        topk = min(self.k_total, P)

        # ------------------------------------------------------------
        # Step 1: Retrieval (q1 vs k1 over full pool) -> select TopK idx
        # ------------------------------------------------------------
        q1_map = self.q1(cur)  # (B,Cr,H,W)
        k1c_map, v1c_map = self.kv1(cur).chunk(2, dim=1)
        k1p_map, v1p_map = self.kv1(prev).chunk(2, dim=1)
        k1n_map, v1n_map = self.kv1(nxt).chunk(2, dim=1)

        Cr = q1_map.shape[1]
        d1 = Cr

        # unfolds: (B, Cr*win2, HW)
        k1p_u = self._unfold(k1p_map)
        v1p_u = self._unfold(v1p_map)
        k1c_u = self._unfold(k1c_map)
        v1c_u = self._unfold(v1c_map)
        k1n_u = self._unfold(k1n_map)
        v1n_u = self._unfold(v1n_map)

        q1_seq = q1_map.view(B, Cr, HW).permute(0, 2, 1)  # (B,HW,Cr)

        # output buffer in refine-dim later; build per chunk
        out_seq = q1_seq.new_empty(B, HW, self.attn_ch_ref)

        cs = self.chunk_size if self.chunk_size > 0 else HW
        for start in range(0, HW, cs):
            end = min(start + cs, HW)
            L = end - start

            # (B,L,win2,Cr)
            k1p = k1p_u[:, :, start:end].view(B, Cr, win2, L).permute(0, 3, 2, 1)
            v1p = v1p_u[:, :, start:end].view(B, Cr, win2, L).permute(0, 3, 2, 1)
            k1c = k1c_u[:, :, start:end].view(B, Cr, win2, L).permute(0, 3, 2, 1)
            v1c = v1c_u[:, :, start:end].view(B, Cr, win2, L).permute(0, 3, 2, 1)
            k1n = k1n_u[:, :, start:end].view(B, Cr, win2, L).permute(0, 3, 2, 1)
            v1n = v1n_u[:, :, start:end].view(B, Cr, win2, L).permute(0, 3, 2, 1)

            # Full pool: (B,L,P,Cr)
            K1pool = torch.cat([k1p, k1c, k1n], dim=2)
            V1pool = torch.cat([v1p, v1c, v1n], dim=2)

            q1_chunk = q1_seq[:, start:end, :]  # (B,L,Cr)

            # retrieval scores: (B,L,P)
            scores1 = (K1pool * q1_chunk.unsqueeze(2)).sum(dim=-1) / (d1 ** 0.5)

            # TopK selection indices in [0..P-1]
            _, idx = torch.topk(scores1, k=topk, dim=-1)  # (B,L,topk)

            # gather selected K1/V1: (B,L,topk,Cr)
            idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, Cr)
            K1sel = torch.gather(K1pool, dim=2, index=idx_exp)
            V1sel = torch.gather(V1pool, dim=2, index=idx_exp)

            # ------------------------------------------------------------
            # Step 2: Full NEW attention on the selected set
            # ------------------------------------------------------------
            # q2 comes from cur (new projection), but we slice it to this chunk
            q2_map = self.q2(cur)  # (B,Cf,H,W)
            Cf = q2_map.shape[1]
            d2 = Cf
            q2_seq = q2_map.view(B, Cf, HW).permute(0, 2, 1)      # (B,HW,Cf)
            q2_chunk = q2_seq[:, start:end, :]                    # (B,L,Cf)

            # project selected tokens into refine space (per token linear)
            # (B,L,topk,Cf)
            K2sel = self.k2(K1sel)
            V2sel = self.v2(V1sel)

            # new scores on selected set: (B,L,topk)
            scores2 = (K2sel * q2_chunk.unsqueeze(2)).sum(dim=-1) / (d2 ** 0.5)

            att2 = torch.softmax(scores2, dim=-1).unsqueeze(-1)   # (B,L,topk,1)
            out_chunk = (att2 * V2sel).sum(dim=2)                 # (B,L,Cf)

            out_seq[:, start:end, :] = out_chunk

        # back to map in refine dim (Cf)
        out = out_seq.permute(0, 2, 1).view(B, self.attn_ch_ref, H, W)
        out = self.proj(out)
        return cur + self.refine(out)
