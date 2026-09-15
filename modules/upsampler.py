import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.blocks import ResBlock

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
        
        # BEFORE STEP
        x = self.act(self.conv_before(feat))
        
        # UPSAMPLING STEPS
        x = self.act(self.ps(self.up1(x)))
        if self.scale == 4:
            x = self.act(self.ps(self.up2(x)))
            
        # AFTER STEP
        x = self.act(self.conv_hr(x))
        x = self.conv_out(x)

        # RESIDUAL CONNECTION WITH BILINEAR UPSAMPLED LOW-RES INPUT
        base = F.interpolate(lq, scale_factor=self.scale, mode="bilinear", align_corners=False)
        return x + base


class UpsamplerBest(nn.Module):
    
    def __init__(
        self,
        feat_ch: int = 64,
        scale: int = 4,
        out_ch: int = 3,
        num_hr_blocks: int = 4,
    ):
        super().__init__()

        assert scale >= 2, "Scale must be >= 2"
        self.scale = scale

        self.act = nn.LeakyReLU(0.1, inplace=True)

        # Initial refinement in LR
        self.conv_before = nn.Conv2d(feat_ch, feat_ch, 3, 1, 1)

        # ----------------------------
        # Dynamic upsampling stages
        # ----------------------------
        self.up_stages = nn.ModuleList()
        remaining_scale = scale

        while remaining_scale > 1:
            if remaining_scale % 2 == 0:
                r = 2
            else:
                r = remaining_scale  # handle 3 or other odd factor

            self.up_stages.append(
                nn.Sequential(
                    nn.Conv2d(feat_ch, feat_ch * (r ** 2), 3, 1, 1),
                    nn.PixelShuffle(r),
                    nn.LeakyReLU(0.1, inplace=True),
                )
            )

            remaining_scale //= r

        # ----------------------------
        # HR Body
        # ----------------------------
        self.hr_body = nn.Sequential(
            *[ResBlock(feat_ch) for _ in range(num_hr_blocks)]
        )

        # Output projection
        self.conv_out = nn.Conv2d(feat_ch, out_ch, 3, 1, 1)


    def forward(self, feat: torch.Tensor, lq: torch.Tensor) -> torch.Tensor:

        # LR refinement
        x = self.act(self.conv_before(feat))

        # Upsampling
        for stage in self.up_stages:
            x = stage(x)

        # HR refinement
        x = self.hr_body(x)

        # Output projection
        x = self.conv_out(x)

        # Bicubic residual base
        base = F.interpolate(
            lq,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False,
        )

        return x + base