import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels: int, act_slope: float = 0.1, res_scale: float = 1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act = nn.LeakyReLU(act_slope, inplace=True)
        self.res_scale = float(res_scale)

    def forward(self, x):
        return x + self.res_scale * self.conv2(self.act(self.conv1(x)))
    
    
class ConvMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None, act_layer=nn.GELU):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))
    
    
    
# --- U-NET ---

class DoubleConv(nn.Module):
    """Classic U-Net double conv block."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """
    Downsample + conv block.
    Uses strided conv for downsampling to keep everything learnable.
    """
    def __init__(self, in_ch: int, out_ch: int, reduction: int = 2):
        super().__init__()
        assert reduction >= 1
        if reduction == 1:
            self.down = nn.Identity()
        else:
            self.down = nn.Conv2d(in_ch, in_ch, kernel_size=reduction, stride=reduction, padding=0)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.conv(x)
        return x


class UpBlock(nn.Module):
    """
    Upsample + skip fusion + conv block.
    Uses bilinear upsampling followed by conv fusion.
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, reduction: int = 2):
        super().__init__()
        assert reduction >= 1
        self.reduction = reduction
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.reduction != 1:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

        
# ---------------------------------

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = ffn_expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

        
        
# ---------------------------------


class SimpleUNet(nn.Module):
    """
    Classic lightweight U-Net for feature refinement.

    Args:
        in_ch: input channels
        out_ch: output channels
        base_ch: channels at first encoder level
        num_levels: number of encoder/decoder levels
            Example:
              num_levels=1 -> only input block + bottleneck + 1 up
              num_levels=2 -> one extra down/up
              num_levels=3 -> two extra down/up
        reduction: spatial reduction factor per level, usually 2
        max_ch: optional cap for internal channels
        use_residual: if True, returns input + projected_output when shapes match

    Input:
        (B, in_ch, H, W)

    Output:
        (B, out_ch, H, W)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int | None = None,
        base_ch: int = 64,
        num_levels: int = 2,
        reduction: int = 2,
        max_ch: int | None = None,
        use_residual: bool = True,
    ):
        super().__init__()
        assert num_levels >= 1, "num_levels must be >= 1"
        assert reduction >= 1, "reduction must be >= 1"

        out_ch = in_ch if out_ch is None else out_ch
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.base_ch = int(base_ch)
        self.num_levels = int(num_levels)
        self.reduction = int(reduction)
        self.use_residual = bool(use_residual)

        def cap_channels(c: int) -> int:
            return min(c, max_ch) if max_ch is not None else c

        # Channel plan:
        # enc_channels[0] = base_ch
        # enc_channels[1] = base_ch*2
        # ...
        enc_channels = [cap_channels(base_ch * (2 ** i)) for i in range(num_levels)]

        self.input_proj = DoubleConv(in_ch, enc_channels[0])

        # Encoder downs
        self.downs = nn.ModuleList()
        for i in range(num_levels - 1):
            self.downs.append(
                DownBlock(
                    in_ch=enc_channels[i],
                    out_ch=enc_channels[i + 1],
                    reduction=reduction,
                )
            )

        # Bottleneck
        bottleneck_in = enc_channels[-1]
        bottleneck_out = cap_channels(enc_channels[-1] * 2)
        self.bottleneck_down = DownBlock(
            in_ch=bottleneck_in,
            out_ch=bottleneck_out,
            reduction=reduction,
        )
        self.bottleneck_channels = bottleneck_out

        # Decoder ups
        self.ups = nn.ModuleList()
        dec_in = bottleneck_out
        for i in reversed(range(num_levels)):
            skip_ch = enc_channels[i]
            outc = enc_channels[i]
            self.ups.append(
                UpBlock(
                    in_ch=dec_in,
                    skip_ch=skip_ch,
                    out_ch=outc,
                    reduction=reduction,
                )
            )
            dec_in = outc

        self.output_proj = nn.Conv2d(enc_channels[0], out_ch, 3, stride=1, padding=1)

        if self.use_residual and self.in_ch != self.out_ch:
            self.res_proj = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.res_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x

        skips = []

        x = self.input_proj(x)
        skips.append(x)

        for down in self.downs:
            x = down(x)
            skips.append(x)

        x = self.bottleneck_down(x)

        # skips were stored shallow -> deep
        # decoder consumes deep -> shallow
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        x = self.output_proj(x)

        if self.use_residual:
            x = x + self.res_proj(inp)

        return x