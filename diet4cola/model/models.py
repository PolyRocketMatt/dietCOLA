import torch
import torch.nn as nn
import torch.nn.functional as F

def get_norm(channels, groups=8):
    groups = min(groups, channels)
    groups = max(1, groups)
    return nn.GroupNorm(groups, channels)


class BaseConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.act = nn.ReLU()

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)

        # GroupNorm because of small batch size
        self.norm1 = get_norm(in_ch)
        self.norm2 = get_norm(out_ch)

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        skip = self.shortcut(x)
        x = self.conv1(self.act(self.norm1(x)))
        x = self.conv2(self.act(self.norm2(x)))
        return x + skip
    

class DownConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.AvgPool2d(kernel_size=2)
        self.conv = BaseConv(in_ch, out_ch)
    
    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x
    

class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = BaseConv(in_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x
    
class UpConvResidual(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = BaseConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        # x = previous features (to be upsampled)
        x = self.up(x)

        # In case of odd spatial sizes, align x to skip's size
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x
    

class UNet(nn.Module):
    def __init__(self, base_ch=32, in_ch = 2, out_ch=2):
        super().__init__()

        self.name = 'UNet'
        self.proj_in = nn.Conv2d(in_ch, base_ch, 1)
        self.proj_out = nn.Conv2d(base_ch, out_ch, 1)

        self.layers = nn.Sequential(
            # Down
            BaseConv(base_ch, base_ch),
            DownConv(base_ch, base_ch * 2),
            DownConv(base_ch * 2, base_ch * 4),
            DownConv(base_ch * 4, base_ch * 8),

            # Bottleneck
            BaseConv(base_ch * 8, base_ch * 8),
            BaseConv(base_ch * 8, base_ch * 8),

            # Up
            UpConv(base_ch * 8, base_ch * 4),
            UpConv(base_ch * 4, base_ch * 2),
            UpConv(base_ch * 2, base_ch),
            BaseConv(base_ch, base_ch)
        )
            
    def forward(self, x):
        x = self.proj_in(x)
        x = self.layers(x)
        x = self.proj_out(x)

        return x
    

class ResidualUNet(nn.Module):
    def __init__(self, base_ch=32, in_ch=2, out_ch=2):
        super().__init__()

        self.proj_in = nn.Conv2d(in_ch, base_ch, 1)
        self.proj_out = nn.Conv2d(base_ch, out_ch, 1)

        # Down path
        self.down1 = BaseConv(base_ch, base_ch)
        self.down2 = DownConv(base_ch, base_ch * 2)
        self.down3 = DownConv(base_ch * 2, base_ch * 4)
        self.down4 = DownConv(base_ch * 4, base_ch * 8)

        # Bottleneck
        self.b1 = BaseConv(base_ch * 8, base_ch * 8)
        self.b2 = BaseConv(base_ch * 8, base_ch * 8)

        # Up path - use UpConvResidual so we properly upsample then concat skip
        self.up1 = UpConvResidual(base_ch * 8, base_ch * 4, base_ch * 4)  # up b (8ch) + s3 (4ch) -> 4ch
        self.up2 = UpConvResidual(base_ch * 4, base_ch * 2, base_ch * 2)  # up prev (4ch) + s2 (2ch) -> 2ch
        self.up3 = UpConvResidual(base_ch * 2, base_ch, base_ch)          # up prev (2ch) + s1 (1ch) -> 1ch
        self.last = BaseConv(base_ch, base_ch)

    def forward(self, x):
        x = self.proj_in(x)

        # Down path (store skips)
        s1 = self.down1(x)          # base_ch
        s2 = self.down2(s1)         # base_ch * 2
        s3 = self.down3(s2)         # base_ch * 4
        s4 = self.down4(s3)         # base_ch * 8

        # Bottleneck
        b = self.b1(s4)
        b = self.b2(b)

        # Up path with skip concatenation using UpConvResidual
        u1 = self.up1(b, s3)
        u2 = self.up2(u1, s2)
        u3 = self.up3(u2, s1)

        out = self.last(u3)
        out = self.proj_out(out)
        return out
    
class ResidualUNetMagnitude(nn.Module):
    def __init__(self, base_ch=32, in_ch=2, out_ch=3):
        super().__init__()

        self.proj_in = nn.Conv2d(in_ch, base_ch, 1)
        self.proj_out = nn.Conv2d(base_ch, out_ch, 1)

        # Down path
        self.down1 = BaseConv(base_ch, base_ch)
        self.down2 = DownConv(base_ch, base_ch * 2)
        self.down3 = DownConv(base_ch * 2, base_ch * 4)
        self.down4 = DownConv(base_ch * 4, base_ch * 8)

        # Bottleneck
        self.b1 = BaseConv(base_ch * 8, base_ch * 8)
        self.b2 = BaseConv(base_ch * 8, base_ch * 8)

        # Up path - use UpConvResidual so we properly upsample then concat skip
        self.up1 = UpConvResidual(base_ch * 8, base_ch * 4, base_ch * 4)  # up b (8ch) + s3 (4ch) -> 4ch
        self.up2 = UpConvResidual(base_ch * 4, base_ch * 2, base_ch * 2)  # up prev (4ch) + s2 (2ch) -> 2ch
        self.up3 = UpConvResidual(base_ch * 2, base_ch, base_ch)          # up prev (2ch) + s1 (1ch) -> 1ch
        self.last = BaseConv(base_ch, base_ch)

    def forward(self, x):
        x = self.proj_in(x)

        # Down path (store skips)
        s1 = self.down1(x)          # base_ch
        s2 = self.down2(s1)         # base_ch * 2
        s3 = self.down3(s2)         # base_ch * 4
        s4 = self.down4(s3)         # base_ch * 8

        # Bottleneck
        b = self.b1(s4)
        b = self.b2(b)

        # Up path with skip concatenation using UpConvResidual
        u1 = self.up1(b, s3)
        u2 = self.up2(u1, s2)
        u3 = self.up3(u2, s1)

        out = self.last(u3)
        out = self.proj_out(out)
        return out
    
class ResidualUNetLarge(nn.Module):
    def __init__(self, base_ch=32, in_ch=2, out_ch=2):
        super().__init__()

        self.proj_in = nn.Conv2d(in_ch, base_ch, 1)
        self.proj_out = nn.Conv2d(base_ch, out_ch, 1)

        # Down path
        self.down1 = BaseConv(base_ch, base_ch)
        self.down2 = DownConv(base_ch, base_ch * 2)
        self.down3 = DownConv(base_ch * 2, base_ch * 4)
        self.down4 = DownConv(base_ch * 4, base_ch * 8)
        self.down5 = DownConv(base_ch * 8, base_ch * 16)
        self.down6 = DownConv(base_ch * 16, base_ch * 32)

        # Bottleneck
        self.b1 = BaseConv(base_ch * 32, base_ch * 32)
        self.b2 = BaseConv(base_ch * 32, base_ch * 32)

        # Up path - use UpConvResidual so we properly upsample then concat skip
        self.up1 = UpConvResidual(base_ch * 32, base_ch * 16, base_ch * 16)
        self.up2 = UpConvResidual(base_ch * 16, base_ch * 8, base_ch * 8)
        self.up3 = UpConvResidual(base_ch * 8, base_ch * 4, base_ch * 4)  # up b (8ch) + s3 (4ch) -> 4ch
        self.up4 = UpConvResidual(base_ch * 4, base_ch * 2, base_ch * 2)  # up prev (4ch) + s2 (2ch) -> 2ch
        self.up5 = UpConvResidual(base_ch * 2, base_ch, base_ch)          # up prev (2ch) + s1 (1ch) -> 1ch
        self.last = BaseConv(base_ch, base_ch)

    def forward(self, x):
        x = self.proj_in(x)

        # Down path (store skips)
        s1 = self.down1(x)          # base_ch
        s2 = self.down2(s1)         # base_ch * 2
        s3 = self.down3(s2)         # base_ch * 4
        s4 = self.down4(s3)         # base_ch * 8
        s5 = self.down5(s4)         # base_ch * 8
        s6 = self.down6(s5)         # base_ch * 8

        # Bottleneck
        b = self.b1(s6)
        b = self.b2(b)

        # Up path with skip concatenation using UpConvResidual
        u1 = self.up1(b, s5)
        u2 = self.up2(u1, s4)
        u3 = self.up3(u2, s3)
        u4 = self.up4(u3, s2)
        u5 = self.up5(u4, s1)

        out = self.last(u5)
        out = self.proj_out(out)
        return out