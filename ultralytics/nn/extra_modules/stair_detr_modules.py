import torch
import torch.nn as nn


__all__ = ("DSEB", "StatisticalFeatureAttention")


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DSEB(nn.Module):
    """STAIR-DETR diverse semantic enhancement block."""

    def __init__(self, c1, c2):
        super().__init__()
        self.main = ConvBNAct(c1, c2, k=3)
        self.point = ConvBNAct(c1, c2, k=1)
        self.pool = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1), ConvBNAct(c1, c2, k=1))
        self.dw = nn.Sequential(ConvBNAct(c1, c1, k=3, g=c1), ConvBNAct(c1, c2, k=1))
        self.fuse = ConvBNAct(c2 * 4, c2, k=1)

    def forward(self, x):
        return self.fuse(torch.cat((self.main(x), self.point(x), self.pool(x), self.dw(x)), dim=1))


class StatisticalFeatureAttention(nn.Module):
    """STAIR-DETR SFA token-statistics attention."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.spatial = nn.Sequential(
            nn.Conv2d(2, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.var(dim=1, keepdim=True, unbiased=False).add(1e-6).sqrt()
        return x * self.spatial(torch.cat((mean, std), dim=1)) * self.channel(x)
