import torch
import torch.nn as nn


__all__ = ("MSFA", "PSConv")


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        if p is None:
            if isinstance(k, tuple):
                p = tuple(((ki - 1) * d) // 2 for ki in k)
            else:
                p = ((k - 1) * d) // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MSFA(nn.Module):
    """IRS-DETR multi-scale feature aggregation module."""

    def __init__(self, c1, c2, reduction=8):
        super().__init__()
        branch = max(c2 // 4, 8)
        self.branches = nn.ModuleList(
            (
                ConvBNAct(c1, branch, k=1),
                ConvBNAct(c1, branch, k=3),
                ConvBNAct(c1, branch, k=5),
                ConvBNAct(c1, branch, k=3, d=2),
            )
        )
        self.fuse = ConvBNAct(branch * 4, c2, k=1)
        hidden = max(c2 // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c2, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))
        return y * self.gate(y)


class PSConv(nn.Module):
    """IRS-DETR pinwheel-shaped convolution downsampling operator."""

    def __init__(self, c1, c2, k=5, s=2):
        super().__init__()
        branch = max(c2 // 4, 8)
        self.paths = nn.ModuleList(
            (
                ConvBNAct(c1, branch, k=(1, k), s=s),
                ConvBNAct(c1, branch, k=(k, 1), s=s),
                ConvBNAct(c1, branch, k=(1, k), s=s),
                ConvBNAct(c1, branch, k=(k, 1), s=s),
            )
        )
        self.fuse = ConvBNAct(branch * 4, c2, k=1)

    def forward(self, x):
        return self.fuse(torch.cat([path(x) for path in self.paths], dim=1))
