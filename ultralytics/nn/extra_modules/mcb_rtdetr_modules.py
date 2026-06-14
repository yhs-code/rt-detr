import torch
import torch.nn as nn


__all__ = ("RFAConv",)


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        if p is None:
            p = ((k - 1) * d) // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class RFAConv(nn.Module):
    """MCB-RT-DETR/RFAConv-style receptive-field attention convolution."""

    def __init__(self, c1, c2, kernels=(3, 5, 7), reduction=8):
        super().__init__()
        self.branches = nn.ModuleList([ConvBNAct(c1, c2, k=k) for k in kernels])
        hidden = max(c1 // reduction, 8)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, len(kernels), 1, bias=True),
        )

    def forward(self, x):
        feats = torch.stack([branch(x) for branch in self.branches], dim=1)
        weights = torch.softmax(self.attn(x), dim=1).unsqueeze(2)
        return (feats * weights).sum(dim=1)
