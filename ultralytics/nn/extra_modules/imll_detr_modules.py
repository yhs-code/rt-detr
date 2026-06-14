import torch
import torch.nn as nn


__all__ = ("MDGA",)


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            if isinstance(k, tuple):
                p = tuple((ki - 1) // 2 for ki in k)
            else:
                p = (k - 1) // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DynamicConv3x3(nn.Module):
    def __init__(self, channels, num_kernels=4, reduction=8):
        super().__init__()
        self.kernels = nn.ModuleList([ConvBNAct(channels, channels, k=3, g=channels) for _ in range(num_kernels)])
        hidden = max(channels // reduction, 8)
        self.weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, num_kernels, 1, bias=True),
        )

    def forward(self, x):
        feats = torch.stack([kernel(x) for kernel in self.kernels], dim=1)
        weight = torch.softmax(self.weight(x), dim=1).unsqueeze(2)
        return (feats * weight).sum(dim=1)


class MDGA(nn.Module):
    """IMLL-DETR multi-scale dynamic convolutional gated attention."""

    def __init__(self, channels, kernels=(7, 11, 19)):
        super().__init__()
        self.local = nn.ModuleList(
            [
                nn.Sequential(
                    ConvBNAct(channels, channels, k=(1, k), g=channels),
                    ConvBNAct(channels, channels, k=(k, 1), g=channels),
                )
                for k in kernels
            ]
        )
        self.local_fuse = ConvBNAct(channels * len(kernels), channels, k=1)
        self.global_context = DynamicConv3x3(channels)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        local = self.local_fuse(torch.cat([branch(x) for branch in self.local], dim=1))
        global_context = self.global_context(x)
        gate = self.gate(torch.cat((local, global_context), dim=1))
        return gate * local + (1.0 - gate) * global_context
