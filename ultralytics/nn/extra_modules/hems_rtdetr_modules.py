import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ("CSPMSEIE",)


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


class CSPMSEIE(nn.Module):
    """HEMS-RTDETR CSP multi-scale edge information enhancement."""

    def __init__(self, channels, bins=(3, 6, 9, 12)):
        super().__init__()
        c_mid = max(channels // 2, 8)
        self.short = ConvBNAct(channels, c_mid, k=1)
        self.edge_in = ConvBNAct(channels, c_mid, k=1)
        self.edge_enhancers = nn.ModuleList([ConvBNAct(c_mid, c_mid, k=3, g=c_mid) for _ in bins])
        self.bins = tuple(int(b) for b in bins)
        self.fuse = ConvBNAct(c_mid * (len(self.bins) + 1), channels, k=1)

    def forward(self, x):
        short = self.short(x)
        feat = self.edge_in(x)
        outs = [short]
        for size, enhance in zip(self.bins, self.edge_enhancers):
            pooled = F.adaptive_avg_pool2d(feat, output_size=(size, size))
            smooth = F.interpolate(pooled, size=feat.shape[-2:], mode="bilinear", align_corners=False)
            edge = feat - smooth
            outs.append(enhance(edge))
        return self.fuse(torch.cat(outs, dim=1))
