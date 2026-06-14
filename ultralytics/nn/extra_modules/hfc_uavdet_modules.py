import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ("CHFB",)


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


class HFB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.local = nn.Sequential(
            ConvBNAct(channels, channels, k=3, g=channels),
            ConvBNAct(channels, channels, k=1),
        )
        self.hf = ConvBNAct(channels, channels, k=1)

    def forward(self, x):
        local = self.local(x)
        max_edge = F.max_pool2d(x, kernel_size=3, stride=1, padding=1) - F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        hf = self.hf(max_edge)
        response = hf.abs().mean(1, keepdim=True)
        steady = response / (F.avg_pool2d(response, kernel_size=3, stride=1, padding=1) + 1e-6)
        gate = torch.sigmoid(steady - 1.0)
        return local + hf * gate


class CHFB(nn.Module):
    """HFC-UAVDet consistent high-frequency enhancement block."""

    def __init__(self, channels, n=2):
        super().__init__()
        c_mid = max(channels // 2, 8)
        self.split = ConvBNAct(channels, c_mid * 2, k=1)
        self.blocks = nn.Sequential(*(HFB(c_mid) for _ in range(int(n))))
        self.fuse = ConvBNAct(c_mid * 2, channels, k=1)

    def forward(self, x):
        x_keep, x_hf = self.split(x).chunk(2, dim=1)
        return self.fuse(torch.cat((x_keep, self.blocks(x_hf)), dim=1))
