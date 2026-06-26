import torch
import torch.nn as nn

__all__ = ("FBRTFCM", "FBRTMKP")


def _autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class _Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, _autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class _Channel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.pool(self.dwconv(x)))


class _Spatial(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, 1, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.bn(self.conv(x)))


class FBRTFCM(nn.Module):
    def __init__(self, channels, out_channels=None):
        super().__init__()
        out_channels = int(out_channels) if out_channels is not None else int(channels)
        self.one = max(1, channels // 4)
        self.two = channels - self.one
        self.conv1 = _Conv(self.one, self.one, 3, 1, 1)
        self.conv12 = _Conv(self.one, self.one, 3, 1, 1)
        self.conv123 = _Conv(self.one, channels, 1, 1)
        self.conv2 = _Conv(self.two, channels, 1, 1)
        self.spatial = _Spatial(channels)
        self.channel = _Channel(channels)
        self.conv3 = _Conv(channels, out_channels, 1, 1)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.one, self.two], dim=1)
        local = self.conv123(self.conv12(self.conv1(x1)))
        semantic = self.conv2(x2)
        return self.conv3(self.spatial(semantic) * local + self.channel(local) * semantic)


class FBRTMKP(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, groups=channels)
        self.conv2 = _Conv(channels, channels, 1, 1)
        self.conv3 = nn.Conv2d(channels, channels, 5, 1, 2, groups=channels)
        self.conv4 = _Conv(channels, channels, 1, 1)
        self.conv5 = nn.Conv2d(channels, channels, 7, 1, 3, groups=channels)
        self.conv6 = _Conv(channels, channels, 1, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        return out + x
