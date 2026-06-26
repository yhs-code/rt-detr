import torch
import torch.nn as nn

__all__ = ("PinwheelShapedConv",)


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


class PinwheelShapedConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        if c2 % 4 != 0:
            raise ValueError("PinwheelShapedConv requires output channels divisible by 4")
        padding = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = nn.ModuleList(nn.ZeroPad2d(p) for p in padding)
        self.cw = _Conv(c1, c2 // 4, (1, k), s=s, p=0)
        self.ch = _Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        self.cat = _Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        yw0 = self.cw(self.pad[0](x))
        yw1 = self.cw(self.pad[1](x))
        yh0 = self.ch(self.pad[2](x))
        yh1 = self.ch(self.pad[3](x))
        return self.cat(torch.cat((yw0, yw1, yh0, yh1), dim=1))
