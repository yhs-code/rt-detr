import torch
import torch.nn as nn

__all__ = ("MRFFE",)


class _ConvBNAct(nn.Sequential):
    def __init__(self, c1, c2, kernel_size, padding=0, dilation=1):
        super().__init__(
            nn.Conv2d(c1, c2, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
        )


class MRFFE(nn.Module):
    def __init__(self, in_channels, out_channels=None, reduction=4):
        super().__init__()
        out_channels = int(out_channels) if out_channels is not None else int(in_channels)
        mid_channels = max(1, int(in_channels) // int(reduction))
        self.reduce = _ConvBNAct(in_channels, mid_channels, 1)
        self.branch1 = nn.Sequential(
            _ConvBNAct(mid_channels, mid_channels, (1, 3), padding=(0, 1)),
            _ConvBNAct(mid_channels, mid_channels, (3, 1), padding=(1, 0)),
            _ConvBNAct(mid_channels, mid_channels, 3, padding=1),
        )
        self.branch2 = nn.Sequential(
            _ConvBNAct(mid_channels, mid_channels, (3, 1), padding=(1, 0)),
            _ConvBNAct(mid_channels, mid_channels, (1, 3), padding=(0, 1)),
            _ConvBNAct(mid_channels, mid_channels, 3, padding=1),
        )
        self.branch3 = nn.Sequential(
            _ConvBNAct(mid_channels, mid_channels, 3, padding=2, dilation=2),
            _ConvBNAct(mid_channels, mid_channels, 3, padding=1),
            _ConvBNAct(mid_channels, mid_channels, 3, padding=1),
        )
        self.fuse = _ConvBNAct(mid_channels * 3, out_channels, 1)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        reduced = self.reduce(x)
        fused = torch.cat((self.branch1(reduced), self.branch2(reduced), self.branch3(reduced)), dim=1)
        return self.fuse(fused) + self.shortcut(x)
