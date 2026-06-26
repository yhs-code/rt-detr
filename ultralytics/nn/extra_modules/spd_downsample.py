import torch.nn as nn
import torch.nn.functional as F


__all__ = ("SPDDownsample",)


class SPDDownsample(nn.Module):
    """Space-to-depth downsampling followed by learned channel compression."""

    def __init__(self, c1, c2, kernel_size=3, act=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(c1 * 4, c2, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        height_pad = x.shape[-2] % 2
        width_pad = x.shape[-1] % 2
        if height_pad or width_pad:
            x = F.pad(x, (0, width_pad, 0, height_pad))
        x = F.pixel_unshuffle(x, 2)
        return self.act(self.bn(self.conv(x)))
