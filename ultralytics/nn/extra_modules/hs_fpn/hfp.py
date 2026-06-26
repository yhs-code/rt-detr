import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("HFP",)


def _valid_groups(channels, groups=32):
    groups = max(1, min(int(groups), int(channels)))
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class _SpatialInteraction(nn.Module):
    def __init__(self, in_channels, ratio=(0.25, 0.25), isdct=False, highpass_kernel=3):
        super().__init__()
        self.ratio = tuple(ratio)
        self.isdct = bool(isdct)
        self.highpass_kernel = int(highpass_kernel)
        self.spatial1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)

    def _high_frequency(self, x):
        kernel = max(3, self.highpass_kernel | 1)
        low = F.avg_pool2d(x, kernel_size=kernel, stride=1, padding=kernel // 2)
        return x - low

    def forward(self, x):
        if not self.isdct:
            return x * torch.sigmoid(self.spatial1x1(x))
        high = self._high_frequency(x)
        return x * torch.sigmoid(high.mean(dim=1, keepdim=True))


class _ChannelInteraction(nn.Module):
    def __init__(self, in_channels, patch=(8, 8), ratio=(0.25, 0.25), isdct=False, groups=32):
        super().__init__()
        self.patch = tuple(patch)
        self.ratio = tuple(ratio)
        self.isdct = bool(isdct)
        groups = _valid_groups(in_channels, groups)
        self.channel1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=groups, bias=False)
        self.channel2x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=groups, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def _high_frequency(self, x):
        low = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        return x - low

    def forward(self, x):
        source = self._high_frequency(x) if self.isdct else x
        if self.isdct:
            pooled_max = F.adaptive_max_pool2d(source, self.patch)
            pooled_avg = F.adaptive_avg_pool2d(source, self.patch)
            pooled_max = torch.sum(self.relu(pooled_max), dim=(2, 3), keepdim=True)
            pooled_avg = torch.sum(self.relu(pooled_avg), dim=(2, 3), keepdim=True)
        else:
            pooled_max = F.adaptive_max_pool2d(source, 1)
            pooled_avg = F.adaptive_avg_pool2d(source, 1)
        channel = self.channel1x1(self.relu(pooled_max)) + self.channel1x1(self.relu(pooled_avg))
        return x * torch.sigmoid(self.channel2x1(channel))


class HFP(nn.Module):
    def __init__(self, in_channels, ratio=(0.25, 0.25), patch=(8, 8), isdct=False, groups=32):
        super().__init__()
        self.spatial = _SpatialInteraction(in_channels, ratio=ratio, isdct=isdct)
        self.channel = _ChannelInteraction(in_channels, patch=patch, ratio=ratio, isdct=isdct, groups=groups)
        self.out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_valid_groups(in_channels, groups), in_channels),
        )

    def forward(self, x):
        return self.out(self.spatial(x) + self.channel(x))
