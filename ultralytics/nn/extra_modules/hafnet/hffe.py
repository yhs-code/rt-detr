import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("HFFENeckFusion",)


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class HardSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3.0) / 6.0


class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.gate = HardSigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.gate(x)


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CoordAttention(nn.Module):
    def __init__(self, channels, reduction=32):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = HardSwish()
        self.conv_h = nn.Conv2d(hidden, channels, 1, bias=True)
        self.conv_w = nn.Conv2d(hidden, channels, 1, bias=True)

    def forward(self, x):
        identity = x
        _, _, h, w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).transpose(2, 3)
        y = torch.cat((x_h, x_w), dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.transpose(2, 3)
        return identity * self.conv_h(x_h).sigmoid() * self.conv_w(x_w).sigmoid()


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7")
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.conv(torch.cat((avg_out, max_out), dim=1)).sigmoid()


class HFFENeckFusion(nn.Module):
    """Two-input HFFE neck fusion adapted for RT-DETR YAML list inputs."""

    def __init__(self, channels, out_channels=256, kernel_size=3, reduction=16):
        super().__init__()
        if not isinstance(channels, (list, tuple)) or len(channels) != 2:
            raise ValueError("HFFENeckFusion expects channels=[semantic_channels, lateral_channels].")
        semantic_channels, lateral_channels = int(channels[0]), int(channels[1])
        out_channels = int(out_channels)
        low_hidden = max(lateral_channels // reduction, 8)
        high_hidden = max(semantic_channels // reduction, 8)

        self.low_spatial = SpatialAttention()
        self.high_spatial = SpatialAttention()
        self.low_gate = nn.Sequential(
            ConvBNAct(lateral_channels, low_hidden, kernel_size),
            nn.Conv2d(low_hidden, 1, 1),
            nn.Sigmoid(),
        )
        self.high_gate = nn.Sequential(
            ConvBNAct(semantic_channels, high_hidden, kernel_size),
            nn.Conv2d(high_hidden, 1, 1),
            nn.Sigmoid(),
        )
        self.low_proj = ConvBNAct(lateral_channels, out_channels, 1)
        self.high_proj = ConvBNAct(semantic_channels, out_channels, 1)
        self.mix_proj = ConvBNAct(lateral_channels + semantic_channels, out_channels, 1)
        self.coord = CoordAttention(out_channels)
        self.final = ConvBNAct(out_channels * 2, out_channels, 1)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("HFFENeckFusion expects [semantic_feature, lateral_feature].")
        semantic, lateral = x
        if semantic.shape[-2:] != lateral.shape[-2:]:
            semantic = F.interpolate(semantic, size=lateral.shape[-2:], mode="nearest")

        lateral_source = lateral
        semantic_source = semantic
        lateral_att = self.low_spatial(lateral)
        semantic_att = self.high_spatial(semantic)
        lateral_map = self.low_gate(lateral_att)
        semantic_map = self.high_gate(semantic_att)
        mixed = torch.cat((lateral_source * semantic_map, semantic_source * lateral_map), dim=1)
        gate = torch.sigmoid(self.coord(self.mix_proj(mixed)))
        lateral_out = gate * self.low_proj(lateral_source + lateral_att)
        semantic_out = gate * self.high_proj(semantic_source + semantic_att)
        return self.final(torch.cat((lateral_out, semantic_out), dim=1))
