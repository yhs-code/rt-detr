import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("SAFFMNeckFusion",)


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, act=True):
        super().__init__()
        if p is None:
            p = k // 2 if isinstance(k, int) else [v // 2 for v in k]
        self.conv = nn.Conv2d(c1, c2, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, x):
        avg_out = self.fc2(self.act(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.act(self.fc1(self.max_pool(x))))
        return torch.sigmoid(avg_out + max_out)


class SpatialCNNAttention(nn.Module):
    def __init__(self, channels, sk_size=3, kernel_size=3):
        super().__init__()
        padding = sk_size // 2
        self.context = nn.Sequential(
            nn.Conv2d(channels, channels, sk_size, padding=padding, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, sk_size, padding=padding, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        x = self.context(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return torch.sigmoid(self.spatial(torch.cat((avg_out, max_out), dim=1)))


class SAFFMNeckFusion(nn.Module):
    """Spatial-channel attention feature fusion adapted for RT-DETR neck inputs."""

    def __init__(self, channels, out_channels=256, sk_size=3, reduction=4):
        super().__init__()
        if not isinstance(channels, (list, tuple)) or len(channels) != 2:
            raise ValueError("SAFFMNeckFusion expects channels=[semantic_channels, lateral_channels].")
        semantic_channels, lateral_channels = int(channels[0]), int(channels[1])
        out_channels = int(out_channels)
        self.semantic_proj = ConvBNAct(semantic_channels, out_channels, 1)
        self.lateral_proj = ConvBNAct(lateral_channels, out_channels, 1)
        self.channel_att = ChannelAttention(out_channels, reduction=reduction)
        self.spatial_att = SpatialCNNAttention(out_channels, sk_size=sk_size)
        self.fuse = ConvBNAct(out_channels * 2, out_channels, 3)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("SAFFMNeckFusion expects [semantic_feature, lateral_feature].")
        semantic, lateral = x
        if semantic.shape[-2:] != lateral.shape[-2:]:
            semantic = F.interpolate(semantic, size=lateral.shape[-2:], mode="nearest")
        semantic = self.semantic_proj(semantic)
        lateral = self.lateral_proj(lateral)
        gate = self.channel_att(lateral) * self.spatial_att(semantic)
        return self.fuse(torch.cat((gate * lateral, gate * semantic), dim=1))
