import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ("MSPConv", "SODGuidedAttention", "P2DetailPrior", "DetailGuidance")


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        if p is None:
            p = ((k - 1) * d) // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.gate(x)


class CoordinateGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.conv1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(hidden, channels, 1, bias=True)
        self.conv_w = nn.Conv2d(hidden, channels, 1, bias=True)

    def forward(self, x):
        _, _, h, w = x.shape
        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, w)).transpose(2, 3)
        y = torch.cat((x_h, x_w), dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.transpose(2, 3)
        return self.conv_h(y_h).sigmoid() * self.conv_w(y_w).sigmoid()


class MSPConv(nn.Module):
    """Multi-scale prior perception convolution for shallow small-object details."""

    def __init__(self, c1, c2, branch_channels=None, dilation=2, reduction=8, init_alpha=0.1):
        super().__init__()
        branch_channels = max(c2 // 4, 8) if branch_channels is None else branch_channels
        self.b1 = ConvBNAct(c1, branch_channels, k=1)
        self.b3 = ConvBNAct(c1, branch_channels, k=3)
        self.bd = ConvBNAct(c1, branch_channels, k=3, d=dilation)
        self.bdw = nn.Sequential(
            ConvBNAct(c1, c1, k=3, g=c1),
            ConvBNAct(c1, branch_channels, k=1),
        )
        self.fuse = ConvBNAct(branch_channels * 4, c2, k=1)
        self.gate = ChannelGate(c2, reduction=reduction)
        self.shortcut = nn.Identity() if c1 == c2 else ConvBNAct(c1, c2, k=1, act=False)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        y = torch.cat((self.b1(x), self.b3(x), self.bd(x), self.bdw(x)), dim=1)
        y = self.fuse(y)
        return self.shortcut(x) + self.alpha * y * self.gate(y)


class SODGuidedAttention(nn.Module):
    """Small-object detail-guided attention with local, strip and global context gates."""

    def __init__(self, channels, reduction=16, init_alpha=0.1):
        super().__init__()
        self.local_detail = nn.Sequential(
            ConvBNAct(channels, channels, k=3, g=channels),
            ConvBNAct(channels, channels, k=1),
        )
        self.coord_gate = CoordinateGate(channels, reduction=reduction)
        hidden = max(channels // reduction, 8)
        self.context_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=True),
            nn.Sigmoid(),
        )
        self.proj = ConvBNAct(channels, channels, k=1)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        detail = self.local_detail(x)
        spatial = self.spatial_gate(torch.cat((detail.mean(1, keepdim=True), detail.amax(1, keepdim=True)), dim=1))
        gated = detail * self.coord_gate(detail) * self.context_gate(x) * spatial
        return x + self.alpha * self.proj(gated)


class P2DetailPrior(nn.Module):
    """Generate a compact small-object detail prior from the high-resolution P2 feature."""

    def __init__(self, c1, hidden_channels=None, reduction=16):
        super().__init__()
        hidden_channels = c1 if hidden_channels is None else hidden_channels
        self.local = ConvBNAct(c1, hidden_channels, k=3)
        self.dilated = ConvBNAct(c1, hidden_channels, k=3, d=2)
        self.context = nn.Sequential(
            ConvBNAct(c1, c1, k=5, g=c1),
            ConvBNAct(c1, hidden_channels, k=1),
        )
        self.fuse = ConvBNAct(hidden_channels * 3, hidden_channels, k=1)
        self.attn = SODGuidedAttention(hidden_channels, reduction=reduction)
        self.prior = nn.Conv2d(hidden_channels, 1, 1, bias=True)

    def forward(self, x):
        detail = torch.cat((self.local(x), self.dilated(x), self.context(x)), dim=1)
        detail = self.attn(self.fuse(detail))
        return self.prior(detail).sigmoid()


class DetailGuidance(nn.Module):
    """Residually guide a feature map with a small-object detail prior."""

    def __init__(self, init_alpha=0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("DetailGuidance expects [prior, feature].")
        prior, feat = x
        prior = F.interpolate(prior, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        return feat + self.alpha * feat * prior
