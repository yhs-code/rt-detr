import torch.nn as nn
import torch.nn.functional as F


__all__ = ("DetailPriorPropagation", "DetailGuidedEnhance")


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


class DetailPriorPropagation(nn.Module):
    """Propagate a compact detail prior from P2 to a target feature scale."""

    def __init__(self, c1, hidden_channels=16):
        super().__init__()
        hidden_channels = max(int(hidden_channels), 8)
        self.proj = nn.Sequential(
            ConvBNAct(c1, hidden_channels, k=3),
            ConvBNAct(hidden_channels, hidden_channels, k=3, d=2),
            nn.Conv2d(hidden_channels, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("DetailPriorPropagation expects [prior, target_feature].")
        prior, target = x
        prior = F.interpolate(prior, size=target.shape[-2:], mode="bilinear", align_corners=False)
        return self.proj(prior)


class DetailGuidedEnhance(nn.Module):
    """Enhance feature responses directly with a propagated detail prior."""

    def __init__(self, gain=1.0):
        super().__init__()
        self.gain = float(gain)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("DetailGuidedEnhance expects [feature, propagated_prior].")
        feat, prior = x
        prior = F.interpolate(prior, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        return feat * (1.0 + self.gain * prior)
