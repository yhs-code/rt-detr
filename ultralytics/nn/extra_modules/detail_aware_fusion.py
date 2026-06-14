import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ("DetailAwareFusionV2",)


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


class DetailAwareFusionV2(nn.Module):
    """Fuse semantic and lateral features under a propagated SODGA detail prior."""

    def __init__(self, channels, out_channels=256, hidden_channels=None, reduction=4):
        super().__init__()
        c_sem, c_lat, c_prior = channels
        out_channels = int(out_channels)
        hidden_channels = out_channels if hidden_channels is None else int(hidden_channels)
        gate_channels = max(out_channels // int(reduction), 16)

        self.sem_proj = ConvBNAct(c_sem, out_channels, k=1)
        self.lat_proj = ConvBNAct(c_lat, out_channels, k=1)
        self.prior_proj = nn.Sequential(
            ConvBNAct(c_prior, gate_channels, k=3),
            nn.Conv2d(gate_channels, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.balance = nn.Sequential(
            ConvBNAct(out_channels * 2 + 1, gate_channels, k=1),
            nn.Conv2d(gate_channels, 2, 1, bias=True),
        )
        self.detail_refine = nn.Sequential(
            ConvBNAct(out_channels, hidden_channels, k=3),
            ConvBNAct(hidden_channels, out_channels, k=1),
        )
        self.semantic_filter = nn.Sequential(
            ConvBNAct(out_channels, gate_channels, k=1),
            nn.Conv2d(gate_channels, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise TypeError("DetailAwareFusionV2 expects [semantic, lateral, propagated_prior].")
        sem, lat, prior = x
        sem = F.interpolate(self.sem_proj(sem), size=lat.shape[-2:], mode="nearest")
        lat = self.lat_proj(lat)
        prior = F.interpolate(prior, size=lat.shape[-2:], mode="bilinear", align_corners=False)
        prior = self.prior_proj(prior)

        weights = torch.softmax(self.balance(torch.cat((sem, lat, prior), dim=1)), dim=1)
        sem_weight = weights[:, 0:1]
        lat_weight = weights[:, 1:2]
        sem_filter = self.semantic_filter(sem)
        guided_lat = lat * (1.0 + prior * sem_filter)
        fused = sem_weight * sem + lat_weight * guided_lat
        return self.detail_refine(fused)
