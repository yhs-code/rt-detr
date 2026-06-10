import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ("SODCalibratedHighFrequencyPrior", "DetailAwareFusion")


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


class SODCalibratedHighFrequencyPrior(nn.Module):
    """Generate a compact P2 high-frequency prior calibrated by SODGA-enhanced semantics."""

    def __init__(self, channels, hidden_channels=32, gate_channels=16, temperature=2.0):
        super().__init__()
        c_p2, c_seed = channels
        hidden_channels = max(int(hidden_channels), 8)
        gate_channels = max(int(gate_channels), 8)
        self.temperature = float(temperature)

        self.p2_proj = ConvBNAct(c_p2, hidden_channels, k=1)
        self.local_hf = ConvBNAct(hidden_channels, hidden_channels, k=3, g=hidden_channels)
        self.hf_mix = ConvBNAct(hidden_channels, hidden_channels, k=1)

        self.seed_proj = nn.Sequential(
            ConvBNAct(c_seed, gate_channels, k=1),
            ConvBNAct(gate_channels, gate_channels, k=3),
            nn.Conv2d(gate_channels, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.threshold = nn.Parameter(torch.tensor(0.02))
        self.prior = nn.Sequential(
            ConvBNAct(hidden_channels + 2, hidden_channels, k=3),
            nn.Conv2d(hidden_channels, 1, 1, bias=True),
        )

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("SODCalibratedHighFrequencyPrior expects [P2, SODGA_seed].")
        p2, seed = x
        p2 = self.p2_proj(p2)

        local_mean = F.avg_pool2d(p2, kernel_size=3, stride=1, padding=1)
        residual = p2 - local_mean
        hf = self.hf_mix(self.local_hf(residual.abs()))

        response = hf.mean(dim=1, keepdim=True)
        threshold = self.threshold.clamp_min(0.0)
        stat_gate = torch.sigmoid((response - threshold) / self.temperature)
        norm_gate = stat_gate / (F.avg_pool2d(stat_gate.abs(), kernel_size=3, stride=1, padding=1) + 1e-6)
        norm_gate = norm_gate.clamp(0.0, 1.0)

        seed_gate = self.seed_proj(seed)
        seed_gate = F.interpolate(seed_gate, size=p2.shape[-2:], mode="bilinear", align_corners=False)

        calibrated = hf * norm_gate * seed_gate
        return torch.sigmoid(self.prior(torch.cat((calibrated, norm_gate, seed_gate), dim=1)))


class DetailAwareFusion(nn.Module):
    """Fuse top-down semantic and lateral features with a propagated detail prior."""

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
        self.weight_net = nn.Sequential(
            ConvBNAct(out_channels * 2 + 1, gate_channels, k=1),
            nn.Conv2d(gate_channels, 2, 1, bias=True),
        )
        self.refine = nn.Sequential(
            ConvBNAct(out_channels, hidden_channels, k=3),
            ConvBNAct(hidden_channels, out_channels, k=1),
        )

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise TypeError("DetailAwareFusion expects [semantic, lateral, detail_prior].")
        sem, lat, prior = x
        sem = F.interpolate(self.sem_proj(sem), size=lat.shape[-2:], mode="nearest")
        lat = self.lat_proj(lat)
        prior = F.interpolate(prior, size=lat.shape[-2:], mode="bilinear", align_corners=False)
        prior = self.prior_proj(prior)

        weights = torch.softmax(self.weight_net(torch.cat((sem, lat, prior), dim=1)), dim=1)
        fused = weights[:, 0:1] * sem + weights[:, 1:2] * (lat * (1.0 + prior))
        return self.refine(fused)
