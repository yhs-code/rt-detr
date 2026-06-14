import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ("SODGAHighFrequencyPrior",)


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        if p is None:
            if isinstance(k, tuple):
                if isinstance(d, tuple):
                    p = tuple(((ki - 1) * di) // 2 for ki, di in zip(k, d))
                else:
                    p = tuple(((ki - 1) * d) // 2 for ki in k)
            else:
                p = ((k - 1) * d) // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SODGAHighFrequencyPrior(nn.Module):
    """Generate a P2 high-frequency prior calibrated by the SODGA semantic response."""

    def __init__(self, channels, hidden_channels=32, gate_channels=16, temperature=2.0):
        super().__init__()
        c_p2, c_seed = channels
        hidden_channels = max(int(hidden_channels), 8)
        gate_channels = max(int(gate_channels), 8)
        self.temperature = float(temperature)

        self.p2_proj = ConvBNAct(c_p2, hidden_channels, k=1)
        self.local_edge = ConvBNAct(hidden_channels, hidden_channels, k=3, g=hidden_channels)
        self.edge_mix = ConvBNAct(hidden_channels, hidden_channels, k=1)
        self.directional_edge = nn.ModuleList(
            (
                ConvBNAct(hidden_channels, hidden_channels, k=(1, 3), g=hidden_channels),
                ConvBNAct(hidden_channels, hidden_channels, k=(3, 1), g=hidden_channels),
            )
        )
        self.seed_gate = nn.Sequential(
            ConvBNAct(c_seed, gate_channels, k=1),
            ConvBNAct(gate_channels, gate_channels, k=3),
            nn.Conv2d(gate_channels, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.noise_gate = nn.Sequential(
            ConvBNAct(hidden_channels + 2, hidden_channels, k=3),
            nn.Conv2d(hidden_channels, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        self.prior = nn.Sequential(
            ConvBNAct(hidden_channels + 2, hidden_channels, k=3),
            nn.Conv2d(hidden_channels, 1, 1, bias=True),
        )
        self.threshold = nn.Parameter(torch.tensor(0.02))

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("SODGAHighFrequencyPrior expects [P2, SODGA_seed].")
        p2, seed = x
        p2 = self.p2_proj(p2)

        local_mean = F.avg_pool2d(p2, kernel_size=3, stride=1, padding=1)
        residual = (p2 - local_mean).abs()
        edge = self.edge_mix(self.local_edge(residual))
        edge = edge + self.directional_edge[0](residual) + self.directional_edge[1](residual)

        response = edge.mean(dim=1, keepdim=True)
        threshold = self.threshold.clamp_min(0.0)
        structured = torch.sigmoid((response - threshold) / self.temperature)
        local_energy = F.avg_pool2d(structured.abs(), kernel_size=3, stride=1, padding=1)
        structured = (structured / (local_energy + 1e-6)).clamp(0.0, 1.0)

        seed = self.seed_gate(seed)
        seed = F.interpolate(seed, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        noise_suppression = self.noise_gate(torch.cat((edge, structured, seed), dim=1))

        calibrated = edge * structured * seed * noise_suppression
        prior = self.prior(torch.cat((calibrated, structured, seed), dim=1))
        return torch.sigmoid(prior)
