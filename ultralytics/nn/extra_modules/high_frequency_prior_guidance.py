import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ("HighFrequencyPrior", "HighFrequencyPriorGuidance")


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class HaarDWT(nn.Module):
    """Fixed Haar DWT that returns LH, HL and HH high-frequency bands."""

    def __init__(self):
        super().__init__()
        ll = torch.tensor([[1.0, 1.0], [1.0, 1.0]]) * 0.5
        lh = torch.tensor([[-1.0, -1.0], [1.0, 1.0]]) * 0.5
        hl = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]) * 0.5
        hh = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]) * 0.5
        weight = torch.stack((lh, hl, hh), dim=0).unsqueeze(1)
        self.register_buffer("weight", weight)

    def forward(self, x):
        c = x.shape[1]
        weight = self.weight.to(dtype=x.dtype).repeat(c, 1, 1, 1)
        y = F.conv2d(x, weight, stride=2, groups=c)
        b, _, h, w = y.shape
        y = y.view(b, c, 3, h, w)
        return y[:, :, 0], y[:, :, 1], y[:, :, 2]


class CoordinateAttention(nn.Module):
    def __init__(self, channels, reduction=32):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(hidden, channels, 1, bias=True)
        self.conv_w = nn.Conv2d(hidden, channels, 1, bias=True)

    def forward(self, x):
        b, c, h, w = x.shape
        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, w)).transpose(2, 3)
        y = torch.cat((x_h, x_w), dim=2)
        y = self.act(self.bn1(self.conv1(y)))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        y_w = y_w.transpose(2, 3)
        a_h = self.conv_h(y_h).sigmoid()
        a_w = self.conv_w(y_w).sigmoid()
        return x * a_h * a_w


class HighFrequencyPrior(nn.Module):
    """Generate a high-frequency prior map from P2 without fusing P2 features."""

    def __init__(self, c1, hidden_channels=None, ca_reduction=32):
        super().__init__()
        hidden_channels = c1 if hidden_channels is None else hidden_channels
        self.dwt = HaarDWT()
        self.lh_conv = ConvBNAct(c1, hidden_channels, k=3)
        self.hl_conv = ConvBNAct(c1, hidden_channels, k=3)
        self.hh_conv = ConvBNAct(c1, hidden_channels, k=3)
        self.fuse = ConvBNAct(hidden_channels * 3, hidden_channels, k=1)
        self.coord_att = CoordinateAttention(hidden_channels, reduction=ca_reduction)
        self.prior = nn.Conv2d(hidden_channels, 1, 1, bias=True)

    def forward(self, x):
        lh, hl, hh = self.dwt(x)
        hf = torch.cat((self.lh_conv(lh), self.hl_conv(hl), self.hh_conv(hh)), dim=1)
        hf = self.coord_att(self.fuse(hf))
        prior = self.prior(hf).sigmoid()
        return F.interpolate(prior, size=x.shape[-2:], mode="bilinear", align_corners=False)


class HighFrequencyPriorGuidance(nn.Module):
    """Residually guide one feature map with a high-frequency prior map."""

    def __init__(self, init_alpha=0.1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError("HighFrequencyPriorGuidance expects [prior, feature].")
        prior, feat = x
        prior = F.interpolate(prior, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        return feat + self.alpha * feat * prior
