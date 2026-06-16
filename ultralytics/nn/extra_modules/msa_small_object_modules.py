import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = (
    "MSPConv",
    "SODGuidedAttention",
    "SODGuidedAttentionStable",
    "SODGuidedAttentionGNLogit",
    "SODGuidedAttentionCalib",
    "P2DetailPrior",
    "DetailGuidance",
    "EncoderFuzzyRefine",
    "CAFM",
)


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


class ConvGNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, gn_groups=16):
        super().__init__()
        if p is None:
            p = ((k - 1) * d) // 2
        groups = min(gn_groups, c2)
        while c2 % groups != 0 and groups > 1:
            groups -= 1
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.norm = nn.GroupNorm(groups, c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class SODGuidedAttentionStable(nn.Module):
    """Stable SODGA variant with batch-independent normalization and bounded positive gain."""

    def __init__(self, channels, reduction=16, init_gain=0.1, max_gain=0.3, gn_groups=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.local_detail = nn.Sequential(
            ConvGNAct(channels, channels, k=3, g=channels, gn_groups=gn_groups),
            ConvGNAct(channels, channels, k=1, gn_groups=gn_groups),
        )
        self.coord_pool = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.SiLU(inplace=True),
        )
        self.coord_h = nn.Conv2d(hidden, channels, 1, bias=True)
        self.coord_w = nn.Conv2d(hidden, channels, 1, bias=True)
        self.context_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        self.spatial_gate = nn.Conv2d(2, 1, 7, padding=3, bias=True)
        self.proj = ConvGNAct(channels, channels, k=1, gn_groups=gn_groups)
        self.max_gain = float(max_gain)
        init_ratio = min(max(float(init_gain) / self.max_gain, 1e-4), 1 - 1e-4)
        self.gain_logit = nn.Parameter(torch.tensor(torch.logit(torch.tensor(init_ratio)).item()))

    def forward(self, x):
        detail = self.local_detail(x)
        _, _, h, w = detail.shape

        x_h = F.adaptive_avg_pool2d(detail, (h, 1))
        x_w = F.adaptive_avg_pool2d(detail, (1, w)).transpose(2, 3)
        coord = self.coord_pool(torch.cat((x_h, x_w), dim=2))
        coord_h, coord_w = torch.split(coord, [h, w], dim=2)
        coord_logit = self.coord_h(coord_h) + self.coord_w(coord_w.transpose(2, 3))

        spatial_input = torch.cat((detail.mean(1, keepdim=True), detail.amax(1, keepdim=True)), dim=1)
        gate = torch.sigmoid((coord_logit + self.context_gate(x) + self.spatial_gate(spatial_input)) / 3.0)
        gain = self.max_gain * torch.sigmoid(self.gain_logit)
        return x + gain * self.proj(detail * gate)


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


class CAFM(nn.Module):
    """Residual convolution-attention fusion for local detail and global context."""

    def __init__(self, channels, groups=4, reduction=4, init_alpha=0.1):
        super().__init__()
        if channels % groups != 0:
            raise ValueError(f"channels ({channels}) must be divisible by groups ({groups}).")

        self.groups = groups
        hidden = max(channels // reduction, 16)
        self.local_pre = ConvBNAct(channels, channels, k=1)
        self.local_group = ConvBNAct(channels, channels, k=3, g=groups)
        self.local_spatial = ConvBNAct(channels, channels, k=3, g=channels)

        self.q = nn.Sequential(ConvBNAct(channels, hidden, k=1), ConvBNAct(hidden, hidden, k=3, g=hidden))
        self.k = nn.Sequential(ConvBNAct(channels, hidden, k=1), ConvBNAct(hidden, hidden, k=3, g=hidden))
        self.v = nn.Sequential(ConvBNAct(channels, channels, k=1), ConvBNAct(channels, channels, k=3, g=channels))
        self.temperature = nn.Parameter(torch.ones(1))

        self.attn_proj = ConvBNAct(channels, channels, k=1)
        self.fuse = ConvBNAct(channels * 2, channels, k=1)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, c // groups, h, w)
        x = x.transpose(1, 2).contiguous()
        return x.reshape(b, c, h, w)

    def forward(self, x):
        local = self.local_pre(x)
        local = self.channel_shuffle(self.local_group(local), self.groups)
        local = self.local_spatial(local)

        b, _, h, w = x.shape
        q = self.q(x).flatten(2)
        k = self.k(x).flatten(2)
        v = self.v(x).flatten(2)
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) / self.temperature.clamp_min(1e-4), dim=-1)
        global_feat = torch.bmm(v, attn.transpose(1, 2)).reshape(b, -1, h, w)
        global_feat = self.attn_proj(global_feat)

        fused = self.fuse(torch.cat((local, global_feat), dim=1))
        return x + self.alpha * fused


class EncoderFuzzyRefine(nn.Module):
    """Refine AIFI encoder features with multi-scale convolution and fuzzy gates."""

    def __init__(self, channels, hidden_channels=None, reduction=8, init_alpha=0.1):
        super().__init__()
        hidden_channels = channels if hidden_channels is None else hidden_channels
        branch_channels = max(hidden_channels // 4, 8)

        self.ms1 = ConvBNAct(channels, branch_channels, k=1)
        self.ms3 = ConvBNAct(channels, branch_channels, k=3)
        self.ms5 = ConvBNAct(channels, branch_channels, k=5)
        self.ms7 = ConvBNAct(channels, branch_channels, k=7, g=1)
        self.fuse = ConvBNAct(branch_channels * 4, hidden_channels, k=1)

        gate_hidden = max(hidden_channels // reduction, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_channels, gate_hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_hidden, hidden_channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.position_gate = CoordinateGate(hidden_channels, reduction=reduction)
        self.uncertainty_gate = nn.Sequential(
            nn.Conv2d(2, 1, 3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.out = ConvBNAct(hidden_channels, channels, k=1)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

    def forward(self, x):
        ms = torch.cat((self.ms1(x), self.ms3(x), self.ms5(x), self.ms7(x)), dim=1)
        feat = self.fuse(ms)
        mean = feat.mean(dim=1, keepdim=True)
        deviation = (feat - mean).abs().mean(dim=1, keepdim=True)
        fuzzy = self.uncertainty_gate(torch.cat((mean, deviation), dim=1))
        feat = feat * self.channel_gate(feat) * self.position_gate(feat) * fuzzy
        return x + self.alpha * self.out(feat)

class SODGuidedAttentionGNLogit(nn.Module):
    """GN + logit-fused SODGA without learnable alpha.

    This variant keeps the SODGA local detail, coordinate, context and spatial
    guidance branches, replaces multiplicative sigmoid gates with averaged
    logits, uses GroupNorm instead of BatchNorm inside the module, and removes
    the learnable alpha parameter.
    """

    def __init__(self, channels, reduction=16, gn_groups=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.local_detail = nn.Sequential(
            ConvGNAct(channels, channels, k=3, g=channels, gn_groups=gn_groups),
            ConvGNAct(channels, channels, k=1, gn_groups=gn_groups),
        )
        self.coord_pool = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.SiLU(inplace=True),
        )
        self.coord_h = nn.Conv2d(hidden, channels, 1, bias=True)
        self.coord_w = nn.Conv2d(hidden, channels, 1, bias=True)
        self.context_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        self.spatial_gate = nn.Conv2d(2, 1, 7, padding=3, bias=True)
        self.proj = ConvGNAct(channels, channels, k=1, gn_groups=gn_groups)
        self._init_identity_projection()

    def _init_identity_projection(self):
        nn.init.zeros_(self.proj.conv.weight)

    def forward(self, x):
        detail = self.local_detail(x)
        _, _, h, w = detail.shape

        x_h = F.adaptive_avg_pool2d(detail, (h, 1))
        x_w = F.adaptive_avg_pool2d(detail, (1, w)).transpose(2, 3)
        coord = self.coord_pool(torch.cat((x_h, x_w), dim=2))
        coord_h, coord_w = torch.split(coord, [h, w], dim=2)
        coord_logit = self.coord_h(coord_h) + self.coord_w(coord_w.transpose(2, 3))

        spatial_input = torch.cat((detail.mean(1, keepdim=True), detail.amax(1, keepdim=True)), dim=1)
        gate = torch.sigmoid((coord_logit + self.context_gate(x) + self.spatial_gate(spatial_input)) / 3.0)
        gated = detail * gate
        return x + self.proj(gated)

class SODGuidedAttentionCalib(nn.Module):
    """Calibration-only SODGA without residual addition or learnable alpha.

    The module keeps SODGA's local detail, coordinate, context and spatial
    guidance ideas. The three guidance branches produce logits, which are
    averaged before a sigmoid. The final feature is calibrated as x * gate, so
    no residual addition and no global alpha parameter are introduced.
    """

    def __init__(self, channels, reduction=16, gn_groups=16, init_std=1e-3):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.local_detail = nn.Sequential(
            ConvGNAct(channels, channels, k=3, g=channels, gn_groups=gn_groups),
            ConvGNAct(channels, channels, k=1, gn_groups=gn_groups),
        )
        self.coord_pool = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.SiLU(inplace=True),
        )
        self.coord_h = nn.Conv2d(hidden, channels, 1, bias=True)
        self.coord_w = nn.Conv2d(hidden, channels, 1, bias=True)
        self.context_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        self.spatial_gate = nn.Conv2d(2, 1, 7, padding=3, bias=True)
        self._init_near_identity(float(init_std))

    def _init_near_identity(self, init_std):
        nn.init.normal_(self.coord_h.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.coord_h.bias)
        nn.init.normal_(self.coord_w.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.coord_w.bias)
        nn.init.normal_(self.context_gate[-1].weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.context_gate[-1].bias)
        nn.init.normal_(self.spatial_gate.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.spatial_gate.bias)

    def forward(self, x):
        detail = self.local_detail(x)
        _, _, h, w = detail.shape

        x_h = F.adaptive_avg_pool2d(detail, (h, 1))
        x_w = F.adaptive_avg_pool2d(detail, (1, w)).transpose(2, 3)
        coord = self.coord_pool(torch.cat((x_h, x_w), dim=2))
        coord_h, coord_w = torch.split(coord, [h, w], dim=2)
        coord_logit = self.coord_h(coord_h) + self.coord_w(coord_w.transpose(2, 3))

        spatial_input = torch.cat((detail.mean(1, keepdim=True), detail.amax(1, keepdim=True)), dim=1)
        gate_logit = (coord_logit + self.context_gate(x) + self.spatial_gate(spatial_input)) / 3.0
        gate = 2.0 * torch.sigmoid(gate_logit)
        return x * gate

