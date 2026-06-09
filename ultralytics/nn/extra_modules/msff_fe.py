import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ("MSFFFE", "FrequencyFocused")


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act="gelu"):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        if act == "gelu":
            self.act = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act is None:
            self.act = nn.Identity()
        else:
            self.act = act

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attn(x)


class FrequencyFocused(nn.Module):
    """Frequency-focused feature refinement from UAV-DETR."""

    def __init__(self, channels):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        dtype = x.dtype
        spatial = self.proj(x).float()
        freq = torch.fft.rfft2(spatial, norm="ortho")
        gate = self.gate(spatial).to(dtype=freq.dtype)
        enhanced = torch.fft.irfft2(freq * gate, s=spatial.shape[-2:], norm="ortho")
        return (self.alpha * enhanced + self.beta * x.float()).to(dtype)


class MSFFFE(nn.Module):
    """Multi-scale feature fusion with frequency enhancement.

    The first input is treated as S2/P2 and compressed with Focus-style
    slicing before concatenation with the remaining same-scale features.
    """

    def __init__(self, ch, c2, focus_channels=None, reduction=8):
        super().__init__()
        if isinstance(ch, int):
            ch = [ch]
        assert len(ch) >= 2, "MSFFFE expects S2 plus at least one same-scale feature."

        focus_channels = ch[0] if focus_channels is None else focus_channels
        self.focus = ConvBNAct(ch[0] * 4, focus_channels, k=1, act="gelu")

        c_in = focus_channels + sum(ch[1:])
        c_branch = max(c_in // reduction, 1)
        c_residual = c_in - c_branch
        self.c_branch = c_branch
        self.c_residual = c_residual

        self.pre = ConvBNAct(c_branch, c_branch, k=1, act="gelu")
        self.freq_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_branch, c_branch, 1, bias=True),
        )

        self.ms1 = ConvBNAct(c_branch, c_branch, k=1, act="gelu")
        self.ms3 = ConvBNAct(c_branch, c_branch, k=3, act="gelu")
        self.ms5 = ConvBNAct(c_branch, c_branch, k=5, act="gelu")
        self.channel_attn = ChannelAttention(c_branch)
        self.ff = FrequencyFocused(c_branch)

        self.large = nn.Sequential(
            ConvBNAct(c_branch, c_branch, k=17, g=c_branch, act="gelu"),
            ConvBNAct(c_branch, c_branch, k=1, act="gelu"),
        )
        self.small = ConvBNAct(c_branch, c_branch, k=1, act="gelu")
        self.fuse = ConvBNAct(c_in, c2, k=1, act="gelu")

    @staticmethod
    def _focus_slice(x):
        return torch.cat(
            (x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]),
            dim=1,
        )

    @staticmethod
    def _match_size(x, size):
        if x.shape[-2:] != size:
            x = F.interpolate(x, size=size, mode="nearest")
        return x

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            raise TypeError("MSFFFE forward expects a list of feature maps.")

        target_size = x[1].shape[-2:]
        s2 = self._match_size(self.focus(self._focus_slice(x[0])), target_size)
        feats = [s2] + [self._match_size(feat, target_size) for feat in x[1:]]
        fused = torch.cat(feats, dim=1)

        x1, x2 = torch.split(fused, [self.c_branch, self.c_residual], dim=1)
        xconv = self.pre(x1)

        freq_input = xconv.float()
        freq = torch.fft.rfft2(freq_input, norm="ortho")
        freq_gate = self.freq_gate(freq_input).to(dtype=freq.dtype)
        xsp = torch.abs(torch.fft.irfft2(freq * freq_gate, s=xconv.shape[-2:], norm="ortho")).to(xconv.dtype)

        xsc = self.ms1(xsp) + self.ms3(xsp) + self.ms5(xsp)
        xsc = self.channel_attn(xsc)
        xf = self.ff(xsc)
        xfinal = x1 + self.large(xconv) + self.small(xconv) + xf

        return self.fuse(torch.cat((xfinal, x2), dim=1))
