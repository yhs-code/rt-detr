import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ("OrthogonalChannelAttention", "HaarWaveletConv")


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        if p is None:
            if isinstance(k, tuple):
                p = tuple(((ki - 1) * d) // 2 for ki in k)
            else:
                p = ((k - 1) * d) // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class OrthogonalChannelAttention(nn.Module):
    """H2Net OCA-style channel attention using fixed orthogonal spatial projections."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        dtype, device = x.dtype, x.device
        yy = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype).view(1, 1, h, 1)
        xx = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype).view(1, 1, 1, w)
        basis = (
            torch.ones(1, 1, h, w, device=device, dtype=dtype),
            yy.expand(1, 1, h, w),
            xx.expand(1, 1, h, w),
            (yy * xx).expand(1, 1, h, w),
        )
        desc = torch.cat([(x * p).mean(dim=(2, 3), keepdim=True) for p in basis], dim=1)
        return x * self.fc(desc)


def _pad_even(x):
    h, w = x.shape[-2:]
    pad_h, pad_w = h % 2, w % 2
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, h, w


def _dwt(x):
    x, h, w = _pad_even(x)
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (-a - b + c + d) * 0.5
    hl = (-a + b - c + d) * 0.5
    hh = (a - b - c + d) * 0.5
    return ll, lh, hl, hh, h, w


def _idwt(ll, lh, hl, hh, h, w):
    out = torch.empty(*ll.shape[:-2], ll.shape[-2] * 2, ll.shape[-1] * 2, device=ll.device, dtype=ll.dtype)
    out[..., 0::2, 0::2] = (ll - lh - hl + hh) * 0.5
    out[..., 0::2, 1::2] = (ll - lh + hl - hh) * 0.5
    out[..., 1::2, 0::2] = (ll + lh - hl - hh) * 0.5
    out[..., 1::2, 1::2] = (ll + lh + hl + hh) * 0.5
    return out[..., :h, :w]


class HaarWaveletConv(nn.Module):
    """H2Net HWConv-style two-level Haar wavelet feature fusion."""

    def __init__(self, channels, levels=2):
        super().__init__()
        self.levels = int(levels)
        self.low_fuse = nn.Sequential(
            ConvBNAct(channels * 4, channels * 4, k=3, g=channels * 4),
            ConvBNAct(channels * 4, channels * 4, k=1),
        )
        self.high_fuse = nn.Sequential(
            ConvBNAct(channels * 4, channels * 4, k=3, g=channels * 4),
            ConvBNAct(channels * 4, channels * 4, k=1),
        )
        self.out = ConvBNAct(channels, channels, k=1)

    def forward(self, x):
        ll1, lh1, hl1, hh1, h1, w1 = _dwt(x)
        if self.levels > 1:
            ll2, lh2, hl2, hh2, h2, w2 = _dwt(ll1)
            low = self.low_fuse(torch.cat((ll2, lh2, hl2, hh2), dim=1))
            ll2, lh2, hl2, hh2 = low.chunk(4, dim=1)
            ll1 = _idwt(ll2, lh2, hl2, hh2, h2, w2)
        high = self.high_fuse(torch.cat((ll1, lh1, hl1, hh1), dim=1))
        ll1, lh1, hl1, hh1 = high.chunk(4, dim=1)
        return self.out(_idwt(ll1, lh1, hl1, hh1, h1, w1))
