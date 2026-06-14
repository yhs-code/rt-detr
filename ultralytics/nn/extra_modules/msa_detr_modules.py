import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ("PercepConv", "MSASODAttention")


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


class PercepConv(nn.Module):
    """MSA-DETR PercepConv with parallel multi-scale and dilated branches."""

    def __init__(self, c1, c2, s=1, reduction=8):
        super().__init__()
        branch = max(c2 // 4, 8)
        self.b1 = ConvBNAct(c1, branch, k=1, s=s)
        self.b3 = ConvBNAct(c1, branch, k=3, s=s)
        self.b5 = ConvBNAct(c1, branch, k=5, s=s)
        self.bd = ConvBNAct(c1, branch, k=3, s=s, d=2)
        self.fuse = ConvBNAct(branch * 4, c2, k=1)
        hidden = max(c2 // reduction, 8)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, c2, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.fuse(torch.cat((self.b1(x), self.b3(x), self.b5(x), self.bd(x)), dim=1))
        return y * self.ca(y)


class MSASODAttention(nn.Module):
    """MSA-DETR SODAttention with global-local mixing and context pooling."""

    def __init__(self, channels, groups=8, reduction=8):
        super().__init__()
        groups = max(1, min(int(groups), channels))
        while channels % groups != 0 and groups > 1:
            groups -= 1
        hidden = max(channels // reduction, 8)
        self.norm = nn.GroupNorm(groups, channels)
        self.local = ConvBNAct(channels, channels, k=3, g=channels)
        self.context_key = nn.Conv2d(channels, 1, 1, bias=True)
        self.context_value = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        self.fuse = ConvBNAct(channels, channels, k=1)

    def forward(self, x):
        local = self.local(self.norm(x))
        b, c, h, w = x.shape
        key = self.context_key(x).view(b, 1, -1)
        attn = torch.softmax(key, dim=-1)
        value = x.view(b, c, -1)
        context = torch.bmm(value, attn.transpose(1, 2)).view(b, c, 1, 1)
        context = self.context_value(context)
        spatial = torch.softmax(F.max_pool2d(local.mean(1, keepdim=True), 3, 1, 1).view(b, 1, -1), dim=-1)
        spatial = spatial.view(b, 1, h, w)
        return self.fuse(local * spatial + context)
