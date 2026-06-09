import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ("DySample",)


class DySample(nn.Module):
    """Dynamic upsampling by learning sampling offsets.

    This is the lightweight DySample upsampler used as a drop-in replacement
    for nearest/bilinear upsampling. The default static range factor is 0.25.
    """

    def __init__(self, in_channels, scale=2, style="lp", groups=4, dyscope=False):
        super().__init__()
        if style not in {"lp", "pl"}:
            raise ValueError(f"DySample style must be 'lp' or 'pl', got {style!r}.")
        if in_channels % groups != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by groups ({groups}).")
        if style == "pl" and in_channels % (scale ** 2) != 0:
            raise ValueError("'pl' style requires in_channels divisible by scale ** 2.")

        self.scale = scale
        self.style = style
        self.groups = groups
        self.dyscope = dyscope

        offset_channels = 2 * groups * (scale ** 2) if style == "lp" else 2 * groups
        conv_channels = in_channels if style == "lp" else in_channels // (scale ** 2)
        self.offset = nn.Conv2d(conv_channels, offset_channels, 1)
        nn.init.normal_(self.offset.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.offset.bias, 0.0)

        if dyscope:
            self.scope = nn.Conv2d(conv_channels, offset_channels, 1, bias=False)
            nn.init.constant_(self.scope.weight, 0.0)

        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        base = torch.stack(torch.meshgrid(h, h, indexing="ij")).transpose(1, 2)
        return base.repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        b, _, h, w = offset.shape
        offset = offset.view(b, 2, -1, h, w)

        coords_h = torch.arange(h, dtype=x.dtype, device=x.device) + 0.5
        coords_w = torch.arange(w, dtype=x.dtype, device=x.device) + 0.5
        coords = torch.stack(torch.meshgrid(coords_w, coords_h, indexing="ij")).transpose(1, 2)
        coords = coords.unsqueeze(1).unsqueeze(0)
        normalizer = torch.tensor([w, h], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(b, -1, h, w), self.scale)
        coords = coords.view(b, 2, -1, self.scale * h, self.scale * w)
        coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)

        sampled = F.grid_sample(
            x.reshape(b * self.groups, -1, h, w),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        )
        return sampled.view(b, -1, self.scale * h, self.scale * w)

    def forward_lp(self, x):
        if self.dyscope:
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if self.dyscope:
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        return self.forward_lp(x) if self.style == "lp" else self.forward_pl(x)
