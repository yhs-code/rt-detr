import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ("LGCDFSODAttention",)


def _valid_groups(channels, groups):
    groups = max(1, min(int(groups), channels))
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class GroupedContextCalibration(nn.Module):
    """Global context calibration branch for grouped SOD attention."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // int(reduction), 1)
        self.conv_mask = nn.Conv2d(channels, 1, kernel_size=1)
        self.channel_add = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.LayerNorm([hidden, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        mask = self.conv_mask(x).view(b, 1, h * w)
        mask = F.softmax(mask, dim=2).unsqueeze(-1)
        value = x.view(b, c, h * w).unsqueeze(1)
        context = torch.matmul(value, mask).view(b, c, 1, 1)
        return x + self.channel_add(context)


class LGCDFSODAttention(nn.Module):
    """Local-Global Contrast Dynamic Fusion SOD attention.

    This module keeps the proven grouped GLM and global context paths of
    MSAOriginalSODAttention, adds a local-contrast detail extractor to the GLM
    path, and replaces direct branch addition with dynamic adaptive fusion.
    """

    def __init__(self, channels, groups=8, reduction=4, proj_kernel=1):
        super().__init__()
        self.channels = channels
        self.groups = _valid_groups(channels, groups)
        self.group_channels = channels // self.groups

        self.glm_coord = nn.Conv2d(self.group_channels, self.group_channels, kernel_size=1)
        self.glm_norm = nn.GroupNorm(1, self.group_channels)
        self.glm_maxpool = nn.AdaptiveMaxPool2d(1)

        self.local_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.local_detail = nn.Sequential(
            nn.Conv2d(
                self.group_channels,
                self.group_channels,
                kernel_size=3,
                padding=1,
                groups=self.group_channels,
                bias=False,
            ),
            nn.GroupNorm(1, self.group_channels),
            nn.ReLU(inplace=True),
        )
        self.lce_fuse = nn.Conv2d(self.group_channels * 2, self.group_channels, kernel_size=1)

        self.context_block = GroupedContextCalibration(self.group_channels, reduction=reduction)
        self.fusion_gate = nn.Conv2d(self.group_channels * 2, self.group_channels, kernel_size=1)

        padding = proj_kernel // 2
        self.project = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=proj_kernel, padding=padding, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self._init_stable_fusion()

    def _init_stable_fusion(self):
        nn.init.zeros_(self.lce_fuse.weight)
        nn.init.zeros_(self.lce_fuse.bias)
        for i in range(self.group_channels):
            self.lce_fuse.weight.data[i, i, 0, 0] = 1.0
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.zeros_(self.fusion_gate.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        grouped = x.view(b * self.groups, self.group_channels, h, w)

        h_pool = grouped.mean(dim=3, keepdim=True)
        w_pool = grouped.mean(dim=2, keepdim=True).transpose(2, 3)
        coord = self.glm_coord(torch.cat((h_pool, w_pool), dim=2))
        h_attn, w_attn = torch.split(coord, [h, w], dim=2)
        coord_attn = torch.sigmoid(h_attn) * torch.sigmoid(w_attn.transpose(2, 3))

        glm = self.glm_norm(grouped * coord_attn)
        glm_weight = self.glm_maxpool(glm)
        glm_weight = F.softmax(glm_weight.flatten(1), dim=1).view_as(glm_weight)
        glm = glm * glm_weight

        local_contrast = grouped - self.local_pool(grouped)
        local_detail = self.local_detail(local_contrast)
        lce = self.lce_fuse(torch.cat((glm, local_detail), dim=1))

        gcc = self.context_block(grouped)
        fusion_weight = torch.sigmoid(self.fusion_gate(torch.cat((lce, gcc), dim=1)))
        out = 2.0 * (fusion_weight * lce + (1.0 - fusion_weight) * gcc)

        out = out.view(b, self.groups, self.group_channels, h, w).reshape(b, c, h, w)
        return self.project(out)
