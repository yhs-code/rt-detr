import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ("BCRSODAttention",)


def _valid_groups(channels, groups):
    groups = max(1, min(int(groups), channels))
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class GroupedContextBlock(nn.Module):
    """ContextBlock branch used by the MSA-DETR SODAttention module."""

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


class BranchConsistencyRecalibration(nn.Module):
    """Bounded channel recalibration from GLM/ContextBlock consistency."""

    def __init__(self, channels, reduction=4, beta=0.1, tau=2.0, init_std=1e-3):
        super().__init__()
        hidden = max(channels // int(reduction), 1)
        self.beta = float(beta)
        self.tau = float(tau)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )
        self._init_near_identity(float(init_std))

    def _init_near_identity(self, init_std):
        nn.init.normal_(self.gate[-1].weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, glm, context):
        stats = torch.cat(
            (
                F.adaptive_avg_pool2d(glm, 1),
                F.adaptive_avg_pool2d(context, 1),
                F.adaptive_avg_pool2d((glm - context).abs(), 1),
            ),
            dim=1,
        )
        logit = self.gate(stats)
        gate = 1.0 + self.beta * torch.tanh(logit / self.tau)
        return (glm + context) * gate


class BCRSODAttention(nn.Module):
    """Branch Consistency Recalibrated SODAttention.

    The original MSA-DETR SODAttention path is preserved: grouped GLM,
    ContextBlock, direct branch addition, and Conv-BN-ReLU projection. BCR only
    adds a bounded channel calibration based on the agreement between the two
    branches before projection.
    """

    def __init__(self, channels, groups=8, reduction=4, proj_kernel=1, beta=0.1, tau=2.0):
        super().__init__()
        self.channels = channels
        self.groups = _valid_groups(channels, groups)
        self.group_channels = channels // self.groups

        self.glm_coord = nn.Conv2d(self.group_channels, self.group_channels, kernel_size=1)
        self.glm_norm = nn.GroupNorm(1, self.group_channels)
        self.glm_maxpool = nn.AdaptiveMaxPool2d(1)
        self.context_block = GroupedContextBlock(self.group_channels, reduction=reduction)
        self.recalibration = BranchConsistencyRecalibration(
            self.group_channels, reduction=reduction, beta=beta, tau=tau
        )

        padding = proj_kernel // 2
        self.project = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=proj_kernel, padding=padding, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

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

        context = self.context_block(grouped)
        out = self.recalibration(glm, context)
        out = out.view(b, self.groups, self.group_channels, h, w).reshape(b, c, h, w)
        return self.project(out)
