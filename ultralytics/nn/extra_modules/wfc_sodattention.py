import math
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ("WFCSODAttention",)


def _valid_groups(channels, groups):
    groups = max(1, min(int(groups), channels))
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return groups


class FastContextBlock(nn.Module):
    """Fast-convergent ContextBlock for grouped SODAttention.

    The mask is initialized to produce uniform spatial context and the final
    channel-add projection is initialized to zero, avoiding noisy global context
    injection during early training while preserving learnability.
    """

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
        self._init_fast_context()

    def _init_fast_context(self):
        nn.init.zeros_(self.conv_mask.weight)
        nn.init.zeros_(self.conv_mask.bias)
        nn.init.normal_(self.channel_add[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.channel_add[-1].bias)

    def forward(self, x):
        b, c, h, w = x.shape
        mask = self.conv_mask(x).view(b, 1, h * w)
        mask = F.softmax(mask, dim=2).unsqueeze(-1)
        value = x.view(b, c, h * w).unsqueeze(1)
        context = torch.matmul(value, mask).view(b, c, 1, 1)
        return x + self.channel_add(context)


class WaterContrastCalibration(nn.Module):
    """Water-scene local contrast calibration for the GLM response.

    It uses local contrast statistics to recalibrate the GLM branch. The final
    gate is identity-initialized with 2 * sigmoid(logits), so the module starts
    from the original GLM response and learns scene-specific calibration.
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // int(reduction), 1)
        self.local_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels + 1, hidden, kernel_size=1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self._init_identity_gate()

    def _init_identity_gate(self):
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, grouped, glm):
        local_contrast = grouped - self.local_pool(grouped)
        contrast_stat = local_contrast.abs().mean(dim=1, keepdim=True)
        logits = self.gate(torch.cat((glm, contrast_stat), dim=1))
        gate = 2.0 * torch.sigmoid(logits)
        return glm * gate


class WFCSODAttention(nn.Module):
    """Water-scene Fast-Convergent SODAttention.

    This variant preserves the MSAOriginalSODAttention core: grouped GLM,
    ContextBlock, direct GLM + context fusion, and Conv-BN-ReLU projection. It
    only adds convergence-oriented changes: mean-preserving GLM competition,
    fast ContextBlock initialization, and identity-initialized water-scene local
    contrast calibration inside the GLM path.
    """

    def __init__(self, channels, groups=8, reduction=4, proj_kernel=1):
        super().__init__()
        self.channels = channels
        self.groups = _valid_groups(channels, groups)
        self.group_channels = channels // self.groups

        self.glm_coord = nn.Conv2d(self.group_channels, self.group_channels, kernel_size=1)
        self.glm_norm = nn.GroupNorm(1, self.group_channels)
        self.glm_maxpool = nn.AdaptiveMaxPool2d(1)
        self.glm_scale = math.sqrt(float(self.group_channels))
        self.water_calibration = WaterContrastCalibration(self.group_channels, reduction=reduction)
        self.context_block = FastContextBlock(self.group_channels, reduction=reduction)

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
        glm = glm * (glm_weight * self.glm_scale)
        glm = self.water_calibration(grouped, glm)

        context = self.context_block(grouped)
        out = glm + context
        out = out.view(b, self.groups, self.group_channels, h, w).reshape(b, c, h, w)
        return self.project(out)
