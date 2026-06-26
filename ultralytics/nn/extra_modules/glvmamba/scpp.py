import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("SCPPFeatureEnhance",)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, channels, bias=False)

    def forward(self, x):
        b, c, _, _ = x.shape
        weight = F.adaptive_avg_pool2d(x, 1).view(b, c)
        weight = self.fc2(self.act(self.fc1(weight))).sigmoid().view(b, c, 1, 1)
        return x * weight


class ScaleAwareModule(nn.Module):
    def __init__(self, channels, dilation_rates=(1, 3, 5), se_reduction=16):
        super().__init__()
        self.branches = nn.ModuleList()
        for dilation in dilation_rates:
            self.branches.append(nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                SEBlock(channels, reduction=se_reduction),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ))
        self.attn_conv = nn.Conv2d(channels, len(dilation_rates), 1, bias=True)

    def forward(self, x):
        branch_feats = [branch(x) for branch in self.branches]
        summed = branch_feats[0]
        for feat in branch_feats[1:]:
            summed = summed + feat
        attn = torch.softmax(self.attn_conv(summed), dim=1)
        fused = 0
        for index, feat in enumerate(branch_feats):
            fused = fused + feat * attn[:, index:index + 1]
        return fused


class SCPPFeatureEnhance(nn.Module):
    """Scale-aware pyramid pooling enhancement for final neck features."""

    def __init__(self, in_channels, out_channels=None, se_reduction=16, dilation_rates=(1, 3, 5)):
        super().__init__()
        out_channels = int(out_channels or in_channels)
        self.local = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.scale = ScaleAwareModule(in_channels, dilation_rates=dilation_rates, se_reduction=se_reduction)
        self.global_fc = nn.Sequential(nn.Linear(in_channels, in_channels, bias=True), nn.ReLU(inplace=True))
        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        local = self.local(x)
        scale = self.scale(x)
        global_context = F.adaptive_avg_pool2d(x, 1).view(b, c)
        global_context = self.global_fc(global_context).view(b, c, 1, 1).expand(-1, -1, h, w)
        return self.out(torch.cat((scale, local, global_context), dim=1))
