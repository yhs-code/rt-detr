import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ("DRPCADynamicSpatialAttention",)


class DRPCADynamicSpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=3):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for shape-preserving dynamic spatial attention")
        self.kernel_size = kernel_size
        self.kernel_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, kernel_size * kernel_size, kernel_size=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, _, height, width = x.shape
        kernels = self.kernel_generator(x).view(batch, 1, self.kernel_size, self.kernel_size)
        x_mean = x.mean(dim=1, keepdim=True).view(1, batch, height, width)
        attn = F.conv2d(x_mean, weight=kernels, padding=self.kernel_size // 2, groups=batch)
        attn = self.sigmoid(attn.view(batch, 1, height, width))
        return x * attn
