"""
Pure-PyTorch fallback for fused_bias_act.

Replaces the CUDA JIT-compiled version shipped with encoder4editing so that
no C++ compiler or CUDA toolkit is required at import time.  Functionally
equivalent for inference (forward pass only).
"""

import torch
from torch import nn
import torch.nn.functional as F


class FusedLeakyReLU(nn.Module):
    """LeakyReLU with a learnable bias, fused into a single op."""

    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    """Bias-add + LeakyReLU + scale, matching the CUDA kernel's behaviour."""
    if input.ndim == 4:
        bias = bias.view(1, -1, 1, 1)      # (B, C, H, W)
    else:
        bias = bias.view(1, -1)             # (B, C)
    return F.leaky_relu(input + bias, negative_slope=negative_slope) * scale
