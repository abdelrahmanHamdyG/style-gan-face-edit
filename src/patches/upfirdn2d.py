"""
Pure-PyTorch fallback for upfirdn2d.

Replaces the CUDA JIT-compiled version shipped with encoder4editing so that
no C++ compiler or CUDA toolkit is required at import time.  Functionally
equivalent for inference (forward pass only).
"""

import torch
import torch.nn.functional as F


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    """Public API — drop-in replacement for the CUDA upfirdn2d."""
    return _upfirdn2d_native(
        input, kernel,
        up_x=up, up_y=up,
        down_x=down, down_y=down,
        pad_x0=pad[0], pad_x1=pad[1],
        pad_y0=pad[0], pad_y1=pad[1],
    )


def _upfirdn2d_native(
    input, kernel,
    up_x, up_y, down_x, down_y,
    pad_x0, pad_x1, pad_y0, pad_y1,
):
    """
    Pure-PyTorch implementation of upfirdn2d.

    Parameters
    ----------
    input  : (batch, channel, H, W)
    kernel : (kH, kW)
    """
    batch, channel, in_h, in_w = input.shape
    kernel_h, kernel_w = kernel.shape

    # ── Upsample (insert zeros between samples) ──
    if up_x > 1 or up_y > 1:
        x = input.view(batch, channel, in_h, 1, in_w, 1)
        x = F.pad(x, [0, up_x - 1, 0, 0, 0, up_y - 1])
        x = x.view(batch, channel, in_h * up_y, in_w * up_x)
    else:
        x = input

    # ── Pad ──
    x = F.pad(
        x,
        [max(pad_x0, 0), max(pad_x1, 0),
         max(pad_y0, 0), max(pad_y1, 0)],
    )

    # Negative pads → crop
    x = x[
        :, :,
        max(-pad_y0, 0): x.shape[2] - max(-pad_y1, 0) if pad_y1 < 0 else x.shape[2],
        max(-pad_x0, 0): x.shape[3] - max(-pad_x1, 0) if pad_x1 < 0 else x.shape[3],
    ]

    # ── Filter (depthwise convolution) ──
    w = torch.flip(kernel, [0, 1]).to(dtype=x.dtype, device=x.device)
    w = w.view(1, 1, kernel_h, kernel_w).expand(channel, 1, kernel_h, kernel_w)
    x = F.conv2d(x, w, groups=channel)

    # ── Downsample ──
    if down_x > 1 or down_y > 1:
        x = x[:, :, ::down_y, ::down_x]

    return x
