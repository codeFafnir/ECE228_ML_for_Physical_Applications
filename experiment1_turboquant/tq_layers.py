"""
TurboQuant-wrapped PyTorch layer replacements.

Each wrapper holds a TurboQuantTensor instead of a float32 weight.
On forward(), it decompresses the weight on-the-fly and applies the
original operation. Biases are kept in FP16/FP32 (small, accuracy-critical).

Design choices:
  - Dequant-on-the-fly: simple, correct, adds ~O(d log d) overhead per forward.
  - block_dim=128 for Linear (rows of width >= 128 typical).
  - block_dim=64 for Conv2d/ConvTranspose2d (smaller filter dims possible).
  - If weight.numel() < block_dim, block_dim is shrunk to next_pow2(numel).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from turboquant import TurboQuantTensor


def _safe_block_dim(numel: int, preferred: int) -> int:
    """Return largest power-of-2 block_dim <= min(preferred, numel)."""
    cap = min(preferred, numel)
    if cap < 1:
        return 1
    # round down to power of 2
    p = 1
    while p * 2 <= cap:
        p *= 2
    return p


class TQLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with TurboQuant 3-bit compressed weight.

    Bias (if present) is kept as-is in the original dtype.
    """

    def __init__(
        self,
        lin: nn.Linear,
        block_dim: int = 128,
        seed: Optional[int] = None,
        n_rotation_blocks: int = 3,
    ):
        super().__init__()
        self.in_features = lin.in_features
        self.out_features = lin.out_features

        # Choose safe block_dim given weight size
        bd = _safe_block_dim(lin.weight.numel(), block_dim)
        self.tq = TurboQuantTensor.compress(
            lin.weight.data,
            block_dim=bd,
            seed=seed,
            n_rotation_blocks=n_rotation_blocks,
        )

        # Bias stays full precision
        if lin.bias is not None:
            self.bias = nn.Parameter(lin.bias.data.clone())
        else:
            self.bias = None

    @property
    def weight(self) -> torch.Tensor:
        # MultiheadAttention and other PyTorch internals access .weight directly
        return self.tq.decompress(dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.tq.decompress(dtype=x.dtype)
        return F.linear(x, W, self.bias)

    def extra_repr(self) -> str:
        bd = self.tq.block_dim
        ratio = self.tq.compression_ratio()
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"block_dim={bd}, compression={ratio:.1f}x"
        )


class TQConv2d(nn.Module):
    """
    Drop-in replacement for nn.Conv2d with TurboQuant 3-bit compressed weight.

    Weight tensor (out_ch, in_ch, kH, kW) is flattened, block-compressed,
    and reshaped back on decompression.
    """

    def __init__(
        self,
        conv: nn.Conv2d,
        block_dim: int = 64,
        seed: Optional[int] = None,
        n_rotation_blocks: int = 3,
    ):
        super().__init__()
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self._weight_shape = tuple(conv.weight.shape)

        bd = _safe_block_dim(conv.weight.numel(), block_dim)
        self.tq = TurboQuantTensor.compress(
            conv.weight.data,
            block_dim=bd,
            seed=seed,
            n_rotation_blocks=n_rotation_blocks,
        )

        if conv.bias is not None:
            self.bias = nn.Parameter(conv.bias.data.clone())
        else:
            self.bias = None

    @property
    def weight(self) -> torch.Tensor:
        return self.tq.decompress(dtype=torch.float32).view(self._weight_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.tq.decompress(dtype=x.dtype).view(self._weight_shape)
        return F.conv2d(x, W, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self) -> str:
        s = self._weight_shape
        ratio = self.tq.compression_ratio()
        return (
            f"weight={s}, stride={self.stride}, "
            f"block_dim={self.tq.block_dim}, compression={ratio:.1f}x"
        )


class TQConvTranspose2d(nn.Module):
    """
    Drop-in replacement for nn.ConvTranspose2d with TurboQuant 3-bit weight.

    Weight tensor (in_ch, out_ch, kH, kW) for transposed convolution.
    """

    def __init__(
        self,
        conv_t: nn.ConvTranspose2d,
        block_dim: int = 64,
        seed: Optional[int] = None,
        n_rotation_blocks: int = 3,
    ):
        super().__init__()
        self.stride = conv_t.stride
        self.padding = conv_t.padding
        self.output_padding = conv_t.output_padding
        self.dilation = conv_t.dilation
        self.groups = conv_t.groups
        self._weight_shape = tuple(conv_t.weight.shape)

        bd = _safe_block_dim(conv_t.weight.numel(), block_dim)
        self.tq = TurboQuantTensor.compress(
            conv_t.weight.data,
            block_dim=bd,
            seed=seed,
            n_rotation_blocks=n_rotation_blocks,
        )

        if conv_t.bias is not None:
            self.bias = nn.Parameter(conv_t.bias.data.clone())
        else:
            self.bias = None

    @property
    def weight(self) -> torch.Tensor:
        return self.tq.decompress(dtype=torch.float32).view(self._weight_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.tq.decompress(dtype=x.dtype).view(self._weight_shape)
        return F.conv_transpose2d(
            x, W, self.bias,
            self.stride, self.padding, self.output_padding,
            self.groups, self.dilation,
        )

    def extra_repr(self) -> str:
        s = self._weight_shape
        ratio = self.tq.compression_ratio()
        return (
            f"weight={s}, stride={self.stride}, "
            f"block_dim={self.tq.block_dim}, compression={ratio:.1f}x"
        )
