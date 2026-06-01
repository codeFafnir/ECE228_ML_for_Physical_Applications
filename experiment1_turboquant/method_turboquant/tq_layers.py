"""
TurboQuant-wrapped PyTorch layer replacements.

Each wrapper holds a TurboQuantTensor instead of a float32 weight.
On forward(), it decompresses the weight on-the-fly and applies the
original operation. Biases are kept in FP16/FP32 (small, accuracy-critical).

Bug 1 fix: tq.idx and tq.norms are registered as persistent nn.Buffers so
.to(device) and state_dict() work correctly. The decompress path reads from
these buffers rather than from the dataclass fields directly.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .turboquant import (
    TurboQuantTensor,
    dequantize_3bit,
    inverse_rotation,
)


def _safe_block_dim(numel: int, preferred: int) -> int:
    """Return largest power-of-2 block_dim <= min(preferred, numel)."""
    cap = min(preferred, numel)
    if cap < 1:
        return 1
    p = 1
    while p * 2 <= cap:
        p *= 2
    return p


def _decompress_from_buffers(
    tq: TurboQuantTensor,
    tq_idx: torch.Tensor,
    tq_norms: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Decompress weight using caller-supplied idx/norms tensors (registered buffers).
    This ensures device correctness without mutating the TurboQuantTensor dataclass.
    """
    scale = math.sqrt(tq.block_dim)
    rotated_scaled = dequantize_3bit(tq_idx)          # (B, block_dim) float32, N(0,1) scale
    rotated_approx = rotated_scaled / scale            # unscale to unit-sphere N(0,1/d)
    unit_approx = inverse_rotation(
        rotated_approx, seed=tq.seed, n_blocks=tq.n_rotation_blocks
    )
    norms = tq_norms.float().unsqueeze(-1)             # (B, 1)
    rows = unit_approx * norms                         # (B, block_dim)
    flat = rows.reshape(-1)
    if tq.pad:
        flat = flat[: flat.numel() - tq.pad]
    return flat.view(tq.orig_shape).to(dtype)


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

        bd = _safe_block_dim(lin.weight.numel(), block_dim)
        self.tq = TurboQuantTensor.compress(
            lin.weight.data,
            block_dim=bd,
            seed=seed,
            n_rotation_blocks=n_rotation_blocks,
        )

        # Bug 1 fix: register as persistent buffers so .to(device) moves them
        self.register_buffer('tq_idx', self.tq.idx)
        self.register_buffer('tq_norms', self.tq.norms)

        if lin.bias is not None:
            self.bias = nn.Parameter(lin.bias.data.clone())
        else:
            self.bias = None

        self._cached_weight: Optional[torch.Tensor] = None

    @property
    def weight(self) -> torch.Tensor:
        return _decompress_from_buffers(self.tq, self.tq_idx, self.tq_norms, torch.float32)

    def _get_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if (
            self._cached_weight is None
            or self._cached_weight.device != device
            or self._cached_weight.dtype != dtype
        ):
            self._cached_weight = _decompress_from_buffers(
                self.tq, self.tq_idx, self.tq_norms, dtype
            )
        return self._cached_weight

    def invalidate_cache(self) -> None:
        self._cached_weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._get_weight(x.dtype, x.device)
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

        self.register_buffer('tq_idx', self.tq.idx)
        self.register_buffer('tq_norms', self.tq.norms)

        if conv.bias is not None:
            self.bias = nn.Parameter(conv.bias.data.clone())
        else:
            self.bias = None

        self._cached_weight: Optional[torch.Tensor] = None

    @property
    def weight(self) -> torch.Tensor:
        return _decompress_from_buffers(
            self.tq, self.tq_idx, self.tq_norms, torch.float32
        ).view(self._weight_shape)

    def _get_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if (
            self._cached_weight is None
            or self._cached_weight.device != device
            or self._cached_weight.dtype != dtype
        ):
            self._cached_weight = _decompress_from_buffers(
                self.tq, self.tq_idx, self.tq_norms, dtype
            ).view(self._weight_shape)
        return self._cached_weight

    def invalidate_cache(self) -> None:
        self._cached_weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._get_weight(x.dtype, x.device)
        return F.conv2d(x, W, self.bias, self.stride, self.padding, self.dilation, self.groups)

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

        self.register_buffer('tq_idx', self.tq.idx)
        self.register_buffer('tq_norms', self.tq.norms)

        if conv_t.bias is not None:
            self.bias = nn.Parameter(conv_t.bias.data.clone())
        else:
            self.bias = None

        self._cached_weight: Optional[torch.Tensor] = None

    @property
    def weight(self) -> torch.Tensor:
        return _decompress_from_buffers(
            self.tq, self.tq_idx, self.tq_norms, torch.float32
        ).view(self._weight_shape)

    def _get_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if (
            self._cached_weight is None
            or self._cached_weight.device != device
            or self._cached_weight.dtype != dtype
        ):
            self._cached_weight = _decompress_from_buffers(
                self.tq, self.tq_idx, self.tq_norms, dtype
            ).view(self._weight_shape)
        return self._cached_weight

    def invalidate_cache(self) -> None:
        self._cached_weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._get_weight(x.dtype, x.device)
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
