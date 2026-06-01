"""
Hadamard+GPTQ inference wrappers (Option C / QuaRot-style).

At quantization time:
  1. Rotate weight W_rot = random_rotation(W, seed)
  2. Run GPTQ on W_rot to get W_rot_q (dequantized)
  3. Store W_rot_q, seed, and GPTQ metadata (scales, zero_points)

At inference time:
  1. Apply inverse_rotation(W_rot_q, seed) to recover approx W
  2. Run the standard linear/conv operation with the recovered weight

This stores W_rot_q in original (rotated) form and applies inverse_rotation
lazily, so that GPU memory holds only the dequantized rotated weight.
The final weight in the original basis is reconstructed per-forward,
but cached to avoid repeated computation.
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_EXP_DIR = Path(__file__).parent.parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

from method_turboquant.turboquant import inverse_rotation


class HadamardGPTQLinear(nn.Module):
    """
    nn.Linear replacement: stores GPTQ-quantized rotated weight.
    At forward(), applies inverse rotation to reconstruct the original-space weight.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        W_rot_q: torch.Tensor,
        seed: int,
        pad: int,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        bias: Optional[torch.Tensor],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.seed = seed
        self._pad = pad                # zero-padding added before rotation

        self.register_buffer("W_rot_q", W_rot_q.float())   # (out, padded_in)
        self.register_buffer("scales", scales.float())
        self.register_buffer("zero_points", zero_points.float())

        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.bias = None

        self._cached_weight: Optional[torch.Tensor] = None
        self._cached_device: Optional[torch.device] = None

    def _get_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self._cached_weight is None or self._cached_device != device:
            W_rot = self.W_rot_q.to(device)                     # (out, padded_in)
            W_orig = inverse_rotation(W_rot, seed=self.seed)    # (out, padded_in)
            if self._pad:
                W_orig = W_orig[:, : W_orig.shape[1] - self._pad]
            self._cached_weight = W_orig.to(dtype)
            self._cached_device = device
        return self._cached_weight

    def invalidate_cache(self) -> None:
        self._cached_weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._get_weight(x.dtype, x.device)
        return F.linear(x, W, self.bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, seed={self.seed}"


class HadamardGPTQConv2d(nn.Module):
    """
    nn.Conv2d replacement: GPTQ on Hadamard-rotated weight (reshaped as 2D).
    At inference, inverse rotation recovers the original weight.
    """

    def __init__(
        self,
        weight_shape: tuple,
        stride,
        padding,
        dilation,
        groups: int,
        W_rot_q: torch.Tensor,
        seed: int,
        pad: int,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        bias: Optional[torch.Tensor],
    ):
        super().__init__()
        self._weight_shape = weight_shape
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.seed = seed
        self._pad = pad

        self.register_buffer("W_rot_q", W_rot_q.float())
        self.register_buffer("scales", scales.float())
        self.register_buffer("zero_points", zero_points.float())

        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.bias = None

        self._cached_weight: Optional[torch.Tensor] = None
        self._cached_device: Optional[torch.device] = None

    def _get_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self._cached_weight is None or self._cached_device != device:
            W_rot = self.W_rot_q.to(device)
            W_orig_2d = inverse_rotation(W_rot, seed=self.seed)
            if self._pad:
                W_orig_2d = W_orig_2d[:, : W_orig_2d.shape[1] - self._pad]
            self._cached_weight = W_orig_2d.view(self._weight_shape).to(dtype)
            self._cached_device = device
        return self._cached_weight

    def invalidate_cache(self) -> None:
        self._cached_weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._get_weight(x.dtype, x.device)
        return F.conv2d(x, W, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self) -> str:
        return f"weight={self._weight_shape}, stride={self.stride}, seed={self.seed}"


class HadamardGPTQConvTranspose2d(nn.Module):
    """
    nn.ConvTranspose2d replacement with Hadamard-rotated GPTQ weight.

    W_rot_q is stored in the transposed form (out_ch*kH*kW, in_ch+pad).
    _get_weight applies inverse_rotation then transposes back to (in_ch, out_ch, kH, kW).
    """

    def __init__(
        self,
        weight_shape: tuple,
        stride,
        padding,
        output_padding,
        dilation,
        groups: int,
        W_rot_q: torch.Tensor,
        seed: int,
        pad: int,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        bias: Optional[torch.Tensor],
    ):
        super().__init__()
        self._weight_shape = weight_shape  # (in_ch, out_ch, kH, kW)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.seed = seed
        self._pad = pad

        # W_rot_q: (out_ch*kH*kW, in_ch+pad) — transposed form
        self.register_buffer("W_rot_q", W_rot_q.float())
        self.register_buffer("scales", scales.float())
        self.register_buffer("zero_points", zero_points.float())

        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.bias = None

        self._cached_weight: Optional[torch.Tensor] = None
        self._cached_device: Optional[torch.device] = None

    def _get_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self._cached_weight is None or self._cached_device != device:
            W_rot = self.W_rot_q.to(device)                       # (out_ch*kH*kW, in_ch+pad)
            W_orig_T = inverse_rotation(W_rot, seed=self.seed)    # (out_ch*kH*kW, in_ch+pad)
            if self._pad:
                W_orig_T = W_orig_T[:, : W_orig_T.shape[1] - self._pad]  # (out_ch*kH*kW, in_ch)
            # Transpose back to (in_ch, out_ch*kH*kW) then reshape to original shape
            self._cached_weight = W_orig_T.T.reshape(self._weight_shape).to(dtype)
            self._cached_device = device
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
        return f"weight={self._weight_shape}, stride={self.stride}, seed={self.seed}"
