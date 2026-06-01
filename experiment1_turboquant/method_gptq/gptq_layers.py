"""
GPTQ inference wrappers for nn.Linear, nn.Conv2d, nn.ConvTranspose2d.

Each wrapper stores the GPTQ-quantized weight as per-group scales and
zero_points alongside the dequantized weight (for fast inference without
recomputing every forward pass). Biases remain full precision.

These layers are used after quantization is complete — they are NOT
themselves responsible for running GPTQ. See gptq_quantize_pinn.py for
how the model is quantized and these wrappers are created.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GPTQLinear(nn.Module):
    """
    nn.Linear replacement with GPTQ-quantized weight.

    Stores dequantized weight as an nn.Buffer for correct .to(device) handling.
    scales and zero_points are stored for analysis/re-quantization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        W_dequant: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        bias: Optional[torch.Tensor],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Dequantized weight as buffer (moves with .to(device))
        self.register_buffer("weight", W_dequant.float())
        self.register_buffer("scales", scales.float())
        self.register_buffer("zero_points", zero_points.float())

        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls,
        lin: nn.Linear,
        W_dequant: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
    ) -> "GPTQLinear":
        return cls(
            in_features=lin.in_features,
            out_features=lin.out_features,
            W_dequant=W_dequant,
            scales=scales,
            zero_points=zero_points,
            bias=lin.bias.data if lin.bias is not None else None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype), self.bias)

    def extra_repr(self) -> str:
        n_groups = self.scales.shape[1] if self.scales.dim() > 1 else 1
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"n_groups={n_groups}"
        )


class GPTQConv2d(nn.Module):
    """
    nn.Conv2d replacement with GPTQ-quantized weight.

    GPTQ treats Conv2d as a linear layer over unfolded patches:
      weight is reshaped to (out_ch, in_ch*kH*kW), quantized, then reshaped back.
    """

    def __init__(
        self,
        weight_shape: tuple,
        stride,
        padding,
        dilation,
        groups: int,
        W_dequant: torch.Tensor,
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

        self.register_buffer("weight", W_dequant.float().view(weight_shape))
        self.register_buffer("scales", scales.float())
        self.register_buffer("zero_points", zero_points.float())

        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.bias = None

    @classmethod
    def from_conv(
        cls,
        conv: nn.Conv2d,
        W_dequant: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
    ) -> "GPTQConv2d":
        return cls(
            weight_shape=tuple(conv.weight.shape),
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            W_dequant=W_dequant,
            scales=scales,
            zero_points=zero_points,
            bias=conv.bias.data if conv.bias is not None else None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.weight.to(x.dtype),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def extra_repr(self) -> str:
        return f"weight={self._weight_shape}, stride={self.stride}"


class GPTQConvTranspose2d(nn.Module):
    """
    nn.ConvTranspose2d replacement with GPTQ-quantized weight.

    Weight shape: (in_ch, out_ch, kH, kW).
    GPTQ quantizes it as (in_ch, out_ch*kH*kW).
    """

    def __init__(
        self,
        weight_shape: tuple,
        stride,
        padding,
        output_padding,
        dilation,
        groups: int,
        W_dequant: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        bias: Optional[torch.Tensor],
    ):
        super().__init__()
        self._weight_shape = weight_shape
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups

        # W_dequant is already in the original weight shape (in_ch, out_ch, kH, kW)
        self.register_buffer("weight", W_dequant.float().reshape(weight_shape))
        self.register_buffer("scales", scales.float())
        self.register_buffer("zero_points", zero_points.float())

        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.bias = None

    @classmethod
    def from_conv_transpose(
        cls,
        conv_t: nn.ConvTranspose2d,
        W_dequant: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
    ) -> "GPTQConvTranspose2d":
        return cls(
            weight_shape=tuple(conv_t.weight.shape),
            stride=conv_t.stride,
            padding=conv_t.padding,
            output_padding=conv_t.output_padding,
            dilation=conv_t.dilation,
            groups=conv_t.groups,
            W_dequant=W_dequant,
            scales=scales,
            zero_points=zero_points,
            bias=conv_t.bias.data if conv_t.bias is not None else None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv_transpose2d(
            x,
            self.weight.to(x.dtype),
            self.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
            self.dilation,
        )

    def extra_repr(self) -> str:
        return f"weight={self._weight_shape}, stride={self.stride}"
