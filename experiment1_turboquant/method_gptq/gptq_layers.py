"""
GPTQ inference wrappers for nn.Linear, nn.Conv2d, nn.ConvTranspose2d.

packed=True (default for 4-bit): stores nibble-packed uint8 weights + fp16 scales.
packed=False: legacy float32 dequantized weights (no storage savings).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gptq_pack import dequant_packed_weight, pack_gptq_tensors


class _GPTQBase(nn.Module):
    num_bits: int
    group_size: int
    packed: bool
    _d_out: int
    _d_in: int

    def _dequant_weight(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if not self.packed:
            return self.weight.to(device=device, dtype=dtype)
        return dequant_packed_weight(
            self.qweight,
            self.scales,
            self.zero_points,
            self._d_out,
            self._d_in,
            self.group_size,
            self.num_bits,
            dtype=dtype,
            device=device,
        )

    def packed_nbytes(self) -> int:
        if not self.packed:
            return self.weight.numel() * self.weight.element_size()
        nb = self.qweight.numel() * self.qweight.element_size()
        nb += self.scales.numel() * self.scales.element_size()
        nb += self.zero_points.numel() * self.zero_points.element_size()
        if self.bias is not None:
            nb += self.bias.numel() * self.bias.element_size()
        return nb

    def fp32_weight_nbytes(self) -> int:
        return self._d_out * self._d_in * 4


class GPTQLinear(_GPTQBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        bias: Optional[torch.Tensor],
        num_bits: int,
        group_size: int,
        packed: bool,
        W_dequant: Optional[torch.Tensor] = None,
        qweight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_bits = num_bits
        self.group_size = group_size
        self.packed = packed
        self._d_out = out_features
        self._d_in = in_features

        if packed:
            assert qweight is not None
            self.register_buffer("qweight", qweight)
            self.register_buffer("scales", scales)
            self.register_buffer("zero_points", zero_points)
        else:
            assert W_dequant is not None
            self.register_buffer("weight", W_dequant.float())
            self.register_buffer("scales", scales.float())
            self.register_buffer("zero_points", zero_points.float())

        self.bias = nn.Parameter(bias.clone()) if bias is not None else None

    @classmethod
    def from_linear(
        cls,
        lin: nn.Linear,
        W_dequant: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        num_bits: int = 4,
        group_size: int = 128,
        packed: bool = True,
    ) -> "GPTQLinear":
        g = min(group_size, lin.in_features)
        if packed:
            qw, sc, zc = pack_gptq_tensors(W_dequant, scales, zero_points, g, num_bits)
            return cls(
                in_features=lin.in_features,
                out_features=lin.out_features,
                scales=sc,
                zero_points=zc,
                bias=lin.bias.data if lin.bias is not None else None,
                num_bits=num_bits,
                group_size=g,
                packed=True,
                qweight=qw,
            )
        return cls(
            in_features=lin.in_features,
            out_features=lin.out_features,
            scales=scales,
            zero_points=zero_points,
            bias=lin.bias.data if lin.bias is not None else None,
            num_bits=num_bits,
            group_size=g,
            packed=False,
            W_dequant=W_dequant,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._dequant_weight(x.dtype, x.device)
        b = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)

    def extra_repr(self) -> str:
        mode = "packed" if self.packed else "fp32"
        return f"in={self.in_features}, out={self.out_features}, {self.num_bits}-bit, {mode}"


class GPTQConv2d(_GPTQBase):
    def __init__(
        self,
        weight_shape: tuple,
        stride,
        padding,
        dilation,
        groups: int,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        bias: Optional[torch.Tensor],
        num_bits: int,
        group_size: int,
        packed: bool,
        W_dequant: Optional[torch.Tensor] = None,
        qweight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self._weight_shape = weight_shape
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_bits = num_bits
        self.group_size = group_size
        self.packed = packed
        self._d_out = weight_shape[0]
        self._d_in = int(torch.tensor(weight_shape[1:]).prod().item())

        if packed:
            assert qweight is not None
            self.register_buffer("qweight", qweight)
            self.register_buffer("scales", scales)
            self.register_buffer("zero_points", zero_points)
        else:
            assert W_dequant is not None
            self.register_buffer("weight", W_dequant.float().view(weight_shape))
            self.register_buffer("scales", scales.float())
            self.register_buffer("zero_points", zero_points.float())

        self.bias = nn.Parameter(bias.clone()) if bias is not None else None

    @classmethod
    def from_conv(
        cls,
        conv: nn.Conv2d,
        W_dequant: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        num_bits: int = 4,
        group_size: int = 128,
        packed: bool = True,
    ) -> "GPTQConv2d":
        shape = tuple(conv.weight.shape)
        g = min(group_size, W_dequant.shape[1])
        if packed:
            qw, sc, zc = pack_gptq_tensors(W_dequant, scales, zero_points, g, num_bits)
            return cls(
                weight_shape=shape,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                scales=sc,
                zero_points=zc,
                bias=conv.bias.data if conv.bias is not None else None,
                num_bits=num_bits,
                group_size=g,
                packed=True,
                qweight=qw,
            )
        return cls(
            weight_shape=shape,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            scales=scales,
            zero_points=zero_points,
            bias=conv.bias.data if conv.bias is not None else None,
            num_bits=num_bits,
            group_size=g,
            packed=False,
            W_dequant=W_dequant,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._dequant_weight(x.dtype, x.device).view(self._weight_shape)
        b = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        return F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self) -> str:
        mode = "packed" if self.packed else "fp32"
        return f"weight={self._weight_shape}, {self.num_bits}-bit, {mode}"


class GPTQConvTranspose2d(_GPTQBase):
    def __init__(
        self,
        weight_shape: tuple,
        d_out: int,
        d_in: int,
        stride,
        padding,
        output_padding,
        dilation,
        groups: int,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        bias: Optional[torch.Tensor],
        num_bits: int,
        group_size: int,
        packed: bool,
        W_dequant: Optional[torch.Tensor] = None,
        qweight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self._weight_shape = weight_shape
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.num_bits = num_bits
        self.group_size = group_size
        self.packed = packed
        self._d_out = d_out
        self._d_in = d_in

        if packed:
            assert qweight is not None
            self.register_buffer("qweight", qweight)
            self.register_buffer("scales", scales)
            self.register_buffer("zero_points", zero_points)
        else:
            assert W_dequant is not None
            self.register_buffer("weight", W_dequant.float().reshape(weight_shape))
            self.register_buffer("scales", scales.float())
            self.register_buffer("zero_points", zero_points.float())

        self.bias = nn.Parameter(bias.clone()) if bias is not None else None

    @classmethod
    def from_conv_transpose(
        cls,
        conv_t: nn.ConvTranspose2d,
        W_dequant: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor,
        num_bits: int = 4,
        group_size: int = 128,
        packed: bool = True,
    ) -> "GPTQConvTranspose2d":
        shape = tuple(conv_t.weight.shape)
        W_2d = W_dequant.reshape(shape[0], -1).T
        d_out, d_in = W_2d.shape
        g = min(group_size, d_in)
        if packed:
            qw, sc, zc = pack_gptq_tensors(W_2d, scales, zero_points, g, num_bits)
            return cls(
                weight_shape=shape,
                d_out=d_out,
                d_in=d_in,
                stride=conv_t.stride,
                padding=conv_t.padding,
                output_padding=conv_t.output_padding,
                dilation=conv_t.dilation,
                groups=conv_t.groups,
                scales=sc,
                zero_points=zc,
                bias=conv_t.bias.data if conv_t.bias is not None else None,
                num_bits=num_bits,
                group_size=g,
                packed=True,
                qweight=qw,
            )
        return cls(
            weight_shape=shape,
            d_out=d_out,
            d_in=d_in,
            stride=conv_t.stride,
            padding=conv_t.padding,
            output_padding=conv_t.output_padding,
            dilation=conv_t.dilation,
            groups=conv_t.groups,
            scales=scales,
            zero_points=zero_points,
            bias=conv_t.bias.data if conv_t.bias is not None else None,
            num_bits=num_bits,
            group_size=g,
            packed=False,
            W_dequant=W_dequant.reshape(shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w2d = self._dequant_weight(x.dtype, x.device)
        w = w2d.T.reshape(self._weight_shape)
        b = self.bias.to(device=x.device, dtype=x.dtype) if self.bias is not None else None
        return F.conv_transpose2d(
            x, w, b, self.stride, self.padding, self.output_padding, self.groups, self.dilation
        )

    def extra_repr(self) -> str:
        mode = "packed" if self.packed else "fp32"
        return f"weight={self._weight_shape}, {self.num_bits}-bit, {mode}"


_GPTQ_LAYER_TYPES = (GPTQLinear, GPTQConv2d, GPTQConvTranspose2d)


def model_storage_bytes(model: nn.Module) -> tuple[int, int, float]:
    """Returns (packed_bytes, fp32_weight_bytes, weight_compression_ratio)."""
    packed = 0
    fp32_equiv = 0
    for module in model.modules():
        if isinstance(module, _GPTQBase):
            packed += module.packed_nbytes()
            fp32_equiv += module.fp32_weight_nbytes()
            if module.bias is not None:
                fp32_equiv += module.bias.numel() * 4
        elif len(list(module.children())) == 0:
            for t in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
                nbytes = t.numel() * t.element_size()
                packed += nbytes
                fp32_equiv += nbytes
    ratio = fp32_equiv / max(packed, 1)
    return packed, fp32_equiv, ratio


def format_storage_report(model_fp32: nn.Module, model_gptq: nn.Module) -> str:
    fp32_all = sum(p.numel() * p.element_size() for p in model_fp32.parameters())
    fp32_all += sum(b.numel() * b.element_size() for b in model_fp32.buffers())
    packed, gptq_fp32_w, ratio = model_storage_bytes(model_gptq)
    gptq_all = sum(p.numel() * p.element_size() for p in model_gptq.parameters())
    gptq_all += sum(b.numel() * b.element_size() for b in model_gptq.buffers())

    lines = [
        "=" * 58,
        "  GPTQ Storage Report",
        "=" * 58,
        f"  FP32 checkpoint (all tensors):  {fp32_all / 1e6:.1f} MB",
        f"  GPTQ model (all tensors):         {gptq_all / 1e6:.1f} MB",
        f"  GPTQ quantizable weights only:    {packed / 1e6:.1f} MB packed",
        f"    vs {gptq_fp32_w / 1e6:.1f} MB if weights were FP32",
        f"  Weight compression ratio:         {ratio:.2f}x",
        f"  Overall model compression:        {fp32_all / max(gptq_all, 1):.2f}x",
        "=" * 58,
    ]
    return "\n".join(lines)
