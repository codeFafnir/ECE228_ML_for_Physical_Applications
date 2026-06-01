"""
Converts an ImprovedPhysicsInformedUNet model to TurboQuant 3-bit form.

Traverses the module tree recursively, replacing:
  nn.Linear         -> TQLinear
  nn.Conv2d         -> TQConv2d
  nn.ConvTranspose2d -> TQConvTranspose2d

Skips (kept full precision):
  - LayerNorm / GroupNorm weight+bias: 1-D, accuracy-critical scaling.
  - BatchNorm variants: statistical normalizers.
  - nn.Embedding.
  - All bias tensors: small, 1-D.

Note: nn.MultiheadAttention's in_proj_weight is a fused (3*d, d) parameter
stored directly on the MHA module, not as a child nn.Linear. It is therefore
NOT reached by this traversal and remains FP32. This is intentional for
stability; to quantize it, wrap MHA separately.

Usage:
    model_q = quantize_model(model, block_dim_linear=128, block_dim_conv=64)
"""

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

PINN_DIR = Path(__file__).parent.parent.parent / "PINN_channel-estimation-main"
if str(PINN_DIR) not in sys.path:
    sys.path.insert(0, str(PINN_DIR))

from .tq_layers import TQLinear, TQConv2d, TQConvTranspose2d


_SKIP_MODULE_TYPES = (
    nn.LayerNorm,
    nn.GroupNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.InstanceNorm2d,
    nn.Embedding,
)


def _should_skip_module(module: nn.Module) -> bool:
    return isinstance(module, _SKIP_MODULE_TYPES)


def _replace_children(
    module: nn.Module,
    block_dim_linear: int,
    block_dim_conv: int,
    parent_name: str = "",
) -> None:
    for child_name, child in list(module.named_children()):
        if _should_skip_module(child):
            continue

        if isinstance(child, nn.Linear):
            setattr(module, child_name, TQLinear(child, block_dim=block_dim_linear))

        elif isinstance(child, nn.ConvTranspose2d):
            # Check ConvTranspose2d before Conv2d (subclass in some torch versions)
            setattr(module, child_name, TQConvTranspose2d(child, block_dim=block_dim_conv))

        elif isinstance(child, nn.Conv2d):
            setattr(module, child_name, TQConv2d(child, block_dim=block_dim_conv))

        else:
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            _replace_children(child, block_dim_linear, block_dim_conv, full_name)


def quantize_model(
    model: nn.Module,
    block_dim_linear: int = 128,
    block_dim_conv: int = 64,
    inplace: bool = False,
) -> nn.Module:
    """
    Return a TurboQuant-compressed copy of the model.

    Args:
        model:             Trained ImprovedPhysicsInformedUNet (or any nn.Module).
        block_dim_linear:  Block size for Linear weight compression (power of 2).
        block_dim_conv:    Block size for Conv2d/ConvTranspose2d weight compression.
        inplace:           If True, modify model in-place (saves peak memory).
    """
    assert (block_dim_linear & (block_dim_linear - 1)) == 0
    assert (block_dim_conv & (block_dim_conv - 1)) == 0

    if not inplace:
        model = copy.deepcopy(model)

    model.eval()
    _replace_children(model, block_dim_linear, block_dim_conv)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_report(model_fp32: nn.Module, model_q: nn.Module) -> str:
    lines = ["=" * 60, "  TurboQuant Compression Report", "=" * 60]

    n_tq_linear = sum(1 for m in model_q.modules() if isinstance(m, TQLinear))
    n_tq_conv = sum(1 for m in model_q.modules() if isinstance(m, TQConv2d))
    n_tq_convt = sum(1 for m in model_q.modules() if isinstance(m, TQConvTranspose2d))
    n_linear = sum(1 for m in model_fp32.modules() if isinstance(m, nn.Linear))
    n_conv = sum(1 for m in model_fp32.modules() if isinstance(m, nn.Conv2d))
    n_convt = sum(1 for m in model_fp32.modules() if isinstance(m, nn.ConvTranspose2d))

    lines.append(f"  Linear:          {n_linear} -> {n_tq_linear} quantized")
    lines.append(f"  Conv2d:          {n_conv} -> {n_tq_conv} quantized")
    lines.append(f"  ConvTranspose2d: {n_convt} -> {n_tq_convt} quantized")

    fp32_bytes = sum(p.numel() * p.element_size() for p in model_fp32.parameters())

    tq_bytes = 0
    for m in model_q.modules():
        if isinstance(m, (TQLinear, TQConv2d, TQConvTranspose2d)):
            tq_bytes += m.tq.nbytes_packed()
            if m.bias is not None:
                tq_bytes += m.bias.numel() * m.bias.element_size()
        else:
            for p in m.parameters(recurse=False):
                tq_bytes += p.numel() * p.element_size()

    lines.append(f"\n  FP32 model size: {fp32_bytes / 1e6:.1f} MB")
    lines.append(f"  TQ-3bit size:    {tq_bytes / 1e6:.1f} MB  (packed)")
    lines.append(f"  Compression:     {fp32_bytes / max(tq_bytes, 1):.1f}x")
    lines.append("=" * 60)
    return "\n".join(lines)
