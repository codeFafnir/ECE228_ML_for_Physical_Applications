"""
Converts an ImprovedPhysicsInformedUNet model to TurboQuant 3-bit form.

Traverses the module tree recursively, replacing:
  nn.Linear         -> TQLinear
  nn.Conv2d         -> TQConv2d
  nn.ConvTranspose2d -> TQConvTranspose2d

Skips (kept full precision):
  - Learnable query Parameters (channel_queries, antenna_queries, frequency_queries):
    small (few KB) and directly participate in transformer decoder attention.
  - LayerNorm / GroupNorm weight+bias: 1-D, accuracy-critical scaling.
  - BatchNorm2d weight, bias, running stats: statistical normalizers.
  - All bias tensors: small, 1-D.
  - PositionalEncoding buffer: non-learnable, deterministic.

Usage:
    model = ImprovedPhysicsInformedUNet(...)
    # load checkpoint...
    model_q = quantize_model(model, block_dim_linear=128, block_dim_conv=64)
    # model_q is a deepcopy with compressed weights, ready for inference
"""

import copy
import sys
from pathlib import Path
from typing import Optional, Set

import torch
import torch.nn as nn

# Allow importing from the PINN source directory
PINN_DIR = Path(__file__).parent.parent / "PINN_channel-estimation-main"
if str(PINN_DIR) not in sys.path:
    sys.path.insert(0, str(PINN_DIR))

from tq_layers import TQLinear, TQConv2d, TQConvTranspose2d


# Module types whose internal parameters must never be quantized.
# Their weights are used for normalization statistics, not linear transforms.
_SKIP_MODULE_TYPES = (
    nn.LayerNorm,
    nn.GroupNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.InstanceNorm2d,
    nn.Embedding,
)

# Parameter names (as seen in named_parameters) to keep full precision.
# These are accessed by exact name suffix match.
_SKIP_PARAM_NAMES: Set[str] = {
    "channel_queries",
    "antenna_queries",
    "frequency_queries",
    "bias",          # all biases stay FP
}


def _should_skip_module(name: str, module: nn.Module) -> bool:
    """Return True if this module should not be recursed into for quantization."""
    return isinstance(module, _SKIP_MODULE_TYPES)


def _replace_children(
    module: nn.Module,
    block_dim_linear: int,
    block_dim_conv: int,
    parent_name: str = "",
) -> None:
    """
    Recursively replace Linear/Conv2d/ConvTranspose2d children in-place.

    We replace at the child level (not grandchild) so we can use setattr.
    For deeper nesting, recursion handles it.
    """
    for child_name, child in list(module.named_children()):
        full_name = f"{parent_name}.{child_name}" if parent_name else child_name

        if _should_skip_module(child_name, child):
            # Keep normalization layers untouched
            continue

        if isinstance(child, nn.Linear):
            replacement = TQLinear(child, block_dim=block_dim_linear)
            setattr(module, child_name, replacement)

        elif isinstance(child, nn.ConvTranspose2d):
            # Must check ConvTranspose2d BEFORE Conv2d (it's a subclass in some torch versions)
            replacement = TQConvTranspose2d(child, block_dim=block_dim_conv)
            setattr(module, child_name, replacement)

        elif isinstance(child, nn.Conv2d):
            replacement = TQConv2d(child, block_dim=block_dim_conv)
            setattr(module, child_name, replacement)

        else:
            # Recurse into compound modules (Sequential, TransformerDecoder, etc.)
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

    Returns:
        Quantized model. Call .eval() and move to device before inference.
    """
    assert (block_dim_linear & (block_dim_linear - 1)) == 0, \
        f"block_dim_linear must be power of 2, got {block_dim_linear}"
    assert (block_dim_conv & (block_dim_conv - 1)) == 0, \
        f"block_dim_conv must be power of 2, got {block_dim_conv}"

    if not inplace:
        model = copy.deepcopy(model)

    model.eval()
    _replace_children(model, block_dim_linear, block_dim_conv)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_report(model_fp32: nn.Module, model_q: nn.Module) -> str:
    """
    Print compression stats: layer counts, total bytes, ratio.
    """
    from tq_layers import TQLinear, TQConv2d, TQConvTranspose2d
    from turboquant import TurboQuantTensor

    lines = ["=" * 60, "  TurboQuant Compression Report", "=" * 60]

    # Count layer types
    n_tq_linear = sum(1 for m in model_q.modules() if isinstance(m, TQLinear))
    n_tq_conv = sum(1 for m in model_q.modules() if isinstance(m, TQConv2d))
    n_tq_convt = sum(1 for m in model_q.modules() if isinstance(m, TQConvTranspose2d))
    n_linear = sum(1 for m in model_fp32.modules() if isinstance(m, nn.Linear))
    n_conv = sum(1 for m in model_fp32.modules() if isinstance(m, nn.Conv2d))
    n_convt = sum(1 for m in model_fp32.modules() if isinstance(m, nn.ConvTranspose2d))

    lines.append(f"  Linear:          {n_linear} -> {n_tq_linear} quantized")
    lines.append(f"  Conv2d:          {n_conv} -> {n_tq_conv} quantized")
    lines.append(f"  ConvTranspose2d: {n_convt} -> {n_tq_convt} quantized")

    # Byte counts
    fp32_bytes = sum(p.numel() * p.element_size()
                     for p in model_fp32.parameters())

    tq_bytes = 0
    for m in model_q.modules():
        if isinstance(m, (TQLinear, TQConv2d, TQConvTranspose2d)):
            tq_bytes += m.tq.nbytes_packed()
            if m.bias is not None:
                tq_bytes += m.bias.numel() * m.bias.element_size()
        elif not isinstance(m, (TQLinear, TQConv2d, TQConvTranspose2d)):
            for p in m.parameters(recurse=False):
                tq_bytes += p.numel() * p.element_size()

    lines.append(f"\n  FP32 model size: {fp32_bytes / 1e6:.1f} MB")
    lines.append(f"  TQ-3bit size:    {tq_bytes / 1e6:.1f} MB  (packed)")
    lines.append(f"  Compression:     {fp32_bytes / max(tq_bytes, 1):.1f}x")
    lines.append("=" * 60)
    return "\n".join(lines)
