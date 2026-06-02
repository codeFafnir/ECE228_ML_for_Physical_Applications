"""
Option A: Pure GPTQ quantization of ImprovedPhysicsInformedUNet.

Algorithm:
  1. For each quantizable layer in forward order:
     a. Register a forward hook on that layer only.
     b. Run calibration data to accumulate the layer Hessian (on CPU).
     c. Remove hook, run gptq_quantize_weight, replace layer.
  2. Only one Hessian is live at a time — safe for large layers on MPS/Mac.

Key implementation notes:
  - Conv2d inputs are unfolded to (batch*H_out*W_out, C_in*kH*kW) for Hessian.
  - ConvTranspose2d inputs are flattened to (batch*H*W, in_ch).
  - Hessians accumulate on CPU to avoid MPS unified-memory OOM.
  - Skipped layers (LayerNorm, GroupNorm, etc.) match the TurboQuant skip list.
"""

import copy
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .gptq_core import gptq_quantize_weight, uniform_quantize_weight
from .gptq_layers import (
    GPTQLinear,
    GPTQConv2d,
    GPTQConvTranspose2d,
    format_storage_report,
)

PINN_DIR = Path(__file__).parent.parent.parent / "PINN_channel-estimation-main"
if str(PINN_DIR) not in sys.path:
    sys.path.insert(0, str(PINN_DIR))


_SKIP_MODULE_TYPES = (
    nn.LayerNorm,
    nn.GroupNorm,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.InstanceNorm2d,
    nn.Embedding,
)

_QUANTIZABLE_TYPES = (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)
_MAX_HESSIAN_DIM = 8192  # skip full GPTQ above this d_in (~256 MB Hessian)


def _is_quantizable(module: nn.Module) -> bool:
    return isinstance(module, _QUANTIZABLE_TYPES) and not isinstance(
        module, _SKIP_MODULE_TYPES
    )


def _collect_quantizable_layers(model: nn.Module) -> list[tuple[str, nn.Module, nn.Module, str]]:
    """
    Collect all quantizable layers in forward-pass order.

    Returns list of (full_name, parent_module, child_module, child_name).
    Respects the same skip-list as TurboQuant.
    """
    layers = []

    def _traverse(module: nn.Module, prefix: str) -> None:
        for child_name, child in module.named_children():
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, _SKIP_MODULE_TYPES):
                continue
            if _is_quantizable(child):
                layers.append((full_name, module, child, child_name))
            else:
                _traverse(child, full_name)

    _traverse(model, "")
    return layers


def _make_hessian_hook(
    hessians: dict,
    counts: dict,
    layer_name: str,
    module_type: type,
    kernel_size=None,
    dilation=None,
    padding=None,
    stride=None,
):
    """
    Forward hook that accumulates H += X_flat^T @ X_flat.

    For Conv2d: input is unfolded to (batch*L, C_in*kH*kW).
    For ConvTranspose2d: input is flattened to (batch*H*W, in_ch).
    For Linear: input is reshaped to (batch*..., in_features).
    """
    def hook_fn(module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
        X = inp[0].detach().float()

        if isinstance(module, nn.Conv2d):
            # Unfold: (B, C_in, H, W) -> (B, C_in*kH*kW, L) -> (B*L, C_in*kH*kW)
            X_unf = F.unfold(
                X,
                kernel_size=module.kernel_size,
                dilation=module.dilation,
                padding=module.padding,
                stride=module.stride,
            )  # (B, C_in*kH*kW, L)
            X_flat = X_unf.permute(0, 2, 1).reshape(-1, X_unf.shape[1])

        elif isinstance(module, nn.ConvTranspose2d):
            # Input: (B, in_ch, H, W) -> flatten spatial -> (B*H*W, in_ch)
            B, C, H, W = X.shape
            X_flat = X.permute(0, 2, 3, 1).reshape(-1, C)

        else:  # Linear
            # (..., in_features) -> (B*..., in_features)
            X_flat = X.reshape(-1, X.shape[-1])

        d_in = X_flat.shape[1]

        if layer_name not in hessians:
            hessians[layer_name] = torch.zeros(d_in, d_in, dtype=torch.float32, device="cpu")
            counts[layer_name] = 0

        x_cpu = X_flat.cpu()
        hessians[layer_name] += x_cpu.T @ x_cpu
        counts[layer_name] += X_flat.shape[0]

    return hook_fn


def _finalize_hessian(H: torch.Tensor, n: int) -> torch.Tensor:
    """Normalize and symmetrize accumulated Hessian."""
    H = H * (2.0 / max(n, 1))
    H = 0.5 * (H + H.T)
    return H


def _layer_input_dim(module: nn.Module) -> int:
    if isinstance(module, nn.Linear):
        return module.in_features
    if isinstance(module, nn.Conv2d):
        kh, kw = module.kernel_size
        return module.in_channels * kh * kw
    if isinstance(module, nn.ConvTranspose2d):
        return module.in_channels
    return 0


def _weight_as_2d(child: nn.Module) -> torch.Tensor:
    if isinstance(child, nn.Conv2d):
        return child.weight.data.float().view(child.weight.shape[0], -1)
    if isinstance(child, nn.ConvTranspose2d):
        return child.weight.data.float().view(child.weight.shape[0], -1).T
    return child.weight.data.float()


def _make_replacement(
    child: nn.Module,
    W_dequant: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    num_bits: int,
    group_size: int,
    packed: bool,
) -> nn.Module:
    g = min(group_size, W_dequant.shape[1])
    if isinstance(child, nn.Conv2d):
        return GPTQConv2d.from_conv(child, W_dequant, scales, zeros, num_bits, g, packed)
    if isinstance(child, nn.ConvTranspose2d):
        W_dequant_orig = W_dequant.T.reshape(child.weight.shape)
        return GPTQConvTranspose2d.from_conv_transpose(
            child, W_dequant_orig, scales, zeros, num_bits, g, packed
        )
    return GPTQLinear.from_linear(child, W_dequant, scales, zeros, num_bits, g, packed)


def _quantize_one_layer(
    child: nn.Module,
    H: torch.Tensor,
    num_bits: int,
    group_size: int,
    block_size: int,
    percdamp: float,
    packed: bool,
) -> nn.Module:
    """Run GPTQ on a single layer and return its inference wrapper."""
    if isinstance(child, nn.Conv2d):
        W_2d = child.weight.data.float().view(child.weight.shape[0], -1)
        W_dequant, scales, zeros = gptq_quantize_weight(
            W_2d, H, num_bits, min(group_size, W_2d.shape[1]), block_size, percdamp
        )
        return GPTQConv2d.from_conv(
            child, W_dequant, scales, zeros, num_bits,
            min(group_size, W_2d.shape[1]), packed,
        )

    if isinstance(child, nn.ConvTranspose2d):
        W_2d = child.weight.data.float().view(child.weight.shape[0], -1).T
        W_dequant, scales, zeros = gptq_quantize_weight(
            W_2d, H, num_bits, min(group_size, W_2d.shape[1]), block_size, percdamp
        )
        W_dequant_orig = W_dequant.T.reshape(child.weight.shape)
        return GPTQConvTranspose2d.from_conv_transpose(
            child, W_dequant_orig, scales, zeros, num_bits,
            min(group_size, W_2d.shape[1]), packed,
        )

    W_2d = child.weight.data.float()
    W_dequant, scales, zeros = gptq_quantize_weight(
        W_2d, H, num_bits, min(group_size, W_2d.shape[1]), block_size, percdamp
    )
    return GPTQLinear.from_linear(
        child, W_dequant, scales, zeros, num_bits,
        min(group_size, W_2d.shape[1]), packed,
    )


def gptq_quantize_pinn(
    model: nn.Module,
    cal_loader: DataLoader,
    num_bits: int = 4,
    group_size: int = 128,
    block_size: int = 128,
    percdamp: float = 0.01,
    device: torch.device = torch.device("cpu"),
    inplace: bool = False,
    verbose: bool = True,
    packed: bool = True,
    model_fp32: nn.Module | None = None,
) -> nn.Module:
    """
    Apply GPTQ to all quantizable layers of the PINN model.

    packed=True (default): uint8-packed storage (4-bit nibble or 8-bit).
    Requires num_bits in {4, 8} when packed=True.
    """
    if packed and num_bits not in (4, 8):
        raise ValueError("packed storage requires num_bits in {4, 8}")
    if not inplace:
        model = copy.deepcopy(model)
    model.eval()
    model.to(device)

    layers = _collect_quantizable_layers(model)
    n_cal_batches = len(cal_loader)
    if verbose:
        print(f"  Found {len(layers)} quantizable layers")
        print(f"  Layer-wise calibration: {n_cal_batches} batches per layer")
        print(f"  Storage: {'packed int' + str(num_bits) if packed else 'fp32 dequant'}")

    # ---- Layer-wise: calibrate one Hessian at a time, then quantize ----
    for full_name, parent, child, child_name in layers:
        d_in = _layer_input_dim(child)
        if d_in > _MAX_HESSIAN_DIM:
            W_2d = _weight_as_2d(child)
            W_dequant, scales, zeros = uniform_quantize_weight(
                W_2d, num_bits, min(group_size, W_2d.shape[1])
            )
            replacement = _make_replacement(
                child, W_dequant, scales, zeros, num_bits,
                min(group_size, W_2d.shape[1]), packed,
            )
            setattr(parent, child_name, replacement)
            if verbose:
                print(f"  [uniform] {full_name:50s} d_in={d_in} -> {num_bits}-bit")
            continue

        hessians: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = {}
        hook = child.register_forward_hook(
            _make_hessian_hook(hessians, counts, full_name, type(child))
        )

        with torch.no_grad():
            for smomp, _accurate, rss in cal_loader:
                model(smomp.to(device), rss.to(device))

        hook.remove()

        if full_name not in hessians:
            if verbose:
                print(f"  SKIP (no Hessian): {full_name}")
            continue

        H = _finalize_hessian(hessians.pop(full_name), counts.pop(full_name))
        replacement = _quantize_one_layer(
            child, H, num_bits, group_size, block_size, percdamp, packed
        )
        setattr(parent, child_name, replacement)
        del H

        if verbose:
            print(f"  [GPTQ] {full_name:50s} {list(child.weight.shape)} -> {num_bits}-bit")

    if verbose and model_fp32 is not None:
        print("\n" + format_storage_report(model_fp32, model))

    return model
