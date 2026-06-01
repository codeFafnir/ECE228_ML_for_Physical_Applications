"""
Option A: Pure GPTQ quantization of ImprovedPhysicsInformedUNet.

Algorithm:
  1. Register forward hooks on all quantizable layers (Linear, Conv2d, ConvTranspose2d).
  2. Run calibration data through the model to accumulate per-layer Hessians.
  3. Remove hooks.
  4. For each layer in forward order: run gptq_quantize_weight(W, H).
  5. Replace the layer with a GPTQ inference wrapper.

Key implementation notes:
  - Conv2d inputs are unfolded to (batch*H_out*W_out, C_in*kH*kW) for Hessian.
  - ConvTranspose2d inputs are flattened to (batch*H*W, in_ch).
  - Layers are processed sequentially to keep only one Hessian in memory at a time.
  - Hessians are deleted immediately after each layer is quantized.
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

from .gptq_core import gptq_quantize_weight
from .gptq_layers import GPTQLinear, GPTQConv2d, GPTQConvTranspose2d

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
            hessians[layer_name] = torch.zeros(d_in, d_in, dtype=torch.float32)
            counts[layer_name] = 0

        hessians[layer_name] += X_flat.T @ X_flat
        counts[layer_name] += X_flat.shape[0]

    return hook_fn


def _finalize_hessian(H: torch.Tensor, n: int) -> torch.Tensor:
    """Normalize and symmetrize accumulated Hessian."""
    H = H * (2.0 / max(n, 1))
    H = 0.5 * (H + H.T)
    return H


def gptq_quantize_pinn(
    model: nn.Module,
    cal_loader: DataLoader,
    num_bits: int = 3,
    group_size: int = 128,
    block_size: int = 128,
    percdamp: float = 0.01,
    device: torch.device = torch.device("cpu"),
    inplace: bool = False,
    verbose: bool = True,
) -> nn.Module:
    """
    Apply GPTQ to all quantizable layers of the PINN model.

    Args:
        model:      Trained FP32 model (ImprovedPhysicsInformedUNet).
        cal_loader: DataLoader yielding (smomp, accurate, rss) for calibration.
        num_bits:   Target bit-width (3 recommended).
        group_size: Columns per quantization group.
        block_size: GPTQ lazy block size (cols processed together).
        percdamp:   Hessian diagonal damping fraction.
        device:     Device for calibration forward passes.
        inplace:    If False, deepcopy the model first.
        verbose:    Print per-layer progress.

    Returns:
        Model with quantizable layers replaced by GPTQ wrappers.
    """
    if not inplace:
        model = copy.deepcopy(model)
    model.eval()
    model.to(device)

    layers = _collect_quantizable_layers(model)
    if verbose:
        print(f"  Found {len(layers)} quantizable layers")

    # ---- Phase 1: Accumulate Hessians via forward hooks ----
    hessians: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    hooks = []

    for full_name, parent, child, child_name in layers:
        h = child.register_forward_hook(
            _make_hessian_hook(hessians, counts, full_name, type(child))
        )
        hooks.append(h)

    if verbose:
        print(f"  Running {len(list(cal_loader))} calibration batches ...")

    with torch.no_grad():
        for smomp, accurate, rss in cal_loader:
            model(smomp.to(device), rss.to(device))

    for h in hooks:
        h.remove()

    if verbose:
        print(f"  Calibration done. Quantizing layers ...")

    # ---- Phase 2: Layer-wise GPTQ ----
    for full_name, parent, child, child_name in layers:
        if full_name not in hessians:
            if verbose:
                print(f"  SKIP (no Hessian): {full_name}")
            continue

        H = _finalize_hessian(hessians.pop(full_name), counts.pop(full_name))
        H = H.to(device)

        if isinstance(child, nn.Conv2d):
            # Reshape weight to (out_ch, in_ch*kH*kW)
            W_2d = child.weight.data.float().view(child.weight.shape[0], -1)
            W_dequant, scales, zeros = gptq_quantize_weight(
                W_2d, H, num_bits, min(group_size, W_2d.shape[1]), block_size, percdamp
            )
            replacement = GPTQConv2d.from_conv(child, W_dequant, scales, zeros)

        elif isinstance(child, nn.ConvTranspose2d):
            # Weight: (in_ch, out_ch, kH, kW) -> transpose to (out_ch*kH*kW, in_ch)
            # so d_out=out_ch*kH*kW, d_in=in_ch, matching H from (batch*H*W, in_ch) activations.
            W_2d = child.weight.data.float().view(child.weight.shape[0], -1).T  # (out_ch*kH*kW, in_ch)
            W_dequant, scales, zeros = gptq_quantize_weight(
                W_2d, H, num_bits, min(group_size, W_2d.shape[1]), block_size, percdamp
            )
            # Transpose back to (in_ch, out_ch*kH*kW) -> reshape to original weight shape
            W_dequant_orig = W_dequant.T.reshape(child.weight.shape)
            replacement = GPTQConvTranspose2d.from_conv_transpose(child, W_dequant_orig, scales, zeros)

        else:  # Linear
            W_2d = child.weight.data.float()  # already (out, in)
            W_dequant, scales, zeros = gptq_quantize_weight(
                W_2d, H, num_bits, min(group_size, W_2d.shape[1]), block_size, percdamp
            )
            replacement = GPTQLinear.from_linear(child, W_dequant, scales, zeros)

        setattr(parent, child_name, replacement)
        del H  # free Hessian memory immediately

        if verbose:
            print(f"  [GPTQ] {full_name:50s} {list(child.weight.shape)} -> {num_bits}-bit")

    return model
