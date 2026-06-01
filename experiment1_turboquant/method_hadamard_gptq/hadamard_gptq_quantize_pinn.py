"""
Option C: Hadamard rotation + GPTQ quantization of ImprovedPhysicsInformedUNet.

QuaRot-style approach:
  1. For each quantizable layer, rotate its weight W using random_rotation(W, seed).
     This makes the weight distribution near-Gaussian, which is optimal for
     uniform GPTQ quantization (no outliers dominate the quantization grid).
  2. Replace the layer in the model with a temporary proxy layer that runs
     the *rotated* weight during calibration (so the Hessian is computed in
     the rotated space, as required by GPTQ).
  3. Accumulate Hessians H = (2/N) sum_i x_i x_i^T over calibration data.
  4. Run GPTQ on each rotated weight W_rot with its Hessian.
  5. Replace proxy layers with HadamardGPTQ inference wrappers that apply
     inverse_rotation at forward time.

Crucial correctness property: the Hessian H used in step 4 is computed
with the *rotated* weights in place, which is the correct Hessian for
the rotated weight optimization problem.

Padding: random_rotation requires the last dimension to be a power of 2.
Weights with non-pow2 column counts are zero-padded before rotation and
un-padded after inverse_rotation.
"""

import copy
import math
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_EXP_DIR = Path(__file__).parent.parent
if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))

PINN_DIR = _EXP_DIR.parent / "PINN_channel-estimation-main"
if str(PINN_DIR) not in sys.path:
    sys.path.insert(0, str(PINN_DIR))

from method_turboquant.turboquant import random_rotation, inverse_rotation
from method_gptq.gptq_core import gptq_quantize_weight
from method_gptq.gptq_layers import (
    GPTQLinear,
    GPTQConv2d,
    GPTQConvTranspose2d,
)
from .hadamard_gptq_layers import (
    HadamardGPTQLinear,
    HadamardGPTQConv2d,
    HadamardGPTQConvTranspose2d,
)

# Layers with d_in larger than this skip Hadamard rotation to avoid OOM when
# zero-padding the (d_in_padded × d_in_padded) Hessian.
_MAX_ROTATION_DIM = 4096


def _zero_pad_hessian(H: torch.Tensor, pad: int) -> torch.Tensor:
    """Zero-pad H from (d, d) to (d+pad, d+pad) to match the padded weight columns."""
    if pad == 0:
        return H
    d = H.shape[0]
    H_new = torch.zeros(d + pad, d + pad, dtype=H.dtype, device=H.device)
    H_new[:d, :d] = H
    return H_new


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


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _rotate_weight_2d(W_2d: torch.Tensor, seed: int) -> tuple[torch.Tensor, int, bool]:
    """
    Apply random_rotation to the rows of W_2d (shape d_out x d_in).

    random_rotation acts on the last dimension. Pads d_in to the next power of 2.
    Returns (W_rot, pad, rotated). If d_in > _MAX_ROTATION_DIM, returns the original
    weight unmodified with pad=0 and rotated=False to avoid OOM when padding the Hessian.
    """
    d_out, d_in = W_2d.shape
    if d_in > _MAX_ROTATION_DIM:
        return W_2d.float(), 0, False
    d_in_padded = _next_pow2(d_in)
    pad = d_in_padded - d_in
    if pad:
        W_2d = F.pad(W_2d, (0, pad))
    W_rot = random_rotation(W_2d.float(), seed=seed)
    return W_rot, pad, True


class _RotatedLinearProxy(nn.Module):
    """
    Temporary proxy that runs a (possibly rotated) weight during calibration.
    When rotated=True: stores W_rot_padded, recovers original via inverse_rotation.
    When rotated=False: stores original weight, used as-is (large-layer fallback).
    Exposes .weight so MultiheadAttention internals don't crash.
    """
    def __init__(self, lin: nn.Linear, W_rot: torch.Tensor, pad: int, seed: int = 0, rotated: bool = True):
        super().__init__()
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self._pad = pad
        self._seed = seed
        self._rotated = rotated
        self.register_buffer("weight_rot", W_rot)
        self.bias = lin.bias

    @property
    def weight(self) -> torch.Tensor:
        if not self._rotated:
            return self.weight_rot
        W_orig = inverse_rotation(self.weight_rot.float(), seed=self._seed)
        if self._pad:
            W_orig = W_orig[:, : W_orig.shape[1] - self._pad]
        return W_orig

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._rotated:
            return F.linear(x, self.weight_rot.to(x.dtype), self.bias)
        W_orig = inverse_rotation(self.weight_rot.float(), seed=self._seed)
        if self._pad:
            W_orig = W_orig[:, : W_orig.shape[1] - self._pad]
        return F.linear(x, W_orig.to(x.dtype), self.bias)


class _RotatedConv2dProxy(nn.Module):
    """Proxy for Conv2d. If rotated=True, stores rotated 2D weight and recovers original via
    inverse_rotation. If rotated=False (large-d_in fallback), stores original weight as-is."""
    def __init__(self, conv: nn.Conv2d, W_rot_2d: torch.Tensor, pad: int, seed: int = 0, rotated: bool = True):
        super().__init__()
        self._weight_shape = tuple(conv.weight.shape)
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self._pad = pad
        self._seed = seed
        self._rotated = rotated
        self.register_buffer("weight_rot_2d", W_rot_2d)
        self.bias = conv.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._rotated:
            return F.conv2d(x, self.weight_rot_2d.view(self._weight_shape).to(x.dtype),
                            self.bias, self.stride, self.padding, self.dilation, self.groups)
        W_orig_2d = inverse_rotation(self.weight_rot_2d.float(), seed=self._seed)
        if self._pad:
            W_orig_2d = W_orig_2d[:, : W_orig_2d.shape[1] - self._pad]
        return F.conv2d(x, W_orig_2d.view(self._weight_shape).to(x.dtype),
                        self.bias, self.stride, self.padding, self.dilation, self.groups)


class _RotatedConvTranspose2dProxy(nn.Module):
    """Proxy for ConvTranspose2d. If rotated=True, stores rotated transposed weight and recovers
    original via inverse_rotation+transpose. If rotated=False, stores original weight as-is."""
    def __init__(self, conv_t: nn.ConvTranspose2d, W_rot_2d: torch.Tensor, pad: int,
                 seed: int = 0, rotated: bool = True):
        super().__init__()
        self._weight_shape = tuple(conv_t.weight.shape)  # (in_ch, out_ch, kH, kW)
        self.stride = conv_t.stride
        self.padding = conv_t.padding
        self.output_padding = conv_t.output_padding
        self.dilation = conv_t.dilation
        self.groups = conv_t.groups
        self._pad = pad
        self._seed = seed
        self._rotated = rotated
        self.register_buffer("weight_rot_2d", W_rot_2d)
        self.bias = conv_t.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._rotated:
            # weight_rot_2d is (out_ch*kH*kW, in_ch) — transpose to original shape
            W_orig = self.weight_rot_2d.T.reshape(self._weight_shape)
        else:
            W_rot = self.weight_rot_2d  # (out_ch*kH*kW, in_ch+pad)
            W_orig_T = inverse_rotation(W_rot.float(), seed=self._seed)
            if self._pad:
                W_orig_T = W_orig_T[:, : W_orig_T.shape[1] - self._pad]
            W_orig = W_orig_T.T.reshape(self._weight_shape)
        return F.conv_transpose2d(
            x, W_orig.to(x.dtype), self.bias,
            self.stride, self.padding, self.output_padding,
            self.groups, self.dilation,
        )


def _collect_quantizable_layers(model: nn.Module) -> list[tuple[str, nn.Module, nn.Module, str]]:
    layers = []

    def _traverse(module: nn.Module, prefix: str) -> None:
        for child_name, child in module.named_children():
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, _SKIP_MODULE_TYPES):
                continue
            if isinstance(child, _QUANTIZABLE_TYPES):
                layers.append((full_name, module, child, child_name))
            else:
                _traverse(child, full_name)

    _traverse(model, "")
    return layers


def _make_hessian_hook(
    hessians: dict,
    counts: dict,
    layer_name: str,
):
    """
    Hook for proxy layers: accumulate H from the actual input activations.
    Proxy Conv2d and ConvTranspose2d proxies handle their own unfolding internally,
    so this hook captures the standard (non-unfolded) input.

    For the proxy classes:
    - _RotatedLinearProxy: input is (batch, ..., in_features) -> flatten
    - _RotatedConv2dProxy: we need unfolded input — hook on proxy's unfold output
    - _RotatedConvTranspose2dProxy: input is (batch, in_ch, H, W) -> flatten spatial

    Simpler: we hook at the proxy level and let each proxy type determine how
    to flatten. We determine the type from the stored weight shape.
    """
    def hook_fn(module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
        X = inp[0].detach().float()

        if isinstance(module, _RotatedLinearProxy):
            X_flat = X.reshape(-1, X.shape[-1])

        elif isinstance(module, _RotatedConv2dProxy):
            kH, kW = module._weight_shape[2], module._weight_shape[3]
            X_unf = F.unfold(
                X,
                kernel_size=(kH, kW),
                dilation=module.dilation,
                padding=module.padding,
                stride=module.stride,
            )
            X_flat = X_unf.permute(0, 2, 1).reshape(-1, X_unf.shape[1])

        elif isinstance(module, _RotatedConvTranspose2dProxy):
            B, C, H, W = X.shape
            X_flat = X.permute(0, 2, 3, 1).reshape(-1, C)

        else:
            return

        if layer_name not in hessians:
            hessians[layer_name] = torch.zeros(X_flat.shape[1], X_flat.shape[1])
            counts[layer_name] = 0
        hessians[layer_name] += X_flat.T @ X_flat
        counts[layer_name] += X_flat.shape[0]

    return hook_fn


def hadamard_gptq_quantize_pinn(
    model: nn.Module,
    cal_loader: DataLoader,
    num_bits: int = 3,
    group_size: int = 128,
    block_size: int = 128,
    percdamp: float = 0.01,
    device: torch.device = torch.device("cpu"),
    base_seed: int = 42,
    inplace: bool = False,
    verbose: bool = True,
) -> nn.Module:
    """
    Apply Hadamard rotation + GPTQ to all quantizable layers of the PINN model.

    Args:
        model:      Trained FP32 model.
        cal_loader: DataLoader yielding (smomp, accurate, rss) triples.
        num_bits:   Target bit-width (3 recommended).
        group_size: GPTQ quantization group size.
        block_size: GPTQ lazy block size.
        percdamp:   Hessian damping fraction.
        device:     Device for calibration passes.
        base_seed:  Base seed; each layer gets seed = base_seed + layer_index.
        inplace:    If False, deepcopy the model.
        verbose:    Print per-layer progress.

    Returns:
        Model with quantizable layers replaced by HadamardGPTQ inference wrappers.
    """
    if not inplace:
        model = copy.deepcopy(model)
    model.eval()
    model.to(device)

    layers = _collect_quantizable_layers(model)
    if verbose:
        print(f"  Found {len(layers)} quantizable layers")

    # ---- Phase 1: Rotate weights and install proxy layers ----
    layer_metadata: dict[str, dict] = {}

    for layer_idx, (full_name, parent, child, child_name) in enumerate(layers):
        seed = base_seed + layer_idx

        if isinstance(child, nn.Conv2d):
            W_2d = child.weight.data.float().view(child.weight.shape[0], -1)
            W_rot, pad, rotated = _rotate_weight_2d(W_2d, seed)
            proxy = _RotatedConv2dProxy(child, W_rot, pad, seed=seed, rotated=rotated)

        elif isinstance(child, nn.ConvTranspose2d):
            # Transpose to (out_ch*kH*kW, in_ch) for GPTQ (matches H from (batch*H*W, in_ch))
            W_2d = child.weight.data.float().view(child.weight.shape[0], -1).T  # (out_ch*kH*kW, in_ch)
            W_rot, pad, rotated = _rotate_weight_2d(W_2d, seed)
            proxy = _RotatedConvTranspose2dProxy(child, W_rot, pad, seed=seed, rotated=rotated)

        else:  # Linear
            W_2d = child.weight.data.float()
            W_rot, pad, rotated = _rotate_weight_2d(W_2d, seed)
            proxy = _RotatedLinearProxy(child, W_rot, pad, seed=seed, rotated=rotated)

        setattr(parent, child_name, proxy)
        layer_metadata[full_name] = {
            "seed": seed,
            "pad": pad,
            "rotated": rotated,
            "orig_layer": child,
            "parent": parent,
            "child_name": child_name,
        }

    # ---- Phase 2: Calibration on the rotated model ----
    hessians: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    hooks = []

    for full_name, parent, _, child_name in layers:
        proxy = getattr(parent, child_name)
        h = proxy.register_forward_hook(
            _make_hessian_hook(hessians, counts, full_name)
        )
        hooks.append(h)

    if verbose:
        print(f"  Running calibration on rotated model ...")

    with torch.no_grad():
        for smomp, accurate, rss in cal_loader:
            model(smomp.to(device), rss.to(device))

    for h in hooks:
        h.remove()

    if verbose:
        print(f"  Calibration done. Quantizing layers ...")

    # ---- Phase 3: GPTQ on (rotated) weights ----
    for full_name, parent, _, child_name in layers:
        meta = layer_metadata[full_name]
        seed = meta["seed"]
        pad = meta["pad"]
        rotated = meta["rotated"]
        orig_layer = meta["orig_layer"]
        proxy = getattr(parent, child_name)

        if full_name not in hessians:
            if verbose:
                print(f"  SKIP (no Hessian): {full_name}")
            setattr(parent, child_name, orig_layer)
            continue

        H = hessians.pop(full_name) * (2.0 / max(counts.pop(full_name), 1))
        H = 0.5 * (H + H.T)
        H = H.to(device)

        if rotated and pad > 0:
            # Zero-pad H to match the padded weight columns.
            # The padded columns of W_rot have no real input, so the padded Hessian
            # entries are 0, which GPTQ's percdamp handles via damping.
            H = _zero_pad_hessian(H, pad)

        if isinstance(orig_layer, nn.Conv2d):
            W_2d = proxy.weight_rot_2d  # (out_ch, d_in or d_in_padded)
            W_dequant, scales, zeros = gptq_quantize_weight(
                W_2d.float(), H, num_bits, min(group_size, W_2d.shape[1]), block_size, percdamp
            )
            if rotated:
                replacement = HadamardGPTQConv2d(
                    weight_shape=tuple(orig_layer.weight.shape),
                    stride=orig_layer.stride,
                    padding=orig_layer.padding,
                    dilation=orig_layer.dilation,
                    groups=orig_layer.groups,
                    W_rot_q=W_dequant,
                    seed=seed,
                    pad=pad,
                    scales=scales,
                    zero_points=zeros,
                    bias=orig_layer.bias.data if orig_layer.bias is not None else None,
                )
            else:
                # Large layer — wrap with pure GPTQ (no rotation).
                # GPTQConv2d.__init__ does .view(weight_shape) internally.
                replacement = GPTQConv2d(
                    weight_shape=tuple(orig_layer.weight.shape),
                    stride=orig_layer.stride,
                    padding=orig_layer.padding,
                    dilation=orig_layer.dilation,
                    groups=orig_layer.groups,
                    W_dequant=W_dequant,
                    scales=scales,
                    zero_points=zeros,
                    bias=orig_layer.bias.data if orig_layer.bias is not None else None,
                )

        elif isinstance(orig_layer, nn.ConvTranspose2d):
            W_2d = proxy.weight_rot_2d  # (out_ch*kH*kW, in_ch or in_ch+pad)
            W_dequant, scales, zeros = gptq_quantize_weight(
                W_2d.float(), H, num_bits, min(group_size, W_2d.shape[1]), block_size, percdamp
            )
            if rotated:
                replacement = HadamardGPTQConvTranspose2d(
                    weight_shape=tuple(orig_layer.weight.shape),
                    stride=orig_layer.stride,
                    padding=orig_layer.padding,
                    output_padding=orig_layer.output_padding,
                    dilation=orig_layer.dilation,
                    groups=orig_layer.groups,
                    W_rot_q=W_dequant,
                    seed=seed,
                    pad=pad,
                    scales=scales,
                    zero_points=zeros,
                    bias=orig_layer.bias.data if orig_layer.bias is not None else None,
                )
            else:
                # W_dequant is (out_ch*kH*kW, in_ch) — transpose to (in_ch, out_ch*kH*kW)
                # then reshape to (in_ch, out_ch, kH, kW) for GPTQConvTranspose2d.
                W_dequant_shaped = W_dequant.T.contiguous().reshape(orig_layer.weight.shape)
                replacement = GPTQConvTranspose2d(
                    weight_shape=tuple(orig_layer.weight.shape),
                    stride=orig_layer.stride,
                    padding=orig_layer.padding,
                    output_padding=orig_layer.output_padding,
                    dilation=orig_layer.dilation,
                    groups=orig_layer.groups,
                    W_dequant=W_dequant_shaped,
                    scales=scales,
                    zero_points=zeros,
                    bias=orig_layer.bias.data if orig_layer.bias is not None else None,
                )

        else:  # Linear
            W_2d = proxy.weight_rot  # (out, d_in or d_in_padded)
            W_dequant, scales, zeros = gptq_quantize_weight(
                W_2d.float(), H, num_bits, min(group_size, W_2d.shape[1]), block_size, percdamp
            )
            if rotated:
                replacement = HadamardGPTQLinear(
                    in_features=orig_layer.in_features,
                    out_features=orig_layer.out_features,
                    W_rot_q=W_dequant,
                    seed=seed,
                    pad=pad,
                    scales=scales,
                    zero_points=zeros,
                    bias=orig_layer.bias.data if orig_layer.bias is not None else None,
                )
            else:
                # Large layer — wrap with pure GPTQ (no rotation)
                replacement = GPTQLinear(
                    in_features=orig_layer.in_features,
                    out_features=orig_layer.out_features,
                    W_dequant=W_dequant,
                    scales=scales,
                    zero_points=zeros,
                    bias=orig_layer.bias.data if orig_layer.bias is not None else None,
                )

        setattr(parent, child_name, replacement)
        del H

        label = "H+GPTQ" if rotated else "GPTQ  "
        if verbose:
            print(
                f"  [{label}] {full_name:48s} {list(orig_layer.weight.shape)} -> {num_bits}-bit"
            )

    return model
