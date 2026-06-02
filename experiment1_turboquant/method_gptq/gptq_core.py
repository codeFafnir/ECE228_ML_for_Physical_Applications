"""
GPTQ core algorithm (Frantar et al., NeurIPS 2022).

References:
  - Paper: https://arxiv.org/abs/2210.17323
  - Official repo: https://github.com/IST-DASLab/gptq

This module is model-agnostic: it operates on a single weight matrix W
and its pre-accumulated Hessian H = (2/N) * sum_i x_i x_i^T.

Public API:
  gptq_quantize_weight(W, H, num_bits, group_size, block_size, percdamp)
    -> (W_dequant, scales, zero_points)

  quantize_uniform(w, scale, zero_point, num_bits) -> w_q (dequantized)
  compute_scales_and_zeros(w_col, num_bits, group_size) -> (scales, zeros)
"""

import torch
import torch.nn.functional as F


def quantize_uniform(
    w: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    num_bits: int,
) -> torch.Tensor:
    """
    Asymmetric uniform quantization + immediate dequantization.

    w:          (...,) float32 weight values
    scale:      (...,) per-group scale (broadcast-compatible with w)
    zero_point: (...,) per-group zero point (integer offset)
    Returns:    (...,) float32 dequantized weights
    """
    q_min = 0
    q_max = 2 ** num_bits - 1
    w_int = torch.clamp(torch.round(w / scale.clamp(min=1e-8) + zero_point), q_min, q_max)
    return (w_int - zero_point) * scale


def compute_scales_and_zeros(
    w_block: torch.Tensor,
    num_bits: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-group asymmetric uniform quantization parameters.

    w_block: (d_out, d_in) weight block
    Returns: scales (d_out, n_groups), zero_points (d_out, n_groups)

    If d_in < group_size, use a single group spanning all columns.
    """
    d_out, d_in = w_block.shape
    g = min(group_size, d_in)
    # Pad d_in to multiple of g
    pad = (-d_in) % g
    if pad:
        w_block = F.pad(w_block, (0, pad))
    n_groups = w_block.shape[1] // g

    # (d_out, n_groups, g)
    w_grouped = w_block.view(d_out, n_groups, g)
    w_min = w_grouped.amin(dim=-1)   # (d_out, n_groups)
    w_max = w_grouped.amax(dim=-1)

    q_max = 2 ** num_bits - 1
    scales = (w_max - w_min).clamp(min=1e-8) / q_max  # (d_out, n_groups)
    zero_points = torch.clamp(torch.round(-w_min / scales.clamp(min=1e-8)), 0, q_max)

    return scales, zero_points


def _expand_group_params(
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    d_in: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Expand per-group (d_out, n_groups) params to per-column (d_out, d_in).
    """
    g = min(group_size, d_in)
    # (d_out, n_groups) -> (d_out, n_groups, 1) -> (d_out, n_groups*g)
    d_out, n_groups = scales.shape
    s_expanded = scales.unsqueeze(-1).expand(d_out, n_groups, g).reshape(d_out, n_groups * g)
    z_expanded = zero_points.unsqueeze(-1).expand(d_out, n_groups, g).reshape(d_out, n_groups * g)
    # Trim to actual d_in (handles padding)
    return s_expanded[:, :d_in], z_expanded[:, :d_in]


def uniform_quantize_weight(
    W: torch.Tensor,
    num_bits: int = 3,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group-wise uniform quant without Hessian (fallback for very wide layers)."""
    W = W.float().clone().cpu()
    scales, zeros = compute_scales_and_zeros(W, num_bits, group_size)
    s_col, z_col = _expand_group_params(scales, zeros, W.shape[1], group_size)
    W_dequant = quantize_uniform(W, s_col, z_col, num_bits)
    return W_dequant, scales, zeros


def gptq_quantize_weight(
    W: torch.Tensor,
    H: torch.Tensor,
    num_bits: int = 3,
    group_size: int = 128,
    block_size: int = 128,
    percdamp: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPTQ: quantize weight matrix W using second-order Hessian H.

    Args:
        W:          (d_out, d_in) weight matrix, float32
        H:          (d_in, d_in) Hessian = (2/N) sum_i x_i x_i^T, float32
        num_bits:   Target bit-width (3 or 4 recommended)
        group_size: Number of columns per quantization group
        block_size: Number of columns processed per GPTQ block (lazy batching)
        percdamp:   Damping fraction of mean diagonal H to prevent singularity

    Returns:
        W_dequant:    (d_out, d_in) dequantized float32 weight (ready for inference)
        scales:       (d_out, n_groups) per-group scales
        zero_points:  (d_out, n_groups) per-group zero points
    """
    W = W.float().clone().cpu()
    H = H.float().clone().cpu()
    d_out, d_in = W.shape

    # 1. Dampen Hessian diagonal to avoid singularity
    damp = percdamp * torch.diag(H).mean()
    H.diagonal().add_(damp)

    # 2. Compute Cholesky of H^{-1} on CPU (MPS lacks linalg.inv/cholesky)
    try:
        H_inv = torch.linalg.inv(H)
        H_inv = 0.5 * (H_inv + H_inv.T)
        H_inv.diagonal().clamp_(min=1e-8)
        Hinv_chol = torch.linalg.cholesky(H_inv, upper=True)
    except torch.linalg.LinAlgError:
        H.diagonal().add_(damp * 10)
        H_inv = torch.linalg.inv(H)
        H_inv = 0.5 * (H_inv + H_inv.T)
        H_inv.diagonal().clamp_(min=1e-8)
        Hinv_chol = torch.linalg.cholesky(H_inv, upper=True)

    # Pre-compute per-group quantization parameters over the full W
    # (used as reference; column-level scales recomputed per block below)
    scales_full, zeros_full = compute_scales_and_zeros(W, num_bits, group_size)

    W_dequant = W.clone()  # accumulate dequantized result
    g = min(group_size, d_in)

    # 3. Process columns in blocks of block_size
    for block_start in range(0, d_in, block_size):
        block_end = min(block_start + block_size, d_in)
        W_block = W[:, block_start:block_end].clone()       # (d_out, bs)
        Hinv_block = Hinv_chol[block_start:block_end, block_start:block_end]  # (bs, bs)

        errors = torch.zeros_like(W_block)

        for j in range(block_end - block_start):
            col_global = block_start + j  # global column index
            w_col = W_block[:, j]         # (d_out,)

            # Per-group scale and zero for this column
            group_idx = col_global // g
            # Clamp to valid group range (handles last partial group)
            group_idx = min(group_idx, scales_full.shape[1] - 1)
            s = scales_full[:, group_idx]   # (d_out,)
            z = zeros_full[:, group_idx]    # (d_out,)

            # Quantize column j
            w_q = quantize_uniform(w_col, s, z, num_bits)

            W_dequant[:, col_global] = w_q
            errors[:, j] = (w_col - w_q) / Hinv_block[j, j].clamp(min=1e-8)

            # Propagate error within the current block
            if j + 1 < block_end - block_start:
                W_block[:, j + 1:] -= (
                    errors[:, j].unsqueeze(1) *
                    Hinv_block[j, j + 1:].unsqueeze(0)
                )

        # Propagate block errors to remaining columns globally
        if block_end < d_in:
            W[:, block_end:] -= errors @ Hinv_chol[block_start:block_end, block_end:]

    return W_dequant, scales_full, zeros_full
