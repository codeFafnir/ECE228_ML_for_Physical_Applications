"""
Pack / unpack GPTQ weight matrices for real storage compression.

4-bit: two weights per uint8 (nibble packing).
8-bit: one weight per uint8.
Scales stored as float16; zero_points as uint8.
"""

import torch
import torch.nn.functional as F


def _expand_group_params(
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    d_in: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    g = min(group_size, d_in)
    d_out, n_groups = scales.shape
    s = scales.unsqueeze(-1).expand(d_out, n_groups, g).reshape(d_out, n_groups * g)
    z = zero_points.unsqueeze(-1).expand(d_out, n_groups, g).reshape(d_out, n_groups * g)
    return s[:, :d_in], z[:, :d_in]


def float_weights_to_qcodes(
    W: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    group_size: int,
    num_bits: int,
) -> torch.Tensor:
    """(d_out, d_in) float32 -> integer codes in [0, 2**num_bits - 1]."""
    d_out, d_in = W.shape
    s_col, z_col = _expand_group_params(scales, zero_points, d_in, group_size)
    q_max = 2 ** num_bits - 1
    q = torch.round(W / s_col.clamp(min=1e-8) + z_col).clamp(0, q_max)
    return q.to(torch.uint8)


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """(d_out, d_in) uint8 codes -> (d_out, ceil(d_in/2)) packed uint8."""
    d_out, d_in = q.shape
    pad = d_in % 2
    if pad:
        q = F.pad(q, (0, 1))
    pairs = q.view(d_out, -1, 2)
    packed = (pairs[:, :, 0] & 0xF) | ((pairs[:, :, 1] & 0xF) << 4)
    return packed.to(torch.uint8)


def unpack_int4(packed: torch.Tensor, d_in: int) -> torch.Tensor:
    """(d_out, ceil(d_in/2)) -> (d_out, d_in) uint8 codes."""
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    q = torch.stack([low, high], dim=-1).reshape(packed.shape[0], -1)
    return q[:, :d_in]


def pack_int8(q: torch.Tensor) -> torch.Tensor:
    """(d_out, d_in) uint8 codes — stored as-is."""
    return q.contiguous().to(torch.uint8)


def unpack_int8(qweight: torch.Tensor, d_in: int) -> torch.Tensor:
    return qweight[:, :d_in]


def unpack_qcodes(qweight: torch.Tensor, d_in: int, num_bits: int) -> torch.Tensor:
    if num_bits == 4:
        return unpack_int4(qweight, d_in)
    if num_bits == 8:
        return unpack_int8(qweight, d_in)
    raise ValueError(f"unsupported num_bits={num_bits}")


def dequant_packed_weight(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    d_out: int,
    d_in: int,
    group_size: int,
    num_bits: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Reconstruct (d_out, d_in) float weight from packed storage."""
    if device is None:
        device = qweight.device
    q = unpack_qcodes(qweight.to(device), d_in, num_bits).float()
    s_col, z_col = _expand_group_params(
        scales.to(device).float(),
        zero_points.to(device).float(),
        d_in,
        group_size,
    )
    return ((q - z_col) * s_col).to(dtype)


def pack_gptq_tensors(
    W_dequant: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    group_size: int,
    num_bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert dequantized GPTQ result to compact storage tensors.

    Returns:
        qweight:     uint8 packed weights
        scales:      float16 (d_out, n_groups)
        zero_points: uint8 (d_out, n_groups)
    """
    if num_bits not in (4, 8):
        raise ValueError(f"packed storage supports num_bits in {{4, 8}}, got {num_bits}")
    q = float_weights_to_qcodes(W_dequant.cpu(), scales.cpu(), zero_points.cpu(), group_size, num_bits)
    qweight = pack_int4(q) if num_bits == 4 else pack_int8(q)
    return (
        qweight,
        scales.cpu().half(),
        zero_points.cpu().to(torch.uint8),
    )


def packed_weight_nbytes(
    d_out: int,
    d_in: int,
    n_groups: int,
    num_bits: int = 4,
) -> int:
    """Theoretical packed byte count for one weight matrix."""
    if num_bits == 4:
        q_bytes = d_out * ((d_in + 1) // 2)
    elif num_bits == 8:
        q_bytes = d_out * d_in
    else:
        q_bytes = (d_out * d_in * num_bits + 7) // 8
    scale_bytes = d_out * n_groups * 2  # fp16
    zero_bytes = d_out * n_groups       # uint8
    return q_bytes + scale_bytes + zero_bytes
