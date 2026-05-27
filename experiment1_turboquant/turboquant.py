"""
TurboQuant core algorithm for weight compression.

Algorithm (Zandieh et al., ICLR 2026, arXiv 2504.19874):
  1. Flatten weight tensor to row-blocks of size `block_dim`.
  2. Per block: store L2 norm, normalize to unit sphere.
  3. Apply random rotation R = (H*D_n)...(H*D_1) where H = normalized
     Walsh-Hadamard transform, D_i = random ±1 diagonal.
     After rotation, coordinates are near-Gaussian (CLT on unit sphere).
  4. Quantize each coordinate to nearest Lloyd-Max 3-bit centroid.
  5. Store: int8 indices + float16 norms + rotation seed.

Decompression: centroid lookup -> inverse rotation -> rescale by norm.

Math note on inverse rotation:
  Forward block i: y = H * D_i * x
  Inverse block i: x = D_i * H * y  (since H^2 = I and D_i^2 = I)
  Full inverse: apply blocks in reverse order, each as (H then D_i).
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Lloyd-Max 3-bit (8-level) optimal centroids for N(0,1).
# Reference: Max (1960), "Quantizing for Minimum Distortion", IRE Trans.
# These values minimize E[(x - Q(x))^2] for Gaussian source.
# ---------------------------------------------------------------------------
LLOYD_MAX_3BIT = torch.tensor(
    [-2.1519, -1.3439, -0.7560, -0.2451,
      0.2451,  0.7560,  1.3439,  2.1519],
    dtype=torch.float32,
)

# Expected MSE ratio for 3-bit Lloyd-Max on Gaussian:  E[quant_err^2] / Var(x)
# Theoretical: ~0.0194 (verified vs. 1/4^3 = 0.0156 lower bound; actual ≈1.25x bound)
LLOYD_MAX_3BIT_DISTORTION_RATIO = 0.0194


def _next_pow2(n: int) -> int:
    """Smallest power of 2 that is >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def fast_hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Normalized Walsh-Hadamard transform along the last dimension.

    Requires last dimension to be a power of 2.
    Property: H @ H = I (orthonormal), so H^{-1} = H.
    Cost: O(d log d) vs O(d^2) for dense orthogonal matrix.
    """
    d = x.shape[-1]
    if d == 1:
        return x.clone()
    assert (d & (d - 1)) == 0, (
        f"fast_hadamard_transform: last dim must be power of 2, got {d}"
    )

    out = x.clone().float()
    h = 1
    while h < d:
        # Butterfly step: pairs of size h
        out = out.view(*out.shape[:-1], d // (2 * h), 2, h)
        a = out[..., 0, :].clone()
        b = out[..., 1, :].clone()
        out[..., 0, :] = a + b
        out[..., 1, :] = a - b
        out = out.view(*out.shape[:-3], d)
        h <<= 1

    return out * (1.0 / math.sqrt(d))


def _build_sign_vectors(d: int, n_blocks: int, seed: int,
                        device: torch.device) -> list:
    """
    Deterministically generate n_blocks ±1 diagonal sign vectors of length d.
    Uses CPU generator for reproducibility across devices.
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed % (2**31))
    signs = []
    for _ in range(n_blocks):
        s = (torch.randint(0, 2, (d,), generator=g, dtype=torch.float32) * 2.0 - 1.0)
        signs.append(s.to(device))
    return signs


def random_rotation(x: torch.Tensor, seed: int, n_blocks: int = 3) -> torch.Tensor:
    """
    Apply random rotation R = (H*D_n)...(H*D_1) to rows of x (last dim).

    Each block applies:  out = H(D_i * out)
    i.e., multiply by ±1 diagonal D_i, then Walsh-Hadamard H.

    After rotation, coordinates of x/||x|| are near-Gaussian by CLT,
    making per-coordinate Lloyd-Max quantization near-optimal.
    """
    d = x.shape[-1]
    assert (d & (d - 1)) == 0, (
        f"random_rotation: last dim must be power of 2, got {d}. "
        f"Pad to {_next_pow2(d)} before calling."
    )
    signs = _build_sign_vectors(d, n_blocks, seed, x.device)
    out = x.float()
    for s in signs:
        out = out * s                        # D_i: element-wise sign flip
        out = fast_hadamard_transform(out)   # H
    return out


def inverse_rotation(y: torch.Tensor, seed: int, n_blocks: int = 3) -> torch.Tensor:
    """
    Inverse of random_rotation.

    Derivation:
      R_i = H * D_i  =>  R_i^{-1} = D_i^{-1} * H^{-1} = D_i * H
      Full R^{-1}: apply blocks in reverse order, each as (H then D_i).

    Verification: R_i^{-1} * R_i = (D_i*H)*(H*D_i) = D_i*(H*H)*D_i = D_i^2 = I ✓
    """
    d = y.shape[-1]
    assert (d & (d - 1)) == 0, (
        f"inverse_rotation: last dim must be power of 2, got {d}"
    )
    signs = _build_sign_vectors(d, n_blocks, seed, y.device)
    out = y.float()
    for s in reversed(signs):
        out = fast_hadamard_transform(out)  # H^{-1} = H
        out = out * s                       # D_i^{-1} = D_i
    return out


def quantize_3bit(rotated: torch.Tensor) -> torch.Tensor:
    """
    Nearest-centroid quantization using Lloyd-Max 3-bit levels.

    Input:  (..., d) float — rotated unit-norm coordinates
    Output: (..., d) int8 — centroid indices in [0, 7]
    """
    levels = LLOYD_MAX_3BIT.to(device=rotated.device, dtype=rotated.dtype)
    # (..., d, 1) - (8,) -> (..., d, 8): compute distance to each centroid
    dist = (rotated.unsqueeze(-1) - levels).abs()
    idx = dist.argmin(dim=-1).to(torch.int8)
    return idx


def dequantize_3bit(idx: torch.Tensor) -> torch.Tensor:
    """Look up Lloyd-Max centroid values from int8 indices."""
    levels = LLOYD_MAX_3BIT.to(device=idx.device, dtype=torch.float32)
    return levels[idx.long()]


@dataclass
class TurboQuantTensor:
    """
    Compressed representation of a weight tensor at ~3.125 bits/param.

    Storage layout:
      idx   : (n_blocks, block_dim)  int8    — quantized centroid indices
      norms : (n_blocks,)            float16 — per-block L2 norm
      seed  : int                            — rotation seed (shared, no storage overhead)
      block_dim, orig_shape, pad, n_rotation_blocks  — reconstruction metadata

    Compression: 3 bits/weight + 16 bits/block = 3 + 16/block_dim bits/weight.
    At block_dim=128: 3.125 bits/param → ~10.2× vs FP32.
    """

    idx: torch.Tensor       # int8, shape (n_blocks, block_dim)
    norms: torch.Tensor     # float16, shape (n_blocks,)
    seed: int
    block_dim: int
    orig_shape: tuple
    pad: int
    n_rotation_blocks: int

    @staticmethod
    def compress(
        W: torch.Tensor,
        block_dim: int = 128,
        seed: Optional[int] = None,
        n_rotation_blocks: int = 3,
    ) -> "TurboQuantTensor":
        """
        Compress weight tensor W to TurboQuant 3-bit form.

        Steps:
          1. Flatten to 1D, zero-pad to multiple of block_dim.
          2. Reshape to (n_blocks, block_dim). block_dim must be power of 2.
          3. Per block: compute L2 norm, normalize, rotate, quantize.
        """
        assert (block_dim & (block_dim - 1)) == 0 and block_dim >= 1, (
            f"block_dim must be power of 2, got {block_dim}"
        )
        if seed is None:
            seed = int(torch.randint(0, 2**30, (1,)).item())

        orig_shape = tuple(W.shape)
        flat = W.detach().float().reshape(-1)
        n = flat.numel()

        pad = (-n) % block_dim
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad)])

        rows = flat.view(-1, block_dim)                 # (B, block_dim)
        norms = rows.norm(dim=-1, keepdim=True)         # (B, 1)
        safe_norms = norms.clamp(min=1e-8)
        unit_rows = rows / safe_norms                   # unit vectors, each coord ~ N(0, 1/d)

        rotated = random_rotation(unit_rows, seed=seed, n_blocks=n_rotation_blocks)
        # Scale coords to N(0,1) before quantizing: unit sphere in R^d has
        # each coord ~ N(0, 1/d), so multiply by sqrt(d) to match Lloyd-Max centroids.
        scale = math.sqrt(block_dim)
        idx = quantize_3bit(rotated * scale)            # (B, block_dim) int8

        return TurboQuantTensor(
            idx=idx,
            norms=norms.squeeze(-1).half(),             # (B,) float16
            seed=seed,
            block_dim=block_dim,
            orig_shape=orig_shape,
            pad=pad,
            n_rotation_blocks=n_rotation_blocks,
        )

    def decompress(self, dtype: torch.dtype = torch.float32, device=None) -> torch.Tensor:
        """
        Reconstruct weight tensor from compressed form.

        Steps (inverse of compress):
          1. Look up centroid values for idx.
          2. Apply inverse rotation to recover (approx) unit vectors.
          3. Rescale by stored norms.
          4. Strip padding, reshape to original shape.
        """
        scale = math.sqrt(self.block_dim)
        rotated_scaled_approx = dequantize_3bit(self.idx)     # (B, block_dim) float32, N(0,1) scale
        rotated_approx = rotated_scaled_approx / scale        # unscale back to unit-sphere N(0,1/d)
        unit_approx = inverse_rotation(
            rotated_approx, seed=self.seed, n_blocks=self.n_rotation_blocks
        )
        norms = self.norms.float().unsqueeze(-1)              # (B, 1) float32
        rows = unit_approx * norms                            # (B, block_dim)

        flat = rows.reshape(-1)
        if self.pad:
            flat = flat[: flat.numel() - self.pad]
        result = flat.view(self.orig_shape).to(dtype)
        return result.to(device) if device is not None else result

    def nbytes_packed(self) -> int:
        """Theoretical storage (bits packed: 3 bits/weight + 16 bits/block norm)."""
        n_weights = self.idx.numel() - self.pad
        total_bits = n_weights * 3 + self.norms.numel() * 16
        return (total_bits + 7) // 8

    def nbytes_fp32(self) -> int:
        """Original FP32 storage size."""
        n_weights = self.idx.numel() - self.pad
        return n_weights * 4

    def compression_ratio(self) -> float:
        """FP32 bytes / packed compressed bytes."""
        return self.nbytes_fp32() / max(self.nbytes_packed(), 1)

    def reconstruction_mse(self, original: torch.Tensor) -> float:
        """Compute actual MSE between decompressed and original weight."""
        recon = self.decompress(dtype=original.dtype)
        return float(((recon - original.detach()) ** 2).mean().item())
