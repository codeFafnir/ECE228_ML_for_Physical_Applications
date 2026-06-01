#!/usr/bin/env python3
"""Generate a synthetic ue_positions_noisy.txt file for PINN training.

This is a fallback when real Wireless InSite UE positions are unavailable.
The output format matches Model.py expectations:
  - 1 header line
  - whitespace-separated columns: x y z
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def infer_num_samples(reference_npy: Path | None, n_samples: int | None) -> int:
    if reference_npy is not None:
        if not reference_npy.exists():
            raise FileNotFoundError(f"Reference npy not found: {reference_npy}")
        arr = np.load(reference_npy, mmap_mode="r")
        if arr.ndim < 1:
            raise ValueError(f"Unexpected npy shape: {arr.shape}")
        return int(arr.shape[0])

    if n_samples is not None:
        if n_samples <= 0:
            raise ValueError("--n-samples must be a positive integer")
        return n_samples

    raise ValueError("Provide --reference-npy or --n-samples to set sample count")


def compute_map_bounds(
    image_path: Path,
    bs_pixel_x: float,
    bs_pixel_y: float,
    bs_real_x: float,
    bs_real_y: float,
    image_width_meters: float,
) -> tuple[float, float, float, float]:
    if not image_path.exists():
        raise FileNotFoundError(f"RSS image not found: {image_path}")

    with Image.open(image_path) as img:
        width, height = img.size

    scale = image_width_meters / float(width)

    # Same coordinate transform used by RSSMapProcessor in find_in_map.py
    origin_x = bs_real_x + bs_pixel_x * scale
    origin_y = bs_real_y - bs_pixel_y * scale

    x_min = origin_x - (width - 1) * scale
    x_max = origin_x
    y_min = origin_y
    y_max = origin_y + (height - 1) * scale
    return x_min, x_max, y_min, y_max


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic UE position file")
    parser.add_argument("--output", default="ue_positions_noisy.txt", help="Output txt path")

    parser.add_argument(
        "--reference-npy",
        default="3D_channel_15GHz_2x2_Pt50.npy",
        help="Numpy file used to infer number of snapshots",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Explicit number of snapshots (use this when reference npy is not available)",
    )

    parser.add_argument("--rss-image", default="Dataset/50_15GHz.jpg", help="RSS map image path")
    parser.add_argument("--bs-pixel-x", type=float, default=287.0)
    parser.add_argument("--bs-pixel-y", type=float, default=293.0)
    parser.add_argument("--bs-real-x", type=float, default=71.06)
    parser.add_argument("--bs-real-y", type=float, default=246.29)
    parser.add_argument("--image-width-meters", type=float, default=527.5)

    parser.add_argument("--z-height", type=float, default=1.5, help="Constant z value in meters")
    parser.add_argument(
        "--noise-std",
        type=float,
        default=3.0,
        help="Gaussian std (m) added to x and y; 3.0 m gives variance 9 like the paper",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_path = Path(args.output)
    reference_npy = Path(args.reference_npy) if args.reference_npy else None
    rss_image = Path(args.rss_image)

    n_samples = infer_num_samples(reference_npy, args.n_samples)
    x_min, x_max, y_min, y_max = compute_map_bounds(
        image_path=rss_image,
        bs_pixel_x=args.bs_pixel_x,
        bs_pixel_y=args.bs_pixel_y,
        bs_real_x=args.bs_real_x,
        bs_real_y=args.bs_real_y,
        image_width_meters=args.image_width_meters,
    )

    rng = np.random.default_rng(args.seed)

    # Uniformly sample positions over the map, then add Gaussian noise.
    x = rng.uniform(x_min, x_max, size=n_samples)
    y = rng.uniform(y_min, y_max, size=n_samples)
    x += rng.normal(0.0, args.noise_std, size=n_samples)
    y += rng.normal(0.0, args.noise_std, size=n_samples)

    # Keep coordinates inside map bounds to avoid heavy clipping in dataloader.
    x = np.clip(x, x_min, x_max)
    y = np.clip(y, y_min, y_max)
    z = np.full(n_samples, args.z_height)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("x y z\n")
        for xi, yi, zi in zip(x, y, z):
            f.write(f"{xi:.6f} {yi:.6f} {zi:.6f}\n")

    print(f"Wrote {n_samples} synthetic UE positions to {out_path}")
    print(f"x range: [{x.min():.3f}, {x.max():.3f}], y range: [{y.min():.3f}, {y.max():.3f}], z={args.z_height}")


if __name__ == "__main__":
    main()