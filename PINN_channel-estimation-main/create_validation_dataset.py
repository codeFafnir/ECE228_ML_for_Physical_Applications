"""
Extract the validation split from full SNR datasets and save standalone files.

Split matches train.py / create_datasets:
  train_ratio=0.8, val_ratio=0.1, random_seed=42

Output (default: validation/snr0/):
  initial_estimate_ls_val.npy   — SMOMP channels, val samples only
  3D_channel_val.npy            — ground-truth channels, val samples only
  ue_positions_val.txt          — UE positions (one row per val sample)
  val_indices.npy               — original indices into the full dataset
  manifest.json                 — split params + normalization constants

Usage (from PINN_channel-estimation-main/):
  python create_validation_dataset.py
  python create_validation_dataset.py --snr 0 --output_dir validation/snr0
"""

import argparse
import json
from pathlib import Path

import numpy as np


TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
RANDOM_SEED = 42


def _load_user_positions(path: Path) -> list[tuple[float, float, float]]:
    positions: list[tuple[float, float, float]] = []
    with open(path, "r") as f:
        for line in f.readlines()[1:]:
            if line.strip():
                x, y, z = map(float, line.strip().split())
                positions.append((x, y, z))
    return positions


def _global_norm_max(smomp: np.ndarray, accurate: np.ndarray, chunk: int = 128) -> tuple[float, float, float]:
    smomp_max = 0.0
    accurate_max = 0.0
    n = smomp.shape[0]
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sr = np.real(smomp[start:end])
        si = np.imag(smomp[start:end])
        smomp_r = np.concatenate([sr, si], axis=1)
        ar = np.real(accurate[start:end])
        ai = np.imag(accurate[start:end])
        accurate_r = np.concatenate([ar, ai], axis=1)
        smomp_max = max(smomp_max, float(np.max(np.abs(smomp_r))))
        accurate_max = max(accurate_max, float(np.max(np.abs(accurate_r))))
    norm_max = max(smomp_max, accurate_max)
    return smomp_max, accurate_max, norm_max


def _val_indices(n_samples: int) -> np.ndarray:
    rng = np.random.RandomState(RANDOM_SEED)
    perm = rng.permutation(n_samples)
    n_train = int(n_samples * TRAIN_RATIO)
    n_val = int(n_samples * VAL_RATIO)
    return perm[n_train : n_train + n_val]


def create_validation_dataset(
    smomp_file: Path,
    accurate_file: Path,
    user_positions_file: Path,
    output_dir: Path,
    snr_db: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {smomp_file.name} (mmap) ...")
    smomp_full = np.load(smomp_file, mmap_mode="r")
    accurate_full = np.load(accurate_file, mmap_mode="r")
    assert smomp_full.shape == accurate_full.shape

    n_samples = smomp_full.shape[0]
    val_idx = _val_indices(n_samples)
    print(f"Val split: {len(val_idx)} samples (train_ratio={TRAIN_RATIO}, val_ratio={VAL_RATIO})")

    print("Computing global normalization (full dataset) ...")
    smomp_max, accurate_max, norm_max = _global_norm_max(smomp_full, accurate_full)

    print("Writing val channel arrays ...")
    smomp_val = np.array(smomp_full[val_idx])
    accurate_val = np.array(accurate_full[val_idx])
    np.save(output_dir / "initial_estimate_ls_val.npy", smomp_val)
    np.save(output_dir / "3D_channel_val.npy", accurate_val)
    np.save(output_dir / "val_indices.npy", val_idx)

    all_positions = _load_user_positions(user_positions_file)
    print("Writing per-sample UE positions ...")
    with open(output_dir / "ue_positions_val.txt", "w") as f:
        f.write("x y z source_index\n")
        for i, real_idx in enumerate(val_idx):
            pos = all_positions[int(real_idx) % len(all_positions)]
            f.write(f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {int(real_idx)}\n")

    manifest = {
        "snr_db": snr_db,
        "split": "val",
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "random_seed": RANDOM_SEED,
        "n_val_samples": int(len(val_idx)),
        "n_total_samples": int(n_samples),
        "norm_max": norm_max,
        "smomp_max": smomp_max,
        "accurate_max": accurate_max,
        "source_smomp": smomp_file.name,
        "source_accurate": accurate_file.name,
        "source_user_positions": user_positions_file.name,
        "rss_image_path": "Dataset/50_15GHz.jpg",
        "channel_shape": list(smomp_val.shape[1:]),
        "dtype": "complex128",
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    smomp_mb = smomp_val.nbytes / 1e6
    print(f"Done. Saved to {output_dir}/")
    print(f"  initial_estimate_ls_val.npy  ({smomp_mb:.1f} MB)")
    print(f"  3D_channel_val.npy         ({accurate_val.nbytes / 1e6:.1f} MB)")
    print(f"  ue_positions_val.txt         ({len(val_idx)} rows)")
    print(f"  manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create standalone validation dataset files")
    parser.add_argument("--snr", type=int, default=0)
    parser.add_argument(
        "--smomp_file", type=str, default="initial_estimate_ls_snr0.npy",
    )
    parser.add_argument(
        "--accurate_file", type=str, default="3D_channel_15GHz_2x2_Pt50.npy",
    )
    parser.add_argument(
        "--user_positions_file", type=str, default="ue_positions_noisy.txt",
    )
    parser.add_argument(
        "--output_dir", type=str, default="validation/snr0",
    )
    args = parser.parse_args()

    root = Path(__file__).parent
    create_validation_dataset(
        smomp_file=root / args.smomp_file,
        accurate_file=root / args.accurate_file,
        user_positions_file=root / args.user_positions_file,
        output_dir=root / args.output_dir,
        snr_db=args.snr,
    )


if __name__ == "__main__":
    main()
