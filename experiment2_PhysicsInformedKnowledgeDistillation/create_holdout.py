"""
Extract a true holdout set (never seen during training) into a self-contained
directory that can be uploaded to Google Drive.

The full dataset is split with RandomState(42):
  train  = indices[:n_train]          (80 %)
  val    = indices[n_train:n_train+n_val]  (10 %) — used as KD val during training
  test   = indices[n_train+n_val:]    (10 %) — TRUE holdout

We take the first --n_holdout samples from the test split (default 790,
which is ~10 % of training set size 7901).

Outputs written to --output_dir:
  holdout_smomp.npy          complex128, shape (N, 16, 4, 576)
  holdout_accurate.npy       complex128, shape (N, 16, 4, 576)
  holdout_ue_positions.txt   N lines of "x y z", one per sample
  holdout_manifest.json      metadata needed by eval_holdout.py

Usage (run from experiment2_PhysicsInformedKnowledgeDistillation/):
    python create_holdout.py

Or with explicit paths:
    python create_holdout.py \\
        --smomp_file    ../PINN_channel-estimation-main/initial_estimate_ls_snr0.npy \\
        --accurate_file ../PINN_channel-estimation-main/3D_channel_15GHz_2x2_Pt50.npy \\
        --user_positions ../PINN_channel-estimation-main/ue_positions_noisy.txt \\
        --output_dir holdout \\
        --n_holdout 790
"""

import argparse
import json
from pathlib import Path

import numpy as np

_EXP2_DIR = Path(__file__).parent
_PINN_DIR = _EXP2_DIR.parent / "PINN_channel-estimation-main"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.10
RANDOM_SEED = 42


def _load_positions(path: str) -> list[tuple[float, float, float]]:
    """Return (x, y, z) tuples from ue_positions_noisy.txt (skip header)."""
    positions: list[tuple[float, float, float]] = []
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines[1:]:
        parts = line.strip().split()
        if parts:
            positions.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return positions


def _compute_norm_max(smomp: np.ndarray, accurate: np.ndarray, chunk: int = 256) -> float:
    """Streaming max-abs without materializing full arrays."""
    global_max = 0.0
    n = smomp.shape[0]
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sm = smomp[start:end]
        ac = accurate[start:end]
        sm_real = np.concatenate([np.real(sm), np.imag(sm)], axis=1).astype(np.float32)
        ac_real = np.concatenate([np.real(ac), np.imag(ac)], axis=1).astype(np.float32)
        global_max = max(global_max, float(np.max(np.abs(sm_real))), float(np.max(np.abs(ac_real))))
    return global_max


def main() -> None:
    parser = argparse.ArgumentParser(description="Create holdout validation bundle")
    parser.add_argument(
        "--smomp_file",
        default=str(_PINN_DIR / "initial_estimate_ls_snr0.npy"),
    )
    parser.add_argument(
        "--accurate_file",
        default=str(_PINN_DIR / "3D_channel_15GHz_2x2_Pt50.npy"),
    )
    parser.add_argument(
        "--user_positions",
        default=str(_PINN_DIR / "ue_positions_noisy.txt"),
    )
    parser.add_argument("--output_dir", default="holdout")
    parser.add_argument(
        "--n_holdout",
        type=int,
        default=790,
        help="Number of holdout samples (default 790 ≈ 10%% of training size)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading smomp  : {args.smomp_file}")
    smomp_mmap = np.load(args.smomp_file, mmap_mode="r")
    print(f"Loading accurate: {args.accurate_file}")
    accurate_mmap = np.load(args.accurate_file, mmap_mode="r")

    n_total = smomp_mmap.shape[0]
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    rng = np.random.RandomState(RANDOM_SEED)
    all_indices = rng.permutation(n_total)

    test_indices = all_indices[n_train + n_val:]
    n_holdout = min(args.n_holdout, len(test_indices))
    holdout_indices = test_indices[:n_holdout]

    print(
        f"\nDataset split  (seed={RANDOM_SEED}): "
        f"total={n_total}, train={n_train}, val={n_val}, test={len(test_indices)}"
    )
    print(f"Holdout size   : {n_holdout} samples (from test split)")

    # ── Extract channel arrays ───────────────────────────────────────────────
    print("\nExtracting smomp channels ...")
    smomp_out = smomp_mmap[holdout_indices]
    print("Extracting accurate channels ...")
    accurate_out = accurate_mmap[holdout_indices]

    smomp_path = out_dir / "holdout_smomp.npy"
    accurate_path = out_dir / "holdout_accurate.npy"
    print(f"Saving {smomp_path} ...")
    np.save(smomp_path, smomp_out)
    print(f"Saving {accurate_path} ...")
    np.save(accurate_path, accurate_out)

    # ── Extract UE positions ─────────────────────────────────────────────────
    print(f"\nLoading positions: {args.user_positions}")
    all_positions = _load_positions(args.user_positions)
    n_pos = len(all_positions)

    pos_path = out_dir / "holdout_ue_positions.txt"
    with open(pos_path, "w", encoding="utf-8") as f:
        f.write("x y z\n")
        for orig_idx in holdout_indices:
            px, py, pz = all_positions[int(orig_idx) % n_pos]
            f.write(f"{px} {py} {pz}\n")
    print(f"Saved positions : {pos_path}  ({n_holdout} rows)")

    # ── Compute (or reuse) norm_max ──────────────────────────────────────────
    existing_manifest = _PINN_DIR / "validation" / "snr0" / "manifest.json"
    if existing_manifest.exists():
        with open(existing_manifest, encoding="utf-8") as f:
            mf = json.load(f)
        norm_max = float(mf["norm_max"])
        print(f"\nReusing norm_max from existing manifest: {norm_max:.6e}")
    else:
        print("\nComputing global norm_max (streaming, may take a moment) ...")
        norm_max = _compute_norm_max(smomp_mmap, accurate_mmap)
        print(f"norm_max = {norm_max:.6e}")

    # ── Write manifest ───────────────────────────────────────────────────────
    manifest = {
        "n_holdout_samples": n_holdout,
        "n_total_samples": n_total,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "random_seed": RANDOM_SEED,
        "norm_max": norm_max,
        "source_smomp": str(Path(args.smomp_file).name),
        "source_accurate": str(Path(args.accurate_file).name),
        "source_user_positions": str(Path(args.user_positions).name),
        "channel_shape": list(smomp_mmap.shape[1:]),
        "dtype": str(smomp_mmap.dtype),
        "split": "test",
    }
    manifest_path = out_dir / "holdout_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest  : {manifest_path}")

    sz_mb = (smomp_path.stat().st_size + accurate_path.stat().st_size) / 1e6
    print(f"\nDone. Holdout bundle: {out_dir}  ({sz_mb:.1f} MB)")
    print(
        "\nUpload the following files to Google Drive:\n"
        f"  {out_dir}/holdout_smomp.npy\n"
        f"  {out_dir}/holdout_accurate.npy\n"
        f"  {out_dir}/holdout_ue_positions.txt\n"
        f"  {out_dir}/holdout_manifest.json\n"
        "  PINN_channel-estimation-main/Dataset/50_15GHz.jpg"
    )


if __name__ == "__main__":
    main()
