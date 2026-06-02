"""
Precompute and cache teacher outputs for PG-KD training.

Runs the frozen teacher once over train and/or val splits in dataset order
and writes fp16 numpy memmaps to teacher_cache/:

  teacher_cache/
    train_out.npy   (N_train, 32, 4, 576)  fp16 — teacher channel estimate
    train_feat.npy  (N_train, 72, 256)     fp16 — TransformerChannelDecoder output
    val_out.npy     (N_val, 32, 4, 576)    fp16
    val_feat.npy    (N_val, 72, 256)       fp16

Estimated runtime on MacBook M4 Air (CPU):
  ~0.1 s/batch × 988 train batches (batch_size=8) ≈ 2 min for train split.
  Pass --device mps to use Apple Silicon GPU (faster).

Usage:
    python precompute_teacher.py \\
        --checkpoint ../simple_ls_0_val.pth \\
        --smomp_file ../PINN_channel-estimation-main/initial_estimate_ls_snr0.npy \\
        --accurate_file ../PINN_channel-estimation-main/3D_channel_15GHz_2x2_Pt50.npy \\
        --user_positions ../PINN_channel-estimation-main/ue_positions_noisy.txt \\
        --rss_image ../PINN_channel-estimation-main/Dataset/50_15GHz.jpg \\
        --cache_dir teacher_cache \\
        --device mps \\
        --batch_size 8 \\
        --splits train val

    # Quick smoke-test with synthetic data (no data files needed):
    python precompute_teacher.py --synthetic --cache_dir teacher_cache_test
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_EXP2_DIR = Path(__file__).parent
_EXP1_SHARED = _EXP2_DIR.parent / "experiment1_turboquant" / "shared"
if str(_EXP1_SHARED) not in sys.path:
    sys.path.insert(0, str(_EXP1_SHARED))
_PINN_DIR = _EXP2_DIR.parent / "PINN_channel-estimation-main"
if str(_PINN_DIR) not in sys.path:
    sys.path.insert(0, str(_PINN_DIR))

from teacher import TeacherWithFeatures
from kd_data import get_train_base_loader, get_val_loader_only
from calibration_data import SyntheticChannelDataset


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def _cache_split(
    teacher: TeacherWithFeatures,
    loader,
    cache_dir: Path,
    split: str,
    device: torch.device,
) -> None:
    n_samples = len(loader.dataset)
    out_file  = cache_dir / f"{split}_out.npy"
    feat_file = cache_dir / f"{split}_feat.npy"

    # Pre-allocate fp16 memmaps (write-mode: creates the file)
    out_map  = np.lib.format.open_memmap(
        str(out_file),  mode="w+", dtype=np.float16, shape=(n_samples, 32, 4, 576)
    )
    feat_map = np.lib.format.open_memmap(
        str(feat_file), mode="w+", dtype=np.float16, shape=(n_samples, 72, 256)
    )

    start = 0
    for batch_idx, batch in enumerate(loader):
        smomp = batch[0].to(device)
        rss   = batch[2].to(device)

        with torch.no_grad():
            out, feat = teacher(smomp, rss)

        b = smomp.shape[0]
        end = start + b
        out_map[start:end]  = out.cpu().float().numpy().astype(np.float16)
        feat_map[start:end] = feat.cpu().float().numpy().astype(np.float16)
        start = end

        if (batch_idx + 1) % 50 == 0 or end == n_samples:
            print(f"  [{split}] {end}/{n_samples} samples cached")

    out_map.flush()
    feat_map.flush()
    print(f"  Saved: {out_file}  ({out_map.shape}, fp16)")
    print(f"  Saved: {feat_file} ({feat_map.shape}, fp16)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute teacher cache for PG-KD")
    parser.add_argument("--checkpoint",  default="../simple_ls_0_val.pth")
    parser.add_argument("--smomp_file",  default="../PINN_channel-estimation-main/initial_estimate_ls_snr0.npy")
    parser.add_argument("--accurate_file", default="../PINN_channel-estimation-main/3D_channel_15GHz_2x2_Pt50.npy")
    parser.add_argument("--user_positions", default="../PINN_channel-estimation-main/ue_positions_noisy.txt")
    parser.add_argument("--rss_image",   default="../PINN_channel-estimation-main/Dataset/50_15GHz.jpg")
    parser.add_argument("--cache_dir",   default="teacher_cache")
    parser.add_argument("--device",      default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batch_size",  type=int, default=8)
    parser.add_argument("--splits",      nargs="+", default=["train", "val"],
                        choices=["train", "val"])
    parser.add_argument("--synthetic",   action="store_true",
                        help="Use synthetic random data (smoke-test without data files)")
    parser.add_argument("--n_synthetic", type=int, default=64,
                        help="Number of synthetic samples per split (smoke-test)")
    args = parser.parse_args()

    device = _select_device(args.device)
    print(f"Device: {device}")

    # Resolve to absolute path NOW, before calibration_data's os.chdir(PINN_DIR)
    # changes the working directory mid-run.
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Load teacher ──────────────────────────────────────────────────────────
    print(f"Loading teacher from {args.checkpoint} ...")
    teacher = TeacherWithFeatures.load(args.checkpoint, device)

    # ── Cache each requested split ────────────────────────────────────────────
    if args.synthetic:
        print("Using synthetic data (smoke-test mode)")
        from torch.utils.data import DataLoader
        for split in args.splits:
            ds = SyntheticChannelDataset(n_samples=args.n_synthetic, seed=99)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
            print(f"\nCaching {split} split ({len(ds)} synthetic samples) ...")
            _cache_split(teacher, loader, cache_dir, split, device)
    else:
        for split in args.splits:
            if split == "train":
                loader = get_train_base_loader(
                    args.smomp_file, args.accurate_file,
                    args.user_positions, args.rss_image,
                    batch_size=args.batch_size,
                )
            else:
                loader = get_val_loader_only(
                    args.smomp_file, args.accurate_file,
                    args.user_positions, args.rss_image,
                    batch_size=args.batch_size,
                )
            print(f"\nCaching {split} split ({len(loader.dataset)} samples) ...")
            _cache_split(teacher, loader, cache_dir, split, device)

    teacher.remove_hook()
    print("\nDone. Cache ready for train_kd.py.")


if __name__ == "__main__":
    main()
