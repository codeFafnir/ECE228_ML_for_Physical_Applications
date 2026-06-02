"""
Train a StudentUNet with Physics-Guided Knowledge Distillation.

Run in the recommended order (start with least compression for baseline):
  python train_kd.py --preset light    [8-10x,  ~36M params]
  python train_kd.py --preset moderate [15-20x, ~18M params]
  python train_kd.py --preset extreme  [35-40x,  ~9M params]

Teacher cache (teacher_cache/) must exist before training. Run:
  python precompute_teacher.py --splits train val ...

Device priority: mps (M4 Air GPU) → cpu (fallback).
  Pass --device cpu to force CPU.
  Note: torch.complex is not supported on MPS; the physics terms inside
  PGKDLoss are automatically computed on CPU and moved back.

Example (full real data):
    python train_kd.py \\
        --preset light \\
        --smomp_file ../PINN_channel-estimation-main/initial_estimate_ls_snr0.npy \\
        --accurate_file ../PINN_channel-estimation-main/3D_channel_15GHz_2x2_Pt50.npy \\
        --user_positions ../PINN_channel-estimation-main/ue_positions_noisy.txt \\
        --rss_image ../PINN_channel-estimation-main/Dataset/50_15GHz.jpg \\
        --cache_dir teacher_cache \\
        --epochs 40 --lr 3e-4 --batch_size 16

Example (synthetic smoke-test):
    python train_kd.py --preset extreme --synthetic --epochs 3 --batch_size 4
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

_EXP2_DIR = Path(__file__).parent
_EXP1_SHARED = _EXP2_DIR.parent / "experiment1_turboquant" / "shared"
if str(_EXP1_SHARED) not in sys.path:
    sys.path.insert(0, str(_EXP1_SHARED))
_PINN_DIR = _EXP2_DIR.parent / "PINN_channel-estimation-main"
if str(_PINN_DIR) not in sys.path:
    sys.path.insert(0, str(_PINN_DIR))

from student_model import build_student, count_parameters, STUDENT_PRESETS
from pgkd_loss import PGKDLoss
from kd_data import get_kd_loaders, get_val_loader_only, get_train_base_loader
from eval_utils import compute_nmse


# ── Device selection ──────────────────────────────────────────────────────────

def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            print("Using MPS (Apple Silicon GPU)")
            return torch.device("mps")
        print("MPS not available — using CPU")
        return torch.device("cpu")
    return torch.device(requested)


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def _validate(
    student: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Returns mean NMSE (dB) over the validation split."""
    student.eval()
    total_nmse = 0.0
    n_batches = 0

    for batch in val_loader:
        smomp   = batch[0].to(device)
        accurate = batch[1].to(device)
        rss     = batch[2].to(device)

        pred = student(smomp, rss)
        total_nmse += compute_nmse(pred.cpu(), accurate.cpu())
        n_batches += 1

    student.train()
    return total_nmse / max(n_batches, 1)


# ── Synthetic cached dataset (smoke-test) ─────────────────────────────────────

class _SyntheticKDDataset(torch.utils.data.Dataset):
    """
    Random 5-tuple dataset matching KDCachedDataset format.
    Used when --synthetic is passed; no real data or cache files needed.
    """
    def __init__(self, n: int = 128, seed: int = 42):
        g = torch.Generator().manual_seed(seed)
        self.smomps    = torch.randn(n, 32, 4, 576, generator=g) * 0.1
        self.accurates = torch.randn(n, 32, 4, 576, generator=g) * 0.1
        self.rss       = torch.rand(n, 2, 30, 30, generator=g) * 2 - 1
        self.t_out     = torch.randn(n, 32, 4, 576, generator=g) * 0.1
        self.t_feat    = torch.randn(n, 72, 256, generator=g) * 0.1

    def __len__(self) -> int:
        return len(self.smomps)

    def __getitem__(self, idx: int) -> tuple:
        return (
            self.smomps[idx], self.accurates[idx], self.rss[idx],
            self.t_out[idx], self.t_feat[idx],
        )


# ── Main training loop ────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = _select_device(args.device)
    preset = args.preset
    cfg    = STUDENT_PRESETS[preset]

    # ── Student ───────────────────────────────────────────────────────────────
    student = build_student(preset)
    student.to(device)
    student.train()
    n_params = count_parameters(student)
    teacher_params = 358_878_624
    print(f"\n{'='*60}")
    print(f"  PG-KD Training  |  preset={preset}")
    print(f"  Student params : {n_params/1e6:.2f}M  ({teacher_params/n_params:.1f}x compression)")
    print(f"  widths={cfg['widths']}, d_bottleneck={cfg['d_bottleneck']}")
    print(f"  device={device}, epochs={args.epochs}, lr={args.lr}, batch={args.batch_size}")
    print(f"  kd_mode={args.kd_mode}, alpha={args.alpha}, beta={args.beta}, gamma={args.gamma}, T={args.T}")
    print(f"{'='*60}\n")

    # ── Loss ─────────────────────────────────────────────────────────────────
    criterion = PGKDLoss(
        d_bottleneck=cfg["d_bottleneck"],
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        T=args.T, kd_mode=args.kd_mode,
    ).to(device)

    # ── Data ─────────────────────────────────────────────────────────────────
    if args.synthetic:
        print("Synthetic data mode (smoke-test)")
        train_ds = _SyntheticKDDataset(n=args.n_synthetic, seed=42)
        val_ds   = _SyntheticKDDataset(n=max(args.n_synthetic // 4, 16), seed=99)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)
    else:
        train_loader, val_loader = get_kd_loaders(
            args.smomp_file, args.accurate_file,
            args.user_positions, args.rss_image,
            cache_dir=args.cache_dir,
            batch_size=args.batch_size,
        )

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(
        list(student.parameters()) + list(criterion.xattn_adapter.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2)

    # ── Training ──────────────────────────────────────────────────────────────
    best_val_nmse = float("inf")
    save_path = Path(args.save_dir) / f"student_{preset}.pth"
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        student.train()
        epoch_losses: dict[str, list] = {
            k: [] for k in ["total", "nmse", "physical", "kd_soft", "xattn_feat"]
        }
        t0 = time.perf_counter()

        for batch in train_loader:
            smomp, accurate, rss, t_out, t_feat = [x.to(device) for x in batch]

            optimizer.zero_grad()
            s_out, s_feat = student(smomp, rss, return_features=True)

            loss, comps = criterion(s_out, s_feat, t_out, t_feat, accurate, rss)
            loss.backward()

            nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(criterion.xattn_adapter.parameters()),
                max_norm=1.0,
            )
            optimizer.step()

            for k, v in comps.items():
                epoch_losses[k].append(v)

        scheduler.step()

        # Validation
        val_nmse = _validate(student, val_loader, device)
        elapsed  = time.perf_counter() - t0

        means = {k: float(np.mean(v)) for k, v in epoch_losses.items()}
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"loss={means['total']:.4f} "
            f"(nmse={means['nmse']:.4f} "
            f"kd={means['kd_soft']:.4f} "
            f"phys={means['physical']:.4f} "
            f"feat={means['xattn_feat']:.4f})  "
            f"val_nmse={val_nmse:.2f}dB  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  "
            f"t={elapsed:.1f}s"
        )

        if val_nmse < best_val_nmse:
            best_val_nmse = val_nmse
            torch.save(
                {
                    "preset": preset,
                    "d_bottleneck": cfg["d_bottleneck"],
                    "widths": cfg["widths"],
                    "epoch": epoch,
                    "val_nmse_db": val_nmse,
                    "model_state_dict": student.state_dict(),
                    "adapter_state_dict": criterion.xattn_adapter.state_dict(),
                },
                save_path,
            )
            print(f"  ✓ Saved best checkpoint (val_nmse={val_nmse:.2f} dB) → {save_path}")

    print(f"\nTraining done. Best val NMSE: {best_val_nmse:.2f} dB")
    print(f"Checkpoint: {save_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train StudentUNet with PG-KD")

    # Preset / model
    parser.add_argument("--preset", required=True,
                        choices=list(STUDENT_PRESETS.keys()),
                        help="Student size preset (light | moderate | extreme)")

    # Data files (real data)
    parser.add_argument("--smomp_file",
                        default="../PINN_channel-estimation-main/initial_estimate_ls_snr0.npy")
    parser.add_argument("--accurate_file",
                        default="../PINN_channel-estimation-main/3D_channel_15GHz_2x2_Pt50.npy")
    parser.add_argument("--user_positions",
                        default="../PINN_channel-estimation-main/ue_positions_noisy.txt")
    parser.add_argument("--rss_image",
                        default="../PINN_channel-estimation-main/Dataset/50_15GHz.jpg")
    parser.add_argument("--cache_dir", default="teacher_cache",
                        help="Directory with pre-computed teacher cache files")

    # Synthetic fallback
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic random data (smoke-test, no data files needed)")
    parser.add_argument("--n_synthetic", type=int, default=128)

    # Hyperparameters
    parser.add_argument("--epochs",     type=int,   default=40)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--kd_mode",    default="mse", choices=["mse", "kl"])
    parser.add_argument("--alpha",      type=float, default=1.0,  help="L^soft_KD weight")
    parser.add_argument("--beta",       type=float, default=0.01, help="L^physical weight")
    parser.add_argument("--gamma",      type=float, default=0.1,  help="L^feat_xattn weight")
    parser.add_argument("--T",          type=float, default=4.0,  help="Distillation temperature")

    # Device / output
    parser.add_argument("--device",   default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--save_dir", default="checkpoints",
                        help="Directory for saved student checkpoints")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
