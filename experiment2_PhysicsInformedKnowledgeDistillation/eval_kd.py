"""
Evaluate trained PG-KD students and print a GPTQ-style comparison table.

Compares the FP32 teacher against all trained student presets found in
checkpoints/.  Any missing checkpoint is listed as "not trained".

Output matches the format from experiment1_turboquant eval_all.py:

  ================================================================
    PG-KD Student Compression Evaluation
  ================================================================
  Method                   NMSE (dB)    Delta   Params     Ratio
  ----------------------------------------------------------------
    Teacher FP32             -14.29      ---    358.88M    1.0x
    Student light (10.0x)    -13.81   +0.48 dB  35.98M   10.0x
    Student moderate (20.1x) -13.50   +0.79 dB  17.83M   20.1x
    Student extreme (38.7x)  -13.12   +1.17 dB   9.29M   38.7x
  ================================================================

Usage (real data):
    python eval_kd.py \\
        --checkpoint ../simple_ls_0_val.pth \\
        --smomp_file ../PINN_channel-estimation-main/initial_estimate_ls_snr0.npy \\
        --accurate_file ../PINN_channel-estimation-main/3D_channel_15GHz_2x2_Pt50.npy \\
        --user_positions ../PINN_channel-estimation-main/ue_positions_noisy.txt \\
        --rss_image ../PINN_channel-estimation-main/Dataset/50_15GHz.jpg \\
        --device mps

Usage (synthetic, no data needed):
    python eval_kd.py --synthetic
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_EXP2_DIR = Path(__file__).parent
_EXP1_SHARED = _EXP2_DIR.parent / "experiment1_turboquant" / "shared"
if str(_EXP1_SHARED) not in sys.path:
    sys.path.insert(0, str(_EXP1_SHARED))
_PINN_DIR = _EXP2_DIR.parent / "PINN_channel-estimation-main"
if str(_PINN_DIR) not in sys.path:
    sys.path.insert(0, str(_PINN_DIR))

from student_model import build_student, count_parameters, STUDENT_PRESETS
from eval_utils import compute_nmse


TEACHER_PARAMS = 358_878_624


# ── Helpers ───────────────────────────────────────────────────────────────────

def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    label: str,
) -> dict:
    model.eval()
    model.to(device)
    total_nmse = 0.0
    n_batches = 0
    t0 = time.perf_counter()

    for batch in loader:
        smomp    = batch[0].to(device)
        accurate = batch[1].to(device)
        rss      = batch[2].to(device)
        pred = model(smomp, rss)
        total_nmse += compute_nmse(pred.cpu(), accurate.cpu())
        n_batches += 1

    elapsed = time.perf_counter() - t0
    nmse_db = total_nmse / max(n_batches, 1)
    print(f"  [{label}] NMSE = {nmse_db:.2f} dB  ({elapsed:.1f}s)")
    return {"nmse_db": nmse_db, "elapsed_s": elapsed}


def _load_student(preset: str, checkpoint_dir: str, device: torch.device) -> Optional[nn.Module]:
    path = Path(checkpoint_dir).resolve() / f"student_{preset}.pth"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location="cpu")
    student = build_student(preset)
    student.load_state_dict(ckpt["model_state_dict"], strict=True)
    student.eval()
    student.to(device)
    return student


def _format_table(
    teacher_result: dict,
    student_results: list[tuple[str, int, dict | None]],
) -> str:
    """
    teacher_result       : dict with nmse_db
    student_results      : list of (preset_label, n_params, result_dict or None)
    """
    teacher_nmse = teacher_result["nmse_db"]
    teacher_ratio = 1.0

    col_w = 32
    header = (
        f"  {'Method':<{col_w}} {'NMSE (dB)':>10} {'Delta':>10} "
        f"{'Params':>10} {'Ratio':>7}"
    )
    sep = "=" * len(header)
    mid = "-" * len(header)

    lines = [
        sep,
        f"  PG-KD Student Compression Evaluation  (Approach 2)",
        sep,
        header,
        mid,
    ]

    def _row(label: str, nmse: float | None, n_params: int, ratio: float) -> str:
        param_str = f"{n_params/1e6:.2f}M"
        ratio_str = f"{ratio:.1f}x"
        if nmse is None:
            return (
                f"  {label:<{col_w}} {'(not trained)':>10} {'':>10} "
                f"{param_str:>10} {ratio_str:>7}"
            )
        delta = nmse - teacher_nmse
        delta_str = "---" if abs(delta) < 1e-9 else f"{delta:+.2f} dB"
        return (
            f"  {label:<{col_w}} {nmse:>10.2f} {delta_str:>10} "
            f"{param_str:>10} {ratio_str:>7}"
        )

    lines.append(_row("Teacher FP32", teacher_nmse, TEACHER_PARAMS, 1.0))
    lines.append(mid)

    for preset_label, n_params, res in student_results:
        ratio = TEACHER_PARAMS / n_params
        nmse  = res["nmse_db"] if res is not None else None
        lines.append(_row(preset_label, nmse, n_params, ratio))

    lines.append(sep)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PG-KD students vs teacher")
    parser.add_argument("--checkpoint",
                        default="../simple_ls_0_val.pth",
                        help="Teacher checkpoint (.pth)")
    parser.add_argument("--smomp_file",
                        default="../PINN_channel-estimation-main/initial_estimate_ls_snr0.npy")
    parser.add_argument("--accurate_file",
                        default="../PINN_channel-estimation-main/3D_channel_15GHz_2x2_Pt50.npy")
    parser.add_argument("--user_positions",
                        default="../PINN_channel-estimation-main/ue_positions_noisy.txt")
    parser.add_argument("--rss_image",
                        default="../PINN_channel-estimation-main/Dataset/50_15GHz.jpg")
    parser.add_argument("--checkpoint_dir", default="checkpoints",
                        help="Directory containing student_{preset}.pth files")
    parser.add_argument("--device",  default="auto",
                        choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (no data files needed)")
    parser.add_argument("--n_synthetic", type=int, default=64)
    parser.add_argument("--presets", nargs="+",
                        default=list(STUDENT_PRESETS.keys()),
                        help="Which presets to evaluate (default: all)")
    args = parser.parse_args()

    device = _select_device(args.device)
    # Resolve before calibration_data's os.chdir(PINN_DIR) changes CWD.
    args.checkpoint_dir = str(Path(args.checkpoint_dir).resolve())
    print(f"Evaluation device: {device}\n")

    # ── Build val loader ──────────────────────────────────────────────────────
    if args.synthetic:
        from calibration_data import SyntheticChannelDataset
        val_ds = SyntheticChannelDataset(n_samples=args.n_synthetic, seed=2)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
        print("Using synthetic data\n")
    else:
        from kd_data import get_val_loader_only
        val_loader = get_val_loader_only(
            args.smomp_file, args.accurate_file,
            args.user_positions, args.rss_image,
            batch_size=args.batch_size,
        )

    # ── Evaluate teacher ──────────────────────────────────────────────────────
    print("Loading teacher ...")
    from model_loader import load_fp32_model
    teacher = load_fp32_model(args.checkpoint if not args.synthetic else None)
    teacher_result = _evaluate(teacher, val_loader, device, "Teacher FP32")
    del teacher
    if device.type == "mps":
        torch.mps.empty_cache()

    # ── Evaluate each student preset ──────────────────────────────────────────
    student_results: list[tuple[str, int, dict | None]] = []

    for preset in args.presets:
        cfg = STUDENT_PRESETS[preset]
        n_params = count_parameters(build_student(preset))
        ratio = TEACHER_PARAMS / n_params
        label = f"Student {preset} ({ratio:.1f}x)"

        student = _load_student(preset, args.checkpoint_dir, device)
        if student is None:
            print(f"  [{preset}] checkpoint not found — skipping")
            student_results.append((label, n_params, None))
            continue

        res = _evaluate(student, val_loader, device, label)
        student_results.append((label, n_params, res))
        del student
        if device.type == "mps":
            torch.mps.empty_cache()

    # ── Print table ───────────────────────────────────────────────────────────
    print()
    print(_format_table(teacher_result, student_results))


if __name__ == "__main__":
    main()
