"""
Evaluate teacher + PG-KD students on the true holdout set.

Loads the bundle produced by create_holdout.py (four files in --holdout_dir)
and prints a GPTQ-style NMSE comparison table identical to eval_kd.py.

Works on both local MPS/CPU (MacBook) and Colab GPU.

Usage (local):
    python eval_holdout.py \\
        --holdout_dir holdout \\
        --rss_image   ../PINN_channel-estimation-main/Dataset/50_15GHz.jpg \\
        --checkpoint  ../simple_ls_0_val.pth \\
        --checkpoint_dir checkpoints \\
        --device mps

Usage (Colab, after Drive mount):
    !python eval_holdout.py \\
        --holdout_dir /content/drive/MyDrive/ECE228/holdout \\
        --rss_image   /content/drive/MyDrive/ECE228/PINN_channel-estimation-main/Dataset/50_15GHz.jpg \\
        --checkpoint  /content/drive/MyDrive/ECE228/simple_ls_0_val.pth \\
        --checkpoint_dir /content/drive/MyDrive/ECE228/checkpoints \\
        --device cuda
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_EXP2_DIR = Path(__file__).parent
_EXP1_SHARED = _EXP2_DIR.parent / "experiment1_turboquant" / "shared"
_PINN_DIR = _EXP2_DIR.parent / "PINN_channel-estimation-main"

for _p in [str(_EXP1_SHARED), str(_PINN_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from student_model import STUDENT_PRESETS, build_student, count_parameters
from eval_utils import compute_nmse

TEACHER_PARAMS = 358_878_624


# ── HoldoutDataset ────────────────────────────────────────────────────────────

class HoldoutDataset(Dataset):
    """
    Reads the four-file bundle produced by create_holdout.py.

    Returns (smomp, accurate, rss) triples identical in dtype and layout to
    MmapChannelDataset, so the same evaluation loop works unchanged.
    """

    def __init__(
        self,
        holdout_dir: str,
        rss_image_path: str,
        crop_size: int = 30,
        use_dbm_values: bool = True,
    ):
        from Model import RSSColorMapper

        hdir = Path(holdout_dir)
        with open(hdir / "holdout_manifest.json", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.smomp = np.load(hdir / "holdout_smomp.npy", mmap_mode="r")
        self.accurate = np.load(hdir / "holdout_accurate.npy", mmap_mode="r")
        self.norm_max = float(self.manifest["norm_max"])
        self.positions = self._load_positions(hdir / "holdout_ue_positions.txt")
        self.rss_processor = self._make_rss_processor(rss_image_path)
        self.crop_size = crop_size
        self.use_dbm_values = use_dbm_values
        self.rss_color_mapper = RSSColorMapper(min_dbm=-110.0, max_dbm=-40.0)

        print(f"Holdout dataset: {len(self.smomp)} samples from {hdir}")

    @staticmethod
    def _load_positions(path: Path) -> list[tuple[float, float]]:
        positions: list[tuple[float, float]] = []
        with open(path, encoding="utf-8") as f:
            for line in f.readlines()[1:]:
                if line.strip():
                    parts = line.strip().split()
                    positions.append((float(parts[0]), float(parts[1])))
        return positions

    @staticmethod
    def _make_rss_processor(rss_image_path: str):
        os.chdir(str(_PINN_DIR))
        from find_in_map import RSSMapProcessor
        return RSSMapProcessor(
            image_path=rss_image_path,
            bs_pixel_coords=(287, 293),
            bs_real_coords=(71.06, 246.29),
            image_width_meters=527.5,
        )

    def __len__(self) -> int:
        return len(self.smomp)

    def __getitem__(self, idx: int):
        smomp_ch = np.concatenate(
            [np.real(self.smomp[idx]), np.imag(self.smomp[idx])], axis=0
        ).astype(np.float32) / self.norm_max

        accurate_ch = np.concatenate(
            [np.real(self.accurate[idx]), np.imag(self.accurate[idx])], axis=0
        ).astype(np.float32) / self.norm_max

        user_x, user_y = self.positions[idx]
        rss_crop = self.rss_processor.crop_around_user(user_x, user_y, self.crop_size)

        if rss_crop is None:
            rss_crop = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.float32)

        if self.use_dbm_values:
            rss_dbm = self.rss_color_mapper.rgb_to_dbm(rss_crop)
            rss_dbm_normalized = self.rss_color_mapper.normalize_dbm(rss_dbm)
            rss_gray = cv2.cvtColor(rss_crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            rss_gray_normalized = rss_gray.astype(np.float32) / 255.0
            rss_tensor = torch.stack([
                torch.from_numpy(rss_gray_normalized).float(),
                torch.from_numpy(rss_dbm_normalized).float(),
            ])
        else:
            rss_gray = cv2.cvtColor(rss_crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            rss_tensor = torch.from_numpy(rss_gray.astype(np.float32) / 255.0).unsqueeze(0)

        return (
            torch.from_numpy(smomp_ch).float(),
            torch.from_numpy(accurate_ch).float(),
            rss_tensor,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
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
    n_holdout: int,
) -> str:
    teacher_nmse = teacher_result["nmse_db"]
    col_w = 32
    header = (
        f"  {'Method':<{col_w}} {'NMSE (dB)':>10} {'Delta':>10} "
        f"{'Params':>10} {'Ratio':>7}"
    )
    sep = "=" * len(header)
    mid = "-" * len(header)

    lines = [
        sep,
        f"  PG-KD Holdout Evaluation  ({n_holdout} samples, true test split)",
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
        nmse = res["nmse_db"] if res is not None else None
        lines.append(_row(preset_label, nmse, n_params, ratio))

    lines.append(sep)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PG-KD models on the true holdout set"
    )
    parser.add_argument(
        "--holdout_dir",
        default="holdout",
        help="Directory with holdout_{smomp,accurate,ue_positions,manifest} files",
    )
    parser.add_argument(
        "--rss_image",
        default=str(_PINN_DIR / "Dataset" / "50_15GHz.jpg"),
        help="Path to 50_15GHz.jpg RSS map image",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(_EXP2_DIR.parent / "simple_ls_0_val.pth"),
        help="Teacher checkpoint (.pth)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints",
        help="Directory containing student_{preset}.pth files",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--presets",
        nargs="+",
        default=list(STUDENT_PRESETS.keys()),
        help="Which student presets to evaluate (default: all)",
    )
    args = parser.parse_args()

    # Resolve before calibration_data's os.chdir(PINN_DIR) fires.
    holdout_dir = str(Path(args.holdout_dir).resolve())
    checkpoint_dir = str(Path(args.checkpoint_dir).resolve())
    checkpoint = str(Path(args.checkpoint).resolve())

    device = _select_device(args.device)
    print(f"Evaluation device: {device}\n")

    # ── Build holdout loader ──────────────────────────────────────────────────
    holdout_ds = HoldoutDataset(
        holdout_dir=holdout_dir,
        rss_image_path=args.rss_image,
    )
    holdout_loader = DataLoader(
        holdout_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    n_holdout = len(holdout_ds)

    # ── Evaluate teacher ──────────────────────────────────────────────────────
    print("\nLoading teacher ...")
    from model_loader import load_fp32_model
    teacher = load_fp32_model(checkpoint)
    teacher_result = _evaluate(teacher, holdout_loader, device, "Teacher FP32")
    del teacher
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Evaluate each student preset ──────────────────────────────────────────
    student_results: list[tuple[str, int, dict | None]] = []

    for preset in args.presets:
        n_params = count_parameters(build_student(preset))
        ratio = TEACHER_PARAMS / n_params
        label = f"Student {preset} ({ratio:.1f}x)"

        student = _load_student(preset, checkpoint_dir, device)
        if student is None:
            print(f"  [{preset}] checkpoint not found in {checkpoint_dir} — skipping")
            student_results.append((label, n_params, None))
            continue

        res = _evaluate(student, holdout_loader, device, label)
        student_results.append((label, n_params, res))
        del student
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Print table ───────────────────────────────────────────────────────────
    print()
    print(_format_table(teacher_result, student_results, n_holdout))


if __name__ == "__main__":
    main()
