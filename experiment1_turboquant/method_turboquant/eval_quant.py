"""
Evaluation: FP32 vs TurboQuant-3bit PINN on channel estimation.

Two modes:
  1. Real data: load actual PINN checkpoint + dataset.
  2. Synthetic data: random tensors of correct shapes (quick smoke-test).

Fixes vs original eval_quant.py:
  - rss_size=30 (matches crop_size=30 in GlobalNormalizedDataset)
  - strict=True in load_state_dict (catch weight shape mismatches early)
  - SyntheticChannelDataset rss_maps shape: (n, 2, 30, 30)

Usage:
    python eval_quant.py --synthetic
    python eval_quant.py --checkpoint ../../simple_ls_0_val.pth --synthetic
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

PINN_DIR = Path(__file__).parent.parent.parent / "PINN_channel-estimation-main"
sys.path.insert(0, str(PINN_DIR))
sys.path.insert(0, str(Path(__file__).parent))

from .quantize_pinn import quantize_model, model_size_report


class SyntheticChannelDataset(torch.utils.data.Dataset):
    """
    Synthetic (initial_estimate, true_channel, rss_map) triples matching PINN shapes.
      smomp / accurate: (32, 4, 576)
      rss_map:          (2, 30, 30)  ← crop_size=30 matches GlobalNormalizedDataset
    """

    def __init__(self, n_samples: int = 64, seed: int = 42):
        torch.manual_seed(seed)
        self.n = n_samples
        self.smomps = torch.randn(n_samples, 32, 4, 576) * 0.1
        self.accurates = torch.randn(n_samples, 32, 4, 576) * 0.1
        self.rss_maps = torch.rand(n_samples, 2, 30, 30) * 2 - 1

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return self.smomps[idx], self.accurates[idx], self.rss_maps[idx]


def compute_nmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    n = pred.shape[1] // 2
    pred_c = torch.complex(pred[:, :n], pred[:, n:])
    target_c = torch.complex(target[:, :n], target[:, n:])
    mse = torch.mean(torch.abs(pred_c - target_c) ** 2)
    power = torch.mean(torch.abs(target_c) ** 2).clamp(min=1e-12)
    return 10 * np.log10((mse / power).item() + 1e-12)


@torch.no_grad()
def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader,
             device: torch.device) -> dict:
    model.eval()
    model.to(device)
    total_nmse = 0.0
    n_batches = 0
    t0 = time.perf_counter()

    for smomp, accurate, rss in loader:
        pred = model(smomp.to(device), rss.to(device))
        total_nmse += compute_nmse(pred, accurate.to(device))
        n_batches += 1

    elapsed = time.perf_counter() - t0
    return {
        "nmse_db": total_nmse / max(n_batches, 1),
        "elapsed_s": elapsed,
        "n_batches": n_batches,
    }


def main():
    parser = argparse.ArgumentParser(description="TurboQuant PINN evaluation")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--block_dim_linear", type=int, default=128)
    parser.add_argument("--block_dim_conv", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    try:
        from Model import ImprovedPhysicsInformedUNet
    except ImportError:
        print(f"ERROR: Cannot import Model.py from {PINN_DIR}")
        sys.exit(1)

    # rss_size=30 matches crop_size=30 in GlobalNormalizedDataset
    model_fp32 = ImprovedPhysicsInformedUNet(
        channel_shape=(32, 4, 576),
        rss_size=30,
        use_dbm_values=True,
    )
    model_fp32.eval()

    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        model_fp32.load_state_dict(state, strict=True)
        print(f"Checkpoint loaded: {args.checkpoint}")
    else:
        print("WARNING: No checkpoint — using random weights.")

    if not args.synthetic:
        print("ERROR: Real data loading not implemented here. Use --synthetic.")
        sys.exit(1)

    dataset = SyntheticChannelDataset(n_samples=args.n_samples)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print("\n--- FP32 Baseline ---")
    fp32_res = evaluate(model_fp32, loader, device)
    print(f"  NMSE: {fp32_res['nmse_db']:.3f} dB")

    print("\nQuantizing (TurboQuant 3-bit) ...")
    t0 = time.perf_counter()
    model_q = quantize_model(
        model_fp32,
        block_dim_linear=args.block_dim_linear,
        block_dim_conv=args.block_dim_conv,
    )
    print(f"  Quantization time: {time.perf_counter() - t0:.2f}s")
    print("\n" + model_size_report(model_fp32, model_q))

    print("\n--- TurboQuant 3-bit ---")
    tq_res = evaluate(model_q, loader, device)
    print(f"  NMSE: {tq_res['nmse_db']:.3f} dB")

    delta = tq_res["nmse_db"] - fp32_res["nmse_db"]
    print(f"\n  Delta (TQ - FP32): {delta:+.3f} dB")


if __name__ == "__main__":
    main()
