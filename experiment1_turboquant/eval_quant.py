"""
Evaluation: FP32 vs TurboQuant-3bit PINN on channel estimation.

Two modes:
  1. Real data: load actual PINN checkpoint + dataset (full evaluation).
  2. Synthetic data: random tensors of the correct shapes (quick smoke-test,
     no dataset required).

Usage:
    # Quick smoke-test (no data needed):
    python eval_quant.py --synthetic

    # Full eval with real checkpoint:
    python eval_quant.py \
        --checkpoint /path/to/best_model.pth \
        --data_dir   /path/to/channel_data \
        --rss_path   /path/to/Dataset/50_15GHz.jpg
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Add PINN source dir to path
PINN_DIR = Path(__file__).parent.parent / "PINN_channel-estimation-main"
sys.path.insert(0, str(PINN_DIR))
sys.path.insert(0, str(Path(__file__).parent))

from quantize_pinn import quantize_model, model_size_report


# -------------------------------------------------------------------------
# Synthetic smoke-test dataset
# -------------------------------------------------------------------------

class SyntheticChannelDataset(torch.utils.data.Dataset):
    """
    Generates random (initial_estimate, true_channel, rss_map) triples
    matching the PINN model's expected input shapes.

    Shapes from Model.py:
      initial_estimate (smomp): (batch, 32, 4, 576)  float32
      true_channel (accurate):  (batch, 32, 4, 576)  float32
      rss_map:                  (batch, 2, 64, 64)   float32
    """

    def __init__(self, n_samples: int = 64, seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.n = n_samples
        # Pre-generate to keep evaluation reproducible
        self.smomps = torch.randn(n_samples, 32, 4, 576) * 0.1
        self.accurates = torch.randn(n_samples, 32, 4, 576) * 0.1
        # RSS map: 2-channel (grayscale + dBm normalized), values in [-1, 1]
        self.rss_maps = torch.rand(n_samples, 2, 64, 64) * 2 - 1

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return self.smomps[idx], self.accurates[idx], self.rss_maps[idx]


# -------------------------------------------------------------------------
# NMSE metric (matches PhysicsInformedLoss.calculate_nmse from Model.py)
# -------------------------------------------------------------------------

def compute_nmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Normalized Mean Square Error for channel tensors.
    Treats interleaved real/imag layout: first half channels = real, second = imag.
    Returns NMSE in dB.
    """
    n = pred.shape[1] // 2
    pred_c = torch.complex(pred[:, :n], pred[:, n:])
    target_c = torch.complex(target[:, :n], target[:, n:])
    mse = torch.mean(torch.abs(pred_c - target_c) ** 2)
    power = torch.mean(torch.abs(target_c) ** 2).clamp(min=1e-12)
    nmse_linear = (mse / power).item()
    nmse_db = 10 * np.log10(nmse_linear + 1e-12)
    return nmse_db


# -------------------------------------------------------------------------
# Evaluation loop
# -------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader,
             device: torch.device, desc: str = "") -> dict:
    """Run inference, return mean NMSE (dB) and throughput."""
    model.eval()
    model.to(device)
    total_nmse = 0.0
    n_batches = 0
    t0 = time.perf_counter()

    for smomp, accurate, rss in loader:
        smomp = smomp.to(device)
        accurate = accurate.to(device)
        rss = rss.to(device)

        pred = model(smomp, rss)
        total_nmse += compute_nmse(pred, accurate)
        n_batches += 1

    elapsed = time.perf_counter() - t0
    avg_nmse = total_nmse / max(n_batches, 1)
    throughput = n_batches * loader.batch_size / elapsed

    return {
        "nmse_db": avg_nmse,
        "elapsed_s": elapsed,
        "throughput_samples_per_s": throughput,
        "n_batches": n_batches,
    }


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TurboQuant PINN evaluation")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (no real dataset needed)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to FP32 model checkpoint (.pth)")
    parser.add_argument("--n_samples", type=int, default=128,
                        help="Number of synthetic samples")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--block_dim_linear", type=int, default=128)
    parser.add_argument("--block_dim_conv", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu or cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    print(f"Device: {device}")

    # --- Load model ---
    try:
        from Model import ImprovedPhysicsInformedUNet
    except ImportError:
        print(f"ERROR: Cannot import Model.py from {PINN_DIR}")
        print("Ensure PINN_channel-estimation-main is alongside this folder.")
        sys.exit(1)

    print("Building FP32 model ...")
    model_fp32 = ImprovedPhysicsInformedUNet(
        channel_shape=(32, 4, 576),
        rss_size=64,
        latent_dim=256,
        use_dbm_values=True,
    )
    model_fp32.eval()

    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        model_fp32.load_state_dict(state, strict=False)
        print("Checkpoint loaded.")
    else:
        print("WARNING: No checkpoint loaded — using random weights.")
        print("Results show algorithm overhead, not model quality.")

    # --- Build dataset ---
    if args.synthetic:
        print(f"\nSynthetic dataset: {args.n_samples} samples")
        dataset = SyntheticChannelDataset(n_samples=args.n_samples)
    else:
        print("ERROR: Real data loading not implemented in this script.")
        print("       Add your dataset loader or use --synthetic.")
        sys.exit(1)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False
    )

    # --- FP32 baseline ---
    print("\n--- FP32 Baseline ---")
    fp32_results = evaluate(model_fp32, loader, device, "FP32")
    print(f"  NMSE:       {fp32_results['nmse_db']:.3f} dB")
    print(f"  Throughput: {fp32_results['throughput_samples_per_s']:.1f} samples/s")

    # --- Quantize ---
    print("\nQuantizing model (TurboQuant 3-bit) ...")
    t_quant_start = time.perf_counter()
    model_q = quantize_model(
        model_fp32,
        block_dim_linear=args.block_dim_linear,
        block_dim_conv=args.block_dim_conv,
        inplace=False,
    )
    t_quant = time.perf_counter() - t_quant_start
    print(f"  Quantization time: {t_quant:.2f} s")

    # --- Compression report ---
    print("\n" + model_size_report(model_fp32, model_q))

    # --- 3-bit evaluation ---
    print("\n--- TurboQuant 3-bit ---")
    tq_results = evaluate(model_q, loader, device, "TQ-3bit")
    print(f"  NMSE:       {tq_results['nmse_db']:.3f} dB")
    print(f"  Throughput: {tq_results['throughput_samples_per_s']:.1f} samples/s")

    # --- Delta ---
    delta = tq_results["nmse_db"] - fp32_results["nmse_db"]
    print(f"\n  NMSE delta (3-bit - FP32): {delta:+.3f} dB")

    if abs(delta) < 1.0:
        print("  STATUS: PASS — degradation < 1 dB (acceptable for PTQ)")
    elif abs(delta) < 3.0:
        print("  STATUS: MARGINAL — consider QAT fine-tuning (qat_fine_tune.py)")
    else:
        print("  STATUS: FAIL — >3 dB degradation; enable QJL residual or QAT")

    return {
        "fp32_nmse_db": fp32_results["nmse_db"],
        "tq_nmse_db": tq_results["nmse_db"],
        "delta_db": delta,
    }


if __name__ == "__main__":
    main()
