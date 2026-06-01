"""
Head-to-head comparison: FP32 vs TurboQuant vs GPTQ vs Hadamard+GPTQ.

Usage (synthetic data, no real dataset needed):
    cd experiment1_turboquant
    python eval_all.py --synthetic

Usage (real data):
    python eval_all.py \\
        --checkpoint ../simple_ls_0_val.pth \\
        --smomp_file  ../PINN_channel-estimation-main/initial_estimate_ls_snr0.npy \\
        --accurate_file ../PINN_channel-estimation-main/3D_channel_15GHz_2x2_Pt50.npy \\
        --user_positions_file ../PINN_channel-estimation-main/ue_positions_noisy.txt \\
        --rss_image_path ../PINN_channel-estimation-main/Dataset/50_15GHz.jpg \\
        --bits 3 --n_cal 128 --device cpu

All four methods are run in sequence. Results are printed as a comparison table.
"""

import argparse
import copy
import sys
import time
from pathlib import Path

import torch

# --- Path setup ---
_THIS_DIR = Path(__file__).parent
_PROJ_DIR = _THIS_DIR.parent
_PINN_DIR = _PROJ_DIR / "PINN_channel-estimation-main"

sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_PINN_DIR))

from shared.model_loader import load_fp32_model
from shared.eval_utils import evaluate, per_layer_reconstruction_mse, format_results_table
from shared.calibration_data import get_calibration_loader, get_val_loader, get_synthetic_loaders
from method_turboquant.quantize_pinn import quantize_model as tq_quantize
from method_gptq.gptq_quantize_pinn import gptq_quantize_pinn
from method_hadamard_gptq.hadamard_gptq_quantize_pinn import hadamard_gptq_quantize_pinn


def main():
    parser = argparse.ArgumentParser(description="Quantization comparison for PINN")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to FP32 checkpoint (.pth). If omitted, uses random weights.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data instead of real dataset.")
    parser.add_argument("--smomp_file", type=str, default=None)
    parser.add_argument("--accurate_file", type=str, default=None)
    parser.add_argument("--user_positions_file", type=str, default=None)
    parser.add_argument("--rss_image_path", type=str, default=None)
    parser.add_argument("--n_cal", type=int, default=128,
                        help="Number of calibration samples for GPTQ methods.")
    parser.add_argument("--n_val_synthetic", type=int, default=256,
                        help="Number of validation samples when using --synthetic.")
    parser.add_argument("--bits", type=int, default=3,
                        help="Quantization bit-width (3 or 4).")
    parser.add_argument("--group_size", type=int, default=128,
                        help="GPTQ group size (columns per quant group).")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--skip_tq", action="store_true",
                        help="Skip TurboQuant (slow on CPU for large models).")
    parser.add_argument("--skip_hadamard_gptq", action="store_true",
                        help="Skip Hadamard+GPTQ (runs TurboQuant rotation + GPTQ).")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # ---- Load FP32 model ----
    print("\nLoading FP32 model ...")
    model_fp32 = load_fp32_model(args.checkpoint)

    # ---- Build data loaders ----
    if args.synthetic:
        print(f"\nUsing synthetic data ({args.n_cal} cal, {args.n_val_synthetic} val samples)")
        cal_loader, val_loader = get_synthetic_loaders(
            n_cal=args.n_cal,
            n_val=args.n_val_synthetic,
            batch_size=args.batch_size,
        )
    else:
        missing = [
            f for f in ["smomp_file", "accurate_file", "user_positions_file", "rss_image_path"]
            if getattr(args, f) is None
        ]
        if missing:
            print(f"ERROR: Missing real-data arguments: {missing}")
            print("       Use --synthetic for a quick smoke-test.")
            sys.exit(1)

        print("\nLoading real dataset ...")
        cal_loader = get_calibration_loader(
            smomp_file=args.smomp_file,
            accurate_file=args.accurate_file,
            user_positions_file=args.user_positions_file,
            rss_image_path=args.rss_image_path,
            n_cal=args.n_cal,
            batch_size=args.batch_size,
        )
        val_loader = get_val_loader(
            smomp_file=args.smomp_file,
            accurate_file=args.accurate_file,
            user_positions_file=args.user_positions_file,
            rss_image_path=args.rss_image_path,
            batch_size=args.batch_size,
        )

    results: dict[str, dict] = {}

    # ================================================================
    # Method 0: FP32 Baseline
    # ================================================================
    print("\n" + "=" * 60)
    print("  [0/3] FP32 Baseline")
    print("=" * 60)
    t0 = time.perf_counter()
    res_fp32 = evaluate(model_fp32, val_loader, device)
    res_fp32["elapsed_s"] = time.perf_counter() - t0
    results["FP32 Baseline"] = res_fp32
    print(f"  NMSE: {res_fp32['nmse_db']:.3f} dB  ({res_fp32['elapsed_s']:.1f}s)")

    # ================================================================
    # Method 1: TurboQuant 3-bit
    # ================================================================
    if not args.skip_tq:
        print("\n" + "=" * 60)
        print("  [1/3] TurboQuant 3-bit (data-oblivious)")
        print("=" * 60)
        t0 = time.perf_counter()
        model_tq = tq_quantize(model_fp32, block_dim_linear=128, block_dim_conv=64)
        quant_time = time.perf_counter() - t0
        print(f"  Quantization time: {quant_time:.2f}s")

        eval_t0 = time.perf_counter()
        res_tq = evaluate(model_tq, val_loader, device)
        res_tq["elapsed_s"] = time.perf_counter() - eval_t0
        res_tq["quant_time_s"] = quant_time
        results["TurboQuant 3-bit"] = res_tq
        print(f"  NMSE: {res_tq['nmse_db']:.3f} dB  (eval: {res_tq['elapsed_s']:.1f}s)")
        del model_tq

    # ================================================================
    # Method 2: Pure GPTQ
    # ================================================================
    print("\n" + "=" * 60)
    print(f"  [2/3] Pure GPTQ {args.bits}-bit (calibration-data-aware)")
    print("=" * 60)
    t0 = time.perf_counter()
    model_gptq = gptq_quantize_pinn(
        model=copy.deepcopy(model_fp32),
        cal_loader=cal_loader,
        num_bits=args.bits,
        group_size=args.group_size,
        device=device,
        verbose=True,
    )
    quant_time = time.perf_counter() - t0
    print(f"  Quantization time: {quant_time:.2f}s")

    eval_t0 = time.perf_counter()
    res_gptq = evaluate(model_gptq, val_loader, device)
    res_gptq["elapsed_s"] = time.perf_counter() - eval_t0
    res_gptq["quant_time_s"] = quant_time
    results[f"GPTQ {args.bits}-bit"] = res_gptq
    print(f"  NMSE: {res_gptq['nmse_db']:.3f} dB  (eval: {res_gptq['elapsed_s']:.1f}s)")
    del model_gptq

    # ================================================================
    # Method 3: Hadamard + GPTQ
    # ================================================================
    if not args.skip_hadamard_gptq:
        print("\n" + "=" * 60)
        print(f"  [3/3] Hadamard+GPTQ {args.bits}-bit (QuaRot-style)")
        print("=" * 60)
        t0 = time.perf_counter()
        model_hgptq = hadamard_gptq_quantize_pinn(
            model=copy.deepcopy(model_fp32),
            cal_loader=cal_loader,
            num_bits=args.bits,
            group_size=args.group_size,
            device=device,
            verbose=True,
        )
        quant_time = time.perf_counter() - t0
        print(f"  Quantization time: {quant_time:.2f}s")

        eval_t0 = time.perf_counter()
        res_hgptq = evaluate(model_hgptq, val_loader, device)
        res_hgptq["elapsed_s"] = time.perf_counter() - eval_t0
        res_hgptq["quant_time_s"] = quant_time
        results[f"Hadamard+GPTQ {args.bits}-bit"] = res_hgptq
        print(f"  NMSE: {res_hgptq['nmse_db']:.3f} dB  (eval: {res_hgptq['elapsed_s']:.1f}s)")
        del model_hgptq

    # ================================================================
    # Results Table
    # ================================================================
    print("\n\n" + format_results_table(results))

    # Per-layer reconstruction MSE (GPTQ vs FP32)
    if f"GPTQ {args.bits}-bit" in results:
        print("\n  Per-layer reconstruction MSE not shown here.")
        print("  Run individual methods with verbose=True and inspect weight MSE manually.")

    return results


if __name__ == "__main__":
    main()
