"""
Shared evaluation utilities for quantization experiments.

compute_nmse: NMSE in dB for complex channel tensors.
evaluate:     Run a model on a DataLoader, return NMSE + timing.
per_layer_reconstruction_mse: Weight-level MSE between FP32 and quantized model.
"""

import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_nmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    NMSE in dB.

    Assumes interleaved real/imaginary layout:
      first n channels = real part, next n channels = imaginary part.
    """
    n = pred.shape[1] // 2
    pred_c = torch.complex(pred[:, :n].float(), pred[:, n:].float())
    target_c = torch.complex(target[:, :n].float(), target[:, n:].float())
    mse = torch.mean(torch.abs(pred_c - target_c) ** 2)
    power = torch.mean(torch.abs(target_c) ** 2).clamp(min=1e-12)
    return 10.0 * float(np.log10((mse / power).item() + 1e-12))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Run inference over the loader and return mean NMSE (dB) + timing.

    Returns dict with keys: nmse_db, elapsed_s, n_batches, n_samples.
    """
    model.eval()
    model.to(device)
    total_nmse = 0.0
    n_batches = 0
    n_samples = 0
    t0 = time.perf_counter()

    for smomp, accurate, rss in loader:
        smomp = smomp.to(device)
        accurate = accurate.to(device)
        rss = rss.to(device)

        pred = model(smomp, rss)
        total_nmse += compute_nmse(pred, accurate)
        n_batches += 1
        n_samples += smomp.shape[0]

    elapsed = time.perf_counter() - t0
    return {
        "nmse_db": total_nmse / max(n_batches, 1),
        "elapsed_s": elapsed,
        "n_batches": n_batches,
        "n_samples": n_samples,
    }


def per_layer_reconstruction_mse(
    model_fp32: nn.Module,
    model_q: nn.Module,
) -> list[dict]:
    """
    Compute weight-level MSE between FP32 and quantized model for each named layer.

    Returns list of dicts: {name, mse, var_fp32, ratio} sorted by ratio descending.
    Only compares layers present in both models with matching shapes.
    """
    fp32_weights = {
        name: param.data.float()
        for name, param in model_fp32.named_parameters()
        if "weight" in name
    }

    results = []
    for name, param_q in model_q.named_parameters():
        if "weight" not in name:
            continue
        if name not in fp32_weights:
            continue
        w_fp = fp32_weights[name]
        w_q = param_q.data.float()
        if w_fp.shape != w_q.shape:
            continue
        mse = float(((w_q - w_fp) ** 2).mean().item())
        var = float(w_fp.var().clamp(min=1e-12).item())
        results.append({
            "name": name,
            "mse": mse,
            "var_fp32": var,
            "ratio": mse / var,
        })

    results.sort(key=lambda x: x["ratio"], reverse=True)
    return results


def format_results_table(results: dict[str, dict]) -> str:
    """
    Format head-to-head comparison table.

    results: {method_name: evaluate() output dict + optional 'quant_time_s'}
    First method is treated as FP32 baseline.
    """
    lines = []
    header = f"{'Method':<25} {'NMSE (dB)':>10} {'Delta':>8} {'Time (s)':>10}"
    lines.append("=" * len(header))
    lines.append("  Head-to-Head Quantization Comparison")
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("-" * len(header))

    methods = list(results.items())
    baseline_nmse = methods[0][1]["nmse_db"] if methods else 0.0

    for method_name, res in methods:
        nmse = res["nmse_db"]
        delta = nmse - baseline_nmse
        delta_str = "---" if abs(delta) < 1e-9 else f"{delta:+.2f} dB"
        elapsed = res.get("elapsed_s", 0.0)
        lines.append(
            f"  {method_name:<23} {nmse:>10.2f} {delta_str:>8} {elapsed:>10.1f}"
        )

    lines.append("=" * len(header))
    return "\n".join(lines)
