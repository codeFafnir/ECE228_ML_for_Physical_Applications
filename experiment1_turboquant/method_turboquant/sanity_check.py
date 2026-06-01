"""
Sanity checks for TurboQuant implementation.

Run from the method_turboquant/ directory or project root:
  python -m experiment1_turboquant.method_turboquant.sanity_check
  # or:
  cd experiment1_turboquant/method_turboquant && python sanity_check.py
"""

import math
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_THIS_DIR = Path(__file__).parent
_EXP_DIR = _THIS_DIR.parent
_PINN_DIR = _EXP_DIR.parent / "PINN_channel-estimation-main"

sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_PINN_DIR))

import os
os.chdir(str(_PINN_DIR))

from turboquant import (
    TurboQuantTensor,
    fast_hadamard_transform,
    random_rotation,
    inverse_rotation,
    quantize_3bit,
    dequantize_3bit,
    LLOYD_MAX_3BIT,
    LLOYD_MAX_3BIT_DISTORTION_RATIO,
)
from tq_layers import TQLinear, TQConv2d, TQConvTranspose2d, _safe_block_dim
from quantize_pinn import quantize_model


_PASS = "\033[92mPASS\033[0m"
_FAIL = "\033[91mFAIL\033[0m"


def run_test(name: str, fn):
    try:
        fn()
        print(f"  [{_PASS}] {name}")
        return True
    except AssertionError as e:
        print(f"  [{_FAIL}] {name}: {e}")
        return False
    except Exception:
        print(f"  [{_FAIL}] {name}")
        traceback.print_exc()
        return False


def test_fht_orthonormality():
    for d in [4, 8, 16, 64, 128]:
        x = torch.randn(d, d)
        err = (fast_hadamard_transform(fast_hadamard_transform(x)) - x).abs().max().item()
        assert err < 1e-5, f"H*H != I for d={d}: max_err={err:.2e}"


def test_fht_self_inverse():
    torch.manual_seed(1)
    for d in [8, 32, 128, 256]:
        v = torch.randn(5, d)
        err = (fast_hadamard_transform(fast_hadamard_transform(v)) - v).abs().max().item()
        assert err < 1e-5, f"H(H(v)) != v for d={d}: err={err:.2e}"


def test_rotation_invertibility():
    torch.manual_seed(2)
    for d in [8, 64, 128]:
        for seed in [0, 42, 12345]:
            x = torch.randn(10, d)
            err = (inverse_rotation(random_rotation(x, seed=seed), seed=seed) - x).abs().max().item()
            assert err < 1e-4, f"Rotation not invertible d={d}, seed={seed}: err={err:.2e}"


def test_lloyd_max_distortion():
    torch.manual_seed(3)
    x = torch.randn(100_000)
    idx = quantize_3bit(x)
    x_hat = dequantize_3bit(idx)
    ratio = ((x - x_hat) ** 2).mean().item() / x.var().item()
    assert 0.005 < ratio < 0.05, f"Distortion ratio {ratio:.4f} out of expected [0.005, 0.05]"


def test_turboquant_tensor_roundtrip():
    torch.manual_seed(4)
    for shape, bd in [
        ((256, 256), 128),
        ((64, 32, 3, 3), 64),
        ((32, 64, 3, 3), 64),
        ((128,), 128),
        ((100, 100), 64),
    ]:
        W = torch.randn(*shape) * 0.02
        tqt = TurboQuantTensor.compress(W, block_dim=bd, seed=7)
        W_hat = tqt.decompress(dtype=torch.float32)
        assert W_hat.shape == W.shape
        ratio = ((W_hat - W) ** 2).mean().item() / max(W.var().item(), 1e-12)
        assert ratio < 0.15, f"Reconstruction ratio too high for shape {shape}: {ratio:.4f}"


def test_compression_ratio():
    torch.manual_seed(5)
    W = torch.randn(256, 256)
    ratio = TurboQuantTensor.compress(W, block_dim=128, seed=1).compression_ratio()
    assert 8.0 <= ratio <= 12.0, f"Unexpected compression ratio: {ratio:.2f}x"


def test_tq_linear_output():
    torch.manual_seed(6)
    lin = nn.Linear(256, 128, bias=True)
    tq_lin = TQLinear(lin, block_dim=128, seed=99)
    x = torch.randn(4, 256)
    with torch.no_grad():
        y_fp32 = lin(x)
        y_tq = tq_lin(x)
    diff = (y_tq - y_fp32).abs().mean().item()
    rel_err = diff / max(y_fp32.abs().mean().item(), 1e-12)
    assert rel_err < 0.5, f"TQLinear rel_err={rel_err:.4f}"
    assert diff > 0, "TQLinear identical to FP32 — not compressed?"


def test_tq_conv2d_output():
    torch.manual_seed(7)
    conv = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    tq_conv = TQConv2d(conv, block_dim=64, seed=77)
    x = torch.randn(2, 32, 16, 16)
    with torch.no_grad():
        y_fp32 = conv(x)
        y_tq = tq_conv(x)
    assert y_tq.shape == y_fp32.shape
    rel_err = (y_tq - y_fp32).abs().mean() / y_fp32.abs().mean().clamp(min=1e-8)
    assert rel_err.item() < 0.5, f"TQConv2d rel_err={rel_err:.4f}"


def test_safe_block_dim():
    assert _safe_block_dim(108, preferred=128) == 64
    assert _safe_block_dim(9, preferred=128) == 8
    assert _safe_block_dim(18432, preferred=128) == 128


def test_bias_not_quantized():
    torch.manual_seed(8)
    lin = nn.Linear(64, 32, bias=True)
    tq_lin = TQLinear(lin, block_dim=64)
    assert isinstance(tq_lin.bias, nn.Parameter)
    err = (tq_lin.bias.data - lin.bias.data).abs().max().item()
    assert err < 1e-6, f"Bias changed after wrapping: {err:.2e}"


def test_tq_idx_norms_are_buffers():
    """Bug 1 fix verification: tq_idx and tq_norms must be registered nn.Buffers."""
    lin = nn.Linear(64, 32)
    tq_lin = TQLinear(lin, block_dim=64)
    buffer_names = {name for name, _ in tq_lin.named_buffers()}
    assert "tq_idx" in buffer_names, "tq_idx not registered as buffer"
    assert "tq_norms" in buffer_names, "tq_norms not registered as buffer"


def test_pinn_model_conversion():
    try:
        from Model import ImprovedPhysicsInformedUNet
    except ImportError:
        raise AssertionError(f"Cannot import Model.py from {_PINN_DIR}")

    torch.manual_seed(9)
    model = ImprovedPhysicsInformedUNet(channel_shape=(32, 4, 576), rss_size=30, use_dbm_values=True)
    model.eval()

    n_orig_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    n_orig_conv = sum(1 for m in model.modules() if isinstance(m, nn.Conv2d))

    model_q = quantize_model(model, block_dim_linear=128, block_dim_conv=64)
    model_q.eval()

    n_tq_lin = sum(1 for m in model_q.modules() if isinstance(m, TQLinear))
    n_tq_conv = sum(1 for m in model_q.modules() if isinstance(m, TQConv2d))
    assert n_tq_lin > 0 and n_tq_conv > 0

    smomp = torch.randn(2, 32, 4, 576) * 0.1
    rss = torch.rand(2, 2, 30, 30) * 2 - 1
    with torch.no_grad():
        pred_fp32 = model(smomp, rss)
        pred_tq = model_q(smomp, rss)

    assert pred_fp32.shape == pred_tq.shape == (2, 32, 4, 576)
    assert torch.isfinite(pred_tq).all(), "TQ model output contains NaN/Inf"


def test_rotation_spreads_outliers():
    torch.manual_seed(10)
    d = 128
    x = torch.zeros(1, d)
    x[0, 0] = 1000.0
    x[0, 1:] = torch.randn(d - 1)
    ratio_before = x.abs().max() / x.abs().mean()
    y = random_rotation(x, seed=42)
    ratio_after = y.abs().max() / y.abs().mean()
    assert ratio_after < ratio_before * 0.1, \
        f"Rotation did not spread outlier: before={ratio_before:.1f}, after={ratio_after:.1f}"


def main():
    print("\n" + "=" * 60)
    print("  TurboQuant Sanity Check Suite")
    print("=" * 60)

    tests = [
        ("FHT orthonormality (H²=I)", test_fht_orthonormality),
        ("FHT self-inverse", test_fht_self_inverse),
        ("Rotation invertibility", test_rotation_invertibility),
        ("Lloyd-Max 3-bit distortion", test_lloyd_max_distortion),
        ("TurboQuantTensor compress/decompress", test_turboquant_tensor_roundtrip),
        ("Compression ratio (~10x)", test_compression_ratio),
        ("TQLinear output vs FP32", test_tq_linear_output),
        ("TQConv2d output vs FP32", test_tq_conv2d_output),
        ("safe_block_dim", test_safe_block_dim),
        ("Bias not quantized", test_bias_not_quantized),
        ("Bug 1 fix: idx/norms as nn.Buffer", test_tq_idx_norms_are_buffers),
        ("PINN model conversion + forward", test_pinn_model_conversion),
        ("Rotation spreads outliers", test_rotation_spreads_outliers),
    ]

    passed, failed = 0, 0
    for name, fn in tests:
        if run_test(name, fn):
            passed += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{passed+failed} passed", end="")
    if failed == 0:
        print(f"  [\033[92mALL PASS\033[0m]")
    else:
        print(f"  [\033[91m{failed} FAILED\033[0m]")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
