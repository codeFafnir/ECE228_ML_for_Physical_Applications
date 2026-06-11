
"""
train_qat.py
============
Quantization-Aware Training (QAT) for ImprovedPhysicsInformedUNet.

Strategy: Physics-Shielded Mixed-Precision QAT
  ┌──────────────────────────────┬───────────┬──────────────────────────────────────────────┐
  │ Submodule                    │ Precision │ Reason                                       │
  ├──────────────────────────────┼───────────┼──────────────────────────────────────────────┤
  │ enc1, enc2, enc3             │ INT8      │ Bulk of FLOPs; linear loss path              │
  │ dec1, dec2, dec3 (ConvT)     │ INT8      │ fbgemm per-tensor for ConvTranspose2d        │
  │ skip_conv1/2, to/from_seq    │ INT8      │ Linear path; low sensitivity                 │
  ├──────────────────────────────┼───────────┼──────────────────────────────────────────────┤
  │ rss_encoder                  │ FP32      │ Feeds RSS power-matching physics term;       │
  │                              │           │ quant noise propagates quadratically through │
  │                              │           │ the physics constraint                       │
  │ transformer_decoder          │ FP32      │ Softmax in attention saturates under INT8    │
  │ cross_attention.multihead_attn│ FP32     │ Same attention-stability reason              │
  │ final_conv                   │ FP32      │ Complex output projection; keeps output      │
  │                              │           │ manifold intact                              │
  └──────────────────────────────┴───────────┴──────────────────────────────────────────────┘

Observer choice: HistogramObserver (activation) + PerChannelMinMaxObserver (weight).
  - HistogramObserver collects a 2048-bin histogram over Phase 1 and computes
    calibrated min/max via percentile clipping; significantly better than
    MovingAverageMinMaxObserver on activations with heavy tails (e.g. transformer
    residuals, skip connections).

Three-phase training schedule:
  Phase 1  epochs [0,               freeze_bn_epoch):   observers ON,  fake-quant ON
  Phase 2  epochs [freeze_bn_epoch, freeze_obs_epoch):  observers OFF, fake-quant ON
  Phase 3  epochs [freeze_obs_epoch, qat_epochs):       observers OFF, fake-quant OFF

Run location: experiment3_qat/ folder.
  - train_qat.py and Model.py both live in experiment3_qat/.
  - PINN data files live in ../PINN_channel-estimation-main/ (one level up).
  - sys.path is set to the experiment3_qat/ directory so Model.py is always found first.
  - All output paths are written relative to experiment3_qat/ (or absolute if preferred).

Usage
-----
  cd experiment3_qat/
  python train_qat.py

Outputs (written to experiment3_qat/)
-------
  pinn_qat_best_val.pth   - best-val fake-quant model state dict
  pinn_int8.pth           - converted INT8 model state dict
  qat_training_log.txt    - epoch-by-epoch log
"""

import os
import sys
import math
import time
import logging
import threading
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Path setup — resolve to the directory containing THIS script (experiment3_qat/).
# This is robust regardless of the cwd when the script is invoked.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../experiment3_qat
PINN_DIR   = os.path.join(SCRIPT_DIR, "..", "PINN_channel-estimation-main")
PINN_DIR   = os.path.normpath(PINN_DIR)                   # clean up the ..

# Insert experiment3_qat/ first so our updated Model.py is always resolved before any
# copy that might exist elsewhere on the path.
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
# Also make the PINN project directory available for find_in_map, etc.
if PINN_DIR not in sys.path:
    sys.path.insert(1, PINN_DIR)

os.chdir(SCRIPT_DIR)
print(f"Working dir : {os.getcwd()}")
print(f"sys.path[0] : {sys.path[0]}   (Model.py resolved from here)")
print(f"sys.path[1] : {sys.path[1]}   (find_in_map, dataset utils)")

# ---------------------------------------------------------------------------
# Keep-alive thread — prevents Colab disconnecting mid-training
# ---------------------------------------------------------------------------
def _keep_alive():
    while True:
        time.sleep(60)
        print(".", end="", flush=True)

threading.Thread(target=_keep_alive, daemon=True).start()

# ---------------------------------------------------------------------------
# Quantization imports
# ---------------------------------------------------------------------------
from torch.ao.quantization import (
    QConfigMapping,
    get_default_qat_qconfig,
    prepare_qat,
    convert,
)
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from torch.ao.quantization.fake_quantize import (
    disable_observer,
    disable_fake_quant,
    enable_observer,
    enable_fake_quant,
)

# ---------------------------------------------------------------------------
# Project imports  (resolved from experiment3_qat/ via sys.path above)
# ---------------------------------------------------------------------------
from Model import (
    ImprovedPhysicsInformedUNet,
    PhysicsInformedLoss,
    GlobalNormalizedDataset,
    create_datasets,
    evaluate_test_set,
    save_checkpoint,
    load_checkpoint,
)
from find_in_map import RSSMapProcessor

# ---------------------------------------------------------------------------
# Logging — written to experiment3_qat/qat_training_log.txt
# ---------------------------------------------------------------------------
_log_path = os.path.join(SCRIPT_DIR, "qat_training_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_log_path, mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ===========================================================================
# 1.  QConfigMapping — physics-shielded with HistogramObserver
# ===========================================================================
def build_qconfig_mapping(backend: str = "x86") -> QConfigMapping:
    """
    Builds a mixed-precision QConfigMapping:

    Activations  — HistogramObserver (2048-bin calibration; percentile clipping)
                   Better than MovingAverageMinMaxObserver for activation
                   distributions with heavy tails (skip connections, residuals).

    Weights (Conv2d / Linear)     — PerChannelMinMaxObserver (per-channel symmetric)
    Weights (ConvTranspose2d)     — MinMaxObserver (per-tensor symmetric)
                                    fbgemm does not support per-channel for transposed conv.

    Physics-shielded FP32 modules (qconfig = None):
      - rss_encoder              : feeds RSS power-matching physics term directly
      - transformer_decoder      : attention softmax unstable under INT8
      - cross_attention.multihead_attn : same attention-stability reason
      - final_conv               : complex output projection
    """
    from torch.ao.quantization.qconfig import QConfig
    from torch.ao.quantization.fake_quantize import FakeQuantize
    from torch.ao.quantization.observer import (
        HistogramObserver,
        MinMaxObserver,
        PerChannelMinMaxObserver,
    )

    # ------------------------------------------------------------------
    # Activation fake-quant: HistogramObserver for accurate range calibration
    # ------------------------------------------------------------------
    histogram_activation = FakeQuantize.with_args(
        observer=HistogramObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=True,
    )

    # ------------------------------------------------------------------
    # Per-channel weight fake-quant for Conv2d / Linear
    # ------------------------------------------------------------------
    per_channel_weight = FakeQuantize.with_args(
        observer=PerChannelMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
    )

    # ------------------------------------------------------------------
    # Per-tensor weight fake-quant for ConvTranspose2d in decoder blocks
    # (fbgemm constraint — per-channel not supported for transposed conv)
    # ------------------------------------------------------------------
    per_tensor_weight = FakeQuantize.with_args(
        observer=MinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
    )

    per_channel_qconfig = QConfig(
        activation=histogram_activation,
        weight=per_channel_weight,
    )

    per_tensor_qconfig = QConfig(
        activation=histogram_activation,
        weight=per_tensor_weight,
    )

    mapping = QConfigMapping()
    mapping.set_global(per_channel_qconfig)

    # ------------------------------------------------------------------
    # Physics shield — these four submodules stay in FP32
    # ------------------------------------------------------------------
    for name in [
        "rss_encoder",                               # RSS physics-matching term
        "transformer_decoder",                       # parent covers pos_encoder + layers
        "transformer_decoder.pos_encoder",           # explicit child coverage
        "transformer_decoder.transformer_decoder",   # explicit child coverage
        "cross_attention.multihead_attn",            # attention stability
        "final_conv",                                # complex output projection
    ]:
        mapping.set_module_name(name, None)

    # ------------------------------------------------------------------
    # ConvTranspose2d blocks in decoder — override to per-tensor weight
    # ------------------------------------------------------------------
    for name in [
        "dec1.conv.0", "dec1.residual.0",
        "dec2.conv.0", "dec2.residual.0",
        "dec3.conv.0", "dec3.residual.0",
    ]:
        mapping.set_module_name(name, per_tensor_qconfig)

    return mapping


# ===========================================================================
# 2.  Model preparation
# ===========================================================================

def prepare_model_for_qat(
    pretrained_path: str,
    qconfig_mapping: QConfigMapping,
    device: str = "cuda",
) -> nn.Module:
    """
    Load pre-trained weights, insert fake-quant observers via FX graph mode,
    move to device.

    FX graph mode is required (not eager mode) because the model has skip
    connections (torch.cat) and residual additions that need graph-level
    quantize/dequantize node placement.
    """
    log.info(f"Loading pre-trained weights from: {pretrained_path}")
    raw = torch.load(pretrained_path, map_location="cpu", weights_only=False)

    model = ImprovedPhysicsInformedUNet(channel_shape=(32, 4, 576))

    if isinstance(raw, dict) and "model_state_dict" in raw:
        model.load_state_dict(raw["model_state_dict"])
        log.info(f"  Loaded from checkpoint dict (epoch {raw.get('epoch', '?')})")
    else:
        model.load_state_dict(raw)
        log.info("  Loaded plain state dict")

    # FX tracing must happen on CPU in train() mode
    model.cpu().train()

    example_smomp = torch.randn(1, 32, 4, 576)
    example_rss   = torch.randn(1, 2, 30, 30)

    log.info("Running prepare_qat_fx (FX graph-mode) ...")
    model_prepared = prepare_qat_fx(
        model,
        qconfig_mapping,
        example_inputs=(example_smomp, example_rss),
    )
    log.info("  Fake-quant nodes inserted successfully.")

    _log_quantization_coverage(model_prepared)

    return model_prepared.to(device)


def _log_quantization_coverage(model: nn.Module) -> None:
    log.info("-" * 60)
    log.info("Quantization coverage (top-level submodules):")
    for name, mod in model.named_children():
        has_fq = any("FakeQuantize" in type(m).__name__ for m in mod.modules())
        tag    = "INT8 (fake-quant)" if has_fq else "FP32 (shielded)"
        log.info(f"  {name:<40s}  ->  {tag}")
    log.info("-" * 60)


# ===========================================================================
# 3.  Observer / fake-quant phase helpers
# ===========================================================================

def _apply_to_fq_nodes(model: nn.Module, fn) -> None:
    for mod in model.modules():
        if "FakeQuantize" in type(mod).__name__:
            fn(mod)

def freeze_observers(model: nn.Module) -> None:
    _apply_to_fq_nodes(model, disable_observer)
    log.info("  [QAT] Observers disabled - quantization ranges now frozen.")

def disable_all_fake_quant(model: nn.Module) -> None:
    _apply_to_fq_nodes(model, disable_fake_quant)
    log.info("  [QAT] Fake-quant disabled - FP32 polish phase.")

def enable_all_fake_quant(model: nn.Module) -> None:
    _apply_to_fq_nodes(model, enable_fake_quant)

def unfreeze_fake_quant(model: nn.Module) -> None:
    _apply_to_fq_nodes(model, enable_fake_quant)


# ===========================================================================
# 4.  Training and validation loops
# ===========================================================================

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: PhysicsInformedLoss,
    optimizer: Optional[torch.optim.Optimizer],
    device: str,
    train: bool,
    clip_norm: float = 0.5,
) -> tuple:
    """Returns (avg_loss, avg_nmse_db)."""
    model.train(train)
    total_loss = 0.0
    total_nmse = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for smomp, accurate, rss in loader:
            smomp    = smomp.to(device)
            accurate = accurate.to(device)
            rss      = rss.to(device)

            if train:
                optimizer.zero_grad()

            pred = model(smomp, rss)
            loss, nmse_linear, _ = criterion(pred, accurate, rss)

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                optimizer.step()

            total_loss += loss.item()
            nmse_val    = nmse_linear.item() if hasattr(nmse_linear, 'item') else float(nmse_linear)
            total_nmse += 10.0 * math.log10(max(nmse_val, 1e-12))
            n_batches  += 1

    if n_batches == 0:
        log.warning("  Warning: loader was empty - returning 0.0 for loss and NMSE.")
        return 0.0, 0.0

    return total_loss / n_batches, total_nmse / n_batches


# ===========================================================================
# 5.  Main QAT training function
# ===========================================================================

def train_qat(
    model_prepared: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    qat_epochs: int       = 100,
    lr: float             = 1e-4,
    freeze_bn_epoch: int  = 60,
    freeze_obs_epoch: int = 80,
    device: str           = "cuda",
    save_best_path: str   = "pinn_qat_best_val.pth",
) -> tuple:
    assert freeze_bn_epoch < freeze_obs_epoch < qat_epochs, (
        "freeze_bn_epoch < freeze_obs_epoch < qat_epochs must hold."
    )

    save_dir = os.path.dirname(save_best_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model_prepared.parameters()),
        lr=lr,
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=qat_epochs, eta_min=lr * 0.05
    )
    criterion = PhysicsInformedLoss(alpha=0.01)

    best_val_nmse_db = float("inf")
    train_losses:     list = []
    val_nmse_db_list: list = []

    log.info("=" * 60)
    log.info(f"Starting QAT fine-tuning for {qat_epochs} epochs")
    log.info(f"  LR = {lr}  |  Device = {device}")
    log.info(f"  Phase 1: epochs 0 - {freeze_bn_epoch - 1}  (observers ON,  fake-quant ON)")
    log.info(f"  Phase 2: epochs {freeze_bn_epoch} - {freeze_obs_epoch - 1}  (observers OFF, fake-quant ON)")
    log.info(f"  Phase 3: epochs {freeze_obs_epoch} - {qat_epochs - 1}  (fake-quant OFF, FP32 polish)")
    log.info("=" * 60)

    for epoch in range(qat_epochs):
        epoch_start = time.time()

        if epoch == freeze_bn_epoch:
            log.info(f"\n[Epoch {epoch}] -> Phase 2: freezing observer ranges.")
            freeze_observers(model_prepared)

        if epoch == freeze_obs_epoch:
            log.info(f"\n[Epoch {epoch}] -> Phase 3: disabling fake-quant (FP32 polish).")
            disable_all_fake_quant(model_prepared)

        train_loss, train_nmse_db = _run_epoch(
            model_prepared, train_loader, criterion,
            optimizer, device, train=True,
        )
        val_loss, val_nmse_db = _run_epoch(
            model_prepared, val_loader, criterion,
            optimizer=None, device=device, train=False,
        )

        scheduler.step()
        train_losses.append(train_loss)
        val_nmse_db_list.append(val_nmse_db)

        elapsed = time.time() - epoch_start
        log.info(
            f"Epoch {epoch + 1:03d}/{qat_epochs}  "
            f"Train Loss={train_loss:.5f}  Train NMSE={train_nmse_db:.2f} dB  |  "
            f"Val Loss={val_loss:.5f}  Val NMSE={val_nmse_db:.2f} dB  "
            f"[{elapsed:.1f}s]"
        )

        if val_nmse_db < best_val_nmse_db:
            best_val_nmse_db = val_nmse_db
            torch.save(model_prepared.state_dict(), save_best_path)
            log.info(f"  + Best val NMSE: {best_val_nmse_db:.2f} dB -> {save_best_path}")

    # weights_only=False required for fake-quant observer state in PyTorch 2.x
    if os.path.exists(save_best_path):
        log.info(f"\nLoading best QAT weights from {save_best_path} ...")
        model_prepared.load_state_dict(
            torch.load(save_best_path, map_location=device, weights_only=False)
        )
    else:
        log.warning(f"  Best checkpoint not found at {save_best_path} - returning last weights.")

    log.info(f"QAT training complete. Best Val NMSE = {best_val_nmse_db:.2f} dB")
    return model_prepared, train_losses, val_nmse_db_list


# ===========================================================================
# 6.  INT8 conversion and export
# ===========================================================================
def export_int8_model(
    model_prepared: nn.Module,
    int8_save_path: str = "pinn_int8.pth",
) -> nn.Module:
    """
    Convert fake-quant model to actual INT8 operators via convert_fx.
    Must be called on a CPU model in eval() mode.
    """
    log.info("Converting fake-quant model to INT8 ...")

    # convert_fx requires CPU + eval mode
    model_prepared = model_prepared.cpu().eval()

    # Re-enable fake-quant so calibrated scale/zero-point values are visible
    enable_all_fake_quant(model_prepared)

    model_int8 = convert_fx(model_prepared)

    out_dir = os.path.dirname(int8_save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.save(model_int8.state_dict(), int8_save_path)
    log.info(f"INT8 model state dict saved to: {int8_save_path}")

    fp32_bytes = sum(
        p.numel() * p.element_size()
        for p in model_int8.parameters()
        if p.dtype == torch.float32
    )
    int8_bytes = sum(
        p.numel() * p.element_size()
        for p in model_int8.parameters()
        if p.dtype == torch.int8
    )
    log.info(f"  INT8 param bytes : {int8_bytes / 1e6:.1f} MB")
    log.info(f"  FP32 param bytes : {fp32_bytes / 1e6:.1f} MB  (physics-shielded layers)")

    return model_int8


# ===========================================================================
# 7.  Evaluation  (in-memory live QAT + INT8 vs FP32 base)
# ===========================================================================
def evaluate_qat_vs_base(
    base_checkpoint:   str,
    qat_int8_path:     str,
    test_loader:       DataLoader,
    device:            str = "cuda",
) -> dict:
    # ------------------------------------------------------------------
    # Base FP32 model
    # ------------------------------------------------------------------
    log.info("Evaluating base FP32 model ...")
    raw = torch.load(base_checkpoint, map_location=device, weights_only=False)
    base_model = ImprovedPhysicsInformedUNet(channel_shape=(32, 4, 576))
    if isinstance(raw, dict) and "model_state_dict" in raw:
        base_model.load_state_dict(raw["model_state_dict"])
    else:
        base_model.load_state_dict(raw)
    base_model = base_model.to(device).eval()

    base_nmse_linear = evaluate_test_set(base_model, test_loader, device=device)
    base_nmse_db     = 10.0 * math.log10(max(base_nmse_linear, 1e-12))
    base_disk_mb     = os.path.getsize(base_checkpoint) / 1e6
    base_param_mb    = sum(p.numel() * p.element_size() for p in base_model.parameters()) / 1e6

    dummy_smomp = torch.randn(1, 32, 4, 576)
    dummy_rss   = torch.randn(1, 2, 30, 30)
    base_cpu    = base_model.cpu()
    with torch.no_grad():
        for _ in range(50):
            base_cpu(dummy_smomp, dummy_rss)
        t0 = time.perf_counter()
        for _ in range(100):
            base_cpu(dummy_smomp, dummy_rss)
        base_latency_ms = (time.perf_counter() - t0) / 100 * 1000

    # ------------------------------------------------------------------
    # INT8 model — rebuild FX structure then load state dict
    # ------------------------------------------------------------------
    log.info("Rebuilding INT8 model structure for evaluation ...")
    qconfig_mapping = build_qconfig_mapping(backend="x86")
    model_for_int8  = prepare_model_for_qat(
        pretrained_path = base_checkpoint,
        qconfig_mapping = qconfig_mapping,
        device          = "cpu",
    )
    model_for_int8.cpu().eval()
    enable_all_fake_quant(model_for_int8)
    model_int8 = convert_fx(model_for_int8)

    loaded_state = torch.load(qat_int8_path, map_location="cpu", weights_only=False)
    model_int8.load_state_dict(loaded_state)
    model_int8.eval()
    log.info("  INT8 state dict loaded successfully.")

    int8_disk_mb  = os.path.getsize(qat_int8_path) / 1e6
    int8_bytes    = sum(p.numel() * p.element_size() for p in model_int8.parameters() if p.dtype == torch.int8)
    fp32_bytes    = sum(p.numel() * p.element_size() for p in model_int8.parameters() if p.dtype == torch.float32)
    int8_param_mb = (int8_bytes + fp32_bytes) / 1e6

    # INT8 inference must run on CPU
    cpu_test_loader = DataLoader(
        test_loader.dataset,
        batch_size  = test_loader.batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = False,
    )
    qat_nmse_linear = evaluate_test_set(model_int8, cpu_test_loader, device="cpu")
    qat_nmse_db     = 10.0 * math.log10(max(qat_nmse_linear, 1e-12))

    with torch.no_grad():
        for _ in range(50):
            model_int8(dummy_smomp, dummy_rss)
        t0 = time.perf_counter()
        for _ in range(100):
            model_int8(dummy_smomp, dummy_rss)
        qat_latency_ms = (time.perf_counter() - t0) / 100 * 1000

    degradation_db  = qat_nmse_db - base_nmse_db
    speedup         = base_latency_ms / qat_latency_ms if qat_latency_ms > 0 else float("nan")
    disk_reduction  = (1 - int8_disk_mb / base_disk_mb) * 100 if base_disk_mb > 0 else float("nan")
    param_reduction = (1 - int8_param_mb / base_param_mb) * 100 if base_param_mb > 0 else float("nan")

    results = {
        "base_nmse_db":       base_nmse_db,
        "qat_nmse_db":        qat_nmse_db,
        "degradation_db":     degradation_db,
        "base_latency_ms":    base_latency_ms,
        "qat_latency_ms":     qat_latency_ms,
        "speedup":            speedup,
        "base_disk_mb":       base_disk_mb,
        "int8_disk_mb":       int8_disk_mb,
        "disk_reduction_pct": disk_reduction,
        "base_param_mb":      base_param_mb,
        "int8_param_mb":      int8_param_mb,
        "param_reduction_pct":param_reduction,
    }

    log.info("=" * 60)
    log.info("Evaluation results")
    log.info(f"  NMSE — Base FP32 : {base_nmse_db:.2f} dB")
    log.info(f"  NMSE — QAT INT8  : {qat_nmse_db:.2f} dB")
    log.info(f"  Degradation      : {degradation_db:+.2f} dB  "
             f"({'OK' if degradation_db < 0.5 else 'review needed'})")
    log.info(f"  Latency — Base   : {base_latency_ms:.2f} ms")
    log.info(f"  Latency — INT8   : {qat_latency_ms:.2f} ms")
    log.info(f"  Speedup          : {speedup:.2f}x")
    log.info(f"  Disk — Base      : {base_disk_mb:.1f} MB")
    log.info(f"  Disk — INT8      : {int8_disk_mb:.1f} MB  ({disk_reduction:.1f}% smaller)")
    log.info(f"  Params — Base    : {base_param_mb:.1f} MB")
    log.info(f"  Params — INT8    : {int8_param_mb:.1f} MB  ({param_reduction:.1f}% smaller)")
    log.info("=" * 60)

    return results


# ===========================================================================
# 8.  Entry point
# ===========================================================================

if __name__ == "__main__":

    # PINN data files live in ../PINN_channel-estimation-main/ relative to experiment3_qat/
    _pinn = PINN_DIR   # already resolved to absolute path above

    config = {
        # ---- Dataset (all relative to PINN_channel-estimation-main/) ----
        "smomp_file":           os.path.join(_pinn, "initial_estimate_ls_snr0.npy"),
        "accurate_file":        os.path.join(_pinn, "3D_channel_15GHz_2x2_Pt50.npy"),
        "user_positions_file":  os.path.join(_pinn, "ue_positions_noisy.txt"),
        "rss_image_path":       os.path.join(_pinn, "Dataset", "50_15GHz.jpg"),
        "bs_pixel_coords":      (287, 293),
        "bs_real_coords":       (71.06, 246.29),
        "image_width_meters":   527.5,
        "batch_size":           32,

        # ---- QAT ----
        "pretrained_checkpoint": os.path.join(_pinn, "simple_ls_0_val.pth"),
        "qat_backend":           "x86",
        "qat_epochs":            100,
        "qat_lr":                1e-4,
        "freeze_bn_epoch":       60,
        "freeze_obs_epoch":      80,

        # ---- Outputs (written to experiment3_qat/) ----
        "name_qat_best_val": os.path.join(SCRIPT_DIR, "pinn_qat_best_val.pth"),
        "name_qat_int8":     os.path.join(SCRIPT_DIR, "pinn_int8.pth"),

        "device": "cuda",
    }

    # ------------------------------------------------------------------
    # Validate all required files up front
    # ------------------------------------------------------------------
    required_files = [
        config["smomp_file"],
        config["accurate_file"],
        config["user_positions_file"],
        config["rss_image_path"],
        config["pretrained_checkpoint"],
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        for f in missing:
            log.error(f"Required file not found: {f}")
        raise FileNotFoundError(
            "One or more required files are missing.\n"
            "Expected layout:\n"
            "  experiment3_qat/\n"
            "    train_qat.py   <-- this file\n"
            "    Model.py\n"
            "  PINN_channel-estimation-main/\n"
            "    initial_estimate_ls_snr0.npy\n"
            "    3D_channel_15GHz_2x2_Pt50.npy\n"
            "    ue_positions_noisy.txt\n"
            "    simple_ls_0_val.pth\n"
            "    Dataset/50_15GHz.jpg\n"
            "Run Steps 1-3 in the README to generate the missing files."
        )

    torch.backends.quantized.engine = config["qat_backend"]
    log.info(f"Quantization backend : {config['qat_backend']}")
    log.info(f"PINN data dir        : {_pinn}")
    log.info(f"Output dir           : {SCRIPT_DIR}")

    # ------------------------------------------------------------------
    # Build datasets and DataLoaders
    # ------------------------------------------------------------------
    log.info("Initialising RSS map processor ...")
    rss_processor = RSSMapProcessor(
        image_path         = config["rss_image_path"],
        bs_pixel_coords    = config["bs_pixel_coords"],
        bs_real_coords     = config["bs_real_coords"],
        image_width_meters = config["image_width_meters"],
    )
    log.info("RSS map processor initialised")

    log.info("Building datasets ...")
    train_dataset, val_dataset, test_dataset = create_datasets(
        config["smomp_file"],
        config["accurate_file"],
        config["user_positions_file"],
        rss_processor,
    )
    log.info("Datasets for train, val, test available")

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"],
        shuffle=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"],
        shuffle=False, num_workers=4, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"],
        shuffle=False, num_workers=4, pin_memory=True,
    )
    log.info("DataLoaders ready (train / val / test)")

    # ------------------------------------------------------------------
    # Build QConfigMapping, prepare model, train, export, evaluate
    # ------------------------------------------------------------------
    log.info("Building QConfigMapping (physics-shielded, HistogramObserver) ...")
    qconfig_mapping = build_qconfig_mapping(backend=config["qat_backend"])

    log.info("Preparing model for QAT ...")
    model_prepared = prepare_model_for_qat(
        pretrained_path = config["pretrained_checkpoint"],
        qconfig_mapping = qconfig_mapping,
        device          = config["device"],
    )

    log.info("Starting QAT fine-tuning ...")
    model_prepared, train_losses, val_nmse_list = train_qat(
        model_prepared   = model_prepared,
        train_loader     = train_loader,
        val_loader       = val_loader,
        qat_epochs       = config["qat_epochs"],
        lr               = config["qat_lr"],
        freeze_bn_epoch  = config["freeze_bn_epoch"],
        freeze_obs_epoch = config["freeze_obs_epoch"],
        device           = config["device"],
        save_best_path   = config["name_qat_best_val"],
    )

    # ------------------------------------------------------------------
    # Export INT8 and run full comparison table
    # ------------------------------------------------------------------
    model_int8 = export_int8_model(
        model_prepared = model_prepared,
        int8_save_path = config["name_qat_int8"],
    )

    evaluation_results = evaluate_qat_vs_base(
        base_checkpoint = config["pretrained_checkpoint"],
        qat_int8_path   = config["name_qat_int8"],
        test_loader     = test_loader,
        device          = config["device"],
    )

    log.info("\nAll done.")
    log.info(f"  Best QAT val NMSE  : {min(val_nmse_list):.2f} dB")
    log.info(f"  QAT INT8 test NMSE : {evaluation_results['qat_nmse_db']:.2f} dB")
    log.info(f"  vs base test NMSE  : {evaluation_results['base_nmse_db']:.2f} dB")
    log.info(f"  NMSE degradation   : {evaluation_results['degradation_db']:+.2f} dB")
    log.info(f"  Inference speedup  : {evaluation_results['speedup']:.2f}x")
    log.info(f"\nOutputs saved to: {SCRIPT_DIR}")
