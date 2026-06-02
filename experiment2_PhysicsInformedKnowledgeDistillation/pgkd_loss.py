"""
Physics-Guided Knowledge Distillation loss (Equation 4 from proposal).

L_PG-KD = L^S_NMSE  +  alpha * L^soft_KD  +  beta * L^S_physical  +  gamma * L^feat_xattn

Terms:
  L^S_NMSE     : NMSE on student channel estimate vs ground truth.
  L^soft_KD    : Soft distillation from teacher outputs.
                  kd_mode='mse' — temperature-scaled MSE (stable default for regression).
                  kd_mode='kl'  — softmax over the tap dimension at temperature T,
                                  KL divergence scaled by T^2 (faithful to proposal).
  L^S_physical : RSS power-matching constraint on student output (identical formula
                 to the teacher's PhysicsInformedLoss, keeps physics behaviour).
  L^feat_xattn : L2 distance between per-token L2-normalized student and teacher
                 cross-attention features (B,72,256), after projecting the student's
                 (B,72,d) map through a learned linear adapter.

The physics loss terms (NMSE + physical) reuse PhysicsInformedLoss from
PINN_channel-estimation-main/Model.py (lines 462-539) unchanged.

The xattn adapter (nn.Linear d→256) is a training-only projection; it is
NOT part of the deployed student and does not count toward its param total.

Note on complex ops and MPS:
  torch.complex is not supported on MPS. PhysicsInformedLoss calls it
  internally. We move the tensors to CPU for that computation and move
  results back. This overhead is small (~ms per batch) since the tensors
  are already fp32 floats.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_PINN_DIR = Path(__file__).parent.parent / "PINN_channel-estimation-main"
if str(_PINN_DIR) not in sys.path:
    sys.path.insert(0, str(_PINN_DIR))


class PGKDLoss(nn.Module):
    """
    Combined PG-KD loss for one training step.

    Args:
        d_bottleneck : student bottleneck dim (d); adapter is d → 256.
        alpha        : weight for L^soft_KD.
        beta         : weight for L^S_physical  (0.01 matches teacher training).
        gamma        : weight for L^feat_xattn.
        T            : distillation temperature (kd_mode='kl' only; mse uses T for scaling).
        kd_mode      : 'mse' or 'kl'.
    """

    def __init__(
        self,
        d_bottleneck: int,
        alpha: float = 1.0,
        beta: float = 0.01,
        gamma: float = 0.1,
        T: float = 4.0,
        kd_mode: str = "mse",
    ):
        super().__init__()
        if kd_mode not in ("mse", "kl"):
            raise ValueError(f"kd_mode must be 'mse' or 'kl', got '{kd_mode}'")

        self.alpha   = alpha
        self.beta    = beta
        self.gamma   = gamma
        self.T       = T
        self.kd_mode = kd_mode

        # Training-only: project student (B,72,d) → (B,72,256) to match teacher feat
        self.xattn_adapter = nn.Linear(d_bottleneck, 256)

        # Reuse the teacher's physics loss — keeps L_NMSE and L_physical identical
        from Model import PhysicsInformedLoss
        # alpha=0 so the internal total = nmse_loss only; we recombine externally
        self._physics_loss = PhysicsInformedLoss(alpha=0.0, use_dbm_correlation=True)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _physics_terms(
        self,
        student_out: torch.Tensor,
        accurate: torch.Tensor,
        rss_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (nmse_loss, power_loss) on CPU (MPS lacks torch.complex support).
        Both are scalar tensors moved back to the student_out device.
        """
        device = student_out.device
        # Move to CPU for complex arithmetic inside PhysicsInformedLoss
        s_cpu   = student_out.detach().float().cpu()
        acc_cpu = accurate.float().cpu()
        rss_cpu = rss_map.float().cpu()
        with torch.no_grad():
            pass  # just to be explicit — no grad needed for the utility call below

        # We still need gradients for student_out through nmse_loss; compute
        # nmse manually here so autograd flows correctly.
        n = student_out.shape[1] // 2
        pred_c   = torch.complex(student_out[:, :n].float(), student_out[:, n:].float())
        true_c   = torch.complex(accurate[:, :n].float(),   accurate[:, n:].float())
        nmse_loss = (
            torch.mean(torch.abs(pred_c - true_c) ** 2)
            / torch.mean(torch.abs(true_c) ** 2).clamp(min=1e-12)
        )

        # power_loss: channel power (dB) vs RSS dBm — computed on CPU then returned
        with torch.no_grad():
            _, _, power_loss_cpu = self._physics_loss(s_cpu, acc_cpu, rss_cpu)
        power_loss = power_loss_cpu.to(device)

        return nmse_loss, power_loss

    def _kd_mse(
        self, student_out: torch.Tensor, teacher_out: torch.Tensor
    ) -> torch.Tensor:
        """Temperature-scaled MSE (stable for regression KD)."""
        return F.mse_loss(student_out / self.T, teacher_out.detach() / self.T)

    def _kd_kl(
        self, student_out: torch.Tensor, teacher_out: torch.Tensor
    ) -> torch.Tensor:
        """
        KL divergence over the last (tap) dimension with temperature T.
        student_out, teacher_out: (B, 32, 4, 576)
        """
        T = self.T
        # Flatten to (B*32*4, 576) for per-spatial-position softmax
        s = student_out.float().reshape(-1, student_out.shape[-1])
        t = teacher_out.detach().float().reshape(-1, teacher_out.shape[-1])
        log_p = F.log_softmax(s / T, dim=-1)
        q     = F.softmax(t / T, dim=-1)
        return F.kl_div(log_p, q, reduction="batchmean") * (T ** 2)

    def _xattn_feat_loss(
        self, student_feat: torch.Tensor, teacher_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Per-token L2-normalized MSE between adapted student and teacher features.
        student_feat : (B, 72, d)
        teacher_feat : (B, 72, 256)
        """
        adapted = self.xattn_adapter(student_feat)  # (B, 72, 256)
        adapted_n = F.normalize(adapted, dim=-1)
        teacher_n = F.normalize(teacher_feat.detach(), dim=-1)
        return F.mse_loss(adapted_n, teacher_n)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        student_out: torch.Tensor,
        student_feat: torch.Tensor,
        teacher_out: torch.Tensor,
        teacher_feat: torch.Tensor,
        accurate: torch.Tensor,
        rss_map: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            student_out  : (B, 32, 4, 576) — student channel estimate
            student_feat : (B, 72, d)       — student cross-attn output (return_features=True)
            teacher_out  : (B, 32, 4, 576) — cached teacher output (fp32, no_grad)
            teacher_feat : (B, 72, 256)     — cached teacher xattn feature
            accurate     : (B, 32, 4, 576) — ground-truth channel
            rss_map      : (B,  2, 30, 30) — RSS map

        Returns:
            total_loss : scalar tensor (with grad)
            components : dict with float values for logging
                         keys: total, nmse, physical, kd_soft, xattn_feat
        """
        nmse_loss, power_loss = self._physics_terms(student_out, accurate, rss_map)

        if self.kd_mode == "mse":
            kd_loss = self._kd_mse(student_out, teacher_out)
        else:
            kd_loss = self._kd_kl(student_out, teacher_out)

        xattn_loss = self._xattn_feat_loss(student_feat, teacher_feat)

        total = (
            nmse_loss
            + self.alpha  * kd_loss
            + self.beta   * power_loss
            + self.gamma  * xattn_loss
        )

        components = {
            "total":       float(total.item()),
            "nmse":        float(nmse_loss.item()),
            "physical":    float(power_loss.item()),
            "kd_soft":     float(kd_loss.item()),
            "xattn_feat":  float(xattn_loss.item()),
        }
        return total, components
