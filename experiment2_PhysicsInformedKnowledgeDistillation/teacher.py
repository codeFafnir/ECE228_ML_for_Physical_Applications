"""
Teacher model loader with forward hook for feature extraction.

Wraps ImprovedPhysicsInformedUNet so that a single forward call returns:
  - output  : (B, 32, 4, 576)  — channel estimate (same as plain model)
  - feat    : (B, 72, 256)     — TransformerChannelDecoder per-token output
                                  used as the xattn distillation target

The hook is registered on `model.transformer_decoder` (the
TransformerChannelDecoder instance), whose forward() returns (B, 72, 256)
after output_proj + LayerNorm.

Usage:
    teacher = TeacherWithFeatures.load(checkpoint_path, device)
    with torch.no_grad():
        out, feat = teacher(smomp, rss)
    teacher.remove_hook()
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

_EXP2_DIR = Path(__file__).parent
_EXP1_SHARED = _EXP2_DIR.parent / "experiment1_turboquant" / "shared"
if str(_EXP1_SHARED) not in sys.path:
    sys.path.insert(0, str(_EXP1_SHARED))

_PINN_DIR = _EXP2_DIR.parent / "PINN_channel-estimation-main"
if str(_PINN_DIR) not in sys.path:
    sys.path.insert(0, str(_PINN_DIR))


class TeacherWithFeatures:
    """
    Thin wrapper around the frozen teacher that exposes the
    TransformerChannelDecoder output (B, 72, 256) via a forward hook.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._feat: Optional[torch.Tensor] = None
        # Hook on the TransformerChannelDecoder instance (not the inner
        # nn.TransformerDecoder); its forward() returns (B, 72, 256).
        self._hook = model.transformer_decoder.register_forward_hook(
            self._capture_hook
        )

    def _capture_hook(
        self, module: nn.Module, inp: tuple, output: torch.Tensor
    ) -> None:
        self._feat = output  # (B, 72, 256)

    def __call__(
        self, initial_channel: torch.Tensor, rss_map: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._feat = None
        out = self.model(initial_channel, rss_map)
        feat = self._feat
        if feat is None:
            raise RuntimeError(
                "TransformerChannelDecoder hook did not fire — check model structure."
            )
        return out, feat

    def remove_hook(self) -> None:
        self._hook.remove()

    @classmethod
    def load(
        cls,
        checkpoint_path: Optional[str],
        device: torch.device,
    ) -> "TeacherWithFeatures":
        """
        Build and load the teacher, move to device, set eval+no_grad.

        Uses load_fp32_model from experiment1_turboquant/shared/model_loader.py
        so checkpoint handling is identical to the quantization experiments.
        """
        from model_loader import load_fp32_model

        model = load_fp32_model(checkpoint_path)
        model.eval()
        model.to(device)
        # Freeze all teacher weights
        for p in model.parameters():
            p.requires_grad_(False)
        return cls(model)
