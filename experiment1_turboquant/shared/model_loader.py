"""
Load the FP32 ImprovedPhysicsInformedUNet from a checkpoint.

Handles both checkpoint formats:
  - val checkpoint: raw model.state_dict()
  - train checkpoint: {'model_state_dict': ..., 'optimizer_state_dict': ..., ...}
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

PINN_DIR = Path(__file__).parent.parent.parent / "PINN_channel-estimation-main"
if str(PINN_DIR) not in sys.path:
    sys.path.insert(0, str(PINN_DIR))


def build_model(rss_size: int = 30) -> nn.Module:
    """
    Instantiate an untrained ImprovedPhysicsInformedUNet.

    rss_size=30 matches crop_size=30 in GlobalNormalizedDataset.
    """
    from Model import ImprovedPhysicsInformedUNet
    return ImprovedPhysicsInformedUNet(
        channel_shape=(32, 4, 576),
        rss_size=rss_size,
        use_dbm_values=True,
    )


def load_fp32_model(checkpoint_path: Optional[str], rss_size: int = 30) -> nn.Module:
    """
    Build the model and optionally load weights from a checkpoint.

    Args:
        checkpoint_path: Path to .pth file, or None for random weights.
        rss_size:        RSS map spatial size (default 30).

    Returns:
        model in eval() mode on CPU with FP32 weights.
    """
    model = build_model(rss_size=rss_size)
    model.eval()

    if checkpoint_path is None:
        print("WARNING: No checkpoint — using random weights.")
        return model

    p = Path(checkpoint_path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    print(f"Loaded checkpoint: {checkpoint_path}")
    return model
