"""Save and load packed GPTQ models for reuse."""

import json
from pathlib import Path

import torch
import torch.nn as nn

from .gptq_layers import model_storage_bytes


def save_gptq_model(
    model: nn.Module,
    save_dir: str | Path,
    name: str,
    num_bits: int,
    group_size: int,
    source_checkpoint: str | None = None,
    eval_nmse_db: float | None = None,
) -> Path:
    """
    Save full GPTQ model (pickle) + state_dict + manifest for reuse.

    Returns path to the main .pth file.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / f"{name}.pth"
    state_path = save_dir / f"{name}_state_dict.pth"
    manifest_path = save_dir / f"{name}_manifest.json"

    packed_bytes, fp32_w_bytes, weight_ratio = model_storage_bytes(model)
    all_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    all_bytes += sum(b.numel() * b.element_size() for b in model.buffers())

    torch.save(model, model_path)
    torch.save(model.state_dict(), state_path)

    manifest = {
        "name": name,
        "num_bits": num_bits,
        "group_size": group_size,
        "packed": True,
        "source_checkpoint": source_checkpoint,
        "eval_nmse_db": eval_nmse_db,
        "model_bytes_mb": round(all_bytes / 1e6, 2),
        "weight_compression_ratio": round(weight_ratio, 2),
        "files": {
            "model": model_path.name,
            "state_dict": state_path.name,
        },
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return model_path


def load_gptq_model(path: str | Path, device: torch.device | str = "cpu") -> nn.Module:
    """Load a packed GPTQ model saved with save_gptq_model."""
    model = torch.load(path, map_location=device, weights_only=False)
    model.eval()
    return model
