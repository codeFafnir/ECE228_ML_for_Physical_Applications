"""
Calibration and validation data loaders for PTQ experiments.

Real data:   reuses GlobalNormalizedDataset from PINN source.
Synthetic:   SyntheticChannelDataset with correct shapes (rss 2×30×30).

Both return DataLoader yielding (smomp, accurate, rss) triples.

Calibration subset uses torch.randperm seeded at 0 to pick 128 samples
from the training split — reproducible across all methods.
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

PINN_DIR = Path(__file__).parent.parent.parent / "PINN_channel-estimation-main"
if str(PINN_DIR) not in sys.path:
    sys.path.insert(0, str(PINN_DIR))

import os
_cwd_backup = os.getcwd()


class SyntheticChannelDataset(torch.utils.data.Dataset):
    """
    Random (smomp, accurate, rss) triples matching the PINN input shapes.
      smomp / accurate: (32, 4, 576)
      rss:              (2, 30, 30)   ← crop_size=30
    """

    def __init__(self, n_samples: int = 256, seed: int = 42):
        g = torch.Generator()
        g.manual_seed(seed)
        self.smomps = torch.randn(n_samples, 32, 4, 576, generator=g) * 0.1
        self.accurates = torch.randn(n_samples, 32, 4, 576, generator=g) * 0.1
        self.rss_maps = torch.rand(n_samples, 2, 30, 30, generator=g) * 2 - 1

    def __len__(self) -> int:
        return len(self.smomps)

    def __getitem__(self, idx: int):
        return self.smomps[idx], self.accurates[idx], self.rss_maps[idx]


def _make_rss_processor(rss_image_path: str):
    """Build RSSMapProcessor with the standard config from train.py."""
    os.chdir(str(PINN_DIR))
    from find_in_map import RSSMapProcessor
    return RSSMapProcessor(
        image_path=rss_image_path,
        bs_pixel_coords=(287, 293),
        bs_real_coords=(71.06, 246.29),
        image_width_meters=527.5,
    )


def get_calibration_loader(
    smomp_file: str,
    accurate_file: str,
    user_positions_file: str,
    rss_image_path: str,
    n_cal: int = 128,
    batch_size: int = 8,
) -> DataLoader:
    """
    128 samples from the training split for Hessian accumulation.

    Uses the same train/val/test split as training (train_ratio=0.7,
    val_ratio=0.15, random_seed=42). Calibration subset is a seeded
    random draw of n_cal indices from the train split.
    """
    from Model import GlobalNormalizedDataset
    rss_proc = _make_rss_processor(rss_image_path)

    train_ds = GlobalNormalizedDataset(
        smomp_file=smomp_file,
        accurate_file=accurate_file,
        user_positions_file=user_positions_file,
        rss_processor=rss_proc,
        crop_size=30,
        split="train",
        train_ratio=0.7,
        val_ratio=0.15,
        random_seed=42,
        use_dbm_values=True,
    )

    g = torch.Generator()
    g.manual_seed(0)
    perm = torch.randperm(len(train_ds), generator=g)
    cal_indices = perm[:n_cal].tolist()
    cal_ds = Subset(train_ds, cal_indices)

    return DataLoader(cal_ds, batch_size=batch_size, shuffle=False, num_workers=0)


def get_val_loader(
    smomp_file: str,
    accurate_file: str,
    user_positions_file: str,
    rss_image_path: str,
    batch_size: int = 8,
) -> DataLoader:
    """Full validation split for NMSE evaluation."""
    from Model import GlobalNormalizedDataset
    rss_proc = _make_rss_processor(rss_image_path)

    val_ds = GlobalNormalizedDataset(
        smomp_file=smomp_file,
        accurate_file=accurate_file,
        user_positions_file=user_positions_file,
        rss_processor=rss_proc,
        crop_size=30,
        split="val",
        train_ratio=0.7,
        val_ratio=0.15,
        random_seed=42,
        use_dbm_values=True,
    )

    return DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)


def get_synthetic_loaders(
    n_cal: int = 128,
    n_val: int = 256,
    batch_size: int = 8,
) -> tuple[DataLoader, DataLoader]:
    """Fallback synthetic loaders when real data is unavailable."""
    cal_ds = SyntheticChannelDataset(n_samples=n_cal, seed=1)
    val_ds = SyntheticChannelDataset(n_samples=n_val, seed=2)
    cal_loader = DataLoader(cal_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return cal_loader, val_loader
