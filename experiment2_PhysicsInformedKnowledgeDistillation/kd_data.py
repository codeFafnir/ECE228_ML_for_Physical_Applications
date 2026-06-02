"""
Data loaders for Physics-Guided Knowledge Distillation (experiment2).

Two access patterns:

1. KDCachedDataset — fastest training path.
   Wraps MmapChannelDataset + pre-computed teacher fp16 caches from
   precompute_teacher.py. Each __getitem__ returns:
     (smomp, accurate, rss, teacher_out, teacher_feat)

2. MmapChannelDataset (re-exported from experiment1 shared) — used for:
   - Building the cache in precompute_teacher.py
   - Eval-only (eval_kd.py doesn't need teacher features at val time)

Both loaders use the 80/10/10 seed-42 split to stay consistent with
experiment1_turboquant.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_EXP2_DIR = Path(__file__).parent
_EXP1_SHARED = _EXP2_DIR.parent / "experiment1_turboquant" / "shared"
if str(_EXP1_SHARED) not in sys.path:
    sys.path.insert(0, str(_EXP1_SHARED))

from calibration_data import (  # noqa: E402 (after sys.path insert)
    MmapChannelDataset,
    _DataBundle,
    _complex_to_real_imag,
)


# ── KD cached dataset ─────────────────────────────────────────────────────────

class KDCachedDataset(Dataset):
    """
    Dataset for distillation training.

    Each sample returns:
        smomp        : (32, 4, 576)  float32
        accurate     : (32, 4, 576)  float32
        rss          : (2, 30, 30)   float32
        teacher_out  : (32, 4, 576)  float32  (from cache, cast from fp16)
        teacher_feat : (72, 256)     float32  (from cache, cast from fp16)

    Args:
        base_dataset : MmapChannelDataset for the channel/RSS tensors.
        cache_dir    : directory containing {split}_out.npy + {split}_feat.npy
                       written by precompute_teacher.py.
        split        : 'train' or 'val' — selects which cache files to load.
    """

    def __init__(
        self,
        base_dataset: MmapChannelDataset,
        cache_dir: str,
        split: str,
    ):
        self.base = base_dataset
        cache_path = Path(cache_dir)

        out_file  = cache_path / f"{split}_out.npy"
        feat_file = cache_path / f"{split}_feat.npy"

        if not out_file.exists() or not feat_file.exists():
            raise FileNotFoundError(
                f"Teacher cache not found in {cache_dir}.\n"
                f"Run precompute_teacher.py first:\n"
                f"  python precompute_teacher.py --split {split} ..."
            )

        # Load as fp16 mmap; cast to fp32 in __getitem__ to avoid RAM doubling
        self.teacher_out  = np.load(out_file,  mmap_mode="r")   # (N, 32, 4, 576) fp16
        self.teacher_feat = np.load(feat_file, mmap_mode="r")   # (N, 72, 256)    fp16

        n_cache = self.teacher_out.shape[0]
        n_base  = len(self.base)
        if n_cache != n_base:
            raise ValueError(
                f"Cache size mismatch: cache has {n_cache} samples, "
                f"base dataset has {n_base}."
            )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple:
        smomp, accurate, rss = self.base[idx]
        teacher_out  = torch.from_numpy(self.teacher_out[idx].astype(np.float32))
        teacher_feat = torch.from_numpy(self.teacher_feat[idx].astype(np.float32))
        return smomp, accurate, rss, teacher_out, teacher_feat


# ── Loader factories ──────────────────────────────────────────────────────────

def _make_bundle(
    smomp_file: str,
    accurate_file: str,
    user_positions_file: str,
    rss_image_path: str,
) -> _DataBundle:
    return _DataBundle.get(smomp_file, accurate_file, user_positions_file, rss_image_path)


def get_kd_loaders(
    smomp_file: str,
    accurate_file: str,
    user_positions_file: str,
    rss_image_path: str,
    cache_dir: str,
    batch_size: int = 16,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and val DataLoaders for KD training.

    Returns (train_loader, val_loader).  Both yield 5-tuples:
      (smomp, accurate, rss, teacher_out, teacher_feat)
    """
    bundle = _make_bundle(smomp_file, accurate_file, user_positions_file, rss_image_path)
    train_ds_base = bundle.make_split_dataset("train")
    val_ds_base   = bundle.make_split_dataset("val")

    train_ds = KDCachedDataset(train_ds_base, cache_dir, split="train")
    val_ds   = KDCachedDataset(val_ds_base,   cache_dir, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
    return train_loader, val_loader


def get_val_loader_only(
    smomp_file: str,
    accurate_file: str,
    user_positions_file: str,
    rss_image_path: str,
    batch_size: int = 16,
    num_workers: int = 0,
) -> DataLoader:
    """
    Val loader returning plain (smomp, accurate, rss) triples.
    Used by eval_kd.py (no teacher features needed at eval time).
    """
    bundle = _make_bundle(smomp_file, accurate_file, user_positions_file, rss_image_path)
    val_ds = bundle.make_split_dataset("val")
    return DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )


def get_train_base_loader(
    smomp_file: str,
    accurate_file: str,
    user_positions_file: str,
    rss_image_path: str,
    batch_size: int = 16,
    num_workers: int = 0,
) -> DataLoader:
    """
    Train loader (no teacher cache) used by precompute_teacher.py.
    Returns (smomp, accurate, rss) triples in dataset order (no shuffle).
    """
    bundle = _make_bundle(smomp_file, accurate_file, user_positions_file, rss_image_path)
    ds = bundle.make_split_dataset("train")
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
    )
