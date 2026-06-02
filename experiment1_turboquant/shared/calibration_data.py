"""
Calibration and validation data loaders for PTQ experiments.

Real data:   mmap-backed channel arrays + shared cache (one load for cal+val).
Synthetic:   SyntheticChannelDataset with correct shapes (rss 2×30×30).

Both return DataLoader yielding (smomp, accurate, rss) triples.

Calibration subset uses torch.randperm seeded at 0 to pick n_cal samples
from the training split — reproducible across all methods.
Evaluation uses the validation split (never test).
"""

import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

PINN_DIR = Path(__file__).parent.parent.parent / "PINN_channel-estimation-main"
if str(PINN_DIR) not in sys.path:
    sys.path.insert(0, str(PINN_DIR))


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


def _complex_to_real_imag(sample: np.ndarray) -> np.ndarray:
    """(C, H, W) complex -> (2C, H, W) float32."""
    return np.concatenate([np.real(sample), np.imag(sample)], axis=0).astype(np.float32)


def _compute_global_norm_max(
    smomp_mmap: np.ndarray,
    accurate_mmap: np.ndarray,
    chunk_size: int = 128,
) -> float:
    """Streaming max-abs over all samples without materializing full arrays."""
    smomp_max = 0.0
    accurate_max = 0.0
    n = smomp_mmap.shape[0]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        smomp_chunk = _complex_to_real_imag(smomp_mmap[start:end])
        accurate_chunk = _complex_to_real_imag(accurate_mmap[start:end])
        smomp_max = max(smomp_max, float(np.max(np.abs(smomp_chunk))))
        accurate_max = max(accurate_max, float(np.max(np.abs(accurate_chunk))))
    return max(smomp_max, accurate_max)


class MmapChannelDataset(Dataset):
    """
    Memory-efficient replacement for GlobalNormalizedDataset.

    Uses np.load(..., mmap_mode='r') and normalizes per-sample on __getitem__.
    Split logic matches train.py create_datasets (train_ratio=0.8, val_ratio=0.1).
    """

    def __init__(
        self,
        smomp_mmap: np.ndarray,
        accurate_mmap: np.ndarray,
        user_positions: list[tuple[float, float]],
        rss_processor,
        norm_max: float,
        split: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.10,
        crop_size: int = 30,
        use_dbm_values: bool = True,
    ):
        from Model import RSSColorMapper

        self.smomp_mmap = smomp_mmap
        self.accurate_mmap = accurate_mmap
        self.user_positions = user_positions
        self.rss_processor = rss_processor
        self.norm_max = norm_max
        self.crop_size = crop_size
        self.use_dbm_values = use_dbm_values
        self.rss_color_mapper = RSSColorMapper(min_dbm=-110.0, max_dbm=-40.0)

        n_samples = smomp_mmap.shape[0]
        rng = np.random.RandomState(42)
        indices = rng.permutation(n_samples)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        if split == "train":
            self.indices = indices[:n_train]
        elif split == "val":
            self.indices = indices[n_train : n_train + n_val]
        elif split == "test":
            self.indices = indices[n_train + n_val :]
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train', 'val', or 'test'")

        print(f"Split '{split}': {len(self.indices)} samples")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = int(self.indices[idx])

        smomp_channel = _complex_to_real_imag(self.smomp_mmap[real_idx]) / self.norm_max
        accurate_channel = _complex_to_real_imag(self.accurate_mmap[real_idx]) / self.norm_max

        user_idx = real_idx % len(self.user_positions)
        user_x, user_y = self.user_positions[user_idx]
        rss_crop = self.rss_processor.crop_around_user(user_x, user_y, self.crop_size)

        if rss_crop is None:
            rss_crop = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.float32)

        if self.use_dbm_values:
            rss_dbm = self.rss_color_mapper.rgb_to_dbm(rss_crop)
            rss_dbm_normalized = self.rss_color_mapper.normalize_dbm(rss_dbm)
            rss_gray = cv2.cvtColor(rss_crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            rss_gray_normalized = rss_gray.astype(np.float32) / 255.0
            rss_tensor = torch.stack([
                torch.from_numpy(rss_gray_normalized).float(),
                torch.from_numpy(rss_dbm_normalized).float(),
            ])
        else:
            rss_gray = cv2.cvtColor(rss_crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            rss_tensor = torch.from_numpy(rss_gray.astype(np.float32) / 255.0).unsqueeze(0)

        return (
            torch.from_numpy(smomp_channel).float(),
            torch.from_numpy(accurate_channel).float(),
            rss_tensor,
        )


class ExplicitValDataset(Dataset):
    """
    Standalone validation files from validation/snr0/ (create_validation_dataset.py).

    Uses normalization constants from manifest.json (computed on the full dataset).
    """

    def __init__(
        self,
        val_dir: str,
        rss_image_path: str,
        crop_size: int = 30,
        use_dbm_values: bool = True,
    ):
        import json
        from Model import RSSColorMapper

        val_path = Path(val_dir)
        with open(val_path / "manifest.json") as f:
            self.manifest = json.load(f)

        self.smomp = np.load(val_path / "initial_estimate_ls_val.npy", mmap_mode="r")
        self.accurate = np.load(val_path / "3D_channel_val.npy", mmap_mode="r")
        self.norm_max = float(self.manifest["norm_max"])
        self.positions = self._load_val_positions(val_path / "ue_positions_val.txt")
        self.rss_processor = _make_rss_processor(rss_image_path)
        self.crop_size = crop_size
        self.use_dbm_values = use_dbm_values
        self.rss_color_mapper = RSSColorMapper(min_dbm=-110.0, max_dbm=-40.0)
        print(f"Explicit val dataset: {len(self.smomp)} samples from {val_path}")

    @staticmethod
    def _load_val_positions(path: Path) -> list[tuple[float, float]]:
        positions: list[tuple[float, float]] = []
        with open(path) as f:
            for line in f.readlines()[1:]:
                if line.strip():
                    x, y, _z, _src = line.strip().split()
                    positions.append((float(x), float(y)))
        return positions

    def __len__(self) -> int:
        return len(self.smomp)

    def __getitem__(self, idx: int):
        smomp_channel = _complex_to_real_imag(self.smomp[idx]) / self.norm_max
        accurate_channel = _complex_to_real_imag(self.accurate[idx]) / self.norm_max

        user_x, user_y = self.positions[idx]
        rss_crop = self.rss_processor.crop_around_user(user_x, user_y, self.crop_size)

        if rss_crop is None:
            rss_crop = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.float32)

        if self.use_dbm_values:
            rss_dbm = self.rss_color_mapper.rgb_to_dbm(rss_crop)
            rss_dbm_normalized = self.rss_color_mapper.normalize_dbm(rss_dbm)
            rss_gray = cv2.cvtColor(rss_crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            rss_gray_normalized = rss_gray.astype(np.float32) / 255.0
            rss_tensor = torch.stack([
                torch.from_numpy(rss_gray_normalized).float(),
                torch.from_numpy(rss_dbm_normalized).float(),
            ])
        else:
            rss_gray = cv2.cvtColor(rss_crop.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            rss_tensor = torch.from_numpy(rss_gray.astype(np.float32) / 255.0).unsqueeze(0)

        return (
            torch.from_numpy(smomp_channel).float(),
            torch.from_numpy(accurate_channel).float(),
            rss_tensor,
        )


class _DataBundle:
    """Shared mmap-backed data; one instance per unique file set."""

    _instances: dict[tuple, "_DataBundle"] = {}

    def __init__(
        self,
        smomp_file: str,
        accurate_file: str,
        user_positions_file: str,
        rss_image_path: str,
    ):
        print("  Loading channel arrays (mmap) ...")
        self.smomp_mmap = np.load(smomp_file, mmap_mode="r")
        self.accurate_mmap = np.load(accurate_file, mmap_mode="r")
        print("  Computing global normalization max ...")
        self.norm_max = _compute_global_norm_max(self.smomp_mmap, self.accurate_mmap)
        self.rss_processor = _make_rss_processor(rss_image_path)
        self.user_positions = self._load_user_positions(user_positions_file)

    @staticmethod
    def _load_user_positions(path: str) -> list[tuple[float, float]]:
        positions: list[tuple[float, float]] = []
        with open(path, "r") as f:
            for line in f.readlines()[1:]:
                if line.strip():
                    x, y, _z = map(float, line.strip().split())
                    positions.append((x, y))
        return positions

    @classmethod
    def get(
        cls,
        smomp_file: str,
        accurate_file: str,
        user_positions_file: str,
        rss_image_path: str,
    ) -> "_DataBundle":
        key = (smomp_file, accurate_file, user_positions_file, rss_image_path)
        if key not in cls._instances:
            cls._instances[key] = cls(
                smomp_file, accurate_file, user_positions_file, rss_image_path
            )
        return cls._instances[key]

    def make_split_dataset(self, split: str) -> MmapChannelDataset:
        return MmapChannelDataset(
            smomp_mmap=self.smomp_mmap,
            accurate_mmap=self.accurate_mmap,
            user_positions=self.user_positions,
            rss_processor=self.rss_processor,
            norm_max=self.norm_max,
            split=split,
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
    n_cal samples from the training split for Hessian accumulation.
    """
    bundle = _DataBundle.get(
        smomp_file, accurate_file, user_positions_file, rss_image_path
    )
    train_ds = bundle.make_split_dataset("train")

    g = torch.Generator()
    g.manual_seed(0)
    perm = torch.randperm(len(train_ds), generator=g)
    cal_indices = perm[:n_cal].tolist()
    cal_ds = Subset(train_ds, cal_indices)

    return DataLoader(cal_ds, batch_size=batch_size, shuffle=False, num_workers=0)


def get_explicit_val_loader(
    val_dir: str,
    rss_image_path: str,
    batch_size: int = 8,
    n_val: Optional[int] = None,
) -> DataLoader:
    """Load pre-built validation files from validation/snr0/."""
    val_ds = ExplicitValDataset(val_dir=val_dir, rss_image_path=rss_image_path)

    if n_val is not None and n_val < len(val_ds):
        g = torch.Generator()
        g.manual_seed(1)
        perm = torch.randperm(len(val_ds), generator=g)
        val_ds = Subset(val_ds, perm[:n_val].tolist())
        print(f"  Using {n_val} validation samples (subset)")

    return DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)


def get_val_loader(
    smomp_file: str,
    accurate_file: str,
    user_positions_file: str,
    rss_image_path: str,
    batch_size: int = 8,
    n_val: Optional[int] = None,
) -> DataLoader:
    """Validation split for NMSE evaluation (never test)."""
    bundle = _DataBundle.get(
        smomp_file, accurate_file, user_positions_file, rss_image_path
    )
    val_ds = bundle.make_split_dataset("val")

    if n_val is not None and n_val < len(val_ds):
        g = torch.Generator()
        g.manual_seed(1)
        perm = torch.randperm(len(val_ds), generator=g)
        val_ds = Subset(val_ds, perm[:n_val].tolist())
        print(f"  Using {n_val} validation samples (subset of val split)")

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
