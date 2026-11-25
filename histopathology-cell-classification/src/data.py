"""Label loading, image loading and PyTorch dataset utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import (EXTRA_LABELS_FILE, IMAGE_SIZE, IMG_DIR, MAIN_LABELS_FILE,
                     SEED, TASKS, DATA_DIR)


# --------------------------------------------------------------------------
# Label tables
# --------------------------------------------------------------------------
def load_label_tables(data_dir: str | Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the main (isCancerous + cellType) and extra (isCancerous only) label files."""
    data_dir = Path(data_dir)
    main_df = pd.read_csv(data_dir / MAIN_LABELS_FILE)
    extra_df = pd.read_csv(data_dir / EXTRA_LABELS_FILE)
    return main_df, extra_df


def check_patient_overlap(main_df: pd.DataFrame, extra_df: pd.DataFrame) -> int:
    """Number of patients that appear in both label files (must be 0)."""
    return len(set(main_df["patientID"]) & set(extra_df["patientID"]))


def build_task_frame(task: str, main_df: pd.DataFrame, extra_df: pd.DataFrame) -> pd.DataFrame:
    """Return the DataFrame used for a task.

    * ``isCancerous`` -> main + extra (all 98 patients, 20,280 images)
    * ``cellType``    -> main only   (60 patients, 9,896 images)
    """
    spec = TASKS[task]
    if spec.uses_extra_data:
        return pd.concat(
            [main_df.assign(source="main"), extra_df.assign(source="extra")],
            axis=0, ignore_index=True,
        )
    return main_df.assign(source="main").reset_index(drop=True)


# --------------------------------------------------------------------------
# Image loading
# --------------------------------------------------------------------------
def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def load_images_chw(df: pd.DataFrame, img_dir: str | Path = IMG_DIR) -> torch.Tensor:
    """Load tiles as a float32 tensor of shape (N, 3, 27, 27) scaled to [0, 1]."""
    img_dir = Path(img_dir)
    X = np.empty((len(df), 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for i, name in enumerate(df["ImageName"].values):
        X[i] = _read_rgb(img_dir / name).transpose(2, 0, 1)
    return torch.from_numpy(X)


def flatten_images(X: torch.Tensor) -> np.ndarray:
    """(N, 3, 27, 27) tensor -> (N, 2187) array for scikit-learn models."""
    return X.numpy().reshape(len(X), -1)


def load_images_flat(df: pd.DataFrame, img_dir: str | Path = IMG_DIR) -> np.ndarray:
    """Load tiles directly as flattened vectors (used for PCA / k-means EDA)."""
    return flatten_images(load_images_chw(df, img_dir))


# --------------------------------------------------------------------------
# PyTorch plumbing
# --------------------------------------------------------------------------
def get_augmentation() -> T.Compose:
    """Training-time augmentation: flips + small rotation.

    Cells have no canonical orientation, so the label is invariant to these
    transforms. Applied to the training set only.
    """
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),
    ])


class ArrayDataset(Dataset):
    """Wraps an in-memory image tensor and a label vector."""

    def __init__(self, X: torch.Tensor, y, transform: Callable | None = None):
        self.X = X
        self.y = torch.as_tensor(np.array(y, dtype=np.int64))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        x = self.X[i]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y[i]


def make_loader(X: torch.Tensor, y, batch_size: int = 128, shuffle: bool = False,
                transform: Callable | None = None, seed: int = SEED) -> DataLoader:
    """DataLoader with a seeded generator so shuffling is reproducible."""
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ArrayDataset(X, y, transform), batch_size=batch_size,
                      shuffle=shuffle, generator=g if shuffle else None)
