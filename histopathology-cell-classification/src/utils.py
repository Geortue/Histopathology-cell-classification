"""Small helpers: reproducibility, device selection, JSON output."""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from .config import SEED


def seed_everything(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_builtin(o):
    """JSON fallback: NumPy scalars/arrays -> plain Python."""
    if hasattr(o, "item") and getattr(o, "ndim", 1) == 0:
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def save_json(obj: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_to_builtin)
    return path
