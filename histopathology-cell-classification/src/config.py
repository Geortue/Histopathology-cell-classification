"""Project-wide constants and default hyperparameters."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SEED = 42

# ---- Paths (all relative to the repository root) ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
IMG_DIR = DATA_DIR / "images"
SPLIT_DIR = PROJECT_ROOT / "splits"
OUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"

MAIN_LABELS_FILE = "data_labels_mainData.csv"
EXTRA_LABELS_FILE = "data_labels_extraData.csv"

# ---- Dataset facts ---------------------------------------------------------
IMAGE_SIZE = 27
CTYPE_NAMES = ["fibroblast", "inflammatory", "epithelial", "others"]
CANCER_NAMES = ["non-cancerous", "cancerous"]


@dataclass(frozen=True)
class TaskSpec:
    """Describes one of the two classification tasks."""

    name: str
    label_col: str
    class_names: tuple[str, ...]
    target_macro_f1: float
    uses_extra_data: bool

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


TASKS: dict[str, TaskSpec] = {
    "isCancerous": TaskSpec(
        name="isCancerous",
        label_col="isCancerous",
        class_names=tuple(CANCER_NAMES),
        target_macro_f1=0.90,
        uses_extra_data=True,  # main + extra labels (98 patients)
    ),
    "cellType": TaskSpec(
        name="cellType",
        label_col="cellType",
        class_names=tuple(CTYPE_NAMES),
        target_macro_f1=0.60,
        uses_extra_data=False,  # only the 60 main-data patients carry cellType
    ),
}


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters shared by every CNN run."""

    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    eval_batch_size: int = 256
    dropout: float = 0.3


# ---- Final recipes reported in the README ----------------------------------
# isCancerous: WiderCNN + augmentation, 15 epochs.
IS_CANCEROUS_FINAL = TrainConfig(epochs=15)
# Transfer source for cellType: SmallCNN trained on isCancerous, no augmentation.
TRANSFER_SOURCE = TrainConfig(epochs=10)
# cellType: SmallCNN initialised from the source, + augmentation, 12 epochs.
CELL_TYPE_FINAL = TrainConfig(epochs=12)

LR_GRID = (1e-2, 1e-3, 1e-4, 1e-5)
