"""Model definitions: scikit-learn baselines and the PyTorch CNNs."""
from __future__ import annotations

import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import SEED


# --------------------------------------------------------------------------
# Classical baselines (operate on flattened 2,187-D pixel vectors)
# --------------------------------------------------------------------------
def build_logreg_pca(n_components: int = 50, C: float = 1.0, seed: int = SEED) -> Pipeline:
    """Standardise -> PCA(50) -> L2 logistic regression (linear baseline)."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=seed)),
        ("lr", LogisticRegression(max_iter=2000, C=C, random_state=seed)),
    ])


def build_svm_pca(n_components: int = 50, C: float = 1.0, seed: int = SEED) -> Pipeline:
    """Standardise -> PCA(50) -> RBF-kernel SVM (non-linear margin baseline)."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=seed)),
        ("svm", SVC(kernel="rbf", C=C, gamma="scale", random_state=seed)),
    ])


def build_random_forest(n_estimators: int = 150, seed: int = SEED, n_jobs: int = 1) -> RandomForestClassifier:
    """Unpruned random forest on raw pixels (high-capacity, overfits)."""
    return RandomForestClassifier(n_estimators=n_estimators, max_features="sqrt",
                                  random_state=seed, n_jobs=n_jobs)


SKLEARN_BUILDERS = {
    "logreg_pca": build_logreg_pca,
    "svm_pca": build_svm_pca,
    "random_forest": build_random_forest,
}


def build_sklearn_model(name: str, **kwargs):
    if name not in SKLEARN_BUILDERS:
        raise KeyError(f"Unknown sklearn model '{name}'. Options: {sorted(SKLEARN_BUILDERS)}")
    return SKLEARN_BUILDERS[name](**kwargs)


# --------------------------------------------------------------------------
# CNNs (input: 3 x 27 x 27)
# --------------------------------------------------------------------------
class _ConvNet(nn.Module):
    """Two double-conv blocks + MLP head. Subclasses only set the widths."""

    def __init__(self, num_classes: int, c1: int, c2: int, hidden: int, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 27 -> 13
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(c2, c2, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 13 -> 6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 6 * 6, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class SmallCNN(_ConvNet):
    """3 -> 32 -> 64 channels, 128-unit head (~361k parameters)."""

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__(num_classes, c1=32, c2=64, hidden=128, dropout=dropout)


class WiderCNN(_ConvNet):
    """3 -> 64 -> 128 channels, 256-unit head (~1.44M parameters, 4x SmallCNN)."""

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__(num_classes, c1=64, c2=128, hidden=256, dropout=dropout)


CNN_REGISTRY = {"small": SmallCNN, "wider": WiderCNN}


def build_cnn(name: str, num_classes: int, dropout: float = 0.3) -> nn.Module:
    if name not in CNN_REGISTRY:
        raise KeyError(f"Unknown CNN '{name}'. Options: {sorted(CNN_REGISTRY)}")
    return CNN_REGISTRY[name](num_classes=num_classes, dropout=dropout)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only))


def make_target_from_source(source: nn.Module, num_classes: int, dropout: float = 0.3) -> nn.Module:
    """Intra-dataset transfer learning.

    Build a fresh network of the same family as ``source`` and copy its convolutional
    feature extractor; the classifier head is re-initialised for ``num_classes``.
    """
    target = type(source)(num_classes=num_classes, dropout=dropout)
    target.features.load_state_dict(source.features.state_dict())
    return target
