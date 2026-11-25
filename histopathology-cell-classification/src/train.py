"""Training loops for the CNNs and helpers for the classical baselines."""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from .config import SEED, TrainConfig
from .data import get_augmentation, make_loader
from .utils import get_device, seed_everything


def compute_class_weights(y, num_classes: int, device: torch.device | None = None) -> torch.Tensor:
    """Inverse-frequency ('balanced') weights  w_c = N / (K * n_c)."""
    w = compute_class_weight("balanced", classes=np.arange(num_classes), y=np.asarray(y))
    return torch.tensor(w, dtype=torch.float32, device=device)


def _run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over ``loader``. Trains when ``optimizer`` is given, else evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, n, ys, ps = 0.0, 0, [], []
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if training:
                optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
            ys.append(yb.cpu().numpy())
            ps.append(out.argmax(1).cpu().numpy())
    f1 = f1_score(np.concatenate(ys), np.concatenate(ps), average="macro")
    return total_loss / n, f1


def train_cnn(model: nn.Module, train_loader, val_loader, epochs: int = 10, lr: float = 1e-3,
              weight_decay: float = 1e-4, freeze_features: bool = False,
              class_weights: torch.Tensor | None = None, restore_best: bool = False,
              device: torch.device | None = None, verbose: bool = True):
    """Train a CNN with Adam and record per-epoch loss and macro-F1.

    Parameters
    ----------
    freeze_features : keep the convolutional trunk fixed (linear-probe transfer).
    class_weights   : optional per-class weights for ``CrossEntropyLoss``.
    restore_best    : reload the weights from the epoch with the best validation
                      macro-F1 before returning. The default (False) returns the
                      *last-epoch* weights, which is what the original notebook evaluated.

    Returns
    -------
    (model, history) where history is a dict of lists plus ``best_epoch``.
    """
    device = device or get_device()
    model = model.to(device)
    if freeze_features:
        for p in model.features.parameters():
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    weight = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_f1, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        tr_loss, tr_f1 = _run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_f1 = _run_epoch(model, val_loader, criterion, device)
        for k, v in zip(history, (ep, tr_loss, va_loss, tr_f1, va_f1)):
            history[k].append(v)
        if va_f1 > best_f1:
            best_f1 = va_f1
            best_state = copy.deepcopy(model.state_dict()) if restore_best else None
            history["best_epoch"] = ep
        if verbose:
            print(f"  ep {ep:>2}/{epochs} | tr_loss {tr_loss:.4f}  va_loss {va_loss:.4f} "
                  f"| tr_f1 {tr_f1:.3f}  va_f1 {va_f1:.3f}")
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def fit_cnn(build_model: Callable[[], nn.Module], X_tr, y_tr, X_val, y_val, cfg: TrainConfig,
            augment: bool = False, class_weights: torch.Tensor | None = None,
            freeze_features: bool = False, restore_best: bool = False,
            seed: int = SEED, verbose: bool = True):
    """Convenience wrapper: seed -> build loaders -> build model -> train.

    ``build_model`` is called *after* seeding so initial weights are reproducible.
    """
    seed_everything(seed)
    transform = get_augmentation() if augment else None
    train_loader = make_loader(X_tr, y_tr, cfg.batch_size, shuffle=True, transform=transform, seed=seed)
    val_loader = make_loader(X_val, y_val, cfg.eval_batch_size, shuffle=False)
    model = build_model()
    t0 = time.time()
    model, history = train_cnn(model, train_loader, val_loader, epochs=cfg.epochs, lr=cfg.lr,
                               weight_decay=cfg.weight_decay, freeze_features=freeze_features,
                               class_weights=class_weights, restore_best=restore_best, verbose=verbose)
    history["train_seconds"] = time.time() - t0
    return model, history


def learning_rate_sweep(build_model: Callable[[], nn.Module], X_tr, y_tr, X_val, y_val,
                        lr_grid=(1e-2, 1e-3, 1e-4, 1e-5), epochs: int = 5, base_cfg: TrainConfig | None = None,
                        verbose: bool = False) -> dict:
    """Controlled experiment over Adam learning rates (same seed, architecture, epochs).

    Returns {'lr_grid', 'train_f1', 'val_f1', 'histories', 'best_lr'} where F1 values are
    taken at each run's best validation epoch.
    """
    base_cfg = base_cfg or TrainConfig()
    out = {"lr_grid": list(lr_grid), "train_f1": [], "val_f1": [], "histories": {}}
    for lr in lr_grid:
        cfg = TrainConfig(**{**base_cfg.__dict__, "epochs": epochs, "lr": lr})
        _, hist = fit_cnn(build_model, X_tr, y_tr, X_val, y_val, cfg, verbose=verbose)
        i = int(np.argmax(hist["val_f1"]))
        out["train_f1"].append(hist["train_f1"][i])
        out["val_f1"].append(hist["val_f1"][i])
        out["histories"][lr] = hist
        print(f"lr={lr:.0e}  best val macro-F1 = {hist['val_f1'][i]:.4f} (epoch {i + 1})")
    out["best_lr"] = lr_grid[int(np.argmax(out["val_f1"]))]
    return out


def fit_sklearn(model, X_tr, y_tr):
    """Fit a scikit-learn estimator and return (model, fit_seconds)."""
    t0 = time.time()
    model.fit(X_tr, y_tr)
    return model, time.time() - t0


def save_checkpoint(model: nn.Module, path: str | Path, **metadata) -> Path:
    """Save weights + metadata. Checkpoints are git-ignored (see .gitignore)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": type(model).__name__, **metadata}, path)
    return path


def load_checkpoint(model: nn.Module, path: str | Path, map_location="cpu") -> nn.Module:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["state_dict"])
    return model
