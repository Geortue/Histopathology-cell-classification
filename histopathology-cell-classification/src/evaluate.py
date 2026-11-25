"""Evaluation helpers: metrics, tables, plots and prediction export."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from .utils import get_device


# --------------------------------------------------------------------------
# Predictions
# --------------------------------------------------------------------------
def predict_cnn(model, X: torch.Tensor, batch_size: int = 256, device: torch.device | None = None) -> np.ndarray:
    """Predicted class indices for an (N, 3, 27, 27) tensor."""
    device = device or get_device()
    model = model.to(device).eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            preds.append(model(X[i:i + batch_size].to(device)).argmax(1).cpu().numpy())
    return np.concatenate(preds)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def score(y_true, y_pred) -> dict:
    """Primary (macro-F1) and secondary (accuracy) metrics."""
    return {"macro_f1": f1_score(y_true, y_pred, average="macro"),
            "accuracy": accuracy_score(y_true, y_pred)}


def evaluate_sklearn(clf, X_tr, y_tr, X_val, y_val, label: str = "") -> dict:
    tr, va = score(y_tr, clf.predict(X_tr)), score(y_val, clf.predict(X_val))
    res = {"train_f1": tr["macro_f1"], "val_f1": va["macro_f1"],
           "train_acc": tr["accuracy"], "val_acc": va["accuracy"]}
    if label:
        print(f"{label}: Train F1={res['train_f1']:.4f} Acc={res['train_acc']:.4f} | "
              f"Val F1={res['val_f1']:.4f} Acc={res['val_acc']:.4f}")
    return res


def evaluate_cnn(model, X_tr, y_tr, X_val, y_val, label: str = "") -> dict:
    tr = score(y_tr, predict_cnn(model, X_tr))
    va = score(y_val, predict_cnn(model, X_val))
    res = {"train_f1": tr["macro_f1"], "val_f1": va["macro_f1"],
           "train_acc": tr["accuracy"], "val_acc": va["accuracy"]}
    if label:
        print(f"{label}: Train F1={res['train_f1']:.4f} Acc={res['train_acc']:.4f} | "
              f"Val F1={res['val_f1']:.4f} Acc={res['val_acc']:.4f}")
    return res


def per_class_table(y_true, y_pred, class_names) -> pd.DataFrame:
    """Per-class precision / recall / F1 / support plus a macro-average row."""
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    df = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=list(class_names))
    df.loc["macro avg"] = [p.mean(), r.mean(), f.mean(), s.sum()]
    return df


def binary_clinical_metrics(y_true, y_pred) -> dict:
    """Sensitivity, specificity, PPV and NPV for a 0/1 task (1 = cancerous)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    div = lambda a, b: float(a / b) if b else float("nan")  # noqa: E731
    return {"sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn)}


def build_test_report(y_true, y_pred, class_names, target_f1: float | None = None) -> dict:
    """Headline test-set summary (dict) that is easy to dump to JSON."""
    out = score(y_true, y_pred)
    out["classification_report"] = classification_report(
        y_true, y_pred, target_names=list(class_names), digits=4, zero_division=0)
    out["per_class"] = per_class_table(y_true, y_pred, class_names).round(4).to_dict()
    if target_f1 is not None:
        out["target_macro_f1"] = target_f1
        out["target_met"] = bool(out["macro_f1"] >= target_f1)
    if len(class_names) == 2:
        out.update(binary_clinical_metrics(y_true, y_pred))
    return out


def results_table(results: dict[str, dict[str, dict]]) -> dict[str, pd.DataFrame]:
    """{task: {model: {'train_f1','val_f1'}}} -> one sorted DataFrame per task with the generalisation gap."""
    tables = {}
    for task, models in results.items():
        df = pd.DataFrame(models).T[["train_f1", "val_f1"]]
        df["gap"] = df["train_f1"] - df["val_f1"]
        tables[task] = df.sort_values("val_f1", ascending=False)
    return tables


# --------------------------------------------------------------------------
# Plots (matplotlib / seaborn imported lazily so the metrics code stays light)
# --------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, class_names, title: str = "", cmap: str = "Blues", ax=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False, ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=12, fontweight="bold")
    return cm


def plot_histories(histories: dict[str, dict], key: str = "val_f1", target: float | None = None,
                   ylabel: str = "Validation macro-F1", title: str = "", ax=None):
    """Overlay one metric across several training runs."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    for name, h in histories.items():
        ax.plot(h["epoch"], h[key], "-o", ms=4, label=name)
    if target is not None:
        ax.axhline(target, color="red", linestyle=":", label=f"target = {target:.2f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_generalisation_gap(histories: dict[str, dict], title: str = "Generalisation gap", ax=None):
    """Train F1 minus validation F1 per epoch."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    for name, h in histories.items():
        ax.plot(h["epoch"], np.asarray(h["train_f1"]) - np.asarray(h["val_f1"]), "-o", ms=4, label=name)
    ax.axhline(0, color="black", linestyle=":")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train F1 - Val F1")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_lr_sweep(sweep: dict, ax=None):
    """Train/val F1 at each learning rate (expects the dict from ``learning_rate_sweep``)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.5))
    labels = [f"{lr:.0e}" for lr in sweep["lr_grid"]]
    ax.plot(labels, sweep["train_f1"], "b-o", label="Train F1")
    ax.plot(labels, sweep["val_f1"], "r-s", label="Val F1")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Macro-F1 (best epoch)")
    ax.set_title("Effect of learning rate", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True)
    return ax


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def save_predictions(df: pd.DataFrame, y_pred, label_col: str, path: str | Path,
                     class_names=None, keep_cols=("patientID", "ImageName")) -> Path:
    """Write per-image predictions next to the ground truth (outputs/ is git-ignored)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in (*keep_cols, label_col, "cellTypeName") if c in df.columns]
    out = df[list(dict.fromkeys(cols))].copy()
    out[f"predicted_{label_col}"] = y_pred
    if class_names is not None:
        out[f"predicted_{label_col}_name"] = [class_names[i] for i in y_pred]
    out.to_csv(path, index=False)
    return path
