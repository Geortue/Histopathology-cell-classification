"""Patient-disjoint train / validation / test splitting.

A row-level ``train_test_split`` would put tiles from the same patient in several
splits and reward within-patient memorisation. We split on ``patientID`` instead.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .config import SEED, SPLIT_DIR


class Splits(NamedTuple):
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def patient_set(df: pd.DataFrame, group_col: str = "patientID") -> set:
    return set(df[group_col])


def assert_patient_disjoint(splits: Splits, group_col: str = "patientID") -> None:
    """Raise AssertionError if any patient appears in more than one split."""
    tr, va, te = (patient_set(d, group_col) for d in splits)
    assert tr.isdisjoint(va), "train/val share patients"
    assert tr.isdisjoint(te), "train/test share patients"
    assert va.isdisjoint(te), "val/test share patients"


def make_patient_disjoint_split(df: pd.DataFrame, test_size: float = 0.20,
                                val_size_within: float = 0.25, seed: int = SEED,
                                group_col: str = "patientID") -> Splits:
    """Nested GroupShuffleSplit giving a 60 / 20 / 20 split *by patient*.

    1. all patients -> 80 % train+val, 20 % test
    2. train+val    -> 75 % train,     25 % val   (= 60 % / 20 % overall)
    """
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trval_idx, test_idx = next(gss1.split(df, groups=df[group_col].values))
    df_trval = df.iloc[trval_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size_within, random_state=seed)
    tr_idx, val_idx = next(gss2.split(df_trval, groups=df_trval[group_col].values))
    df_train = df_trval.iloc[tr_idx].reset_index(drop=True)
    df_val = df_trval.iloc[val_idx].reset_index(drop=True)

    splits = Splits(df_train, df_val, df_test)
    assert_patient_disjoint(splits, group_col)
    return splits


def subsample_patients(df: pd.DataFrame, frac: float, seed: int = SEED,
                       group_col: str = "patientID") -> pd.DataFrame:
    """Keep a random fraction of *patients* (all of their rows). Used by quick/smoke-test modes."""
    keep = df[group_col].drop_duplicates().sample(frac=frac, random_state=seed)
    return df[df[group_col].isin(keep)].reset_index(drop=True)


def exclude_patients(df: pd.DataFrame, patients: set, group_col: str = "patientID") -> pd.DataFrame:
    """Drop every row belonging to ``patients``."""
    return df[~df[group_col].isin(patients)].reset_index(drop=True)


def cross_task_overlap(source: Splits, target: Splits, group_col: str = "patientID") -> dict[str, int]:
    """Count source-training patients that are held-out (val/test) patients in the target task.

    The two tasks are split independently, so a patient can be in the isCancerous
    training set *and* the cellType test set. If the isCancerous model is used as a
    transfer source, those patients leak into the cellType evaluation.
    """
    src_train = patient_set(source.train, group_col)
    return {
        "source_train_in_target_val": len(src_train & patient_set(target.val, group_col)),
        "source_train_in_target_test": len(src_train & patient_set(target.test, group_col)),
    }


def describe_split(splits: Splits, label_col: str, group_col: str = "patientID") -> pd.DataFrame:
    """One summary row per split: images, patients and positive rate / class mix."""
    rows = []
    for name, df in zip(("train", "val", "test"), splits):
        row = {"split": name, "images": len(df), "patients": df[group_col].nunique()}
        if df[label_col].nunique() <= 2:
            row["positive_rate"] = round(float(df[label_col].mean()), 3)
        else:
            for k, v in df[label_col].value_counts(normalize=True).sort_index().items():
                row[f"class_{k}"] = round(float(v), 3)
        rows.append(row)
    return pd.DataFrame(rows).set_index("split")


def save_splits(splits: Splits, prefix: str, split_dir: str | Path = SPLIT_DIR) -> None:
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    for name, df in zip(("train", "val", "test"), splits):
        df.to_csv(split_dir / f"{prefix}_{name}.csv", index=False)


def load_splits(prefix: str, split_dir: str | Path = SPLIT_DIR) -> Splits:
    split_dir = Path(split_dir)
    return Splits(*(pd.read_csv(split_dir / f"{prefix}_{n}.csv") for n in ("train", "val", "test")))
