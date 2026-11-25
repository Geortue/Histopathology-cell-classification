import pandas as pd

from src.data import build_task_frame, check_patient_overlap, load_label_tables
from src.splits import (assert_patient_disjoint, cross_task_overlap, describe_split,
                        exclude_patients, make_patient_disjoint_split, subsample_patients)


def test_label_tables_and_task_frames(synthetic_dataset):
    main_df, extra_df = load_label_tables(synthetic_dataset)
    assert check_patient_overlap(main_df, extra_df) == 0
    assert len(build_task_frame("isCancerous", main_df, extra_df)) == len(main_df) + len(extra_df)
    assert len(build_task_frame("cellType", main_df, extra_df)) == len(main_df)


def test_split_is_patient_disjoint_and_exhaustive(synthetic_dataset):
    main_df, extra_df = load_label_tables(synthetic_dataset)
    df = build_task_frame("isCancerous", main_df, extra_df)
    s = make_patient_disjoint_split(df, seed=1)
    assert_patient_disjoint(s)
    assert sum(len(x) for x in s) == len(df)
    assert set(pd.concat(s).patientID) == set(df.patientID)


def test_split_is_deterministic(synthetic_dataset):
    main_df, _ = load_label_tables(synthetic_dataset)
    a = make_patient_disjoint_split(main_df, seed=7)
    b = make_patient_disjoint_split(main_df, seed=7)
    assert all(x.equals(y) for x, y in zip(a, b))


def test_cross_task_overlap_and_exclusion(synthetic_dataset):
    main_df, extra_df = load_label_tables(synthetic_dataset)
    can = make_patient_disjoint_split(build_task_frame("isCancerous", main_df, extra_df), seed=3)
    ct = make_patient_disjoint_split(build_task_frame("cellType", main_df, extra_df), seed=3)
    counts = cross_task_overlap(can, ct)
    assert set(counts) == {"source_train_in_target_val", "source_train_in_target_test"}
    held_out = set(ct.val.patientID) | set(ct.test.patientID)
    cleaned = exclude_patients(can.train, held_out)
    assert set(cleaned.patientID).isdisjoint(held_out)


def test_describe_and_subsample(synthetic_dataset):
    main_df, _ = load_label_tables(synthetic_dataset)
    s = make_patient_disjoint_split(main_df, seed=0)
    assert list(describe_split(s, "isCancerous").index) == ["train", "val", "test"]
    sub = subsample_patients(main_df, 0.5, seed=0)
    assert sub.patientID.nunique() == round(main_df.patientID.nunique() * 0.5)
