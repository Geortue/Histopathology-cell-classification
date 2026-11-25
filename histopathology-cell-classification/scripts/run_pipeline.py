#!/usr/bin/env python
"""End-to-end reproduction of the final models and held-out test evaluation.

Examples
--------
    python scripts/run_pipeline.py --task both
    python scripts/run_pipeline.py --task isCancerous --save-models
    python scripts/run_pipeline.py --task both --leakage-safe --restore-best
    python scripts/run_pipeline.py --task both --quick        # 1 epoch, 10% of patients (smoke test)

Pipeline
--------
1. Patient-disjoint 60/20/20 splits, built separately for each task.
2. isCancerous: WiderCNN + augmentation (15 epochs).
3. cellType:    SmallCNN pre-trained on isCancerous (10 epochs, no augmentation),
                convolutional trunk transferred, then fine-tuned with augmentation (12 epochs).
4. Evaluation on the held-out test patients; metrics, predictions and (optionally)
   weights are written to --out-dir / --model-dir (both git-ignored).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402
from src.data import (build_task_frame, check_patient_overlap, load_images_chw,  # noqa: E402
                      load_label_tables)
from src.evaluate import evaluate_cnn, predict_cnn, save_predictions, build_test_report  # noqa: E402
from src.models import SmallCNN, WiderCNN, count_parameters, make_target_from_source  # noqa: E402
from src.splits import (cross_task_overlap, describe_split, exclude_patients,  # noqa: E402
                        make_patient_disjoint_split, save_splits, subsample_patients, Splits)
from src.train import fit_cnn, save_checkpoint  # noqa: E402
from src.utils import get_device, save_json, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=["isCancerous", "cellType", "both"], default="both")
    p.add_argument("--data-dir", type=Path, default=C.DATA_DIR)
    p.add_argument("--split-dir", type=Path, default=C.SPLIT_DIR)
    p.add_argument("--out-dir", type=Path, default=C.OUT_DIR)
    p.add_argument("--model-dir", type=Path, default=C.MODEL_DIR)
    p.add_argument("--seed", type=int, default=C.SEED)
    p.add_argument("--restore-best", action="store_true",
                   help="evaluate best-validation-epoch weights instead of last-epoch weights")
    p.add_argument("--leakage-safe", action="store_true",
                   help="remove cellType val/test patients from the isCancerous transfer-source training data")
    p.add_argument("--save-models", action="store_true", help="write checkpoints to --model-dir")
    p.add_argument("--quick", action="store_true", help="smoke test: 1 epoch per model, 10%% of patients")
    return p.parse_args()


def _load(splits: Splits, label_col: str, img_dir: Path):
    return [(load_images_chw(d, img_dir), d[label_col].values.astype("int64")) for d in splits]


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device()
    img_dir = args.data_dir / "images"
    print(f"Device: {device}")

    # ---- 1. labels and patient-disjoint splits ------------------------------
    main_df, extra_df = load_label_tables(args.data_dir)
    assert check_patient_overlap(main_df, extra_df) == 0, "main/extra label files share patients"
    frames = {t: build_task_frame(t, main_df, extra_df) for t in C.TASKS}
    if args.quick:
        frames = {t: subsample_patients(f, 0.10, args.seed) for t, f in frames.items()}

    splits = {t: make_patient_disjoint_split(f, seed=args.seed) for t, f in frames.items()}
    for t, s in splits.items():
        save_splits(s, prefix=("cancer" if t == "isCancerous" else "ctype"), split_dir=args.split_dir)
        print(f"\n{t} splits (patient-disjoint):")
        print(describe_split(s, C.TASKS[t].label_col).to_string())

    def cfg(base: C.TrainConfig) -> C.TrainConfig:
        return replace(base, epochs=1) if args.quick else base

    metrics: dict = {"seed": args.seed, "restore_best": args.restore_best,
                     "leakage_safe": args.leakage_safe, "quick": args.quick}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 2. isCancerous ---------------------------------------------------------
    spec_c = C.TASKS["isCancerous"]
    (Xc_tr, yc_tr), (Xc_va, yc_va), (Xc_te, yc_te) = _load(splits["isCancerous"], spec_c.label_col, img_dir)

    if args.task in ("isCancerous", "both"):
        print("\n=== isCancerous: WiderCNN + augmentation ===")
        model_c, hist_c = fit_cnn(lambda: WiderCNN(spec_c.num_classes), Xc_tr, yc_tr, Xc_va, yc_va,
                                  cfg(C.IS_CANCEROUS_FINAL), augment=True, restore_best=args.restore_best,
                                  seed=args.seed)
        print(f"parameters: {count_parameters(model_c):,}")
        evaluate_cnn(model_c, Xc_tr, yc_tr, Xc_va, yc_va, label="isCancerous val")
        yp = predict_cnn(model_c, Xc_te, device=device)
        rep = build_test_report(yc_te, yp, spec_c.class_names, spec_c.target_macro_f1)
        print("\nTEST isCancerous\n" + rep["classification_report"])
        print(f"macro-F1 = {rep['macro_f1']:.4f} (target {spec_c.target_macro_f1})  "
              f"sensitivity = {rep['sensitivity']:.3f}  specificity = {rep['specificity']:.3f}")
        metrics["isCancerous"] = {**{k: v for k, v in rep.items() if k != "classification_report"},
                                  "val_f1_last_epoch": hist_c["val_f1"][-1],
                                  "val_f1_best_epoch": max(hist_c["val_f1"])}
        save_predictions(splits["isCancerous"].test, yp, spec_c.label_col,
                         args.out_dir / "test_predictions_isCancerous.csv", spec_c.class_names)
        if args.save_models:
            save_checkpoint(model_c, args.model_dir / "iscancerous_widercnn_aug.pt", task="isCancerous")

    # ---- 3. cellType (transfer from isCancerous) -------------------------------
    if args.task in ("cellType", "both"):
        spec_t = C.TASKS["cellType"]
        (Xt_tr, yt_tr), (Xt_va, yt_va), (Xt_te, yt_te) = _load(splits["cellType"], spec_t.label_col, img_dir)

        overlap = cross_task_overlap(splits["isCancerous"], splits["cellType"])
        print(f"\nTransfer-source overlap with cellType held-out patients: {overlap}")
        if args.leakage_safe:
            held_out = set(splits["cellType"].val["patientID"]) | set(splits["cellType"].test["patientID"])
            src = Splits(*(exclude_patients(d, held_out) for d in splits["isCancerous"]))
            (Xs_tr, ys_tr), (Xs_va, ys_va), _ = _load(src, spec_c.label_col, img_dir)
            print("--leakage-safe: transfer source trained without cellType val/test patients "
                  f"({len(src.train)} train images)")
        elif overlap["source_train_in_target_test"] or overlap["source_train_in_target_val"]:
            print("WARNING: the transfer source saw cellType val/test patients during pre-training. "
                  "Re-run with --leakage-safe for an unbiased estimate.")
            Xs_tr, ys_tr, Xs_va, ys_va = Xc_tr, yc_tr, Xc_va, yc_va
        else:
            Xs_tr, ys_tr, Xs_va, ys_va = Xc_tr, yc_tr, Xc_va, yc_va
        metrics["transfer_source_overlap"] = overlap

        print("\n=== Transfer source: SmallCNN on isCancerous (no augmentation) ===")
        source, _ = fit_cnn(lambda: SmallCNN(spec_c.num_classes), Xs_tr, ys_tr, Xs_va, ys_va,
                            cfg(C.TRANSFER_SOURCE), augment=False, seed=args.seed)

        print("\n=== cellType: transfer + augmentation ===")
        model_t, hist_t = fit_cnn(lambda: make_target_from_source(source, spec_t.num_classes),
                                  Xt_tr, yt_tr, Xt_va, yt_va, cfg(C.CELL_TYPE_FINAL), augment=True,
                                  restore_best=args.restore_best, seed=args.seed)
        evaluate_cnn(model_t, Xt_tr, yt_tr, Xt_va, yt_va, label="cellType val")
        yp = predict_cnn(model_t, Xt_te, device=device)
        rep = build_test_report(yt_te, yp, spec_t.class_names, spec_t.target_macro_f1)
        print("\nTEST cellType\n" + rep["classification_report"])
        print(f"macro-F1 = {rep['macro_f1']:.4f} (target {spec_t.target_macro_f1})")
        metrics["cellType"] = {**{k: v for k, v in rep.items() if k != "classification_report"},
                               "val_f1_last_epoch": hist_t["val_f1"][-1],
                               "val_f1_best_epoch": max(hist_t["val_f1"])}
        save_predictions(splits["cellType"].test, yp, spec_t.label_col,
                         args.out_dir / "test_predictions_cellType.csv", spec_t.class_names)
        if args.save_models:
            save_checkpoint(model_t, args.model_dir / "celltype_transfer_aug.pt", task="cellType")

    save_json(metrics, args.out_dir / "metrics.json")
    print(f"\nMetrics written to {args.out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
