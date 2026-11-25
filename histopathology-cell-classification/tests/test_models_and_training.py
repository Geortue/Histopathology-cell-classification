import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src import config as C  # noqa: E402
from src.data import (build_task_frame, flatten_images, get_augmentation, load_images_chw,  # noqa: E402
                      load_label_tables, make_loader)
from src.evaluate import (binary_clinical_metrics, evaluate_cnn, per_class_table, predict_cnn,  # noqa: E402
                          results_table, build_test_report)
from src.models import (SmallCNN, WiderCNN, build_sklearn_model, count_parameters,  # noqa: E402
                        make_target_from_source)
from src.splits import make_patient_disjoint_split  # noqa: E402
from src.train import compute_class_weights, fit_cnn, learning_rate_sweep  # noqa: E402


def test_parameter_counts_match_reported_architecture():
    assert count_parameters(SmallCNN(2)) == 360_866
    assert count_parameters(WiderCNN(2)) == 1_440_578


def test_forward_shapes():
    x = torch.rand(4, 3, 27, 27)
    assert SmallCNN(2)(x).shape == (4, 2)
    assert WiderCNN(4)(x).shape == (4, 4)


def test_transfer_copies_features_and_resets_head():
    src = SmallCNN(2)
    tgt = make_target_from_source(src, num_classes=4)
    for k, v in src.features.state_dict().items():
        assert torch.equal(v, tgt.features.state_dict()[k])
    assert tgt.classifier[-1].out_features == 4


def test_augmentation_preserves_shape():
    x = torch.rand(3, 27, 27)
    assert get_augmentation()(x).shape == x.shape


def test_class_weights_formula():
    y = np.array([0] * 6 + [1] * 2 + [2] * 2)
    w = compute_class_weights(y, 3).numpy()
    np.testing.assert_allclose(w, 10 / (3 * np.array([6, 2, 2])), rtol=1e-5)


def test_binary_clinical_metrics():
    m = binary_clinical_metrics([1, 1, 0, 0, 0], [1, 0, 0, 0, 1])
    assert m["sensitivity"] == pytest.approx(0.5)
    assert m["specificity"] == pytest.approx(2 / 3)


def test_end_to_end_smoke(synthetic_dataset):
    main_df, extra_df = load_label_tables(synthetic_dataset)
    df = build_task_frame("isCancerous", main_df, extra_df)
    s = make_patient_disjoint_split(df, seed=0)
    img_dir = synthetic_dataset / "images"
    (Xtr, ytr), (Xva, yva), (Xte, yte) = [(load_images_chw(d, img_dir), d.isCancerous.values) for d in s]
    assert Xtr.shape[1:] == (3, 27, 27) and Xtr.min() >= 0 and Xtr.max() <= 1
    assert flatten_images(Xtr).shape[1] == 2187

    cfg = C.TrainConfig(epochs=2, batch_size=16)
    model, hist = fit_cnn(lambda: SmallCNN(2), Xtr, ytr, Xva, yva, cfg, augment=True, verbose=False)
    assert len(hist["val_f1"]) == 2 and hist["best_epoch"] in (1, 2)
    assert set(evaluate_cnn(model, Xtr, ytr, Xva, yva)) == {"train_f1", "val_f1", "train_acc", "val_acc"}

    yp = predict_cnn(model, Xte)
    rep = build_test_report(yte, yp, C.CANCER_NAMES, 0.90)
    assert 0.0 <= rep["macro_f1"] <= 1.0 and "sensitivity" in rep and "target_met" in rep
    assert list(per_class_table(yte, yp, C.CANCER_NAMES).index) == [*C.CANCER_NAMES, "macro avg"]


def test_same_seed_gives_identical_training(synthetic_dataset):
    main_df, extra_df = load_label_tables(synthetic_dataset)
    s = make_patient_disjoint_split(build_task_frame("isCancerous", main_df, extra_df), seed=0)
    img_dir = synthetic_dataset / "images"
    (Xtr, ytr), (Xva, yva) = [(load_images_chw(d, img_dir), d.isCancerous.values) for d in s[:2]]
    cfg = C.TrainConfig(epochs=2, batch_size=16)
    _, h1 = fit_cnn(lambda: SmallCNN(2), Xtr, ytr, Xva, yva, cfg, seed=5, verbose=False)
    _, h2 = fit_cnn(lambda: SmallCNN(2), Xtr, ytr, Xva, yva, cfg, seed=5, verbose=False)
    assert h1["train_loss"] == h2["train_loss"]


def test_lr_sweep_and_sklearn_baselines(synthetic_dataset):
    main_df, extra_df = load_label_tables(synthetic_dataset)
    s = make_patient_disjoint_split(build_task_frame("isCancerous", main_df, extra_df), seed=0)
    img_dir = synthetic_dataset / "images"
    (Xtr, ytr), (Xva, yva) = [(load_images_chw(d, img_dir), d.isCancerous.values) for d in s[:2]]

    sweep = learning_rate_sweep(lambda: SmallCNN(2), Xtr, ytr, Xva, yva, lr_grid=(1e-2, 1e-3), epochs=1,
                                base_cfg=C.TrainConfig(batch_size=16))
    assert sweep["best_lr"] in (1e-2, 1e-3)

    res = {}
    for name in ("logreg_pca", "random_forest"):
        clf = build_sklearn_model(name).fit(flatten_images(Xtr), ytr)
        res[name] = {"train_f1": 0.9, "val_f1": float((clf.predict(flatten_images(Xva)) == yva).mean())}
    table = results_table({"isCancerous": res})["isCancerous"]
    assert "gap" in table.columns


def test_frozen_trunk_does_not_change_and_head_does(synthetic_dataset):
    main_df, _ = load_label_tables(synthetic_dataset)
    s = make_patient_disjoint_split(build_task_frame("cellType", main_df, main_df.iloc[0:0]), seed=0)
    img_dir = synthetic_dataset / "images"
    (Xtr, ytr), (Xva, yva) = [(load_images_chw(d, img_dir), d.cellType.values) for d in s[:2]]

    source = SmallCNN(2)
    target = make_target_from_source(source, num_classes=4)
    trunk_before = [p.detach().clone() for p in target.features.parameters()]
    head_before = [p.detach().clone() for p in target.classifier.parameters()]

    cfg = C.TrainConfig(epochs=2, batch_size=16)
    fit_cnn(lambda: target, Xtr, ytr, Xva, yva, cfg, freeze_features=True, verbose=False)

    assert all(torch.equal(a, b) for a, b in zip(trunk_before, target.features.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(head_before, target.classifier.parameters()))


def test_restore_best_returns_best_epoch_weights(synthetic_dataset):
    main_df, extra_df = load_label_tables(synthetic_dataset)
    s = make_patient_disjoint_split(build_task_frame("isCancerous", main_df, extra_df), seed=0)
    img_dir = synthetic_dataset / "images"
    (Xtr, ytr), (Xva, yva) = [(load_images_chw(d, img_dir), d.isCancerous.values) for d in s[:2]]
    cfg = C.TrainConfig(epochs=3, batch_size=16, lr=1e-2)
    model, hist = fit_cnn(lambda: SmallCNN(2), Xtr, ytr, Xva, yva, cfg, restore_best=True, verbose=False)
    # Re-evaluating the restored weights must reproduce the best recorded validation F1
    from sklearn.metrics import f1_score
    f1 = f1_score(yva, predict_cnn(model, Xva), average="macro")
    assert f1 == pytest.approx(max(hist["val_f1"]), abs=1e-6)
