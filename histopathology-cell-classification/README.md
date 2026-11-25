# Histopathology Cell Classification

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-CNN-ee4c2c)
![scikit-learn](https://img.shields.io/badge/scikit--learn-baselines-f7931e)
![Evaluation](https://img.shields.io/badge/evaluation-patient--disjoint-success)

Two related classification tasks on **27×27 RGB H&E tiles of colon cells**, evaluated on **patients the model has never seen**:

| Task | Type | Data | Metric | Target |
|---|---|---|---|---|
| `isCancerous` | binary (cancerous vs non-cancerous) | 20,280 tiles · 98 patients | macro-F1 | ≥ 0.90 |
| `cellType` | 4-class (fibroblast, inflammatory, epithelial, others) | 9,896 tiles · 60 patients | macro-F1 | ≥ 0.60 |

The project compares classical baselines with small CNNs, runs controlled experiments (learning rate, augmentation, transfer learning, class weighting, model capacity), and reports results honestly, **including where the targets were missed and why**.

![Sample tiles per cell type](docs/figures/sample_tiles.png)

## Highlights

- **Patient-disjoint evaluation.** Splits are made with nested `GroupShuffleSplit` on `patientID` (60/20/20 by patient) and disjointness is asserted in code. A random row split would reward within-patient memorisation.
- **Macro-F1 as the primary metric**, because both targets are class-imbalanced (65/35 and 41/26/19/14 %).
- **A broad model comparison** with train/validation gaps: logistic regression, SVM-RBF, random forest, small and wider CNNs, augmentation, transfer learning (frozen and fine-tuned) and class-weighted loss.
- **Controlled experiments** with one variable changed at a time (learning-rate sweep, epochs, capacity).
- **Findings grounded in theory:** random-forest overfitting, the U-shaped learning-rate curve, augmentation narrowing the generalisation gap, and capacity not being the bottleneck.
- **Clean, tested, modular code** in `src/`, a one-command pipeline, and a narrative notebook.

## Results

### Final models on held-out test patients

| Task | Final model | Test macro-F1 | Target | Met? |
|---|---|---|---|---|
| `isCancerous` | WiderCNN + augmentation (15 epochs) | **0.854** | 0.90 | No |
| `cellType` | SmallCNN transferred from isCancerous + augmentation (12 epochs) | **0.516** | 0.60 | No |

**isCancerous (20 test patients, 3,956 tiles):** accuracy 0.878 · sensitivity 0.892 · specificity 0.874 · PPV 0.717 · NPV 0.957.

**cellType (12 test patients, 1,839 tiles):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| epithelial | 0.932 | 0.750 | 0.831 |
| inflammatory | 0.478 | 0.741 | 0.581 |
| fibroblast | 0.512 | 0.563 | 0.537 |
| others | 0.153 | 0.095 | **0.117** |

<p align="center">
  <img src="docs/figures/iscancerous_test_confusion.png" width="38%" alt="isCancerous test confusion matrix">
  <img src="docs/figures/celltype_test_results.png" width="60%" alt="cellType test confusion matrix and per-class metrics">
</p>

### Validation comparison (macro-F1 of the weights at the end of training)

| Model | isCancerous train | isCancerous val | cellType train | cellType val |
|---|---|---|---|---|
| Logistic regression + PCA(50) | 0.840 | 0.814 | n/a | n/a |
| SVM-RBF + PCA(50) | n/a | n/a | 0.833 | 0.570 |
| Random forest (150 trees, raw pixels) | 1.000 | 0.828 | 1.000 | 0.508 |
| SmallCNN | 0.899 | 0.852 | 0.675 | 0.548 |
| SmallCNN + augmentation (10 ep) | 0.888 | 0.862 | 0.635 | 0.552 |
| SmallCNN + augmentation (15 ep) | 0.894 | 0.874 | n/a | n/a |
| **WiderCNN + augmentation (15 ep)** | 0.897 | **0.876** | n/a | n/a |
| Transfer, frozen trunk | n/a | n/a | 0.736 | 0.576 |
| Transfer, fine-tune all | n/a | n/a | 0.792 | 0.624 |
| **Transfer + augmentation** | n/a | n/a | 0.740 | 0.586 |
| Transfer + augmentation + class-weighted loss | n/a | n/a | 0.734 | 0.578 |

### Key findings

1. **Patient-disjoint splits matter.** Class mix shifts between patient groups (for example, cancer rate is 37 % in train but 27 % in test), which is exactly the generalisation problem the evaluation is meant to expose.
2. **Raw pixels do not cluster by class.** k-means on PCA(50) of pixels has an Adjusted Rand Index of about 0.04 against both labels, motivating learned representations.
3. **More capacity ≠ better.** The random forest has the largest train/val gap (cellType: 1.00 → 0.51). `WiderCNN` has 4× the parameters of `SmallCNN` (1.44M vs 0.36M) and improves validation F1 by only 0.002.
4. **The learning-rate sweep shows the predicted U-shape** (best-epoch val F1: 0.802 at 1e-2, **0.852 at 1e-3**, 0.817 at 1e-4, 0.611 at 1e-5).
5. **Augmentation acts as regularisation:** training F1 drops (harder task) while validation F1 holds or rises and the generalisation gap shrinks.
6. **`isCancerous` plateaus around 0.85–0.88 across very different CNNs**, pointing to the 27×27 resolution and patient variability, not model size, as the bottleneck.
7. **`cellType` is limited by the `others` class** (test F1 0.12), a heterogeneous catch-all. Class-weighted loss raised its recall but lowered its precision and did not improve macro-F1, so the problem is visual signal rather than imbalance.

<p align="center">
  <img src="docs/figures/learning_rate_sweep.png" width="49%" alt="Learning rate sweep">
  <img src="docs/figures/iscancerous_cnn_regimes.png" width="49%" alt="isCancerous CNN regimes">
</p>
<p align="center">
  <img src="docs/figures/celltype_transfer_variants.png" width="49%" alt="cellType transfer learning variants">
  <img src="docs/figures/celltype_class_weighting.png" width="49%" alt="cellType class-weighted loss comparison">
</p>

## Notes on the results (read before citing the numbers)

These are limitations of the original experiment that this repository documents rather than hides:

- **Both accuracy targets were missed** (0.854 vs 0.90; 0.516 vs 0.60).
- **Selection vs. evaluation mismatch (cellType).** The final model was *selected* by best-epoch validation F1 (0.694 at epoch 10), but the weights *evaluated* on test are the last-epoch weights (validation F1 0.586). Against 0.586, the val-to-test drop is about 0.07, not 0.18. Validation F1 on this task also swings by 0.05–0.15 between epochs (12 validation patients), so single "best epoch" numbers are optimistic. `scripts/run_pipeline.py --restore-best` makes selection and evaluation consistent.
- **Possible transfer-source leakage.** The two tasks are split independently, so patients in the `isCancerous` *training* set can also be `cellType` *validation/test* patients. Since cancer status equals "epithelial" in the main data, the transferred trunk may have seen labels of held-out patients. The pipeline reports this overlap and `--leakage-safe` removes those patients from the source model's training data. The headline numbers above were produced **without** that flag; a leakage-safe re-run is on the roadmap.
- **Single split, single seed, no confidence intervals.** With 12 test patients for `cellType`, one or two atypical patients move the metric noticeably.
- **Single-institution data, no external validation.** This is a research exercise, not a diagnostic tool. In screening, a missed cancer costs far more than a false alarm, so a deployed operating point should favour sensitivity.

## Repository structure

```
histopathology-cell-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── pytest.ini
├── notebooks/
│   └── showcase.ipynb          # narrative: EDA → splits → baselines → CNNs → advanced techniques → test
├── src/
│   ├── config.py               # paths, task specs, default hyperparameters, final recipes
│   ├── data.py                 # label loading, image loading, Dataset / DataLoader, augmentation
│   ├── splits.py               # patient-disjoint splitting, overlap checks, split summaries
│   ├── models.py               # sklearn baselines, SmallCNN, WiderCNN, transfer-learning helper
│   ├── train.py                # CNN training loop, LR sweep, class weights, checkpoints
│   ├── evaluate.py             # metrics, per-class tables, clinical metrics, plots, prediction export
│   └── utils.py                # seeding, device, JSON helper
├── scripts/
│   └── run_pipeline.py         # end-to-end CLI: splits → train final models → test evaluation
├── tests/                      # pytest suite on a tiny synthetic dataset (no real data needed)
├── docs/figures/               # figures used in this README
├── data/                       # (git-ignored) dataset goes here, see data/README.md
├── splits/                     # (git-ignored) generated patient-disjoint split CSVs
├── models/                     # (git-ignored) saved weights
└── outputs/                    # (git-ignored) predictions and metrics
```

## Quick start

```bash
git clone https://github.com/<your-username>/histopathology-cell-classification.git
cd histopathology-cell-classification

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**1. Add the data** following [`data/README.md`](data/README.md).

**2. Run the tests** (they use synthetic data, so no dataset is needed):

```bash
pytest
```

**3. Explore the notebook:**

```bash
jupyter lab notebooks/showcase.ipynb      # set QUICK = True in the first code cell for a ~2 min smoke run
```

**4. Or reproduce the final models from the command line:**

```bash
python scripts/run_pipeline.py --task both                       # as originally reported (~10 min on CPU)
python scripts/run_pipeline.py --task both --leakage-safe --restore-best   # stricter protocol
python scripts/run_pipeline.py --task isCancerous --save-models  # also write weights to models/
python scripts/run_pipeline.py --task both --quick               # 1 epoch, 10% of patients
```

Outputs: `outputs/metrics.json`, `outputs/test_predictions_*.csv`, `splits/*.csv`, and optionally `models/*.pt`, all git-ignored.

## Method summary

| Component | Details |
|---|---|
| Splits | Nested `GroupShuffleSplit` on `patientID`; 60/20/20 by patient, per task; `SEED = 42` |
| Input | 3×27×27 tensors scaled to [0, 1] |
| Baselines | StandardScaler → PCA(50) → logistic regression / SVM-RBF; random forest on raw pixels |
| CNN | 2 double-conv blocks (3→32→64), 3×3 kernels, max-pool, 128-unit head, dropout 0.3 (361k params). `WiderCNN`: 3→64→128, 256-unit head (1.44M) |
| Optimiser | Adam, lr 1e-3 (chosen by sweep), weight decay 1e-4, batch 128 |
| Augmentation | Horizontal + vertical flip, rotation ±15°, training set only |
| Transfer | Conv trunk of the isCancerous CNN copied into the cellType CNN; head re-initialised; frozen vs fine-tuned compared |
| Class weights | `w_c = N / (K · n_c)` in `CrossEntropyLoss` |
| Metric | Macro-F1 (primary), accuracy, per-class P/R/F1, sensitivity/specificity for isCancerous |

## Roadmap

- [ ] Re-run cellType with `--leakage-safe --restore-best` and report the difference
- [ ] Grouped k-fold over patients with confidence intervals and multiple seeds
- [ ] Colour/stain augmentation and higher-resolution inputs
- [ ] Split the `others` class into homogeneous sub-classes
- [ ] Decision-threshold tuning for sensitivity on isCancerous

## Author

**Huu Duc Anh Le**. Originally developed for a university machine learning course (COSC2793). The dataset is not redistributed here.
