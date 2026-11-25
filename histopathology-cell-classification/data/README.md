# Data

The dataset is **not** included in this repository (it is git-ignored). Place it here with this layout:

```
data/
├── data_labels_mainData.csv     # columns: InstanceID, patientID, ImageName, cellTypeName, cellType, isCancerous
├── data_labels_extraData.csv    # columns: InstanceID, patientID, ImageName, isCancerous
└── images/
    ├── 1.png
    ├── 2.png
    └── ...                      # 27x27 RGB H&E tiles of colon cells
```

| File | Patients | Images | Labels |
|---|---|---|---|
| `data_labels_mainData.csv` | 60 | 9,896 | `isCancerous` + `cellType` |
| `data_labels_extraData.csv` | 38 | 10,384 | `isCancerous` only |

`cellType` encoding: `0` fibroblast, `1` inflammatory, `2` epithelial, `3` others.
`isCancerous` encoding: `0` non-cancerous, `1` cancerous.

The two label files share no patients (checked in code), which is what allows them to be combined for the `isCancerous` task without patient-level leakage.

The data was supplied as part of a university course assessment. Check the licence and terms attached to your copy before redistributing it.
