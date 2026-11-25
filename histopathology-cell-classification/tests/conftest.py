"""Shared fixtures: a tiny synthetic dataset that mimics the real label files and image folder."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory):
    """12 'main' patients (cellType + isCancerous) and 8 'extra' patients (isCancerous only)."""
    root = tmp_path_factory.mktemp("data")
    img_dir = root / "images"
    img_dir.mkdir()
    rng = np.random.default_rng(0)

    rows_main, rows_extra, idx = [], [], 0
    for pid in range(1, 21):
        is_main = pid <= 12
        for _ in range(rng.integers(8, 14)):
            idx += 1
            name = f"{idx}.png"
            if is_main:
                ctype = int(rng.integers(0, 4))
                cancer = int(ctype == 2)  # epithelial <-> cancerous, as in the real data
            else:
                cancer = int(rng.integers(0, 2))
            # Class-dependent colour so models have *something* to learn
            base = np.array([200, 100, 180]) if cancer else np.array([120, 140, 200])
            arr = np.clip(base + rng.normal(0, 25, (27, 27, 3)), 0, 255).astype("uint8")
            Image.fromarray(arr, "RGB").save(img_dir / name)
            if is_main:
                rows_main.append({"InstanceID": idx, "patientID": pid, "ImageName": name,
                                  "cellTypeName": ["fibroblast", "inflammatory", "epithelial", "others"][ctype],
                                  "cellType": ctype, "isCancerous": cancer})
            else:
                rows_extra.append({"InstanceID": idx, "patientID": pid, "ImageName": name, "isCancerous": cancer})

    pd.DataFrame(rows_main).to_csv(root / "data_labels_mainData.csv", index=False)
    pd.DataFrame(rows_extra).to_csv(root / "data_labels_extraData.csv", index=False)
    return root
