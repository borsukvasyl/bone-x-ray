import glob
import os

import pandas as pd


def parse_mura_dataset(dataset_labels_path: str, prefix: str):
    data = pd.read_csv(dataset_labels_path, header=None, names=["study", "label"])
    result = []
    for _, row in data.iterrows():
        images = glob.glob(os.path.join(prefix, row.study, "*"))
        part = row.study.split(os.path.sep)[2]
        parsed = {
            "images": images,
            "study": row.study,
            "label": row.label,
            "part": part,
        }
        result.append(parsed)
    return result
