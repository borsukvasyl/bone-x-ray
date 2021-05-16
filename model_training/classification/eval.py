import glob
import os

import cv2
import numpy as np
import pandas as pd
import tqdm
from fire import Fire
from skimage.io import imread
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, classification_report

from model_training.classification.predictor import CkptClassificationPredictor
from bone_xray.utils import load_yaml


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


def make_predictions(predictor, data):
    predictions = []
    labels = []
    for study in tqdm.tqdm(data):
        study_predictions = []
        for img_path in study["images"]:
            img = imread(img_path)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            pred = predictor(img)
            study_predictions.append(pred)

        prediction = np.mean(study_predictions)
        prediction = int(prediction > 0.5)

        predictions.append(prediction)
        labels.append(study["label"])
    return np.array(predictions), np.array(labels)


def print_scores(predictions, targets, title=None):
    accuracy = accuracy_score(targets, predictions)
    kappa = cohen_kappa_score(targets, predictions)
    cf_matrix = confusion_matrix(targets, predictions)
    report = classification_report(targets, predictions)

    if title is not None:
        print("-" * 10, title, "-" * 10)
    print(f"Accuracy: {accuracy}")
    print(f"Kappa: {kappa}")
    print(f"Confusion matrix:")
    print(cf_matrix)
    print(report)


def evaluate_classification(predictor, data):
    body_parts = np.array([i["part"] for i in data])
    unique_parts = np.unique(body_parts).tolist()

    predictions, targets = make_predictions(predictor, data)
    for part in unique_parts:
        mask = body_parts == part
        print_scores(predictions[mask], targets[mask], title=part)
    print_scores(predictions, targets, title="Total")


def main(config_path: str, checkpoint_path: str, csv_path: str, prefix: str):
    config = load_yaml(config_path)
    predictor = CkptClassificationPredictor(config, checkpoint_path)
    data = parse_mura_dataset(csv_path, prefix)
    evaluate_classification(predictor, data)


if __name__ == '__main__':
    Fire(main)
