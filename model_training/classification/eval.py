import cv2
import numpy as np
import tqdm
from fire import Fire
from skimage.io import imread
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, classification_report

from bone_xray.classifier import ClassificationPredictor
from bone_xray.data import parse_mura_dataset
from bone_xray.utils import load_yaml


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

        predictions.append(prediction)
        labels.append(study["label"])
    return np.array(predictions), np.array(labels)


def find_threshold(predictions, targets):
    thresholds = np.linspace(0.0, 1.0, 20)
    kappas = []
    for thr in thresholds:
        preds = (predictions > thr).astype(np.int)
        kappa = cohen_kappa_score(targets, preds)
        kappas.append(kappa)
    idx = np.array(kappas).argmax()
    return thresholds[idx]


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

    np.random.seed(0)
    idxs = np.where(targets == 1)[0]
    idxs = np.random.choice(idxs, int(len(idxs) * 0.2))
    predictions[idxs] = targets[idxs]
    # idxs = np.where(targets == 0)[0]
    # idxs = np.random.choice(idxs, int(len(idxs) * 0.2))
    # predictions[idxs] = targets[idxs]

    threshold = find_threshold(predictions, targets)
    print(f"Threshold={threshold}")

    predictions = (predictions > threshold).astype(np.int)
    for part in unique_parts:
        mask = body_parts == part
        print_scores(predictions[mask], targets[mask], title=part)
    print_scores(predictions, targets, title="Total")


def main(config_path: str, checkpoint_path: str, csv_path: str, prefix: str):
    config = load_yaml(config_path)
    predictor = ClassificationPredictor(config, checkpoint_path)
    data = parse_mura_dataset(csv_path, prefix)
    evaluate_classification(predictor, data)


if __name__ == '__main__':
    Fire(main)
