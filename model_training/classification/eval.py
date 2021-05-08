import os
from collections import defaultdict

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
import torch
import tqdm
from fire import Fire
from pytorch_toolbelt.utils import image_to_tensor
from skimage.io import imread
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix

from model_training.classification.model import ClassificationLightningModel
from model_training.common.models import get_model
from model_training.common.utils import load_yaml


def iterate_dataset(prefix, data):
    for _, row in data.iterrows():
        image_path = os.path.join(prefix, row.file_path)
        img = imread(image_path)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        label = int("_positive/" in image_path)
        study = "/".join(image_path.split("/")[-3:-1])
        yield img, label, study


def make_predictions(model, prefix, data):
    transform = albu.Compose([
        albu.Resize(384, 384),
        albu.Normalize(mean=0.5, std=0.5)
    ])

    model_predictions = defaultdict(list)
    target_labels = dict()
    for img, label, study in tqdm.tqdm(iterate_dataset(prefix, data), total=len(data)):
        img = transform(image=img)["image"]
        img = image_to_tensor(img)[None, ...]
        with torch.no_grad():
            pred = model(img.cuda()).sigmoid().cpu()
            pred = np.squeeze(pred.numpy())
        model_predictions[study].append(pred)
        target_labels[study] = label

    predictions = []
    labels = []
    for s in model_predictions.keys():
        predictions.append(np.mean(model_predictions[s]))
        labels.append(target_labels[s])
    return np.array(predictions), np.array(labels)


def evaluate_classification(pl_model, prefix, data):
    predictions, targets = make_predictions(pl_model, prefix, data)

    predicted_classes = (predictions > 0.5).astype(np.int)
    targets_classes = targets.astype(np.int)

    accuracy = accuracy_score(targets_classes, predicted_classes)
    kappa = cohen_kappa_score(targets_classes, predicted_classes)
    cf_matrix = confusion_matrix(targets_classes, predicted_classes)

    print(f"Accuracy: {accuracy}")
    print(f"Kappa: {kappa}")
    print(f"Confusion matrix:")
    print(cf_matrix)


def main(config_path: str, checkpoint_path: str, prefix: str, csv_path: str):
    config = load_yaml(config_path)
    model_config = config["model"]
    model = get_model(model_config["name"], model_config)

    pl_model = ClassificationLightningModel.load_from_checkpoint(
        checkpoint_path,
        model=model,
        config=config
    )
    pl_model = pl_model.eval().cuda()

    data = pd.read_csv(csv_path, header=None, names=["file_path"])
    evaluate_classification(pl_model, prefix, data)


if __name__ == '__main__':
    Fire(main)
