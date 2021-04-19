import numpy as np
import torch
from fire import Fire
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix

from model_training.classification.data import BonesDataModule
from model_training.classification.model import ClassificationLightningModel
from model_training.common.models import get_model
from model_training.common.utils import load_yaml


def make_predictions(model, pl_dataset):
    model_predictions = []
    target_labels = []
    for batch in pl_dataset.val_dataloader():
        imgs, targets = batch
        imgs = imgs.cuda()
        with torch.no_grad():
            preds = model(imgs).sigmoid().cpu().numpy()
        model_predictions.append(preds)
        target_labels.append(targets)
    model_predictions = np.vstack(model_predictions)
    target_labels = np.vstack(target_labels)
    return model_predictions, target_labels


def evaluate_classification(pl_model, pl_dataset):
    predictions, targets = make_predictions(pl_model, pl_dataset)

    predicted_classes = np.squeeze(predictions > 0.5).astype(np.int)
    targets_classes = np.squeeze(targets).astype(np.int)

    accuracy = accuracy_score(targets_classes, predicted_classes)
    kappa = cohen_kappa_score(targets_classes, predicted_classes)
    cf_matrix = confusion_matrix(targets_classes, predicted_classes)

    print(f"Accuracy: {accuracy}")
    print(f"Kappa: {kappa}")
    print("Confusion matrix:")
    print(cf_matrix)


def main(config_path: str, checkpoint_path: str):
    config = load_yaml(config_path)
    model_config = config["model"]
    model = get_model(model_config["name"], model_config)
    pl_model = ClassificationLightningModel.load_from_checkpoint(
        checkpoint_path,
        model=model,
        config=config
    )
    pl_model = pl_model.eval().cuda()

    pl_dataset = BonesDataModule(config=config)
    pl_dataset.setup()

    evaluate_classification(pl_model, pl_dataset)


if __name__ == '__main__':
    Fire(main)
