import albumentations as albu
import numpy as np
import torch
from pytorch_toolbelt.utils import image_to_tensor

from model_training.classification.model import ClassificationLightningModel
from model_training.common.models import get_model


class ClassificationPredictor:
    def __init__(self, config, checkpoint_path):
        model_config = config["model"]
        model = get_model(model_config["name"], model_config)

        self.pl_model = ClassificationLightningModel.load_from_checkpoint(
            checkpoint_path,
            model=model,
            config=config
        )
        self.pl_model = self.pl_model.eval().cuda()

        self.transform = albu.Compose([
            albu.Resize(config["img_size"], config["img_size"]),
            albu.Normalize(mean=0.5, std=0.5)
        ])

    def __call__(self, img):
        img = self.transform(image=img)["image"]
        img = image_to_tensor(img)[None, ...]

        with torch.no_grad():
            pred = self.pl_model(img.cuda()).sigmoid().cpu()
            pred = np.squeeze(pred.numpy())
        return pred
