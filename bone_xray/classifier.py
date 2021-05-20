from typing import Dict

import torch

from bone_xray.base import BasePredictor, to_device
from bone_xray.models import get_model


class ClassificationPredictor(BasePredictor):
    def __init__(self, config: Dict, checkpoint_path: str):
        model_config = config["model"]
        model = get_model(model_config["name"], model_config, model_weights=checkpoint_path)
        super().__init__(model, config["img_size"])

    def _process(self, x: torch.Tensor):
        with torch.no_grad():
            x = to_device(x)
            pred = self.model(x).softmax(1).cpu()
            pred = pred[0, 1]  # probability of positive label
        return pred
