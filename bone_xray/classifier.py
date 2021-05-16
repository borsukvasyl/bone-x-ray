from typing import Dict

from bone_xray.base import BaseClassificationPredictor
from bone_xray.models import get_model


class ClassificationPredictor(BaseClassificationPredictor):
    def __init__(self, config: Dict, checkpoint_path: str):
        model_config = config["model"]
        model = get_model(model_config["name"], model_config, model_weights=checkpoint_path)
        super().__init__(model, config["img_size"])
