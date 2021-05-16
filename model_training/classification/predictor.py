from typing import Dict

from bone_xray.base import BaseClassificationPredictor, BaseLocalizationPredictor
from model_training.common.models import get_model


class CkptClassificationPredictor(BaseClassificationPredictor):
    def __init__(self, config: Dict, checkpoint_path: str):
        model_config = config["model"]
        model = get_model(model_config["name"], model_config, model_weights=checkpoint_path)
        super().__init__(model, config["img_size"])


class CkptLocalizationPredictor(BaseLocalizationPredictor):
    def __init__(self, config: Dict, checkpoint_path: str):
        model_config = config["model"]
        model = get_model(model_config["name"], model_config, model_weights=checkpoint_path)
        cam_layer = self.model.model.backbone[4].unit16.conv2.conv
        super().__init__(model, cam_layer, config["img_size"])
