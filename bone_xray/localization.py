from typing import Dict

import cv2
import numpy as np
import torch

from bone_xray.base import BasePredictor, to_device
from bone_xray.models import get_model
from bone_xray.models.cam import ScoreCAM


def visualize_heatmap(img: np.ndarray,
                      mask: np.ndarray,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    shape = img.shape[:-1][::-1]
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, dsize=shape, interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, dsize=shape, interpolation=cv2.INTER_CUBIC)
    return (img * (1 - mask[..., np.newaxis]) + heatmap * mask[..., np.newaxis]).astype(np.uint8)


class LocalizationPredictor(BasePredictor):
    def __init__(self, config: Dict, checkpoint_path: str):
        model_config = config["model"]
        model = get_model(model_config["name"], model_config, model_weights=checkpoint_path)
        cam_layer = model.backbone[4].unit16.conv2.conv
        cam_model = ScoreCAM(model, cam_layer)
        super().__init__(cam_model, config["img_size"])

    def _process(self, x: torch.Tensor):
        with torch.no_grad():
            x = to_device(x)
            cam, _ = self.model.forward(x, idx=1)
        return cam.squeeze().numpy()
